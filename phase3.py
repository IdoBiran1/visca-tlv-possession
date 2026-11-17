"""Phase 3: learned evidence. A gradient-boosted classifier gives P(H), P(A), P(D) per frame; Phase 1's decoder turns them into labels.

    model, dec, _ = fit(train, score)                       # train: name -> cleaned match (IDSSE), labels used
    labels = predict(c, model, dec)

Training: every 5th labelled frame of the training matches, plus its mirror (teams swapped, pitch
turned 180 degrees), so the model can't learn a home/away bias. The decoder's settings are tuned on
out-of-fold probabilities (the training matches split in two, each half predicted by a model fitted on
the other), so they are not fitted to a model's over-confident view of its own training data.

Two options close the gap between IDSSE and our tracker (docs/2026-10-22-phase3-report.html):
  drop=BALL_CUES  train without the ball cues that only IDSSE shows: its dead ball is tracked lying still,
                  often just outside the lines, while trackers like ours lose it
  blend=True      mix Phase 1's speed-based dead-ball log-odds into the model's with weight w, tuned together
                  with the decoder; `extra` matches (Metrica) join the tuning objective
"""
import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

import features as fe
import phase1

STEP = 5
CLASSES = np.array(["H", "A", "D"])
PAIRS = ["d", "within2", "within5", "n", "cx", "length", "width", "line", "advance", "speed", "icpt_t"]   # *_h / *_a
CONTEXT = {"d_margin": "mean", "last_touch_team": "mean", "ctrl_team": "mean", "ball_seen": "mean", "speed_all_2s": "mean",
           "slow_share": "mean", "ball_out_m": "max", "ball_speed": "mean"}
SIGNED = ["d_margin", "last_touch_team", "ctrl_team", "last_ctrl_team", "next_ctrl_team", "cc_team", "cc_score", "icpt_margin"]   # +1 home / -1 away
BALL_CUES = ("ball_speed", "ball_out_m", "ball_fast")      # dead-ball AUC IDSSE 0.004 / 0.83 vs Metrica 0.43 / 0.52
MODEL = dict(max_iter=200, learning_rate=0.1, max_leaf_nodes=31, min_samples_leaf=100, l2_regularization=1.0,
             early_stopping=False, random_state=0)
DEC_GRID = {"tau": [1.0, 2.0, 3.0], "c_switch": [6.0, 9.0, 12.0, 16.0], "c_to_dead": [9.0, 12.0, 16.0, 24.0],
            "c_from_dead": [4.0, 8.0, 12.0, 16.0], "min_poss_s": [0.2, 0.4, 1.0], "min_dead_s": [0.6, 1.0, 2.0, 4.0]}
BLEND_GRID = {"w": [0.0, 0.25, 0.5, 0.75], "v0": [1.2, 1.5, 1.8], "k_speed": [0.75, 1.5, 3.0]}
DEC_DEFAULT = {"tau": 2.0, "c_switch": 9.0, "c_to_dead": 16.0, "c_from_dead": 12.0, "min_poss_s": 0.4, "min_dead_s": 1.0,
               "w": 0.0, "v0": 1.5, "k_speed": 1.5}
TUNE_SMOOTH = False   # tune: pick by the mean objective of a setting and its one-step grid neighbours, so retraining doesn't flip the pick
TUNE_ON = ()       # name prefixes of the training matches the decoder is tuned on, e.g. ("idsse_",) for a pooled predictor
                   # (docs/2026-10-24-model-averaging.md); (): all of them
POSS_ALPHA = 0.0   # train each frame with weight (its spell's length)^-POSS_ALPHA; 1: every possession counts equally, as in spellF1
_MODELS = {}   # (drop, training names) -> fitted model, shared by every predictor in one process
AUG = {}       # training set name -> the same match under other corruption draws, fitted on beside it (validate.AUG_DRAWS)
_AUG_FEATS = {}


def with_aug(n, c, F):
    """[(c, F)] for training match n, plus each of its extra corruption draws with their frame and raw features (computed
    once). The draws follow their match, so a match left out of a fit leaves out its draws too."""
    for x in AUG.get(n, ()):
        if id(x) not in _AUG_FEATS:
            _AUG_FEATS[id(x)] = (x, fe.frame_features(x).join(fe.raw_features(x)))
    return [(c, F)] + [_AUG_FEATS[id(x)] for x in AUG.get(n, ())]


def with_context(F, fps=25, drop=()):
    """Frame features plus centred rolling summaries over 1 s and 4 s (offline: the future is fair game)."""
    out = {c: F[c] for c in F.columns if c != "forced_dead" and not c.startswith("raw_")}   # raw_: phase4's raw member only
    out["within2_diff"] = F.within2_h - F.within2_a
    for col, how in CONTEXT.items():
        for s in (1, 4):
            r = F[col].rolling(int(s * fps) | 1, center=True, min_periods=1)
            out["{}_{}{}s".format(col, how, s)] = r.max() if how == "max" else r.mean()
    X = pd.DataFrame(out).astype("float32")
    return X[[c for c in X.columns if not c.startswith(tuple(drop))]] if drop else X


def mirror(F, length, width):
    """The same frames with home and away swapped and the pitch turned 180 degrees (home still attacks +x)."""
    G = F.copy()
    for p in PAIRS:
        if p + "_h" in F:   # icpt_t: only with features.INTERCEPT
            G[p + "_h"], G[p + "_a"] = F[p + "_a"].values, F[p + "_h"].values
    G["cy_h"], G["cy_a"] = width - F.cy_a.values, width - F.cy_h.values
    for col in SIGNED:
        if col in F:   # cc_ and icpt_: only with features.CONTROL_CHANGE and features.INTERCEPT
            G[col] = -F[col]
    G["ball_x"], G["ball_y"] = length - F.ball_x, width - F.ball_y
    raw = [c for c in F.columns if c.startswith("raw_")]
    if raw:   # features.raw_features: offsets, velocities and sides turn; distances and ball_seen stay
        G[raw] = -F[raw]
        keep = [c for c in raw if c.endswith("_d") or c == "raw_ball_seen"]
        G[keep] = F[keep]
        G["raw_ball_x"], G["raw_ball_y"] = length - F.raw_ball_x, width - F.raw_ball_y
    return G


def _rows(c):
    keep = c.graded & np.isin(c.labels, CLASSES)
    keep[np.arange(len(keep)) % STEP != 0] = False
    return keep


def spell_weights(labels, fps, alpha):
    """Per frame: (seconds in its run of identical labels)^-alpha; 1 everywhere when alpha is 0."""
    w = np.ones(len(labels))
    if alpha:
        edges = np.r_[0, np.flatnonzero(labels[1:] != labels[:-1]) + 1, len(labels)]
        for a, b in zip(edges[:-1], edges[1:]):
            w[a:b] = ((b - a) / fps) ** -alpha
    return w


PREBIN = True   # trees_fit: weighted fits see unweighted quantile codes, as sklearn 1.9 bins weighted data one percentile at a
                # time. tell_apart 1.4x faster; fit_model doesn't use it (no faster on its 1.9M rows, and it keeps the baselines)


def bin_edges(X, n_bins=255, rows=200_000):
    """Column -> unweighted quantile edges (n_bins - 1 at most) from a fixed sample of `rows` rows, NaN ignored."""
    k = np.sort(np.random.default_rng(0).choice(len(X), rows, replace=False)) if len(X) > rows else slice(None)
    qs = np.linspace(0, 1, n_bins + 1)[1:-1]
    edges = {}
    for col in X.columns:
        v = X[col].to_numpy(np.float64)[k]
        v = v[~np.isnan(v)]
        edges[col] = np.unique(np.quantile(v, qs)) if len(v) else np.array([])
    return edges


def prebin(X, edges):
    """X with each value replaced by its bin's index (NaN stays NaN): at most 255 values a column, so the trees'
    own binning keeps them as they are and skips the weighted percentiles."""
    out = {}
    for col in X.columns:
        v = X[col].to_numpy(np.float64)
        out[col] = np.where(np.isnan(v), np.nan, np.searchsorted(edges[col], v)).astype(np.float32)
    return pd.DataFrame(out, index=X.index)


def trees_fit(X, y, w=None, **params):
    """HistGradientBoostingClassifier(**params) on X; a weighted fit goes through prebin (when PREBIN)."""
    edges = bin_edges(X) if w is not None and PREBIN else None
    model = HistGradientBoostingClassifier(**params).fit(X if edges is None else prebin(X, edges), y, sample_weight=w)
    model.prebin_edges = edges
    return model


def trees_proba(model, X):
    """predict_proba on X binned the way the model was fitted (log_probs uses it, so model_screen's trees_fit models work)."""
    edges = getattr(model, "prebin_edges", None)
    return model.predict_proba(X if edges is None else prebin(X, edges))


def fit_model(train, feats, drop=()):
    key = (tuple(drop), tuple(sorted((n, id(c)) for n, c in train.items())), POSS_ALPHA)   # id: clean and degraded copies differ
    if key in _MODELS:
        return _MODELS[key]
    X, y, w = [], [], []
    for n, c0 in train.items():
        for c, F in with_aug(n, c0, feats[n]):
            k = _rows(c)
            X += [with_context(F, drop=drop)[k], with_context(mirror(F, c.length, c.width), drop=drop)[k]]
            lab = c.labels[k]
            y += [lab, np.where(lab == "H", "A", np.where(lab == "A", "H", lab))]
            w += [spell_weights(c.labels, c.fps, POSS_ALPHA)[k]] * 2
    w = np.concatenate(w)
    model = HistGradientBoostingClassifier(**MODEL)
    model.fit(pd.concat(X, ignore_index=True), np.concatenate(y), sample_weight=w / w.mean() if POSS_ALPHA else None)
    _MODELS[key] = model
    return model


def log_probs(model, F, drop=()):
    """[T, 3] log P(H), P(A), P(D) per frame."""
    P = trees_proba(model, with_context(F, drop=drop))
    order = [list(model.classes_).index(k) for k in CLASSES]
    return np.log(np.clip(P[:, order], 1e-4, 1.0))


def blend(logp, F, d):
    """Mix Phase 1's speed-based dead-ball log-odds into the model's: logit(D) = (1-w)*model + w*Phase 1."""
    if not d.get("w"):
        return logp
    p = np.exp(logp)
    pd_m = np.clip(p[:, 2], 1e-4, 1 - 1e-4)
    logit_m = np.log(pd_m / (1 - pd_m))
    p1 = dict(phase1.DEFAULTS, v0=d["v0"], k_speed=d["k_speed"])
    pd_1 = np.clip(np.exp(phase1.evidence(F, p1)[:, 2]), 1e-4, 1 - 1e-4)
    logit_1 = np.log(pd_1 / (1 - pd_1))
    p_dead = 1 / (1 + np.exp(-((1 - d["w"]) * logit_m + d["w"] * logit_1)))
    home = p[:, 0] / np.clip(p[:, 0] + p[:, 1], 1e-8, None)
    out = np.stack([(1 - p_dead) * home, (1 - p_dead) * (1 - home), p_dead], axis=1)
    return np.log(np.clip(out, 1e-4, 1.0))


def decode(c, logp, F, d):
    p = dict(phase1.DEFAULTS, **{k: v for k, v in d.items() if k in phase1.DEFAULTS})
    return phase1.decode(c, blend(logp, F, d) * d["tau"], p)


def fit(train, score, drop=(), use_blend=False, extra=None, log=None, backend=None):
    """(model fitted on all of `train`, decoder (and blend) settings, tuning score).

    Settings are tuned on out-of-fold probabilities for `train`, plus `extra` matches (never trained on).
    backend: a module or class with fit_model / log_probs like this one's (phase4: a TCN); default the trees."""
    be = backend or sys.modules[__name__]
    extra = extra or {}
    feats = {n: getattr(be, "features", fe.frame_features)(c) for n, c in {**train, **extra}.items()}
    names = sorted(train)
    halves = [names[0::2], names[1::2]]
    logp = {}
    for a, b in ((0, 1), (1, 0)):
        m = be.fit_model({n: train[n] for n in halves[a]}, feats, drop)
        for n in halves[b]:
            logp[n] = be.log_probs(m, feats[n], drop)
    full = be.fit_model(train, feats, drop)
    for n in extra:
        logp[n] = be.log_probs(full, feats[n], drop)
    every = {**train, **extra}
    tuned = [n for n in logp if n in extra or not TUNE_ON or n.startswith(TUNE_ON)]
    objective = lambda d: float(np.mean([score(every[n].labels, decode(every[n], logp[n], feats[n], d))["spellF1"] for n in tuned]))
    best, best_score = tune(objective, dict(DEC_GRID, **(BLEND_GRID if use_blend else {})), dict(DEC_DEFAULT, w=0.5 if use_blend else 0.0))
    if log:
        log("  tuned on {} matches ({} out-of-fold, {} extra): spellF1 {:.4f} {}".format(len(tuned), len(train), len(extra), best_score, best))
    return full, best, best_score


def tune(objective, grid, best):
    """Coordinate descent over `grid` from settings `best`, two passes at most: (best settings, their objective)."""
    objective = smoothed(objective, grid) if TUNE_SMOOTH else objective
    best_score = objective(best)
    for _ in range(2):
        improved = False
        for key, values in grid.items():
            for v in values:
                if v == best[key]:
                    continue
                trial = dict(best, **{key: v}); s = objective(trial)
                if s > best_score + 1e-4:
                    best, best_score, improved = trial, s, True
        if not improved:
            break
    return best, best_score


def smoothed(objective, grid):
    """`objective` averaged over a setting and its one-step neighbours along each axis of `grid`, each setting evaluated once."""
    seen = {}

    def raw(d):
        k = tuple(sorted(d.items()))
        if k not in seen:
            seen[k] = objective(d)
        return seen[k]

    def mean_near(d):
        near = [d] + [dict(d, **{key: vals[j]}) for key, vals in grid.items() if d[key] in vals
                      for j in (vals.index(d[key]) - 1, vals.index(d[key]) + 1) if 0 <= j < len(vals)]
        return float(np.mean([raw(x) for x in near]))
    return mean_near


def predict(c, model, d, F=None, drop=(), backend=None):
    be = backend or sys.modules[__name__]
    F = getattr(be, "features", fe.frame_features)(c) if F is None else F
    return decode(c, be.log_probs(model, F, drop), F, d)
