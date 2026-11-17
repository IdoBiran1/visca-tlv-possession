"""Phase 1: rule-based evidence per frame, then a duration-aware Viterbi decoder.

    labels = predict(clean_match(...))                 # "H" / "A" / "D" / "X" per frame
    validate.py runs it as the predictor `phase1`.

Evidence (log-odds, hand-built from features.py):
  dead ball   slow players (2 s mean speed), the last ball event being a ball-out rather than a restart,
              the ball unseen; pushed towards alive by a recent touch; forced D by rule
  team        the team that last touched the ball, and the nearest player's team when the ball is seen
Decoder: states H, A, D, each a chain of sub-states so a spell lasts at least its minimum duration;
switching team, stopping play and restarting each cost a fixed log-penalty. Decoded at 5 Hz.
Optionally X through live ball gaps, so the scorer carries the current team.
Optionally, on our match only, evidence files out/evidence/<name>.npz (lane C) added to the per-frame
log-probabilities before decoding, one weight per file (docs/2026-10-23-decoder-evidence-hook.md).
"""
import os

import numpy as np

import features as fe

STEP = 5                     # decode every 5 frames (0.2 s at 25 fps)
DEFAULTS = {
    # dead-ball log-odds
    "v0": 1.5,               # m/s: below this the players look stopped
    "k_speed": 3.0,          # log-odds per m/s below v0
    "k_out": 2.0,            # the last ball event is a ball-out, with no restart since
    "k_unseen": 0.5,         # the ball isn't seen
    "k_touch_live": 1.5,     # a touch in the last second argues for live play
    # team log-odds (+ = home)
    "k_touch": 1.5,          # last_touch_team (+1 / -1)
    "k_near": 1.5,           # nearest-player margin (tanh of metres / 2), when the ball is seen within 3 m
    # decoder
    "min_poss_s": 1.0,
    "min_dead_s": 2.0,
    "c_switch": 6.0,         # log-penalty for H <-> A
    "c_to_dead": 6.0,        # alive -> D
    "c_from_dead": 4.0,      # D -> alive
    "x_live_gaps": 0,        # 1: write X where the ball is unseen and play is alive
    "evidence": {},          # our match only: {file name: weight}, e.g. {"broadcast": 1.0}; off when empty
}
EVIDENCE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out", "evidence")
OURS = "BAR-ATH (ours)"      # the only match with evidence files


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def evidence(F, p):
    """Per-frame log-probabilities [T, 3] for H, A, D from a features.frame_features table."""
    v = F.speed_all_2s.to_numpy(float)
    seen = F.ball_seen.to_numpy(float) > 0
    last_out = (F.t_since_ball_out_s < F.t_since_restart_s).to_numpy() & (F.t_since_ball_out_s < fe.CAP_S).to_numpy()
    recent_touch = F.t_since_touch_s.to_numpy(float) < 1.0
    dead_logit = (p["k_speed"] * np.nan_to_num(p["v0"] - v, nan=0.0) + p["k_out"] * last_out
                  + p["k_unseen"] * ~seen - p["k_touch_live"] * recent_touch)
    p_dead = _sigmoid(dead_logit)
    p_dead = np.where(F.forced_dead.to_numpy() > 0, 1.0, p_dead)
    margin = F.d_margin.to_numpy(float)
    near_ok = seen & (np.fmin(F.d_h.to_numpy(float), F.d_a.to_numpy(float)) < 3)
    team_logit = p["k_touch"] * F.last_touch_team.to_numpy(float) + p["k_near"] * np.where(near_ok, np.tanh(np.nan_to_num(margin) / 2), 0.0)
    p_home = _sigmoid(team_logit)
    eps = 1e-4
    probs = np.stack([(1 - p_dead) * p_home, (1 - p_dead) * (1 - p_home), p_dead], axis=1)
    return np.log(np.clip(probs, eps, 1.0))


def _transitions(p, fps):
    """Dense log-transition matrix over chained sub-states; returns (matrix, state -> label index)."""
    m_poss = max(1, int(round(p["min_poss_s"] * fps / STEP)))
    m_dead = max(1, int(round(p["min_dead_s"] * fps / STEP)))
    sizes = [m_poss, m_poss, m_dead]
    first = np.cumsum([0] + sizes[:-1]); last = first + np.array(sizes) - 1
    S = sum(sizes)
    label = np.repeat([0, 1, 2], sizes)
    T = np.full((S, S), -np.inf)
    for k in range(3):
        for i in range(first[k], last[k]):
            T[i, i + 1] = 0.0                          # inside a chain: forced forward
        T[last[k], last[k]] = 0.0                      # stay
    cost = {(0, 1): p["c_switch"], (1, 0): p["c_switch"], (0, 2): p["c_to_dead"], (1, 2): p["c_to_dead"],
            (2, 0): p["c_from_dead"], (2, 1): p["c_from_dead"]}
    for (a, b), c in cost.items():
        T[last[a], first[b]] = -c
    return T, label, first


def viterbi(logp, T, label, first, bonus=None):
    """Most likely label sequence for one period. logp: [n, 3] per decode step.
    bonus: optional [n] log-bonus per step on both team switches (H -> A and A -> H), or [n, 2] on switches to H and
    to A apart; None leaves T as it is."""
    n, S = len(logp), len(label)
    emit = logp[:, label]                              # [n, S]
    delta = np.full(S, -np.inf); delta[first] = emit[0, first]
    back = np.zeros((n, S), np.int16)
    for t in range(1, n):
        scores = delta[:, None] + T                    # [from, to]
        if bonus is not None:
            to_h, to_a = bonus[t] if bonus.ndim == 2 else (bonus[t], bonus[t])
            scores[first[1] - 1, first[1]] += to_a   # last H sub-state -> first A
            scores[first[2] - 1, first[0]] += to_h   # last A sub-state -> first H
        back[t] = np.argmax(scores, axis=0)
        delta = scores[back[t], np.arange(S)] + emit[t]
    path = np.empty(n, np.int64); path[-1] = int(np.argmax(delta))
    for t in range(n - 1, 0, -1):
        path[t - 1] = back[t, path[t]]
    return label[path]


GRID = {   # second round: ranges widened where the first round's best sat on an edge
    "v0": [1.2, 1.5, 1.8], "k_speed": [0.75, 1.5, 3.0, 5.0], "k_out": [1.0, 2.0, 3.0, 5.0, 8.0], "k_unseen": [0.0, 0.5, 1.0],
    "k_touch_live": [1.5, 3.0, 5.0, 8.0], "k_touch": [0.75, 1.5, 3.0], "k_near": [0.75, 1.5, 3.0],
    "min_poss_s": [0.2, 0.4, 1.0], "min_dead_s": [1.0, 2.0, 4.0],
    "c_switch": [6.0, 9.0, 12.0, 16.0, 24.0], "c_to_dead": [6.0, 9.0, 12.0, 16.0, 24.0], "c_from_dead": [4.0, 8.0, 12.0, 16.0, 24.0],
}


def fit(train, score, passes=2, log=None):
    """Coordinate descent over GRID, maximising the mean spellF1 over `train` (name -> cleaned match).

    `score(gt, pred)` returns a dict with "spellF1" (validate.score). Features are computed once per match.
    Labels of the matches being predicted are never seen: callers pass only training matches.
    """
    feats = {n: fe.frame_features(c) for n, c in train.items()}
    objective = lambda p: float(np.mean([score(c.labels, predict(c, p, feats[n]))["spellF1"] for n, c in train.items()]))
    best = dict(DEFAULTS); best_score = objective(best)
    for _ in range(passes):
        improved = False
        for key, values in GRID.items():
            for v in values:
                if v == best[key]:
                    continue
                trial = dict(best, **{key: v}); s = objective(trial)
                if s > best_score + 1e-4:
                    best, best_score, improved = trial, s, True
            if log:
                log("  {} -> {} (train spellF1 {:.4f})".format(key, best[key], best_score))
        if not improved:
            break
    return best, best_score


def add_evidence(c, logp, weights):
    """Per file: w * dead_logodds onto D, +/- w * team_logodds / 2 onto H / A, where the file is valid.

    Applied on the decoder's own scale (after phase3's tau), and only to our match: no other has these files."""
    if c.name != OURS or not any(weights.values()):
        return logp
    logp = logp.copy()
    for name, w in weights.items():
        with np.load(os.path.join(EVIDENCE_DIR, name + ".npz")) as e:
            d = w * np.where(e["valid"], e["dead_logodds"], 0.0)
            t = w * np.where(e["valid"], e["team_logodds"], 0.0) / 2
        assert len(d) == len(logp), (name, len(d), len(logp))
        logp[:, 0] += t; logp[:, 1] -= t; logp[:, 2] += d
    return logp


def decode(c, logp, p):
    """Viterbi labels per frame from per-frame log-probabilities [T, 3] (H, A, D); X outside the periods.

    c.switch_bonus, if the match has one: a per-frame log-bonus on team switches (situations.py's prior), so a switch
    costs c_switch - bonus there; [T, 2] for switches to H and to A apart (external/switch_screen.py). Absent (every match unless set): the decoder is unchanged."""
    logp = add_evidence(c, logp, p.get("evidence") or {})
    T, label, first = _transitions(p, c.fps)
    out = np.full(len(logp), "X", dtype="<U1")
    names = np.array(["H", "A", "D"])
    for a, b in c.periods:
        n = (b - a) // STEP
        if n < 2:
            continue
        blocks = logp[a:a + n * STEP].reshape(n, STEP, 3).mean(1)
        bonus = getattr(c, "switch_bonus", None)
        if bonus is not None:
            bonus = bonus[a:a + n * STEP].reshape(n, STEP, -1).mean(1)
            bonus = bonus[:, 0] if bonus.shape[1] == 1 else bonus
        lab = np.repeat(names[viterbi(blocks, T, label, first, bonus)], STEP)
        out[a:a + n * STEP] = lab
        out[a + n * STEP:b] = lab[-1]
    out[c.forced_dead] = "D"
    return out


def predict(c, params=None, F=None):
    """Labels per frame for a cleaned match. F: its frame_features table, if already computed."""
    p = dict(DEFAULTS, **(params or {}))
    F = fe.frame_features(c) if F is None else F
    out = decode(c, evidence(F, p), p)
    if p["x_live_gaps"]:
        unseen = F.ball_seen.to_numpy() == 0
        out[unseen & np.isin(out, ["H", "A"])] = "X"
    return out
