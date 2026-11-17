"""A learned detector of the moment possession changes team (docs/2026-11-02-change-detector.md).

features.touches finds the new team's touch at only 71% of the starts of short possessions, even on the true ball
(docs/2026-11-02-short-possession-gap.md). This learns the same moment from the labels instead: trees on a ±0.4 s
window of fine-grained ball movement and each team's nearest player, trained on IDSSE and Metrica's live changes of
team. Every ball but ours is first blurred like our tracker's (clean.smooth_ball, SMOOTH_S), so the kicks it learns
look like ours. Nothing is set from our match or the public window.

    W = window(c)                          # detector inputs, one row per frame
    m = fit({name: (c, W)})                # on training matches
    P = proba(m, W)                        # [T, 2]: P(possession changes to home / to away around this frame)
    python change.py test
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

import clean
import features as fe

SMOOTH_S = 0.28        # clean.smooth_ball's match to our tracker's ball
OURS = "BAR-ATH"       # our match's name prefix: its ball is already the tracker's
NEAR_S = 0.2           # frames this close to a change are its positives
OFFSETS_S = (-0.4, -0.2, 0.0, 0.2, 0.4)
NEG_EVERY = 5          # every 5th other frame as a negative
MODEL = dict(max_iter=200, learning_rate=0.1, max_leaf_nodes=31, min_samples_leaf=200, l2_regularization=1.0, random_state=0)
PAIRED = ["d{}", "dmin", "close", "pspeed", "within2"]   # _h / _a columns; d{} per offset
SIGNED = ["touch_who", "ctrl", "last_touch", "last_ctrl", "next_ctrl"]


def shift(x, k):
    """x[t + k] at t, NaN outside."""
    out = np.full(x.shape, np.nan)
    if k > 0:
        out[:-k] = x[k:]
    elif k < 0:
        out[-k:] = x[:k]
    else:
        out[:] = x
    return out


def window(c):
    """DataFrame of detector inputs per frame of a cleaned match."""
    fps, T = c.fps, len(c.bxy)
    b = c.bxy if str(c.name).startswith(OURS) else clean.smooth_ball(c.bxy, c.periods, fps, SMOOTH_S)
    h1, h2 = max(1, round(0.1 * fps)), max(1, round(0.2 * fps))
    f = {}
    vel = (shift(b, h1) - shift(b, -h1)) * fps / (2 * h1)
    speed = np.linalg.norm(vel, axis=1)
    for s in (-0.4, -0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3, 0.4):
        f["bspeed{:+.1f}".format(s)] = shift(speed, round(s * fps))
    before, after = (b - shift(b, -h2)) * fps / h2, (shift(b, h2) - b) * fps / h2
    sb, sa = np.linalg.norm(before, axis=1), np.linalg.norm(after, axis=1)
    f["dv"], f["speed_before"], f["speed_after"] = np.linalg.norm(after - before, axis=1), sb, sa
    with np.errstate(invalid="ignore", divide="ignore"):
        cos = np.sum(before * after, axis=1) / (sb * sa)
    f["turn"] = np.where((sb > 0.5) & (sa > 0.5), np.arccos(np.clip(cos, -1, 1)), np.nan)
    seen = (~np.isnan(b[:, 0])).astype(float)
    f["seen"] = pd.Series(seen).rolling(2 * h2 * 2 + 1, center=True, min_periods=1).mean().to_numpy()

    d = np.linalg.norm(c.pxy - b[:, None], axis=2)
    ps = clean.speed(c.pxy, fps)
    rows = np.arange(T)
    for t, tag in ((0, "h"), (1, "a")):
        dt = np.where(c.team[None] == t, d, np.inf)
        dt[np.isnan(dt)] = np.inf
        i = dt.argmin(1)
        dn = np.where(np.isfinite(dt[rows, i]), dt[rows, i], np.nan)
        for s in OFFSETS_S:
            f["d{:+.1f}_{}".format(s, tag)] = shift(dn, round(s * fps))
        f["dmin_" + tag] = pd.Series(dn).rolling(2 * h2 * 2 + 1, center=True, min_periods=1).min().to_numpy()
        f["close_" + tag] = (shift(dn, -h2) - shift(dn, h2)) * fps / (2 * h2)
        f["pspeed_" + tag] = ps[rows, i]
        f["within2_" + tag] = (dt < 2).sum(1).astype(float)

    hit, who = fe.touches(c.bxy, c.pxy, c.team, fps)       # the hand detector, as the frame model sees it
    ctl = fe.control(c.bxy, c.pxy, c.team, fps)
    f["touch_who"] = np.where(hit, who, 0.0)
    f["ctrl"] = ctl
    f["last_touch"] = fe.carry(np.r_[False, hit[:-1]], np.r_[0.0, who[:-1]], c.periods, 0.0)   # before this frame
    f["last_ctrl"] = fe.carry(np.r_[False, (ctl != 0)[:-1]], np.r_[0.0, ctl[:-1]], c.periods, 0.0)
    f["next_ctrl"] = fe.carry_next(ctl != 0, ctl, c.periods, 0.0)
    f["t_since_touch"] = fe.since(hit, c.periods, fps)
    return pd.DataFrame(f).astype("float32")


def targets(labels, graded, fps):
    """Per frame: 1 within NEAR_S of a live change of team to home, 2 to away, 0 otherwise."""
    lab = np.where(graded, labels, "X")
    y = np.zeros(len(lab), int)
    r = max(1, round(NEAR_S * fps))
    for t in np.flatnonzero((lab[1:] != lab[:-1]) & np.isin(lab[1:], ["H", "A"]) & np.isin(lab[:-1], ["H", "A"])) + 1:
        y[max(0, t - r):t + r + 1] = 1 if lab[t] == "H" else 2
    return y


def mirror(W):
    """Home and away swapped (the window has no pitch position, so nothing turns)."""
    G = W.copy()
    for p in PAIRED:
        for s in (OFFSETS_S if "{}" in p else (None,)):
            k = p.format("{:+.1f}".format(s)) if s is not None else p
            G[k + "_h"], G[k + "_a"] = W[k + "_a"].values, W[k + "_h"].values
    G[SIGNED] = -W[SIGNED]
    return G


def fit(train):
    """Trees on {name: (cleaned match, window)}: every positive frame and every NEG_EVERY-th other graded frame, each
    with its mirror."""
    X, y = [], []
    for c, W in train.values():
        t = targets(c.labels, c.graded, c.fps)
        k = c.graded & ((t > 0) | (np.arange(len(t)) % NEG_EVERY == 0))
        X += [W[k], mirror(W[k])]
        y += [t[k], np.array([0, 2, 1])[t[k]]]
    return HistGradientBoostingClassifier(**MODEL).fit(pd.concat(X, ignore_index=True), np.concatenate(y))


def proba(model, W):
    P = model.predict_proba(W)
    return P[:, [list(model.classes_).index(1), list(model.classes_).index(2)]]


def test():
    lab = np.array(list("HHHHHAAAHHDDAA"))
    y = targets(lab, np.ones(14, bool), fps=10.0)   # r = 2 frames
    assert list(y) == [0, 0, 0, 2, 2, 2, 1, 1, 1, 1, 1, 0, 0, 0], y   # the later change wins where they overlap
    W = pd.DataFrame({c: np.arange(3.0) for c in ["d{:+.1f}_{}".format(s, t) for s in OFFSETS_S for t in "ha"]
                      + [p + "_" + t for p in PAIRED[1:] for t in "ha"] + SIGNED})
    W["dmin_h"] = 5.0
    M = mirror(W)
    assert (M.dmin_a == 5).all() and (M.touch_who == -W.touch_who).all() and (mirror(M) == W).all().all()
    print("ok")


if __name__ == "__main__":
    test()
