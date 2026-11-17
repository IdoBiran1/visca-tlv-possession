"""python test_phase3.py -- the mirror augmentation: twice is the identity, and it swaps the teams."""
import warnings
from types import SimpleNamespace

import numpy as np

import clean
import features as fe
import phase3

warnings.filterwarnings("ignore")
fe.CONTROL_CHANGE = fe.INTERCEPT = True   # every SIGNED and PAIRS column present
rng = np.random.default_rng(0)
T, L, W = 300, 105.0, 68.0
pxy = rng.uniform([10, 10], [95, 58], size=(1, 6, 2)) + rng.normal(0, 0.3, (T, 6, 2)).cumsum(0)   # 6 players wandering
ball = rng.uniform([10, 10], [95, 58], size=(1, 2)) + rng.normal(0, 0.5, (T, 2)).cumsum(0)
c = SimpleNamespace(name="s", fps=25, length=L, width=W, pxy=pxy, team=np.array([0, 0, 0, 1, 1, 1]), bxy=ball,
                    bdet=np.ones(T, bool), periods=[(0, T)], labels=None)
F = fe.frame_features(clean.clean_match(c))
M = phase3.mirror(F, L, W)
back = phase3.mirror(M, L, W)
assert np.allclose(back.fillna(-9).values, F.fillna(-9).values), "mirror twice is not the identity"
assert np.allclose(M.icpt_t_h.fillna(-9), F.icpt_t_a.fillna(-9)) and np.allclose(M.icpt_dist_m.fillna(-9), F.icpt_dist_m.fillna(-9))
assert np.allclose(M.d_h.fillna(-9), F.d_a.fillna(-9)) and np.allclose(M.d_margin.fillna(-9), -F.d_margin.fillna(9))
assert all(np.allclose(M[k].fillna(-9), -F[k].fillna(9)) for k in phase3.SIGNED)
assert np.allclose(M.ball_x.fillna(-9), (L - F.ball_x).fillna(-9))
assert list(phase3.with_context(F).columns) == list(phase3.with_context(M).columns)

# prebin: order kept, NaN kept, at most 255 values a column; a weighted fit predicts through the same bins
X = phase3.with_context(F).dropna(axis=1, how="all")   # line_h / line_a are all NaN here, and sklearn's weighted fit rejects those
B =phase3.prebin(X, phase3.bin_edges(X))
for col in X.columns:
    v, b = X[col].to_numpy(float), B[col].to_numpy()
    assert (np.isnan(v) == np.isnan(b)).all() and len(np.unique(b[~np.isnan(b)])) <= 255
    o = np.argsort(v[~np.isnan(v)], kind="stable")
    assert (np.diff(b[~np.isnan(b)][o]) >= 0).all(), col
y = (F.d_margin.fillna(0).to_numpy() > 0).astype(int)
m = phase3.trees_fit(X, y, np.ones(len(y)), max_iter=20, random_state=0)
assert m.prebin_edges is not None and (phase3.trees_proba(m, X).argmax(1) == y).mean() > 0.9
assert phase3.trees_fit(X, y, max_iter=20, random_state=0).prebin_edges is None   # unweighted fits are untouched
# tune: the plain search takes a narrow spike (a = 0), the smoothed one the broad plateau (a = 3)
f = {0: 1.0, 1: 0.0, 2: 0.8, 3: 0.9, 4: 0.8}
grid = {"a": [0, 1, 2, 3, 4], "b": [0, 1]}
assert phase3.tune(lambda d: f[d["a"]], grid, {"a": 2, "b": 0})[0]["a"] == 0
phase3.TUNE_SMOOTH = True
assert phase3.tune(lambda d: f[d["a"]], grid, {"a": 2, "b": 0})[0]["a"] == 3
phase3.TUNE_SMOOTH = False
print("ok")

# with_aug: a match brings its own extra draws, and only its own
phase3.AUG["s"] = [clean.clean_match(c)]
assert len(phase3.with_aug("s", c, F)) == 2 and len(phase3.with_aug("t", c, F)) == 1
assert list(phase3.with_aug("s", c, F)[1][1].columns) == list(fe.frame_features(phase3.AUG["s"][0]).join(fe.raw_features(phase3.AUG["s"][0])).columns)
phase3.AUG.clear()
print("with_aug ok")
