"""python test_phase4.py -- the TCN learns a trivially separable sequence and returns per-frame log-probabilities."""
import warnings
from types import SimpleNamespace

import numpy as np

import clean
import features as fe
import phase3
import phase4

warnings.filterwarnings("ignore")
rng = np.random.default_rng(0)
T, L, W = 4000, 105.0, 68.0
pxy = rng.uniform([10, 10], [95, 58], size=(1, 6, 2)) + rng.normal(0, 0.3, (T, 6, 2)).cumsum(0)
ball = np.where(np.arange(T)[:, None] < T // 2, pxy[:, 0], pxy[:, 3]) + 0.5   # with home, then with away
labels = np.where(np.arange(T) < T // 2, "H", "A")
c = SimpleNamespace(name="s", fps=25, length=L, width=W, pxy=pxy, team=np.array([0, 0, 0, 1, 1, 1]), bxy=ball,
                    bdet=np.ones(T, bool), periods=[(0, T)], labels=labels)
cm = clean.clean_match(c); cm.labels, cm.graded = labels, np.ones(T, bool)
F = fe.frame_features(cm)
model = phase4.fit_model({"s": cm}, {"s": F})
lp = phase4.log_probs(model, F)
assert lp.shape == (T, 3) and np.allclose(np.exp(lp).sum(1), 1, atol=1e-3)
pred = np.array(["H", "A", "D"])[lp.argmax(1)]
assert np.mean(pred[200:T // 2 - 200] == "H") > 0.9 and np.mean(pred[T // 2 + 200:-200] == "A") > 0.9

# N_SEEDS: nets for seeds SEED, SEED + 1, ... with their log-probabilities averaged; the first is the default model's
phase4.N_SEEDS = 2
two = phase4.fit_model({"s": cm}, {"s": F})
phase4.N_SEEDS = 1
assert len(two["nets"]) == 2 and np.array_equal(phase4.log_probs(dict(two, nets=two["nets"][:1]), F), lp)
pred = np.array(["H", "A", "D"])[phase4.log_probs(two, F).argmax(1)]
assert np.mean(pred[200:T // 2 - 200] == "H") > 0.9 and np.mean(pred[T // 2 + 200:-200] == "A") > 0.9

# the raw member (features.raw_features): the same sequence from the ball and its nearest players alone
R = fe.raw_features(cm)
assert R.shape[1] == 5 + 6 * 5 and np.allclose(phase3.mirror(phase3.mirror(F.join(R), L, W), L, W)[R.columns], R, atol=1e-4, equal_nan=True)
FR = F.join(R)
raw = phase4.fit_model({"s": cm}, {"s": FR}, raw=True)
pred = np.array(["H", "A", "D"])[phase4.log_probs(raw, FR).argmax(1)]
assert np.mean(pred[200:T // 2 - 200] == "H") > 0.9 and np.mean(pred[T // 2 + 200:-200] == "A") > 0.9

# Seeds: K whole SetAvg models at seeds SEED, SEED + 1 averaged; K = 1 is SetAvg itself. SetAvg.MIRROR: the mean with the
# mirrored match's prediction (H and A swapped back), which MIRROR_DUMP keeps in LAST_MIRROR
import setnet
phase4.DEVICE = setnet.DEVICE = "cpu"
setnet.K = 6                  # the toy match has 6 players
cm.name = "idsse_s"
cm.labels = np.where(np.arange(T) < 100, "D", labels)   # the trees need all three classes
FS = phase4.SetAvg.features(cm)
one = phase4.SetAvg.fit_model({"idsse_s": cm}, {"idsse_s": FS})
lp1 = phase4.SetAvg.log_probs(one, FS)
phase4.Seeds.K = 1
assert np.array_equal(phase4.Seeds.log_probs(phase4.Seeds.fit_model({"idsse_s": cm}, {"idsse_s": FS}), FS), lp1)
phase4.Seeds.K = 2
two = phase4.Seeds.fit_model({"idsse_s": cm}, {"idsse_s": FS})
assert len(two) == 2 and phase4.SEED == 0 and np.allclose(phase4.Seeds.log_probs(two, FS), (lp1 + phase4.SetAvg.log_probs(two[1], FS)) / 2)
# the predictor's path (phase4_v13_seeds: one decode of the members' averaged log-probabilities) gives the labels
# external/replay.py --with gives from the members' saved (float32) log-probabilities
import json
d = dict(phase3.DEC_DEFAULT, tau=3.0, c_switch=6.0)
cs = cm
cs.graded, cs.forced_dead = np.ones(T, bool), np.zeros(T, bool)
mine = phase3.decode(cs, phase4.Seeds.log_probs(two, FS), FS, d)
saved = [phase4.SetAvg.log_probs(m, FS).astype(np.float32) for m in two]
replayed = phase3.decode(cs, np.mean([x.astype(np.float64) for x in saved], 0), None, json.loads(json.dumps(d)))
assert np.array_equal(mine, replayed), int((mine != replayed).sum())
phase4.SetAvg.MIRROR = phase4.SetAvg.MIRROR_DUMP = True
lpm = phase4.SetAvg.log_probs(one, FS)
assert np.allclose(lpm, (lp1 + phase4.LAST_MIRROR) / 2)
for x in (lp1, phase4.LAST_MIRROR, phase4.Seeds.log_probs(two, FS)):
    pred = np.array(["H", "A", "D"])[x.argmax(1)]
    assert np.mean(pred[200:T // 2 - 200] == "H") > 0.9 and np.mean(pred[T // 2 + 200:-200] == "A") > 0.9
phase4.SetAvg.MIRROR = phase4.SetAvg.MIRROR_DUMP = False
print("ok")
