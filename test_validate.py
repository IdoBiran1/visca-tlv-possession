"""MTAH_REPO=... python test_validate.py -- error_segments on synthetic labels with a known answer."""
import warnings

import numpy as np

import validate as V

warnings.filterwarnings("ignore")
lab = lambda *parts: np.array(list("".join(c * n for c, n in parts)))

# truth: H 100 frames, D 25, A 100. Prediction: H runs 10 frames long, then straight into A (no dead ball)
gt = lab(("H", 100), ("D", 25), ("A", 100))
pred = lab(("H", 110), ("A", 115))
df = V.error_segments(gt, pred)
assert list(df.kind) == ["matched", "matched"], df
h, a = df.iloc[0], df.iloc[1]
assert h.team == "H" and h.end_offset_s == 0.4 and h.start_offset_s == 0      # 10 frames late to end
assert a.team == "A" and a.start_offset_s == -0.6                              # starts 15 frames early

# a blip of the other team splits one true possession: the pieces fail IoU 0.5 and the blip is spurious
gt = lab(("H", 100))
pred = lab(("H", 40), ("A", 40), ("H", 20))
df = V.error_segments(gt, pred)
assert list(df.kind) == ["missed", "spurious", "spurious", "spurious"], list(df.kind)
miss = df[df.kind == "missed"].iloc[0]
assert miss.pred_pieces == 2 and miss.pred_other_team == 0.4
blip = df[(df.kind == "spurious") & (df.team == "A")].iloc[0]
assert blip.truth_other_team == 1.0                                            # really H's ball
# pool_prefixes: which sets a predictor trains on
from types import SimpleNamespace
assert V.pool_prefixes(SimpleNamespace(pool=False)) == ("idsse_",)
assert V.pool_prefixes(SimpleNamespace(pool=True)) == ("idsse_", "metrica")
assert V.pool_prefixes(V.PREDICTORS["phase4_a05_st"]) == ("idsse_", "st_")
assert V.pool_prefixes(V.PREDICTORS["phase4_raw3_pooled"]) == ("idsse_", "metrica")
assert V.VERDICT_GROUPS == ("idsse_", "metrica", "st_", "pff_")   # four equal judges

# truth: the judge labels JUDGE_BOUNCE_S gave a set, else its own (training) labels
c = SimpleNamespace(labels=np.array(list("HAH")))
assert V.truth(c) is c.labels and V.JUDGE_BOUNCE_S == V.BOUNCE_S == 0
c.judge = np.array(list("HHH"))
assert V.truth(c) is c.judge

# judge: a small KEEP asks for a second seed; a large one doesn't
def _scores(f1s, poss=250):
    s = {"m{}".format(i): {"spellF1": f, "poss_per90_true": poss} for i, f in enumerate(f1s)}
    s = {("idsse_" + k): v for k, v in s.items()}
    s.update(ours_label_free={"poss_per90": 280, "D_share": 0.46, "pre_kickoff_all_D": True}, ours_public={"spellF1": 0.7})
    return s
base = _scores([0.70, 0.71, 0.72, 0.69, 0.70, 0.71, 0.70])
small = V.judge(base, _scores([0.71, 0.72, 0.73, 0.70, 0.71, 0.72, 0.71]))
big = V.judge(base, _scores([0.73, 0.74, 0.75, 0.72, 0.73, 0.74, 0.73]))
assert small["verdict"] == "KEEP" and small["second_seed_needed"]
assert big["verdict"] == "KEEP" and not big["second_seed_needed"]

# baseline v6 trains like phase4_avg but tunes its decoder on IDSSE only, without touching the module's default
assert V.PREDICTORS["phase4_v6"].pool and V.phase3.TUNE_ON == ()
assert V.PREDICTORS["phase4_v7"].pool and V.fe.INTERCEPT   # baseline v7: v6 + intercept features
assert V.PREDICTORS["phase4_v8"].pool and V.AUG_DRAWS == 2   # baseline v8: v7 + two more corruption draws

# candidate v9: the copy trained on moves in stoppages, the match itself keeps its ball
import numpy as np
c = type("C", (), {})(); c.labels = np.array(["H"] * 5 + ["D"] * 40 + ["H"] * 5); c.bdet = np.ones(50, bool); c.fps = 25; c.bxy = np.zeros((50, 2)); c.pxy = np.tile(np.arange(50.)[:, None, None], (1, 1, 2))
moved = V.carried("m", c)
assert V.PREDICTORS["phase4_v9"].pool and moved is not c and moved is V.carried("m", c) and not c.bxy.any()

# heavy_slot: a held slot blocks the next run, and is freed when its holder dies
import fcntl, os, subprocess, sys, tempfile
d = tempfile.mkdtemp()
holder = subprocess.Popen([sys.executable, "-c", "import validate as V, time; V.SLOT_DIR, V.MAX_JOBS = {!r}, 1; V.heavy_slot(); time.sleep(60)".format(d)],
                          cwd=V.HERE, stdout=subprocess.PIPE, text=True)
assert "slot 0 of 1" in holder.stdout.readline()
fh = open(os.path.join(d, "slot0.lock"), "w")
try:
    fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
    raise AssertionError("slot not held")
except BlockingIOError:
    pass
holder.kill(); holder.wait()
fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)                                 # freed by the crash
sys.modules.pop("phase4", None)                           # a module run hasn't imported yet still takes settings
V.apply_settings([("phase4.EPOCHS", "12")])
assert sys.modules["phase4"].EPOCHS == 12
V.apply_settings([("phase4.Raw3.W_RAW", "0.5"), ("POSS_ALPHA", "0.25")])   # a class's setting, and this module's
assert sys.modules["phase4"].Raw3.W_RAW == 0.5 and sys.modules["phase4"].Ensemble.W_RAW == 0.0 and V.POSS_ALPHA == 0.25
for bad in ("phase4.Raw3.NOPE", "phase9.X", "NOPE"):
    try:
        V.apply_settings([(bad, "1")])
        raise AssertionError(bad)
    except SystemExit:
        pass
print("ok")
