"""python test_phase1.py -- the decoder on synthetic evidence with a known answer."""
import tempfile
from types import SimpleNamespace

import numpy as np

import phase1

p = dict(phase1.DEFAULTS, min_poss_s=1.0, min_dead_s=2.0, c_switch=6.0, c_to_dead=6.0, c_from_dead=4.0)
T, label, first = phase1._transitions(p, 25)          # 0.2 s steps: possessions >= 5 steps, dead >= 10 steps


def evid(seq, strength=2.0):
    """log-probs per step that favour the given labels."""
    out = np.full((len(seq), 3), -strength)
    for i, s in enumerate(seq):
        out[i, "HAD".index(s)] = 0.0
    return out


decode = lambda seq: "".join(np.array(list("HAD"))[phase1.viterbi(evid(seq), T, label, first)])

# a 2-step blip of the other team inside a long possession is absorbed (too short, and switching costs)
assert decode("H" * 30 + "AA" + "H" * 30) == "H" * 62
# a long, clear change of possession goes through, at the right step
assert decode("H" * 30 + "A" * 30) == "H" * 30 + "A" * 30
# a stoppage shorter than the minimum dead duration is not called; a long one is
assert "D" not in decode("H" * 30 + "D" * 4 + "H" * 30)
assert decode("H" * 30 + "D" * 20 + "A" * 30) == "H" * 30 + "D" * 20 + "A" * 30
# every decoded spell respects the minimum durations
path = decode("HAHAHAHAHA" * 6 + "D" * 3 + "A" * 20)
runs = [len(r) for r in "".join(c if i == 0 or c == path[i - 1] else "|" + c for i, c in enumerate(path)).split("|")]
assert min(runs) >= 5, path

# the evidence hook: a file's dead and team log-odds move the decoded labels, on our match only, where valid
phase1.EVIDENCE_DIR = tempfile.mkdtemp()
n = 400
dead, team = np.zeros(n, np.float32), np.zeros(n, np.float32)
dead[100:200], team[250:] = 6.0, -8.0
np.savez(phase1.EVIDENCE_DIR + "/cam.npz", dead_logodds=dead, team_logodds=team, valid=np.ones(n, bool))
np.savez(phase1.EVIDENCE_DIR + "/off.npz", dead_logodds=dead, team_logodds=team, valid=np.zeros(n, bool))
m = lambda name: SimpleNamespace(name=name, fps=25, periods=[(0, n)], forced_dead=np.zeros(n, bool))
home = np.log(np.tile([0.8, 0.1, 0.1], (n, 1)))       # the model says home in possession throughout
run = lambda name, ev: "".join(phase1.decode(m(name), home, dict(p, evidence=ev)))
assert run(phase1.OURS, {}) == "H" * n
assert run(phase1.OURS, {"cam": 1.0}) == "H" * 100 + "D" * 100 + "H" * 50 + "A" * 150
assert run(phase1.OURS, {"cam": 0.0}) == run(phase1.OURS, {"off": 1.0}) == run("idsse_J03WMX", {"cam": 1.0}) == "H" * n

# the switch-bonus hook: a 2 s away blip too weak to pay two switches, taken where the bonus makes them cheap
blip = home.copy(); blip[200:250] = np.log([0.35, 0.55, 0.1])
q = dict(p, c_switch=9.0, min_poss_s=0.4)
with_bonus = lambda b: "".join(phase1.decode(SimpleNamespace(**vars(m("x")), switch_bonus=b), blip, q))
assert with_bonus(None) == with_bonus(np.zeros(n)) == "".join(phase1.decode(m("x"), blip, q)) == "H" * n
cheap = np.zeros(n); cheap[190:260] = 8.0
assert with_bonus(cheap) == "H" * 200 + "A" * 50 + "H" * 150, with_bonus(cheap)
print("ok")
