"""python test_boundary.py -- boundary.shift: on our match only, turnovers earlier, never removing a possession; D stays."""
from types import SimpleNamespace

import numpy as np

import boundary

c = SimpleNamespace(fps=25, name="idsse_x")
lab = np.array(["H"] * 50 + ["A"] * 10 + ["H"] * 40 + ["D"] * 50 + ["A"] * 150)
c.name = boundary.OURS
assert (boundary.shift(c, lab) == lab).all()                       # off by default
boundary.SHIFT_S = 0.6                                             # 15 frames
c.name = "idsse_x"
assert (boundary.shift(c, lab) == lab).all()                       # not our match
c.name = boundary.OURS
out = boundary.shift(c, lab)
assert list(np.flatnonzero(out[1:] != out[:-1]) + 1) == [35, 51, 100, 150], np.flatnonzero(out[1:] != out[:-1]) + 1
print("ok")
