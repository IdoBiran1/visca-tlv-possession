"""python test_features.py -- the event detectors and helpers on a synthetic play with known answers."""
import warnings
from types import SimpleNamespace

import numpy as np

import clean
import features as fe

warnings.filterwarnings("ignore")
fps, T, L, W = 25, 500, 105.0, 68.0

# since / carry inside periods
ev = np.zeros(10, bool); ev[[2, 7]] = True
assert list(fe.since(ev, [(0, 5), (5, 10)], 1, cap_s=9)) == [9, 9, 0, 1, 2, 9, 9, 0, 1, 2]
assert list(fe.carry(ev, np.arange(10.0), [(0, 10)], -1)) == [-1, -1, 2, 2, 2, 2, 2, 7, 7, 7]

# a play: ball unseen for 2 s, then still at (50, 34) for 2 s; a home player kicks it towards +x at
# 12 m/s (frame 100); an away player stops it and sends it back at -12 m/s (frame 150); it reaches
# the home player again (frame 200) and rolls off over y = 68 (out near frame 261)
ball = np.full((T, 2), np.nan)
ball[50:100] = [50, 34]
for i in range(100, 150):
    ball[i] = [50 + 12 * (i - 100) / fps, 34]
for i in range(150, 200):
    ball[i] = [74 - 12 * (i - 150) / fps, 34]
for i in range(200, 300):
    ball[i] = [50, 34 + 14 * (i - 200) / fps]
players = np.full((T, 3, 2), np.nan)
players[:, 0] = [49.2, 34]                            # home player, 0.8 m from the kick (inside CTRL_RADIUS_M)
players[:, 1] = [75.0, 34]                            # away player, 1 m beyond the turn
players[:, 2] = [10, 10]
team = np.array([0, 1, 0])

c = SimpleNamespace(name="synthetic", fps=fps, length=L, width=W, pxy=players, team=team, bxy=ball,
                    bdet=~np.isnan(ball[:, 0]), periods=[(0, T)], labels=None)
cm = clean.clean_match(c)
F = fe.frame_features(cm)

hit, who = fe.touches(cm.bxy, cm.pxy, cm.team, fps)
touch_frames = np.flatnonzero(hit)
assert len(touch_frames) == 3 and all(abs(t - e) <= 3 for t, e in zip(touch_frames, (100, 150, 200))), touch_frames
assert list(who[touch_frames]) == [1.0, -1.0, 1.0]
assert F.last_touch_team[170] == -1 and F.last_touch_team[120] == 1 and F.last_touch_team[60] == 0

rs = np.flatnonzero(fe.restarts(cm.ball_speed, fps))
assert len(rs) == 1 and abs(rs[0] - 100) <= 6, rs     # the kick after 4 s unseen or still

assert F.t_since_ball_out_s[255] == fe.CAP_S and F.t_since_ball_out_s[280] < 1.5
assert F.ball_out_m[290] > 0 and F.ball_out_m[150] == 0
# the home player stands beside the still ball (control), not beside the kicked one
assert F.ctrl_team[75] == 1 and F.ctrl_team[125] == 0 and F.ctrl_team[20] == 0
assert F.next_ctrl_team[20] == 1 and F.last_ctrl_team[125] == 1
assert F.n_h[0] == 2 and F.n_a[0] == 1 and abs(F.d_h[60] - 0.8) < 1e-6

# control_change: runs restart at each period, every count stays inside its run
ctl = np.array([0, 1, 1, 0, 0, 0, 0, -1, 0, 0.0])
hit = np.zeros(10, bool); hit[2] = True
cc = fe.control_change(hit, ctl, ctl, [(0, 5), (5, 10)], 1)
assert list(cc["cc_team"]) == [0, 1, 1, 1, 1, 0, 0, -1, -1, -1], cc["cc_team"]
assert list(cc["cc_since_s"]) == [0, 0, 1, 2, 3, 0, 1, 0, 1, 2] and list(cc["cc_until_s"]) == [1, 4, 3, 2, 1, 2, 1, 3, 2, 1]
assert list(cc["cc_run_ev_s"]) == [0, 2, 2, 2, 2, 0, 0, 1, 1, 1] and list(cc["cc_run_touches"]) == [0, 1, 1, 1, 1, 0, 0, 0, 0, 0]
assert list(cc["cc_score"]) == [0, 2, 2, 2, 2, 0, 0, -1, -1, -1]
# the play: home controls the still ball and kicks it, away turns it over at frame 150 with one touch, home gets it back at 200
assert not any(c.startswith("cc_") for c in F.columns)   # off by default
fe.CONTROL_CHANGE = True
G = fe.frame_features(cm)
fe.CONTROL_CHANGE = False
assert list(G.cc_team[[120, 170, 220]]) == [1, -1, 1]
assert abs(G.cc_since_s[170] - 0.8) <= 0.12 and abs(G.cc_until_s[170] - 1.2) <= 0.12
assert G.cc_run_touches[170] == 1 and G.cc_run_ev_s[170] < 0.1 < 1 < G.cc_run_ev_s[120]   # one touch against 2 s of control
assert np.allclose(G.drop(columns=[c for c in G if c.startswith("cc_")]).fillna(-9), F.fillna(-9))
print("ok")

# intercept: a ball passed at 12 m/s along +x from (30, 34) reaches the away player 15 m ahead first, not the home
# player standing 4 m behind it; a ball at the home player's feet is his
T = 50
bxy = np.c_[30 + 12 * np.arange(T) / 25, np.full(T, 34.0)]
pxy = np.array([[[26.0, 34.0], [45.0, 34.5]]]).repeat(T, 0)
i = fe.intercept(bxy, pxy, np.array([0, 1]), 25)
k = 10   # 0.4 s in: 4.8 m travelled, 10.2 m left to the away player
assert i["icpt_t_a"][k] < i["icpt_t_h"][k] and i["icpt_margin"][k] < 0 and i["icpt_dist_m"][k] > 5, {a: b[k] for a, b in i.items()}
still = fe.intercept(np.full((T, 2), [26.5, 34.0]), pxy, np.array([0, 1]), 25)
assert still["icpt_margin"][k] > 0 and still["icpt_dist_m"][k] == 0
print("intercept ok")
