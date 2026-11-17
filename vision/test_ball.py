"""python vision/test_ball.py -- camera from homography, flight fit, label snapping, search mask, and a
tiny training run that has to learn to find a moving white disc. Synthetic only, no match data."""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ball  # noqa: E402
import frames  # noqa: E402
import radar  # noqa: E402
from frames import PITCH, project3d  # noqa: E402

rng = np.random.default_rng(0)

# A camera high behind one touchline, looking at the centre spot; its ground homography in the
# challenge's normalised convention; camera() has to give back the same 3D projection.
C, target = np.array([52.0, -40.0, 30.0]), np.array([52.0, 34.0, 0.0])
z = (target - C) / np.linalg.norm(target - C)
x = np.cross(z, [0, 0, 1.0])
x /= np.linalg.norm(x)
R = np.stack([x, np.cross(z, x), z])
P = np.array([[1500, 0, 960], [0, 1500, 540], [0, 0, 1.0]]) @ np.column_stack([R, -R @ C])
Hn = P[:, [0, 1, 3]] @ np.diag([PITCH[0], PITCH[1], 1.0])
Pc = frames.camera(Hn / Hn[2, 2])
pts = rng.uniform([0, 0, 0], [PITCH[0], PITCH[1], 10], (50, 3))
assert np.abs(project3d(Pc, pts) - project3d(P, pts)).max() < 0.5

# A lofted ball (4 m up, rising) seen with 1 px noise: the flight fit explains it, and puts the
# ground point under it far closer than the ground-roll fit does (depth is the weak direction).
t = (np.arange(25) - 12) / 25
X = np.stack([40 + 15 * t, 30 + 5 * t, 4 + 3 * t - 9.81 * t ** 2 / 2], 1)
uv = project3d(P, X) + rng.normal(0, 1, (25, 2))
Ps = np.repeat(Pc[None], 25, 0)
(fx, ef), (gx, eg) = frames.fit_track(Ps, uv, t, True), frames.fit_track(Ps, uv, t, False)
assert ef < 1.5 and eg > 3 * ef and 1.5 < fx[12, 2] < 6
assert 2 * np.linalg.norm(fx[12, :2] - X[12, :2]) < np.linalg.norm(gx[12, :2] - X[12, :2])

# Snapping: a white 7 px disc 30 px above the projected label is found to the pixel.
grass = np.full((300, 300, 3), (70, 130, 50), np.uint8)
yy, xx = np.mgrid[:300, :300]
img = grass.copy()
img[(xx - 150) ** 2 + (yy - 120) ** 2 <= 12] = 235
assert np.abs(ball.snap(img, (152, 150))[:2] - (150, 120)).max() <= 1
line = np.array([[100, 100, 90], [110, 105, 90], [120, 110, 90]], np.float32)
assert ball.smooth(*line) and not ball.smooth(line[0], line[1] + (30, 0, 0), line[2])

# Search mask: near a player yes, far from everything no, near an anchor yes, off the pitch and
# on the score bug no (Hn's camera sees the whole pitch).
m = ball.search_mask([(960, 500), (40, 40)], [(1500, 800, 100)], Hn / Hn[2, 2])
assert m[500, 960 + ball.RP - 20] and not m[1000, 300] and m[800, 1450] and m.shape == (1080, 1920)
assert not m[40, 40] and not m[1079, 960]
post = np.round(project3d(P, [0.0, PITCH[1] / 2 + 3.66, 0])).astype(int)
assert not ball.search_mask([tuple(post)], [], Hn / Hn[2, 2])[post[1], post[0]]

# Tiny training run: the net, the target, the loss and the (u, v) order have to agree.
def disc_clip(n, size=32):
    X = np.empty((n, 3, size, size, 3), np.uint8)
    lab = rng.uniform(6, size - 6, (n, 2))
    vel = rng.uniform(-3, 3, (n, 2))
    for i in range(n):
        for k in range(3):
            X[i, k] = (70, 130, 50)
            c = lab[i] + (k - 1) * vel[i]
            X[i, k][(xx[:size, :size] - c[0]) ** 2 + (yy[:size, :size] - c[1]) ** 2 <= 6] = 235
    return X, lab.round().astype(np.float32)


torch.manual_seed(0)
net = ball.Net(8).to(ball.DEV)
opt = torch.optim.Adam(net.parameters(), 3e-3)
for step in range(150):
    Xs, ys = disc_clip(16)
    loss = ball.focal(net(ball.to_input(Xs)), ball.heatmap_target(ys, 32))
    opt.zero_grad()
    loss.backward()
    opt.step()
Xs, ys = disc_clip(32)
old, ball.CROP = ball.CROP, 32
assert ball.evaluate_crops(net, Xs, ys) > 0.9, ball.evaluate_crops(net, Xs, ys)
ball.CROP = old
# tidy: a spike is dropped, a 0.2 s hole bridged, trusted frames untouched, and a video run that is
# 1 m off the tracker's ground point is shifted to meet it at both joins.
T = 100
true = np.stack([np.linspace(10, 30, T), np.full(T, 34.0)], 1)
obs = true + [0, 1.0] + rng.normal(0, 0.15, (T, 2))
trusted = np.zeros(T, bool)
trusted[:30] = trusted[70:] = True
obs[trusted] = true[trusted]
obs[50] += 8
obs[60:65] = np.nan
out = ball.tidy(obs, trusted, [(0, T)])
assert np.isfinite(out).all() and (out[trusted] == obs[trusted]).all()
assert np.abs(out - true).max() < 1.0 and np.abs(np.diff(out, axis=0)).max() < 0.6

# plausible: a detection 30 m from the tracked ball one frame later is rejected, so is one 20 m from
# every player in live play; a smooth one next to a player is kept.
xy = np.zeros((10, 2)) + [50.0, 30.0]
xy[:, 0] += np.arange(10) * 0.5
trusted = np.zeros(10, bool)
trusted[[0, 9]] = True
xy[8] += [30, 0]
pl = np.repeat(np.array([[[50.0, 31.0], [55.0, 30.0]]]), 10, 0)
pl[4] += [-40, -20]                                        # everyone far away at frame 4
keep = ball.plausible(xy, trusted, pl, np.ones(10, bool), [(0, 10)])
assert keep[[0, 1, 2, 3, 9]].all() and not keep[8] and not keep[4]
assert ball.plausible(xy, trusted, pl, np.zeros(10, bool), [(0, 10)])[4]   # dead ball: far is fine

# tidy keeps a restart: a jittery ball still for 3 s, then kicked at 15 m/s, is still until the kick
# and features.restarts fires within 0.2 s of it.
import clean, features
features.STILL_SPEED, features.MOVING_SPEED, features.QUIET_S = 1.0, 5.0, 2.0   # lane B's tidy (not in the final recipe) was built on these
T = 150
kick = np.stack([np.r_[np.full(75, 40.0), 40 + 0.6 * np.arange(1, 76)], np.full(T, 30.0)], 1)
obs = kick + rng.normal(0, 0.1, (T, 2))
out = ball.tidy(obs, np.zeros(T, bool), [(0, T)])
sp = clean.ball_speed_clean(out, 25)[0]
assert np.nanmax(sp[5:70]) < features.STILL_SPEED
fired = np.flatnonzero(features.restarts(sp, 25))
assert len(fired) == 1 and abs(fired[0] - 75) <= 5, fired

# Ball from players: features are scaled, velocity needs LAG frames of history, missing players
# are masked, and the net's ball is a weighted mix of present players' positions plus a small offset.
pxy = np.zeros((10, 3, 2))
pxy[:, 0] = [52.5, 34.0]
pxy[:, 1, 0] = 10 + np.arange(10) * 0.2                 # 1.0 m per 0.2 s = 5 m/s
pxy[:, 1, 1] = 34.0
pxy[:, 2] = np.nan
f, m = radar.player_table(pxy, np.array([0, 1, 0]), 105.0, 68.0)
assert m[radar.LAG:, :2].all() and not m[:radar.LAG].any() and not m[:, 2].any()
assert np.allclose(f[-1, 0, :2], 0) and np.isclose(f[-1, 1, 2], 0.5) and (f[:, 2] == 0).all()
xy, _ = radar.Radar().to(ball.DEV)(torch.from_numpy(f[-1:]).to(ball.DEV), torch.from_numpy(m[-1:]).to(ball.DEV))
assert -0.95 < xy[0, 0].item() < 0.15 and abs(xy[0, 1].item()) < 0.15
# link: the ball rolls on from the tracked ball; a brighter sock 10 m away wins frame by frame, but
# the linked path follows the ball, and says "none" where nothing fits.
T = 30
cand = np.full((T, 2, 2), np.nan)
cand[:, 0] = np.stack([20 + 0.4 * np.arange(T), np.full(T, 30.0)], 1)      # the ball, 10 m/s
cand[:, 1] = cand[:, 0] + [0, 10]                                            # a sock, 10 m away
pp_ = np.tile([0.5, 0.8], (T, 1))
pp_[10:13] = 0                                                               # nothing seen for 3 frames
path = ball.link(cand, pp_, start=np.array([19.6, 30.0]), end=np.array([32.0, 30.0]))
assert (path[:10] == 0).all() and (path[10:13] == 2).all() and (path[13:] == 0).all()

# peaks: two separated discs come out as the two strongest local maxima, in pixel (u, v) order.
two = np.full((1, 3, 1080, 1920, 3), (70, 130, 50), np.uint8)
for c in ((300, 200), (900, 600)):
    two[0, :, c[1] - 2:c[1] + 3, c[0] - 2:c[0] + 3] = 235
class Blob(torch.nn.Module):                                 # a stand-in net: whiteness, as logits
    def forward(self, x):
        return (x[:, 3:6].mean(1) - 0.2) * 20
pk = ball.peaks(Blob(), two[0], np.ones((1080, 1920), bool), k=3)
assert {tuple(pk[0, :2]), tuple(pk[1, :2])} <= {(u, v) for u in range(298, 303) for v in range(198, 203)} | \
       {(u, v) for u in range(898, 903) for v in range(598, 603)} and pk[0, 2] > 0.5 and pk[1, 2] > 0.5
assert abs(pk[0, 0] - pk[1, 0]) > 100

print("ok")
