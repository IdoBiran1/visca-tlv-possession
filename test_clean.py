"""python test_clean.py -- one check per cleaning step, on synthetic cases with a known answer."""
import numpy as np
import pandas as pd
import clean

L, W, fps = 105.0, 68.0, 25

# speed: 1 m per frame at 25 fps is 25 m/s, whatever the window
xy = np.c_[np.arange(50.0), np.zeros(50)]
assert np.allclose(clean.speed(xy, fps)[5:], 25.0)

# direction: home keeper at x=2 in half 1, at x=103 in half 2 -> attacks +x then -x
T, P = 20, 4
team = np.array([0, 0, 1, 1])
p = np.full((2 * T, P, 2), 30.0)
p[:T, 0, 0], p[:T, 1, 0], p[:T, 2, 0], p[:T, 3, 0] = 2, 40, 103, 60    # half 1: home defends x=0
p[T:, 0, 0], p[T:, 1, 0], p[T:, 2, 0], p[T:, 3, 0] = 103, 60, 2, 40    # half 2: ends swapped
periods = [(0, T), (T, 2 * T)]
assert (clean.attack_direction(p, team, periods, L) == [[1, -1], [-1, 1]]).all()
p2, b2, flipped = clean.normalise_direction(p, np.zeros((2 * T, 2)), team, periods, L, W)
assert flipped == [False, True] and (clean.attack_direction(p2, team, periods, L)[:, 0] == 1).all()
assert np.allclose(b2[T:], [L, W])  # the ball is rotated with the players

# ball speed: a 20 m teleport is flagged, normal motion is kept
b = np.c_[np.arange(60.0) * 0.5, np.zeros(60)]
b[30:, 0] += 20
v, bad = clean.ball_speed_clean(b, fps)
assert bad[30:35].all() and not bad[:30].any() and not bad[36:].any()

# extra players: 12 home slots, the newest one goes; a slot 5 m off the pitch goes too
T, P = 30, 13
p = np.tile(np.linspace(10, 90, P)[None, :, None], (T, 1, 2)); p[..., 1] = 30
team = np.r_[np.zeros(12, int), [1]]
p[:20, 11] = np.nan                      # slot 11 appears at frame 20: the youngest
p[:, 12, 1] = W + 5                       # the away slot stands 5 m outside the touchline
out, removed = clean.drop_extra_players(p, team, L, W, fps)
assert removed[20:, 11].all() and not removed[:, :11].any() and removed[:, 12].all()
assert ((~np.isnan(out[..., 0]))[:, team == 0].sum(1) <= 11).all()

# glides: a slot sliding 1 m per frame (25 m/s) is hidden, with 0.2 s either side; a 10 m/s sprint is kept
p = np.zeros((100, 2, 2))
p[:, 0, 0] = np.r_[np.zeros(40), np.arange(1, 11), np.full(50, 10.0)]
p[:, 1, 0] = np.arange(100) * 0.4
out, removed = clean.drop_glides(p, 25)
assert list(np.flatnonzero(removed[:, 0])) == list(range(34, 55)) and not removed[:, 1].any()
assert np.isnan(out[34:55, 0]).all() and not np.isnan(out[:34, 0]).any()

# gap features
f = clean.ball_gap_features(np.array([1, 1, 0, 0, 0, 1], bool), 1)
assert list(f["gap_len_s"]) == [0, 0, 3, 3, 3, 0] and list(f["time_to_gap_end_s"]) == [0, 0, 3, 2, 1, 0]

# degrade: hides about the requested share, mostly slow frames, and bridges short live breaks
slow = np.zeros(10000, bool); slow[1000:2000] = slow[2010:3000] = slow[6000:7000] = True
vis = clean.degrade_ball(slow, [(0, 10000)], 25, np.array([1.0]), 0.3, 0.8, seed=1)
assert 0.3 <= (~vis).mean() < 0.32
hid = ~vis
assert (hid & slow).sum() / hid.sum() > 0.6
assert not (hid[1000:2000].all() and hid[2010:3000].all()) or hid[2000:2010].all()  # the 0.4 s break goes with its stoppage

# ball gaps in context: a gap from 2 s before a stoppage to 0.4 s before its end is a stoppage gap, the other is live;
# the simulation hides a stoppage with those edges and reaches the live missing share
seen = np.ones(10000, bool); seen[950:1990] = seen[3000:3050] = seen[6000:6050] = False   # the last: inside a stoppage
stopped = np.zeros(10000, bool); stopped[1000:2000] = stopped[6000:7000] = True
ctx = np.zeros(10000, int); ctx[2999] = 8
gm = clean.ball_gap_model(seen, stopped, ctx, np.ones(10000, bool), 25)
assert np.allclose(gm.edges, [[-2, -0.4, 40]]) and list(gm.live_s) == [2.0] and gm.stop_s.tolist() == [[0.0, 2.0, 40.0]] and gm.propensity[8] > 1 > gm.propensity[0]
hid = ~clean.degrade_ball_context(stopped, ctx, [(0, 10000)], 25, gm, seed=0)
assert any(hid[a - 50:b - 10].all() and not hid[b - 10:b].any() for a, b in [(1000, 2000), (6000, 7000)])
assert list(gm.cover[4:]) == [0.5, 0.5] and abs(hid[~stopped].mean() - gm.miss_live) < 0.01 and hid[stopped].mean() >= gm.miss_stopped

# video ball: only the video runs needed for the share change, with lane B's error size, some frames locked on a
# player and one long lock on the touchline
b = np.c_[np.linspace(10, 90, 5000), np.full(5000, 30.0)]
p = (b - [0, 10.0])[:, None]                                 # one player running alongside, 10 m from the ball
video = np.zeros(5000, bool); video[1000:1500] = video[3000:3500] = True
vb, changed = clean.video_ball(b, p, video, [(0, 5000)], 68.0, 25, share=0.1, burst_share=0, long_per_min=0, seed=0)
assert changed.sum() == 500 and (changed <= video).all() and (vb[~changed] == b[~changed]).all()
assert 0.2 < np.median(np.linalg.norm(vb[changed] - b[changed], axis=1)) < 0.35   # net A: sd 0.23 m per axis, median 0.27 m
vb, changed = clean.video_ball(b, p, video, [(0, 5000)], 68.0, 25, share=0.1, long_per_min=0.3, long_s=(2.0, 2.0), seed=0)
err = np.linalg.norm(vb[changed] - b[changed], axis=1)
assert (err > 8).any() and (err > 25).any()                  # locks on the player (10 m) and on the touchline (30 m)

# bounces: a short A between two H spells goes; next to D, or at the edge, it stays
lab = np.array(list("HHHHAAHHHH"))
assert "".join(clean.drop_bounces(lab, 3, 1)) == "HHHHHHHHHH"
assert "".join(clean.drop_bounces(lab, 2, 1)) == "HHHHAAHHHH"   # 2 s is not shorter than 2 s
assert "".join(clean.drop_bounces(np.array(list("HHDAAHH")), 9, 1)) == "HHDAAHH"
assert "".join(clean.drop_bounces(np.array(list("AAHHHH")), 9, 1)) == "AAHHHH"

# tags: a defending twin with the same frames and the other side is dropped, nothing else
tags = pd.DataFrame([("PROGRESSION", "H", 10, 50), ("DEFENDING IN MIDDLE THIRD", "A", 10, 50),
                     ("DEFENDING IN MIDDLE THIRD", "H", 70, 90), ("SET PIECES", "N/A", 5, 9)],
                    columns=["code", "side", "start_frame", "end_frame"])
kept, matched, unmatched = clean.dedupe_tags(tags)
assert matched == 1 and unmatched == 0 and len(kept) == 3 and 1 not in kept.index
# smooth_ball rounds a kick's corner and doesn't blur across a period boundary
kick = np.zeros((100, 2)); kick[50:, 0] = np.arange(50) * 0.5
sm = clean.smooth_ball(kick, [(0, 100)], 25, 0.28)
assert np.abs(np.diff(sm[:, 0], 2)).max() < 0.2 * np.abs(np.diff(kick[:, 0], 2)).max()
two = np.r_[np.zeros((50, 2)), np.ones((50, 2))]
assert np.allclose(clean.smooth_ball(two, [(0, 50), (50, 100)], 25, 0.28), two)
gappy = kick.copy(); gappy[60:70] = np.nan                     # ... nor across a gap, which stays a gap
sm = clean.smooth_ball(gappy, [(0, 100)], 25, 0.28)
assert np.isnan(sm[60:70]).all() and not np.isnan(sm[:60]).any() and not np.isnan(sm[70:]).any()
# degrade_players replays the missing counts and keeps the same player hidden while the count holds
rng = np.random.default_rng(0)
pxy = rng.uniform([20, 20], [80, 50], size=(1, 6, 2)).repeat(50, axis=0)
team = np.array([0, 0, 0, 1, 1, 1])
missing = np.zeros((50, 2), int); missing[10:30, 0] = 1; missing[20:25, 1] = 2
model = (np.ones((len(clean.CROWD_BINS) - 1, len(clean.EDGE_BINS) - 1)), missing)
gone = np.isnan(clean.degrade_players(pxy, team, 105.0, 68.0, model, seed=0)[..., 0])
assert sorted(np.bincount(gone[:, :3].sum(1), minlength=2)[:2]) == [20, 30]       # one home player out for 20 frames
assert len({tuple(r) for r in gone[gone[:, :3].any(1), :3]}) == 1                   # ... always the same one
assert gone[:, 3:].sum() == 10
# swap_teams: only opponents that pass close swap, the player moves into the opponent's slot, and the share is met
pxy = np.full((500, 4, 2), 50.0); pxy[:, 1, 0] = 51.0; pxy[:, 2, 0] = 10.0; pxy[:, 3, 0] = 90.0   # 0 and 1 close
team = np.array([0, 1, 0, 1])
sw = clean.swap_teams(pxy, team, 25, 0.05, seed=1)
moved = np.isnan(sw[:, :2, 0]).any(1)
assert moved.sum() >= 0.05 * 2000 and np.array_equal(sw[:, 2:], pxy[:, 2:])        # far players untouched
assert all(np.isnan(sw[f, :2, 0]).sum() == 1 and (sw[f, :2, 0] == sw[f, :2, 0]).sum() == 1 for f in np.flatnonzero(moved))
assert np.array_equal(clean.swap_teams(pxy, team, 25, 0.0), pxy)
# move_slots: a stretch goes to an empty slot of the other team, or a new one; the rest stays where it was
pxy = np.full((200, 3, 2), np.nan); pxy[:, 0] = 10.0; pxy[:, 1] = 20.0; pxy[150:, 2] = 30.0   # 0 home; 1, 2 away
mv, tm = clean.move_slots(pxy, np.array([0, 1, 1]), [(0, 50, 100)])
assert np.isnan(mv[50:100, 0]).all() and (mv[50:100, 2] == 10.0).all() and (mv[150:, 2] == 30.0).all() and len(tm) == 3
mv, tm = clean.move_slots(pxy, np.array([0, 1, 1]), [(0, 120, 160)])                        # slot 2 busy: a new one
assert tm.tolist() == [0, 1, 1, 1] and (mv[120:160, 3] == 10.0).all() and np.isnan(mv[120:160, 0]).all()
print("ok")
