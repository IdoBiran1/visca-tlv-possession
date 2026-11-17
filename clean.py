"""Cleaning steps shared by our match and the external datasets.

Every function works on plain arrays in one convention: metres from a pitch corner, one row per
frame, NaN where nothing was observed, `team` = 0 home / 1 away per player slot. The loaders in
external/profile_datasets.py (and preprocess.load_tracking for our match) produce exactly that.
docs/ explains why each step exists and how it was checked; test_clean.py checks the logic.
"""
from types import SimpleNamespace

import numpy as np

MAX_BALL_SPEED = 40.0     # m/s; the hardest shots reach about 35
MAX_PLAYER_SPEED = 12.0   # m/s; sprinters peak near 10
OFF_PITCH_M = 4.0         # real players (throw-ins, corners) stay within ~4.5 m of the lines in IDSSE


def runs(mask):
    """Half-open [start, stop) runs where mask is True."""
    m = np.r_[False, np.asarray(mask, bool), False]
    e = np.flatnonzero(m[1:] != m[:-1])
    return list(zip(e[::2], e[1::2]))


def speed(xy, fps, window_s=0.2):
    """Speed in m/s from positions, over a fixed time window so 10 and 25 fps read the same.

    Works for players [T, P, 2] and the ball [T, 2]. NaN where either end is missing.
    """
    step = max(1, int(round(window_s * fps)))
    v = np.full(xy.shape[:-1], np.nan)
    v[step:] = np.linalg.norm(xy[step:] - xy[:-step], axis=-1) / (step / fps)
    return v


def attack_direction(pxy, team, periods, length):
    """+1 if a team attacks towards x = length in a period, -1 if towards x = 0. Shape [n_periods, 2].

    Read from the goalkeeper: each team's rearmost player sits by the goal it defends, so the
    side with the smaller median gap between a team's deepest player and its goal line is home.
    """
    out = np.zeros((len(periods), 2), int)
    for k, (a, b) in enumerate(periods):
        for t in (0, 1):
            x = pxy[a:b][:, team == t, 0]
            near_zero = np.nanmedian(np.nanmin(x, axis=1))           # deepest player's distance to x = 0
            near_far = np.nanmedian(length - np.nanmax(x, axis=1))   # ... to x = length
            out[k, t] = 1 if near_zero < near_far else -1            # defends x = 0, so attacks +x
    return out


def normalise_direction(pxy, bxy, team, periods, length, width, side=0):
    """Rotate each period so team `side` always attacks towards +x. Returns (pxy, bxy, flipped)."""
    d = attack_direction(pxy, team, periods, length)
    pxy, bxy = pxy.copy(), bxy.copy()
    flipped = []
    for k, (a, b) in enumerate(periods):
        if d[k, side] < 0:  # rotate 180 degrees about the centre spot: x and y both flip
            pxy[a:b] = [length, width] - pxy[a:b]
            bxy[a:b] = [length, width] - bxy[a:b]
        flipped.append(bool(d[k, side] < 0))
    return pxy, bxy, flipped


def smooth_ball(bxy, periods, fps, sigma_s, kernel="gauss"):
    """A ball path blurred like our tracker's: Gaussian over time, inside each period and each run of seen frames.
    kernel "box": a moving average with the same standard deviation (sigma_s * sqrt 12 wide), for a test generator.

    Our ball has no sharp kicks: its velocity changes across a touch, and its accelerations, are far smaller than
    IDSSE's and Metrica's. A small sigma_s brings IDSSE to ours on both."""
    from scipy.ndimage import gaussian_filter1d, uniform_filter1d
    out = bxy.copy()
    for a, b in periods:
        for x, y in runs(~np.isnan(bxy[a:b, 0])):
            seg = bxy[a + x:a + y]
            out[a + x:a + y] = (uniform_filter1d(seg, int(round(sigma_s * fps * 12 ** .5)) | 1, axis=0, mode="nearest") if kernel == "box"
                                else gaussian_filter1d(seg, sigma_s * fps, axis=0, mode="nearest"))
    return out


def tracker_alone(n, template, share, rng, how="blocks", block=750):
    """[n] bool: which of n tracker-seen frames, in order, read the tracker alone (blurred) rather than net A (sharp).

    template: the same flag over ours' tracker-seen in-play frames in order (shared/ours_ball_neta_v2.npz source
    "track" against "net"). how "blocks": blocks of `block` frames cut from the template at random places, each next
    one from the blocks above or below the template's median share as the running share sits under or over `share`,
    so run lengths and order are ours. "fitted" (a different generator, for test sets): alternating runs of lognormal
    lengths fitted to the template's net and track runs, both scaled so the expected track share is `share`."""
    t = np.asarray(template, bool)
    if how == "fitted":
        logs = {v: np.log([b - a for a, b in runs(t == v)]) for v in (False, True)}
        mu, sd = {v: logs[v].mean() for v in logs}, {v: logs[v].std() for v in logs}
        mean = {v: np.exp(mu[v] + sd[v] ** 2 / 2) for v in logs}
        k = np.log(share * mean[False] / ((1 - share) * mean[True])) / 2   # half on each side, so neither run length moves far
        mu[True], mu[False] = mu[True] + k, mu[False] - k
        out, have, v = [], 0, bool(rng.random() < share)
        while have < n:
            out.append(np.full(max(1, int(round(rng.lognormal(mu[v], sd[v])))), v))
            have, v = have + len(out[-1]), not v
        return np.concatenate(out)[:n] if out else np.zeros(0, bool)
    starts = np.arange(len(t) - block)
    shares = np.convolve(t, np.ones(block), "valid")[:len(starts)] / block
    lo, hi = starts[shares <= np.median(shares)], starts[shares > np.median(shares)]
    out = np.zeros(n, bool)
    for k in range(0, n, block):
        s = rng.choice(hi if out[:k].sum() < share * k else lo)
        out[k:k + block] = t[s:s + min(block, n - k)]
    return out


def ball_speed_clean(bxy, fps):
    """Ball speed with physically impossible values (> MAX_BALL_SPEED) set to NaN.

    Positions are left alone: the direction of travel is still useful, only the magnitude is not.
    Returns (speed, implausible_mask).
    """
    v = speed(bxy, fps)
    bad = v > MAX_BALL_SPEED
    return np.where(bad, np.nan, v), bad


def drop_glides(pxy, fps, pad_s=0.2):
    """Hide a slot while it moves faster than MAX_PLAYER_SPEED from one frame to the next. Returns (pxy, removed).

    Our tracker swaps identities by sliding the slot from one person to another far faster than anyone runs
    for about a second; the slot is no one while it slides. IDSSE's only such steps are
    slots reused at substitutions. `pad_s` either side catches the slide's slower start and end.
    """
    step = np.zeros(pxy.shape[:2])
    step[1:] = np.nan_to_num(np.linalg.norm(pxy[1:] - pxy[:-1], axis=2))
    removed = np.zeros(step.shape, bool)
    pad = int(round(pad_s * fps))
    for s in range(step.shape[1]):
        for a, b in runs(step[:, s] > MAX_PLAYER_SPEED / fps):
            removed[max(0, a - 1 - pad):b + pad, s] = True
    removed &= ~np.isnan(pxy[..., 0])
    pxy = pxy.copy()
    pxy[removed] = np.nan
    return pxy, removed


def drop_extra_players(pxy, team, length, width, fps):
    """Remove slots that cannot be players in play, so each team has at most 11. Returns (pxy, removed).

    1. A slot more than OFF_PITCH_M outside the lines is someone else (bench, staff, warm-up).
    2. If a team still shows more than 11, the youngest tracklets go first: a new track that
       appears while 11 are already tracked is a ghost or a duplicate far more often than not.
    """
    pxy = pxy.copy()
    present = ~np.isnan(pxy[..., 0])
    x, y = pxy[..., 0], pxy[..., 1]
    outside = np.maximum.reduce([-x, x - length, -y, y - width]) > OFF_PITCH_M
    removed = present & outside
    pxy[removed] = np.nan
    present &= ~removed
    age = np.zeros(present.shape, int)
    for s in range(present.shape[1]):
        for a, b in runs(present[:, s]):
            age[a:b, s] = np.arange(b - a)
    for t in (0, 1):
        cols = np.flatnonzero(team == t)
        for f in np.flatnonzero(present[:, cols].sum(1) > 11):
            live = cols[present[f, cols]]
            extra = live[np.argsort(age[f, live], kind="stable")][:len(live) - 11]
            pxy[f, extra] = np.nan
            removed[f, extra] = True
    return pxy, removed


def ball_gap_features(seen, fps):
    """Per-frame confidence features for the ball: gap length, time into the gap, time to its end (s)."""
    T = len(seen)
    gap_len = np.zeros(T); into = np.zeros(T); left = np.zeros(T)
    for a, b in runs(~np.asarray(seen, bool)):
        n = b - a
        gap_len[a:b] = n / fps
        into[a:b] = np.arange(n) / fps
        left[a:b] = np.arange(n, 0, -1) / fps
    return {"ball_seen": np.asarray(seen, bool), "gap_len_s": gap_len, "time_in_gap_s": into, "time_to_gap_end_s": left}


CROWD_BINS, EDGE_BINS = (0, 1.0, 1.5, 2.0, 3.0, 5.0, np.inf), (0, 3.0, 8.0, np.inf)   # metres


def _crowd_edge(pxy, length, width):
    """[T, P] distance to the nearest other tracked player, and to the nearest touchline or goal line."""
    T, P = pxy.shape[:2]
    near = np.full((T, P), np.inf)
    for i in range(P):
        d = np.linalg.norm(pxy - pxy[:, i:i + 1], axis=2)
        d[:, i] = np.inf
        near[:, i] = np.nanmin(np.where(np.isnan(d), np.inf, d), axis=1)
    x, y = pxy[..., 0], pxy[..., 1]
    edge = np.minimum.reduce([x, length - x, y, width - y])
    return near, edge


def _bins(near, edge):
    return (np.clip(np.digitize(near, CROWD_BINS) - 1, 0, len(CROWD_BINS) - 2),
            np.clip(np.digitize(np.nan_to_num(edge, nan=99.0), EDGE_BINS) - 1, 0, len(EDGE_BINS) - 2))


def player_loss_model(c):
    """How our tracker loses players, from our cleaned match `c`: (relative propensity per crowd x edge bin,
    [n, 2] players missing per team on each graded frame). Losses are tracklets that end while their team drops
    below 11 (not a hand-over to a new slot). Our tracker loses players in crowds and near the lines."""
    vis = ~np.isnan(c.pxy[..., 0])
    near, edge = _crowd_edge(c.pxy, c.length, c.width)
    ib, jb = _bins(near, edge)
    exposure, events = np.zeros((len(CROWD_BINS) - 1, len(EDGE_BINS) - 1)), np.zeros((len(CROWD_BINS) - 1, len(EDGE_BINS) - 1))
    g = c.graded[:-1]
    np.add.at(exposure, (ib[:-1][vis[:-1] & g[:, None]], jb[:-1][vis[:-1] & g[:, None]]), 1)
    count = np.stack([vis[:, c.team == t].sum(1) for t in (0, 1)], axis=1)
    for f, s_ in zip(*np.nonzero(vis[:-1] & ~vis[1:] & g[:, None])):
        if count[f + 1, c.team[s_]] < min(11, count[f, c.team[s_]]):
            events[ib[f, s_], jb[f, s_]] += 1
    return events / np.maximum(exposure, 1), np.clip(11 - count[c.graded], 0, 11)


def degrade_players(pxy, team, length, width, model, seed=0):
    """Hide players in a clean match the way our tracker loses them (player_loss_model). Our per-team missing
    counts are replayed from a random offset, which keeps how bursty the losses are; which players go missing
    is drawn by crowd x edge propensity, and a hidden player stays hidden while the count calls for it."""
    propensity, missing = model
    rng = np.random.default_rng(seed)
    near, edge = _crowd_edge(pxy, length, width)
    w = propensity[_bins(near, edge)] * ~np.isnan(pxy[..., 0])
    m = np.roll(missing, rng.integers(len(missing)), axis=0)
    out = pxy.copy()
    for t in (0, 1):
        slots = np.flatnonzero(team == t)
        hidden = []
        for f in range(len(pxy)):
            k = m[f % len(m), t]
            hidden = [s for s in hidden if w[f, s] > 0][len(hidden) - k if len(hidden) > k else 0:]   # release the oldest
            free = [s for s in slots if w[f, s] > 0 and s not in hidden]
            while len(hidden) < k and free:
                p = w[f, free] / w[f, free].sum()
                hidden.append(free.pop(rng.choice(len(free), p=p)))
            out[f, hidden] = np.nan
    return out


def swap_teams(pxy, team, fps, share, seed=0, near_m=1.5, median_s=3.2, sigma=1.27, max_s=60.0):
    """Give players the other team's slot the way our tracker does. Where two opponents pass within `near_m`, one's
    slot slides onto the other and follows him, so he is counted with the wrong team while the opponent goes
    untracked, for a lognormal time (median `median_s`). On ours, shirt colours under the slots show
    a small share of tracked player-time in the other team's slot, almost never as a swapped pair. Episodes are added until `share` of tracked player-time is moved."""
    rng = np.random.default_rng(seed)
    out = pxy.copy()
    vis = ~np.isnan(pxy[..., 0])
    h, a = np.flatnonzero(team == 0), np.flatnonzero(team == 1)
    cand = []
    for i in range(0, len(pxy), 20000):   # (frame, home slot, away slot) of opponents within near_m
        q = pxy[i:i + 20000].astype(np.float32)
        with np.errstate(invalid="ignore"):
            f, j, k = np.nonzero(np.linalg.norm(q[:, h, None] - q[:, None, a], axis=-1) < near_m)
        cand.append(np.c_[f + i, h[j], a[k]])
    cand = np.concatenate(cand)
    busy = np.zeros(vis.shape, bool)
    left = share * vis.sum()
    for f, x, b in cand[rng.permutation(len(cand))]:
        if left <= 0:
            break
        if rng.random() < 0.5:
            x, b = b, x                       # b's slot takes over x
        e = min(len(pxy), f + int(np.clip(rng.lognormal(np.log(median_s * fps), sigma), 1, max_s * fps)))
        if busy[f:e, x].any() or busy[f:e, b].any():
            continue
        out[f:e, b], out[f:e, x] = pxy[f:e, x], np.nan
        busy[f:e, [x, b]] = True
        left -= vis[f:e, x].sum()
    return out


def move_slots(pxy, team, segments, margin=25):
    """Undo swap_teams on our match: move each (slot, start, stop) stretch into a slot of the other team that is
    empty from `margin` frames before to after it (so it joins no other tracklet), or a new slot if none is.
    The stretches come from shirt colours under the slots (docs/2026-10-23-wrong-team-slots.md). Returns (pxy, team)."""
    pxy, team = pxy.copy(), np.asarray(team).copy()
    for s, a, b in segments:
        lo, hi = max(0, a - margin), b + margin
        free = [c for c in np.flatnonzero(team != team[s]) if np.isnan(pxy[lo:hi, c, 0]).all()]
        if not free:
            pxy = np.concatenate([pxy, np.full((len(pxy), 1, 2), np.nan)], axis=1)
            team = np.append(team, 1 - team[s])
            free = [pxy.shape[1] - 1]
        pxy[a:b, free[0]] = pxy[a:b, s]
        pxy[a:b, s] = np.nan
    return pxy, team


def stopped_play(slow, in_play, fps, bridge_s=2.0):
    """Slow frames inside in_play, with live breaks shorter than `bridge_s` inside a stoppage bridged."""
    stopped = np.asarray(slow, bool) & in_play
    for a, b in runs(~stopped & in_play):
        if (b - a) < bridge_s * fps and a > 0 and b < len(stopped) and stopped[a - 1] and stopped[b]:
            stopped[a:b] = True
    return stopped


def ball_context(bxy, pxy, length, width):
    """Per frame: 3 * zone (0 middle third, 1 final third, 2 penalty box) + crowd (players within 3 m of the ball: 0, 1, 2+)."""
    x, y = bxy[:, 0], bxy[:, 1]
    with np.errstate(invalid="ignore"):
        box = ((x < 16.5) | (x > length - 16.5)) & (np.abs(y - width / 2) < 20.16)
        zone = np.where(box, 2, ((x < length / 3) | (x > 2 * length / 3)).astype(int))
        crowd = np.minimum((np.linalg.norm(pxy - bxy[:, None], axis=2) < 3).sum(1), 2)
    return 3 * zone + crowd


STOP_BINS_S = (2, 4, 8, 16, 32)


def ball_gap_model(seen, stopped, ctx, in_play, fps):
    """How our tracker loses the ball, measured on our match. A gap holding a stoppage's middle frame is a stoppage
    gap; any other is a gap inside a stoppage or a live gap, by where it starts. Returns a SimpleNamespace:
      cover       share of stoppages with a gap, per STOP_BINS_S length bin
      miss_stopped, miss_live  share of stopped / live in-play frames with no ball
      edges       [n, 3] s: a stoppage gap's start and end minus its stoppage's, and the stoppage's length
      stop_s      [n, 3] s: gaps that start inside a stoppage without holding its middle: start minus the stoppage's
                  start, gap length, stoppage length (ours with the recovered ball: in long stoppages, while the ball is off
                  the pitch)
      live_s      live gap lengths, s
      propensity  relative rate a live gap starts, per ball_context bin of the last seen frame"""
    miss = ~np.asarray(seen, bool) & in_play
    st = runs(stopped)
    mids = np.array([(a + b) // 2 for a, b in st])
    edges, stop_s, live_s = [], [], []
    starts = np.zeros(len(miss), bool)
    for a, b in runs(miss):
        k = np.flatnonzero((mids >= a) & (mids < b))
        if len(k):
            edges.append(((a - st[k[0]][0]) / fps, (b - st[k[-1]][1]) / fps, (st[k[-1]][1] - st[k[0]][0]) / fps))
        elif stopped[a]:
            c, d = st[np.searchsorted([x for x, _ in st], a, side="right") - 1]
            stop_s.append(((a - c) / fps, (b - a) / fps, (d - c) / fps))
        else:
            live_s.append((b - a) / fps)
            starts[max(a - 1, 0)] = True
    open_ = ~miss & in_play & ~stopped
    ev, ex = np.bincount(ctx[starts & open_], minlength=9), np.bincount(ctx[open_], minlength=9)
    rate = ev.sum() / ex.sum()
    length = np.digitize([(b - a) / fps for a, b in st], STOP_BINS_S)
    cover = [miss[mids[length == k]].mean() if (length == k).any() else 0.5 for k in range(len(STOP_BINS_S) + 1)]
    return SimpleNamespace(cover=np.array(cover), miss_stopped=float(miss[stopped].mean()), miss_live=float(miss[in_play & ~stopped].mean()),
                           edges=np.array(edges), stop_s=np.array(stop_s).reshape(-1, 3), live_s=np.array(live_s),
                           propensity=(ev + 1) / (ex * rate + 1))   # +1: shrink thin bins to 1


def carry_ball(c, move=.55, carry=.6, seed=0):
    """A ball that moves during stoppages, as our tracker's does: each visible stretch of at least 0.4 s inside dead ball, with probability `move`, is carried by the
    nearest player (share `carry`) or rolls away at 1 to 5 m/s. Edits c.bxy, c.ball_speed and c.ball_fast in place. Returns frames changed."""
    rng, n = np.random.default_rng(seed), 0
    for a0, b0 in runs(np.asarray(c.labels) == "D"):
        for a, b in [(a0 + x, a0 + y) for x, y in runs(c.bdet[a0:b0])]:
            if b - a < .4 * c.fps or rng.random() >= move:
                continue
            start = c.bxy[a].copy(); n += b - a
            near = np.linalg.norm(c.pxy[a] - start, axis=1)
            if rng.random() < carry and np.isfinite(near).any():
                held = c.pxy[a:b, int(np.nanargmin(near))]
                c.bxy[a:b] = np.where(np.isfinite(held), held + rng.normal(0, .3, 2), start)
            else:
                t = rng.uniform(0, 2 * np.pi); v = rng.uniform(1, 5) * np.array([np.cos(t), np.sin(t)])
                c.bxy[a:b] = start + v * np.arange(b - a)[:, None] / c.fps
    c.ball_speed, c.ball_fast = ball_speed_clean(c.bxy, c.fps)
    return n


def degrade_ball_context(stopped, ctx, periods, fps, model, seed=0, near=10):
    """Hide a clean ball the way ball_gap_model says our tracker loses it. Returns the mask of frames left visible.

    A stoppage is hidden with probability `cover` of its length, with the edges of one of the `near` stoppages of ours
    closest in length; gaps from `stop_s` start at ours' offsets in one of the `near` stoppages closest in length to
    ours, until `miss_stopped` of the stopped frames are hidden; then live gaps start where ours do (`propensity` of the frame's ball_context bin) until `miss_live` of the
    live frames are hidden."""
    rng = np.random.default_rng(seed)
    T = len(stopped)
    in_play, period = np.zeros(T, bool), np.zeros(T, int)
    for k, (a, b) in enumerate(periods):
        in_play[a:b], period[a:b] = True, k
    live, hidden = in_play & ~stopped, np.zeros(T, bool)
    for a, b in runs(stopped):
        if rng.random() >= model.cover[np.digitize((b - a) / fps, STOP_BINS_S)]:
            continue
        alike = np.argsort(np.abs(model.edges[:, 2] - (b - a) / fps))[:near]
        d0, d1 = np.round(model.edges[rng.choice(alike), :2] * fps).astype(int)
        lo, hi = periods[period[a]]
        hidden[max(a + d0, lo):max(min(b + d1, hi), lo)] = True
    inside, st = np.flatnonzero(stopped & in_play), runs(stopped & in_play)
    lens = np.array([b - a for a, b in st])
    for _ in range(20 * len(st) if len(model.stop_s) else 0):   # ponytail: bounded, so an unreachable target can't hang
        if hidden[inside].mean() >= model.miss_stopped:
            break
        d0, n, length = np.round(model.stop_s[rng.integers(len(model.stop_s))] * fps).astype(int)
        a, b = st[rng.choice(np.argsort(np.abs(lens - length))[:near])]
        if a + d0 < b:   # the offset lands inside this stoppage
            hidden[a + d0:a + d0 + max(1, n)] = True
    stop_gaps, w = hidden.copy(), np.cumsum(model.propensity[ctx] * live)
    for _ in range(20 * len(runs(live))):
        if hidden[live].mean() >= model.miss_live:
            break
        s = int(np.searchsorted(w, rng.uniform(0, w[-1]), side="right"))
        n = int(round(rng.choice(model.live_s) * fps))
        if not stop_gaps[s:s + n + 2].any():   # ours never merge with a stoppage gap: they'd be one gap
            hidden[s + 1:s + 1 + n] = True
    return ~(hidden & in_play)


def video_ball(bxy, pxy, video, periods, width, fps, share, sd_m=(0.23, 0.23), bias_m=(0.0, 0.0), burst_share=0.047,
               burst_frames=(2, 30), long_per_min=0.0, long_s=(2.0, 10.0), sigma_s=0.04, seed=0):
    """A clean ball as net A's video ball reads it where our tracker loses it. Returns (bxy, frames changed).

    Runs of `video` (frames our tracker would lose but the video finds), in random order, until `share` of the
    in-play frames, get net A's error model (neta/st_port.py and st_hard.py: SoccerTrack 132877 half 1 reviewed frames,
    the same reweighted to our gaps' crowd): Gaussian error of sd `sd_m` along x / y,
    smooth over `sigma_s`, plus `bias_m` (an airborne ball unprojects further from the camera); on `burst_share` of
    them a false lock on the nearest player for `burst_frames`; and `long_per_min` long locks per minute of play,
    held for `long_s` on a still point by the near touchline (a sock, the bench).
    `long_per_min` is 0: lane B's 0.1 was measured on our match, and SoccerTrack's reviewed half doesn't measure long
    locks (docs/2026-11-10-compliance.md)."""
    rng = np.random.default_rng(seed)
    in_play = np.zeros(len(bxy), bool)
    for a, b in periods:
        in_play[a:b] = True
    kept, out = np.zeros(len(bxy), bool), bxy.copy()
    for a, b in (runs(video)[i] for i in rng.permutation(len(runs(video)))):
        if kept.sum() >= share * in_play.sum():
            break
        kept[a:b] = True
    frames = np.flatnonzero(kept)
    locked = 0
    while len(frames) and locked < burst_share * len(frames):
        s, n = int(rng.choice(frames)), int(rng.integers(burst_frames[0], burst_frames[1] + 1))
        d = np.nan_to_num(np.linalg.norm(pxy[s] - bxy[s], axis=1), nan=np.inf)
        if np.isfinite(d.min()):
            span = np.arange(s, min(s + n, len(kept)))
            span = span[kept[span]]
            out[span] = pxy[span, int(np.argmin(d))]
            locked += len(span)
    for _ in range(int(round(long_per_min * in_play.sum() / fps / 60))):
        s = int(rng.choice(frames))
        span = np.arange(s, s + int(rng.uniform(*long_s) * fps))
        span = span[span < len(out)]
        out[span] = [out[s, 0], -0.5 if out[s, 1] < width / 2 else width + 0.5]
        kept[span] = True
    noise = smooth_ball(np.where(kept[:, None], rng.normal(size=bxy.shape), np.nan), periods, fps, sigma_s)[kept]
    noise = noise / noise.std(axis=0) * sd_m + bias_m
    out[kept] = smooth_ball(np.where(kept[:, None], out, np.nan), periods, fps, sigma_s)[kept] + noise
    return out, kept


def degrade_ball(slow, periods, fps, gap_lengths_s, missing_share, slow_share_of_missing, seed=0, bridge_s=2.0):
    """Hide a clean ball the way our tracker loses it. Returns the mask of frames left visible.

    Our tracker loses the ball mostly while play is stopped. So whole slow stretches (players'
    2 s mean speed under 1.5 m/s, with breaks shorter than `bridge_s` bridged) are hidden first,
    until `slow_share_of_missing` of the target is slow; the rest are gaps of lengths drawn from
    `gap_lengths_s`, started in live play, until `missing_share` of the frames in `periods` is hidden.
    """
    rng = np.random.default_rng(seed)
    T = len(slow)
    in_play = np.zeros(T, bool)
    for a, b in periods:
        in_play[a:b] = True
    target = missing_share * in_play.sum()
    hidden = np.zeros(T, bool)
    stopped = stopped_play(slow, in_play, fps, bridge_s)
    stretches = runs(stopped)
    for i in rng.permutation(len(stretches)):
        if hidden.sum() >= slow_share_of_missing * target:
            break
        a, b = stretches[i]
        hidden[a:b] = True
    live = np.flatnonzero(in_play & ~stopped)
    while (hidden & in_play).sum() < target:
        start = int(rng.choice(live))
        hidden[start:start + int(round(rng.choice(gap_lengths_s) * fps))] = True
    return ~(hidden & in_play)


def drop_bounces(labels, max_s, fps):
    """Give a possession shorter than `max_s` back to the team on both sides of it.

    H H H A A H H H with the A spell under max_s becomes all H: a touch or a deflection that
    ends with the same team in control is not a change of possession (CHALLENGE.md). Spells
    next to a D or at the edge of the array are kept.
    """
    lab = np.array(labels, copy=True)
    for _ in range(5):  # removing one bounce can expose another
        changed = False
        b = np.r_[0, np.flatnonzero(lab[1:] != lab[:-1]) + 1, len(lab)]
        spans = [(b[i], b[i + 1], lab[b[i]]) for i in range(len(b) - 1)]
        for (a0, _, t0), (a1, b1, t1), (_, _, t2) in zip(spans, spans[1:], spans[2:]):
            if t1 in ("H", "A") and t0 == t2 and t0 in ("H", "A") and t0 != t1 and (b1 - a1) / fps < max_s:
                lab[a1:b1] = t0
                changed = True
        if not changed:
            break
    return lab


def mean_player_speed(pxy, fps, smooth_s=2.0):
    """Mean speed of all tracked players per frame (jumps over MAX_PLAYER_SPEED ignored), then a centred mean."""
    s = speed(pxy, fps)
    with np.errstate(invalid="ignore"):
        per_frame = np.nanmean(np.where(s < MAX_PLAYER_SPEED, s, np.nan), axis=1)
    w = int(smooth_s * fps) | 1
    ok = ~np.isnan(per_frame)
    num = np.convolve(np.where(ok, per_frame, 0), np.ones(w), "same")
    den = np.convolve(ok.astype(float), np.ones(w), "same")
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(den > 0, num / den, np.nan)


def clean_match(m, bounce_s=None, ball_visible=None, dead_frames=None):
    """Every cleaning step, in order, for one match from any dataset. Returns a SimpleNamespace.

    `m` needs: name, fps, length, width, pxy, team, bxy, bdet, periods, labels (may be None).
      bounce_s      give label spells shorter than this back to the team either side (IDSSE)
      ball_visible  extra mask of frames where the ball may be seen (degrade_ball, for IDSSE)
      dead_frames   frames that are dead ball by rule (ours: 110,808-113,741, before the second-half kick-off)
    The ball is set to NaN wherever it isn't seen, so no feature can peek at a hidden ball.
    """
    pxy, glides = drop_glides(m.pxy, m.fps)
    pxy, removed = drop_extra_players(pxy, m.team, m.length, m.width, m.fps)
    removed |= glides
    bdet = np.asarray(m.bdet, bool).copy()
    if ball_visible is not None:
        bdet &= ball_visible
    bxy = np.where(bdet[:, None], m.bxy, np.nan)
    pxy, bxy, flipped = normalise_direction(pxy, bxy, m.team, m.periods, m.length, m.width)
    ball_v, ball_fast = ball_speed_clean(bxy, m.fps)
    graded = np.zeros(len(bxy), bool)
    for a, b in m.periods:
        graded[a:b] = True
    forced_dead = np.zeros(len(bxy), bool)
    if dead_frames is not None:
        forced_dead[dead_frames] = True
    labels = None if m.labels is None else np.asarray(m.labels).copy()
    if labels is not None and bounce_s:
        labels = drop_bounces(labels, bounce_s, m.fps)
    if labels is not None:
        labels[~graded] = "X"
    return SimpleNamespace(name=m.name, fps=m.fps, length=m.length, width=m.width, pxy=pxy, team=np.asarray(m.team),
                           bxy=bxy, bdet=bdet, ball_speed=ball_v, ball_fast=ball_fast, periods=list(m.periods),
                           graded=graded, forced_dead=forced_dead, labels=labels, removed=removed, flipped=flipped)


MIRRORED_CODES = {  # attacking code -> the defending code Smart Tagging writes for the other team
    "PROGRESSION": "DEFENDING IN MIDDLE THIRD",
    "BUILD UP": "DEFENDING IN ATTACKING THIRD",
    "ATTACKING TRANSITION": "DEFENSIVE TRANSITION",
    "CREATING CHANCES": "DEFENDING IN DEFENSIVE THIRD",
}


def dedupe_tags(tags):
    """Drop the defending half of each mirrored Smart Tagging pair. Returns (kept, n_matched, n_unmatched).

    `tags` is a DataFrame with code, side, start_frame, end_frame (preprocess.load_tags). A
    defending tag is dropped only when its attacking twin has the same frames and the other side.
    """
    drop, unmatched = set(), 0
    other = {"H": "A", "A": "H"}
    for att, dfn in MIRRORED_CODES.items():
        a = tags[tags.code == att]
        d = tags[tags.code == dfn]
        keys = {(r.start_frame, r.end_frame, r.side): i for i, r in d.iterrows()}
        for _, r in a.iterrows():
            i = keys.get((r.start_frame, r.end_frame, other.get(r.side, r.side)))
            if i is None:
                unmatched += 1
            else:
                drop.add(i)
    return tags.drop(index=sorted(drop)), len(drop), unmatched
