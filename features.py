"""Per-frame features from a cleaned match (clean.clean_match), the same for every dataset.

Identity-free: nothing follows a track slot over time, only per-team aggregates and the nearest
player of each team, so our anonymous slots and the external datasets' named players give the
same inputs. Coordinates are the cleaned ones: metres from a corner, home attacking +x.

    from features import frame_features
    F = frame_features(clean.clean_match(match))     # one row per frame
"""
import numpy as np
import pandas as pd

import clean

# Tuned so the detectors work on our tracker, which smooths the ball's path (docs/2026-10-21-phase0-report.html):
TOUCH_WINDOW_S = 0.4     # ball velocity over the 0.4 s before vs the 0.4 s after: a smoothed kick is a curve, not a corner
TOUCH_DV = 2.0           # m/s change in ball velocity across the touch
TOUCH_RADIUS_M = 2.5     # nearest player this close
TOUCH_GAP_S = 0.3        # two touches closer than this are one
OUT_M = 0.3              # beyond the line by more than this: the ball is out (line noise is ~0.1 m)
STILL_SPEED, MOVING_SPEED, QUIET_S = 0.5, 6.0, 1.0   # restart: ball still or unseen for 1 s, then faster than 6 m/s
CAP_S = 60.0             # "time since" before the first event of a period
CTRL_RADIUS_M, CTRL_DV = 1.0, 8.0   # control: a player this close to the ball, moving with it to within 8 m/s
CONTROL_CHANGE = False   # add control_change's cc_ columns: NEUTRAL twice (docs/2026-10-24-turnover-evidence.md)
CC_SCORE_CAP_S = 2.0
INTERCEPT = True         # add intercept()'s icpt_ columns (B1, docs/2026-10-24-intercept.md); on since baseline v7
REACT_S, VMAX = 0.7, 7.0                 # a player: reaction time, then top speed (Spearman 2017's pitch-control defaults)
BALL_DECEL = 3.0                         # m/s^2, a ground ball rolling out
ICPT_HORIZON_S, ICPT_STEP_S = 3.0, 0.2   # path sampled this far ahead, this finely


def since(event, periods, fps, cap_s=CAP_S):
    """Seconds since the last True frame, counted inside each period; cap_s before the first."""
    out = np.full(len(event), cap_s)
    for a, b in periods:
        ev = np.flatnonzero(event[a:b])
        if not len(ev):
            continue
        i = np.arange(b - a)
        pos = np.searchsorted(ev, i, side="right") - 1
        t = np.where(pos >= 0, (i - ev[np.maximum(pos, 0)]) / fps, cap_s)
        out[a:b] = np.minimum(t, cap_s)
    return out


def carry(event, value, periods, default=0.0):
    """value at the last True frame, carried forward inside each period; default before the first."""
    out = np.full(len(event), default, dtype=float)
    for a, b in periods:
        ev = np.flatnonzero(event[a:b])
        if not len(ev):
            continue
        pos = np.searchsorted(ev, np.arange(b - a), side="right") - 1
        out[a:b] = np.where(pos >= 0, value[a + ev[np.maximum(pos, 0)]], default)
    return out


def carry_next(event, value, periods, default=0.0):
    """value at the next True frame, carried backward inside each period (offline); default after the last."""
    T = len(event)
    return carry(event[::-1], value[::-1], [(T - b, T - a) for a, b in periods], default)[::-1]


def ball_out_distance(bxy, length, width):
    x, y = bxy[:, 0], bxy[:, 1]
    with np.errstate(invalid="ignore"):
        return np.maximum.reduce([-x, x - length, -y, y - width, np.zeros_like(x)])  # NaN where unseen


def velocity_change(bxy, fps):
    """|velocity over the next TOUCH_WINDOW_S - velocity over the previous TOUCH_WINDOW_S| per frame, m/s."""
    h = max(1, int(round(TOUCH_WINDOW_S * fps)))
    before, after = np.full(bxy.shape, np.nan), np.full(bxy.shape, np.nan)
    before[h:] = (bxy[h:] - bxy[:-h]) / (h / fps)
    after[:-h] = (bxy[h:] - bxy[:-h]) / (h / fps)
    return np.linalg.norm(after - before, axis=1)


def touches(bxy, pxy, team, fps):
    """(touch frame mask, team of the toucher: +1 home / -1 away / 0). See TOUCH_* above.

    Tuned on IDSSE and Metrica for the worst case: last_touch_team separates H from A with AUC 0.927 on both,
    at 1,800-2,100 touches per 90 (docs/2026-11-10-compliance.md; until 10 Nov the window was 0.2 s, set with our
    public window too)."""
    dv = velocity_change(bxy, fps)
    w = max(1, int(TOUCH_GAP_S * fps))
    peak = pd.Series(dv).rolling(2 * w + 1, center=True, min_periods=1).max().to_numpy()
    with np.errstate(invalid="ignore"):
        cand = (dv > TOUCH_DV) & (dv >= peak)
    d = np.linalg.norm(pxy - bxy[:, None, :], axis=2)
    d = np.where(np.isnan(d), np.inf, d)
    nearest = np.argmin(d, axis=1)
    close = d[np.arange(len(d)), nearest] < TOUCH_RADIUS_M
    hit = cand & close
    who = np.where(hit, np.where(team[nearest] == 0, 1.0, -1.0), 0.0)
    return hit, who


def control(bxy, pxy, team, fps):
    """+1 / -1 where the nearest player moving with the ball (see CTRL_*) is home / away, else 0.

    Catches the receptions and dribbles that touches() misses on our smoothed ball path, where the ball
    slows into a player's stride without a sharp velocity change. CTRL_* chosen on IDSSE and Metrica (damaged like
    our tracking) for the worst group's last_ctrl_team H/A AUC, .885; right on 93-97% of the live frames where it
    fires (34-44% of them). Until 10 Nov 2 m and 2 m/s, checked on our public window too (docs/2026-11-10-compliance.md)."""
    h = max(1, int(round(0.1 * fps)))
    vb, vp = np.full(bxy.shape, np.nan), np.full(pxy.shape, np.nan)
    vb[h:-h] = (bxy[2 * h:] - bxy[:-2 * h]) * fps / (2 * h)
    vp[h:-h] = (pxy[2 * h:] - pxy[:-2 * h]) * fps / (2 * h)
    with np.errstate(invalid="ignore"):
        ok = (np.linalg.norm(pxy - bxy[:, None], axis=2) < CTRL_RADIUS_M) & (np.linalg.norm(vp - vb[:, None], axis=2) < CTRL_DV)
    d = np.where(ok, np.linalg.norm(pxy - bxy[:, None], axis=2), np.inf)
    k = np.argmin(d, axis=1)
    has = np.isfinite(d[np.arange(len(d)), k])
    return np.where(has, np.where(team[k] == 0, 1.0, -1.0), 0.0)


def control_change(hit, who, ctl, periods, fps):
    """The team in control and the moments it changes: touches and control fused into one stream, carried forward.

    A run is a stretch with the same cc_team. Any opponent touch or control frame starts a new one, so a
    passing opponent makes a run a frame or two long with almost no evidence, while a real possession,
    even a 1 s one, gathers touches and frames moving with the ball. On IDSSE, Metrica, SoccerTrack and
    PFF, cc_run_ev_s tells whether cc_team is right with AUC 0.65-0.78.
      cc_team         +1 home / -1 away: the team of the latest touch or control (0 before the first)
      cc_since_s      seconds since cc_team changed; cc_until_s: until it next changes (offline)
      cc_run_ev_s     seconds of touch or control evidence in the current run
      cc_run_touches  touches in the current run (a pass chain)
      cc_score        cc_team * cc_run_ev_s, capped at CC_SCORE_CAP_S"""
    e = np.where(hit, who, ctl)
    team = carry(e != 0, e, periods, 0.0)
    T = len(e)
    start = np.r_[True, team[1:] != team[:-1]]
    for a, b in periods:
        start[a] = True
        if b < T:
            start[b] = True
    end = np.r_[start[1:], True]
    t = np.arange(T) / fps
    run = np.cumsum(start) - 1
    ev = np.bincount(run, weights=e != 0)[run] / fps
    return {"cc_team": team, "cc_since_s": since(start, periods, fps),
            "cc_until_s": np.minimum(carry_next(end, t, periods, np.inf) - t + 1 / fps, CAP_S),
            "cc_run_ev_s": ev, "cc_run_touches": np.bincount(run, weights=hit)[run],
            "cc_score": team * np.minimum(ev, CC_SCORE_CAP_S)}


def intercept(bxy, pxy, team, fps, chunk=20000):
    """Who reaches the ball first on its current path: {icpt_t_h, icpt_t_a, icpt_margin, icpt_dist_m}.

    The ball rolls on from its velocity over 0.2 s, slowing at BALL_DECEL. A player reaches a point after REACT_S
    (carried on by his velocity), then at VMAX. icpt_t_*: the earliest path time any seen player of the team can be
    there, s, ICPT_HORIZON_S + ICPT_STEP_S if none can. icpt_margin: away's minus home's (> 0: home first).
    icpt_dist_m: how far the ball travels before the first team gets it (far: a pass on its way). NaN where the
    ball is unseen. ponytail: no aerial balls (our tracker gives the point under them), no player acceleration."""
    T, h = len(bxy), max(1, int(round(0.1 * fps)))
    vb, vp = np.full(bxy.shape, np.nan), np.full(pxy.shape, np.nan)
    vb[h:-h] = (bxy[2 * h:] - bxy[:-2 * h]) * fps / (2 * h)
    vp[h:-h] = (pxy[2 * h:] - pxy[:-2 * h]) * fps / (2 * h)
    vb, vp = np.nan_to_num(vb), np.nan_to_num(vp)            # unknown velocity: standing still
    tau = np.arange(0, ICPT_HORIZON_S + 1e-9, ICPT_STEP_S)
    cap = ICPT_HORIZON_S + ICPT_STEP_S
    out = {k: np.full(T, np.nan) for k in ("t_h", "t_a", "dist")}
    for a in range(0, T, chunk):
        b, v, q = bxy[a:a + chunk], vb[a:a + chunk], pxy[a:a + chunk] + REACT_S * vp[a:a + chunk]
        speed = np.linalg.norm(v, axis=1, keepdims=True)
        u = np.divide(v, speed, out=np.zeros_like(v), where=speed > 0.5)
        tc = np.minimum(tau, speed / BALL_DECEL)                                                  # (n, K): rolling, then stopped
        s = speed * tc - 0.5 * BALL_DECEL * tc ** 2                                               # metres travelled
        path = b[:, None] + u[:, None] * s[..., None]                                             # (n, K, 2)
        with np.errstate(invalid="ignore"):
            ok = REACT_S + np.linalg.norm(path[:, :, None] - q[:, None], axis=3) / VMAX <= tau[None, :, None]   # (n, K, N)
        for t, tag in ((0, "t_h"), (1, "t_a")):
            any_k = ok[:, :, team == t].any(2)                                                    # (n, K)
            first = np.where(any_k.any(1), any_k.argmax(1), -1)
            out[tag][a:a + chunk] = np.where(first >= 0, tau[first], cap)
        k = np.round(np.minimum(out["t_h"][a:a + chunk], out["t_a"][a:a + chunk]) / ICPT_STEP_S).astype(int)
        out["dist"][a:a + chunk] = s[np.arange(len(s)), np.minimum(k, len(tau) - 1)]
    seen = ~np.isnan(bxy[:, 0])
    f = {"icpt_t_h": out["t_h"], "icpt_t_a": out["t_a"], "icpt_margin": out["t_a"] - out["t_h"], "icpt_dist_m": out["dist"]}
    return {k: np.where(seen, x, np.nan) for k, x in f.items()}


def restarts(ball_speed, fps):
    """Frames where a ball that was still or unseen for QUIET_S starts moving faster than MOVING_SPEED.

    "Unseen" counts as quiet because trackers like ours lose the ball during most stoppages. Against
    the labels' dead-to-live changes (within 2 s), on IDSSE and Metrica damaged like our tracking: F1 .63 / .90,
    the worse group's F1 chosen there (docs/2026-11-10-compliance.md; until 10 Nov 1.0, 5.0, 2.0 with our per-90 rate)."""
    s = pd.Series(ball_speed)
    n, ramp = int(QUIET_S * fps), int(0.3 * fps)   # speed is a 0.2 s average, so it needs ~0.3 s to climb past MOVING_SPEED
    quiet = (s.isna() | (s < STILL_SPEED)).astype(float)
    was_quiet = (quiet.shift(ramp).rolling(n, min_periods=n).min() == 1).to_numpy()
    moving = (s > MOVING_SPEED).to_numpy()
    prev_moving = np.r_[False, moving[:-1]]
    return was_quiet & moving & ~prev_moving


def team_block(pxy, team, t, length, fps):
    """Shape and movement of one team, measured from the goal it defends."""
    x, y = pxy[:, team == t, 0], pxy[:, team == t, 1]
    n = (~np.isnan(x)).sum(1).astype(float)
    with np.errstate(invalid="ignore"), np.testing.suppress_warnings() as sup:
        sup.filter(RuntimeWarning)
        cx, cy = np.nanmean(x, 1), np.nanmean(y, 1)
        length_m = np.nanmax(x, 1) - np.nanmin(x, 1)
        width_m = np.nanmax(y, 1) - np.nanmin(y, 1)
    if t == 0:   # home defends x = 0: its last outfield line is the 2nd smallest x
        line = np.sort(np.where(np.isnan(x), np.inf, x), axis=1)[:, 1] if x.shape[1] > 1 else np.full(len(x), np.nan)
    else:        # away defends x = length
        line = length - np.sort(np.where(np.isnan(x), -np.inf, x), axis=1)[:, -2] if x.shape[1] > 1 else np.full(len(x), np.nan)
    line = np.where(np.isfinite(line), line, np.nan)
    k = int(0.5 * fps)
    adv = np.full(len(x), np.nan)
    adv[k:-k] = (cx[2 * k:] - cx[:-2 * k]) / (2 * k / fps) * (1 if t == 0 else -1)   # + = moving towards the goal it attacks
    s = clean.speed(pxy[:, team == t], fps)
    with np.errstate(invalid="ignore"), np.testing.suppress_warnings() as sup:
        sup.filter(RuntimeWarning)
        spd = np.nanmean(np.where(s < clean.MAX_PLAYER_SPEED, s, np.nan), 1)
    tag = "h" if t == 0 else "a"
    attack_x = cx if t == 0 else length - cx      # centroid measured towards the goal this team attacks
    return {"n_" + tag: n, "cx_" + tag: attack_x, "cy_" + tag: cy, "length_" + tag: length_m, "width_" + tag: width_m,
            "line_" + tag: line, "advance_" + tag: adv, "speed_" + tag: spd}


def frame_features(c):
    """DataFrame, one row per frame of a cleaned match `c` (clean.clean_match)."""
    fps, T = c.fps, len(c.bxy)
    f = {}
    seen = c.bdet & ~np.isnan(c.bxy[:, 0])
    f["ball_seen"] = seen.astype(float)
    f["ball_x"], f["ball_y"] = c.bxy[:, 0], c.bxy[:, 1]
    f["ball_speed"] = c.ball_speed
    f["ball_fast"] = c.ball_fast.astype(float)
    gaps = clean.ball_gap_features(seen, fps)
    f["gap_len_s"], f["time_in_gap_s"], f["time_to_gap_end_s"] = gaps["gap_len_s"], gaps["time_in_gap_s"], gaps["time_to_gap_end_s"]
    out_m = ball_out_distance(c.bxy, c.length, c.width)
    f["ball_out_m"] = out_m

    # nearest player of each team, and how crowded the ball is
    d = np.linalg.norm(c.pxy - c.bxy[:, None, :], axis=2)
    for t, tag in ((0, "h"), (1, "a")):
        dt = d[:, c.team == t]
        with np.errstate(invalid="ignore"), np.testing.suppress_warnings() as sup:
            sup.filter(RuntimeWarning)
            f["d_" + tag] = np.nanmin(dt, 1)
        f["within2_" + tag] = (dt < 2).sum(1).astype(float)
        f["within5_" + tag] = (dt < 5).sum(1).astype(float)
    f["d_margin"] = f["d_a"] - f["d_h"]   # > 0: a home player is nearer

    for t in (0, 1):
        f.update(team_block(c.pxy, c.team, t, c.length, fps))
    f["speed_all_2s"] = clean.mean_player_speed(c.pxy, fps)
    s = clean.speed(c.pxy, fps)
    with np.errstate(invalid="ignore"), np.testing.suppress_warnings() as sup:
        sup.filter(RuntimeWarning)
        f["slow_share"] = np.nanmean(np.where(np.isnan(s), np.nan, s < 1.0), 1)

    # events
    with np.errstate(invalid="ignore"):
        out = out_m > OUT_M
    out_start = out & ~np.r_[False, out[:-1]]
    f["t_since_ball_out_s"] = since(out_start, c.periods, fps)
    rs = restarts(c.ball_speed, fps)
    f["t_since_restart_s"] = since(rs, c.periods, fps)
    hit, who = touches(c.bxy, c.pxy, c.team, fps)
    f["t_since_touch_s"] = since(hit, c.periods, fps)
    f["last_touch_team"] = carry(hit, who, c.periods, 0.0)       # +1 home, -1 away, 0 none yet
    ctl = control(c.bxy, c.pxy, c.team, fps)
    f["ctrl_team"] = ctl                                           # +1 home, -1 away, 0 no one
    f["last_ctrl_team"] = carry(ctl != 0, ctl, c.periods, 0.0)
    f["next_ctrl_team"] = carry_next(ctl != 0, ctl, c.periods, 0.0)
    if CONTROL_CHANGE:
        f.update(control_change(hit, who, ctl, c.periods, fps))
    if INTERCEPT:
        f.update(intercept(c.bxy, c.pxy, c.team, fps))
    if getattr(c, "ball_source", None) is not None:   # validate.BALL_SOURCE_FLAG: which source gave the ball
        f["src_track"], f["src_video"] = (c.ball_source == 2).astype(float), (c.ball_source == 3).astype(float)
    f["forced_dead"] = c.forced_dead.astype(float)
    return pd.DataFrame(f).astype("float32")


RAW_NEAREST = 5   # players in raw_features


def raw_features(c):
    """Ball-centric raw inputs (Sanford 2020) for phase4's raw member, every column prefixed raw_: the ball, its last
    seen place carried through gaps, and the RAW_NEAREST players nearest it, nearest first: offset from the ball,
    velocity, distance and side (+1 home / -1 away, 0 where fewer players are tracked). phase3.mirror turns them."""
    T, h = len(c.bxy), max(1, int(round(0.1 * c.fps)))
    seen = ~np.isnan(c.bxy[:, 0])
    last = np.maximum.accumulate(np.where(seen, np.arange(T), -1))   # ponytail: carried across half-time, which isn't graded
    b = np.where(last[:, None] >= 0, c.bxy[np.maximum(last, 0)], [c.length / 2, c.width / 2])
    pad = lambda xy: np.full((h,) + xy.shape[1:], np.nan)
    vel = lambda xy: np.concatenate([pad(xy), (xy[2 * h:] - xy[:-2 * h]) * c.fps / (2 * h), pad(xy)])
    f = {"raw_ball_x": b[:, 0], "raw_ball_y": b[:, 1], "raw_ball_seen": seen.astype(float)}
    f["raw_ball_vx"], f["raw_ball_vy"] = np.nan_to_num(vel(c.bxy)).T
    rel, vp = c.pxy - b[:, None], vel(c.pxy)
    d = np.linalg.norm(rel, axis=2)
    order = np.argsort(np.where(np.isnan(d), np.inf, d), axis=1)[:, :RAW_NEAREST]
    side, t = np.where(c.team == 0, 1.0, -1.0), np.arange(T)
    for k in range(order.shape[1]):
        i = order[:, k]
        f["raw_p{}_dx".format(k)], f["raw_p{}_dy".format(k)] = rel[t, i].T
        f["raw_p{}_vx".format(k)], f["raw_p{}_vy".format(k)] = vp[t, i].T
        f["raw_p{}_d".format(k)] = d[t, i]
        f["raw_p{}_side".format(k)] = np.where(np.isfinite(d[t, i]), side[i], 0.0)
    return pd.DataFrame(f).astype("float32")


def tag_features(tags, n_frames):
    """Our match only: one 0/1 column per Smart Tagging code and team (twins removed), plus set pieces."""
    kept, _, _ = clean.dedupe_tags(tags)
    cols = {}
    for code in sorted(set(kept.code)):
        for side in ("H", "A", "N/A"):
            sub = kept[(kept.code == code) & (kept.side == side)]
            if len(sub):
                m = np.zeros(n_frames, np.float32)
                for a, b in zip(sub.start_frame, sub.end_frame):
                    m[max(0, a):min(n_frames, b)] = 1
                cols["tag_{}_{}".format(code.lower().replace(" ", "_"), side.replace("/", "").lower())] = m
    return pd.DataFrame(cols)
