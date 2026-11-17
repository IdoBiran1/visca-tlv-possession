"""Load our match and the public external datasets into one shape, and profile them the same way.

Every loader returns a Match: metres from the pitch corner, a team per player slot, the ball with a
"detected" mask, and per-frame labels H / A / D / X (X = no label) where the dataset has any. The
profile then measures each the same way, so the numbers sit side by side.

    MTAH_REPO=... python external/profile_datasets.py            # writes out/dataset_profiles.json

External data lives outside the repo (EXT, below). Our match's numbers are competition data: the
output goes to out/, which is gitignored.
"""
import csv
import json
import os
import sys
from dataclasses import dataclass

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
EXT = os.path.expanduser(os.environ.get("MTAH_EXT", "~/Github/external_data"))
CACHE = os.path.join(EXT, "_cache")


@dataclass
class Match:
    name: str
    source: str
    fps: float
    length: float
    width: float
    pxy: np.ndarray          # [T, P, 2] metres, NaN where the slot is empty
    team: np.ndarray         # [P] 0 home, 1 away
    bxy: np.ndarray          # [T, 2] metres, NaN where missing
    bdet: np.ndarray         # [T] the ball was actually observed (not filled in)
    pdet: np.ndarray         # [T, P] the player was actually observed
    periods: list            # [(start, stop)] half-open in-play windows
    labels: np.ndarray = None  # [T] of H / A / D / X, or None
    label_note: str = ""


# ---------------------------------------------------------------- loaders

def load_ours():
    import preprocess as pp
    trk = pp.load_tracking()
    n = len(trk["ball"])
    gt = pp.read_labels(os.path.join(pp.REPO, "evaluation", "gt", "BAR-ATH_possession_gt_public.csv"))
    periods = [(a, b + 1) for _, a, b in pp.half_video_blocks(pp.GAME)]
    pdet = ~np.isnan(trk["xy"][..., 0])
    return Match("BAR-ATH (ours)", "Metrica ATD, tactical camera", pp.FPS, pp.PITCH[0], pp.PITCH[1],
                 trk["xy"], trk["team"].astype(int), trk["ball"], ~np.isnan(trk["ball"][:, 0]), pdet,
                 periods, gt, "official answer key, public 5 minutes only")


def load_idsse(match_id):
    path = os.path.join(CACHE, "idsse_{}.npz".format(match_id))
    if not os.path.isfile(path):
        from floodlight.io import dfl
        d = os.path.join(EXT, "idsse")
        pos = [f for f in os.listdir(d) if f.startswith("DFL_04_03") and match_id in f][0]
        info = [f for f in os.listdir(d) if f.startswith("DFL_02_01") and match_id in f][0]
        xy, poss, status, _, pitch = dfl.read_position_data_xml(os.path.join(d, pos), os.path.join(d, info))
        halves = [h for h in ("firstHalf", "secondHalf") if h in xy]
        p_parts, b_parts, lab_parts, periods, t0 = [], [], [], [], 0
        width_slots = max(xy[h]["Home"].xy.shape[1] for h in halves) // 2, max(xy[h]["Away"].xy.shape[1] for h in halves) // 2
        for h in halves:
            home, away, ball = xy[h]["Home"].xy, xy[h]["Away"].xy, xy[h]["Ball"].xy
            T = len(ball)
            frame = np.full((T, sum(width_slots), 2), np.nan)
            frame[:, :home.shape[1] // 2] = home.reshape(T, -1, 2)
            frame[:, width_slots[0]:width_slots[0] + away.shape[1] // 2] = away.reshape(T, -1, 2)
            p_parts.append(frame); b_parts.append(ball[:, :2])
            pc, sc = poss[h].code, status[h].code
            lab_parts.append(np.where(sc == 0, "D", np.where(pc == 1, "H", np.where(pc == 2, "A", "X"))))
            periods.append((t0, t0 + T)); t0 += T
        # floodlight puts the origin at the centre spot; move it to the corner like ours
        x0, y0 = pitch.xlim[0], pitch.ylim[0]
        pxy = np.concatenate(p_parts) - [x0, y0]
        bxy = np.concatenate(b_parts) - [x0, y0]
        team = np.r_[np.zeros(width_slots[0], int), np.ones(width_slots[1], int)]
        os.makedirs(CACHE, exist_ok=True)
        np.savez(path, pxy=pxy, bxy=bxy, team=team, labels=np.concatenate(lab_parts), periods=np.array(periods),
                 L=pitch.xlim[1] - pitch.xlim[0], W=pitch.ylim[1] - pitch.ylim[0], fps=xy[halves[0]]["Ball"].framerate)
    z = np.load(path)
    bxy = z["bxy"]
    return Match("IDSSE " + match_id, "DFL / TRACAB optical, multi-camera", float(z["fps"]), float(z["L"]),
                 float(z["W"]), z["pxy"], z["team"], bxy, ~np.isnan(bxy[:, 0]), ~np.isnan(z["pxy"][..., 0]),
                 [tuple(p) for p in z["periods"]], z["labels"], "DFL BallPossession + BallStatus flags, every frame")


def _metrica_csv(game):
    base = os.path.join(EXT, "metrica", "data", "Sample_Game_{}".format(game), "Sample_Game_{}_".format(game))
    sides = []
    for side in ("Home", "Away"):
        rows = list(csv.reader(open(base + "RawTrackingData_{}_Team.csv".format(side))))
        data = np.array(rows[3:], dtype=float)
        sides.append(data)
    period, home, away = sides[0][:, 0], sides[0][:, 3:-2], sides[1][:, 3:-2]
    ball = sides[0][:, -2:]
    L, W = 105.0, 68.0
    T = len(period)
    to_m = lambda a: a.reshape(T, -1, 2) * [L, W]
    pxy = np.concatenate([to_m(home), to_m(away)], axis=1)
    team = np.r_[np.zeros(home.shape[1] // 2, int), np.ones(away.shape[1] // 2, int)]
    bxy = ball * [L, W]
    periods = [(int(np.flatnonzero(period == p)[0]), int(np.flatnonzero(period == p)[-1]) + 1) for p in (1, 2)]
    events = list(csv.DictReader(open(base + "RawEventsData.csv")))
    return T, L, W, pxy, team, bxy, periods, events


METRICA_MIN_RESTART_S = 3.0   # a PASS / BALL LOST / SHOT this soon after a stoppage starts is part of it, not a restart


def metrica_labels(T, events, fps=25):
    """H / A / D from Metrica's event stream (an approximation: these games ship no possession labels).

    The team of the last on-ball event owns the ball. Dead ball runs from a stoppage to the restart.
    - Stoppages: the ball crossing a line (a BALL OUT, or a shot out or in, at the event's end frame), a foul (FAULT
      RECEIVED, or a CHALLENGE ...-FAULT-... without advantage), a hand ball, an offside (BALL LOST subtypes, the
      offside at its end frame) or the end of a half.
    - Restarts: a SET PIECE, a goal kick (Metrica types them PASS / BALL LOST / BALL OUT with "GOAL KICK" in the
      subtype), or the first PASS, BALL LOST or SHOT at least METRICA_MIN_RESTART_S after the stoppage began (a keeper's
      kick or a throw-in Metrica didn't type as a set piece).
    - A set piece with no stoppage open (one Metrica didn't record) makes D from the end of the last event before it.
    """
    lab = np.full(T, "X", dtype="<U1")
    side = {"Home": "H", "Away": "A"}
    dead_from, last_end = None, None
    marks = []  # (frame, label from here on)
    for e in sorted(events, key=lambda e: int(e["Start Frame"])):
        f, t, sub = int(e["Start Frame"]) - 1, e["Type"], e["Subtype"]
        end = max(f, int(e["End Frame"] or e["Start Frame"]) - 1)
        late = dead_from is not None and f >= dead_from + METRICA_MIN_RESTART_S * fps
        if t == "SET PIECE" or "GOAL KICK" in sub or (late and t in ("PASS", "BALL LOST", "SHOT")):
            if dead_from is None and t == "SET PIECE" and last_end is not None and last_end < f:
                dead_from = last_end
            if dead_from is not None:
                marks.append((dead_from, "D")); dead_from = None
            marks.append((f, side[e["Team"]]))
        elif t in ("PASS", "RECOVERY", "SHOT") and dead_from is None:
            marks.append((f, side[e["Team"]]))
        if dead_from is None:
            if t == "BALL OUT" or (t == "SHOT" and ("OUT" in sub or "GOAL" in sub)) or "END HALF" in sub or "OFFSIDE" in sub:
                dead_from = end
            elif t == "FAULT RECEIVED" or "HAND BALL" in sub or (t == "CHALLENGE" and "FAULT" in sub and "ADVANTAGE" not in sub):
                dead_from = f
        if t != "CARD":
            last_end = end
    if dead_from is not None:   # a stoppage still open when the events end (the final whistle)
        marks.append((dead_from, "D"))
    marks.sort()
    for (f, l), (g, _) in zip(marks, marks[1:] + [(T, None)]):
        lab[f:g] = l
    return lab


def load_metrica(game):
    T, L, W, pxy, team, bxy, periods, events = _metrica_csv(game)
    lab = metrica_labels(T, events)
    return Match("Metrica sample {}".format(game), "Metrica, anonymised match", 25.0, L, W, pxy, team, bxy,
                 ~np.isnan(bxy[:, 0]), ~np.isnan(pxy[..., 0]), periods, lab, "derived from Metrica events (approximate)")


def load_metrica3():
    """Metrica sample game 3: EPTS tracking (read with kloppy) and JSON events, in the same shape as games 1-2.

    Its EPTS layout differs from ours (x,y per player, no speed; the player set changes at each
    substitution), so preprocess.parse_atd can't read it; external/test_parsers.py checks that
    kloppy reads our own match exactly like parse_atd does before we trust it here.
    """
    path = os.path.join(CACHE, "metrica3_v2.npz")   # v2 keeps the events: labels are derived on load, never cached
    base = os.path.join(EXT, "metrica", "data", "Sample_Game_3", "Sample_Game_3_")
    if not os.path.isfile(path):
        from kloppy import metrica
        ds = metrica.load_tracking_epts(meta_data=base + "metadata.xml", raw_data=base + "tracking.txt", coordinates="metrica")
        df = ds.to_df(engine="pandas")
        players = [p for t in ds.metadata.teams for p in t.players]
        home = ds.metadata.teams[0]
        L, W = 105.0, 68.0
        pxy = np.stack([np.c_[df["{}_x".format(p.player_id)].to_numpy(float), df["{}_y".format(p.player_id)].to_numpy(float)]
                        for p in players], axis=1) * [L, W]
        team = np.array([0 if p.team == home else 1 for p in players])
        bxy = np.c_[df["ball_x"].to_numpy(float), df["ball_y"].to_numpy(float)] * [L, W]
        per = df["period_id"].to_numpy()
        periods = np.array([(int(np.flatnonzero(per == k)[0]), int(np.flatnonzero(per == k)[-1]) + 1) for k in (1, 2)])
        shift = int(df["frame_id"].iloc[0]) - 1  # events count frames like the tracking, from 1
        events = []
        for e in json.load(open(base + "events.json"))["data"]:
            subs = e["subtypes"] or []
            subs = subs if isinstance(subs, list) else [subs]
            start = e["start"]["frame"] - shift
            end = ((e.get("end") or {}).get("frame") or e["start"]["frame"]) - shift
            events.append({"Team": "Home" if e["team"]["id"] == home.team_id else "Away", "Type": e["type"]["name"],
                           "Subtype": " ".join(s["name"] for s in subs), "Start Frame": str(start), "End Frame": str(end)})
        os.makedirs(CACHE, exist_ok=True)
        np.savez(path, pxy=pxy, bxy=bxy, team=team, periods=periods, events=json.dumps(events))
    z = np.load(path)
    bxy = z["bxy"]
    return Match("Metrica sample 3", "Metrica, anonymised match (EPTS)", 25.0, 105.0, 68.0, z["pxy"], z["team"], bxy,
                 ~np.isnan(bxy[:, 0]), ~np.isnan(z["pxy"][..., 0]), [tuple(p) for p in z["periods"]],
                 metrica_labels(len(bxy), json.loads(str(z["events"]))), "derived from Metrica events (approximate)")


SOCCERTRACK_LABELLED = ("117092", "117093", "118575", "118576", "118577", "118578", "128057", "128058")
# 132831 and 132877 carry ballStatus INPLAY on every frame: no possession labels


def dequantize(xy, periods, fps, sigma_s):
    """Gaussian over time along each unbroken run of positions, inside each period (NaN stays NaN)."""
    from scipy.ndimage import gaussian_filter1d
    out = xy.copy()
    flat = out.reshape(len(out), -1, 2)
    for a, b in periods:
        for s in range(flat.shape[1]):
            seen = ~np.isnan(flat[a:b, s, 0])
            e = np.flatnonzero(np.diff(np.r_[0, seen.astype(np.int8), 0]))
            for x, y in zip(e[::2] + a, e[1::2] + a):
                if y - x > 2:
                    flat[x:y, s] = gaussian_filter1d(flat[x:y, s], sigma_s * fps, axis=0, mode="nearest")
    return out


def soccertrack_labels(status, period_starts):
    """H / A / D / X per frame from SoccerTrack's ballStatus: BALLOUT -> D, NEUTRAL keeps the last team of its half
    (X before the first one), anything else (INPLAY) -> X."""
    out, last, starts = [], None, set(int(v) for v in period_starts)
    for t, st in enumerate(status):
        if t in starts:
            last = None
        if st in ("HOME", "AWAY"):
            last = "H" if st == "HOME" else "A"
        out.append("D" if st == "BALLOUT" else (last or "X") if st in ("HOME", "AWAY", "NEUTRAL") else "X")
    return np.array(out, dtype="<U1")


def load_soccertrack(match_id, smooth_s=0.3):
    """SoccerTrack v2 match `match_id` (atomscott/soccertrack-v2 on Hugging Face, CC BY 4.0): ten university matches
    from a fixed panoramic camera over the whole pitch. Uses the provider's raw tracking export, whose every frame
    carries a ballStatus: HOME / AWAY / BALLOUT / NEUTRAL. Labels: BALLOUT -> D; NEUTRAL (loose or contested,
    5-9% of frames) stays with the last team, as the challenge rules say; X before the first team in a half.

    The export rounds positions to 0.01 of the pitch (1.05 m x 0.68 m), which over a 0.2 s window looks like
    5 m/s jumps, so every track is smoothed with a Gaussian of `smooth_s`. The ball is a reconstruction anchored
    on events (zero acceleration on ~58% of frames, steps far beyond any kick), not a measurement.
    Only the two halves are kept (one export also has an unfilmed third period)."""
    import re
    import xml.etree.ElementTree as ET
    path = os.path.join(CACHE, "soccertrack_{}.npz".format(match_id))
    if not os.path.isfile(path):
        base = os.path.join(EXT, "soccertrack", "raw", match_id, match_id + "_tracker_box_")
        meta = ET.parse(base + "metadata.xml").getroot()
        side = {t.get("id"): t.get("side") for t in meta.iter("team")}
        team_of = {p.get("id"): 0 if side[p.get("teamId")] == "home" else 1 for p in meta.find("players")}
        loc = re.compile(r'loc="\[([^,\]]+), ([^\]]+)\]"')
        ids, per, status, players, ball, keep = {}, [], [], [], [], False
        for line in open(base + "data.xml"):
            s = line.lstrip()
            if s.startswith("<frame"):
                p = re.search(r'eventPeriod="([^"]*)"', s).group(1)
                keep = p in ("FIRST_HALF", "SECOND_HALF")
                if keep:
                    per.append(p); status.append(re.search(r'ballStatus="([^"]*)"', s).group(1))
                    players.append([]); ball.append((np.nan, np.nan))
            elif keep and s.startswith("<player"):
                m = loc.search(s)
                players[-1].append((ids.setdefault(re.search(r'playerId="([^"]*)"', s).group(1), len(ids)),
                                    float(m.group(1)), float(m.group(2))))
            elif keep and s.startswith("<ball"):
                m = loc.search(s)
                if m:
                    ball[-1] = (float(m.group(1)), float(m.group(2)))
        T = len(per)
        pxy = np.full((T, len(ids), 2), np.nan, np.float32)
        for t, row in enumerate(players):
            for i, x, y in row:
                pxy[t, i] = x, y
        pxy *= [105.0, 68.0]
        bxy = np.array(ball, np.float32) * [105.0, 68.0]
        team = np.array([team_of[pid] for pid in ids])
        per = np.array(per)
        periods = np.array([(int(np.flatnonzero(per == p)[0]), int(np.flatnonzero(per == p)[-1]) + 1)
                            for p in ("FIRST_HALF", "SECOND_HALF")])
        labels = soccertrack_labels(status, periods[:, 0])
        os.makedirs(CACHE, exist_ok=True)
        np.savez(path, pxy=pxy, bxy=bxy, team=team, periods=periods, labels=labels)
    z = np.load(path)
    periods = [tuple(int(v) for v in p) for p in z["periods"]]
    pxy = dequantize(z["pxy"].astype(float), periods, 25.0, smooth_s)
    bxy = dequantize(z["bxy"].astype(float), periods, 25.0, smooth_s)
    return Match("SoccerTrack v2 {}".format(match_id), "SoccerTrack v2, fixed panoramic camera (provider export)", 25.0,
                 105.0, 68.0, pxy, z["team"], bxy, ~np.isnan(bxy[:, 0]), ~np.isnan(pxy[..., 0]), periods, z["labels"],
                 "the provider's per-frame ballStatus (NEUTRAL kept with the last team)")


PFF_HELD_OUT = ("10502", "10506", "10510", "10514", "3812", "3816", "3820", "3824",
                "3828", "3832", "3836", "3840", "3844", "3848", "3852", "3856")
# every 4th of the 64 match ids in sorted order: a fixed quarter, so a harness run stays affordable


def pff_labels(events, times, periods):
    """H / A / D / X at `times` (video seconds) from PFF's event data, in `periods` [(start, end)] seconds.

    PFF's game events mark a player in control of the ball (OTB, and the kick-offs), a ball going out (OUT) and
    the restart's set-piece type. Possession is the team of the last control event, so a loose ball or a touch
    without control keeps the last team, as the challenge rules say. D runs from an OUT to the next control
    event, or, for a set-piece restart with no OUT before it (foul, offside, goal), from the end of the last
    control event to the restart."""
    seen, ge = set(), []
    for e in sorted(events, key=lambda e: e["startTime"]):
        g = e.get("gameEvents") or {}
        if e["gameEventId"] in seen or not g.get("gameEventType"):
            continue
        seen.add(e["gameEventId"])
        ge.append((e["startTime"], e.get("endTime") or e["startTime"], g["gameEventType"], g.get("setpieceType"), g.get("homeTeam")))
    marks = []                                   # (time, label from here on)
    last_end, out_at = None, None
    for start, end, typ, setpiece, home in ge:
        if typ == "OUT":
            out_at = start
        elif typ == "OTB" or typ.endswith("KICKOFF"):
            if home is None:
                continue
            dead_from = out_at if out_at is not None else last_end if setpiece not in (None, "O") else None
            if dead_from is not None and dead_from < start:
                marks.append((dead_from, "D"))
            marks.append((start, "H" if home else "A"))
            last_end, out_at = end, None
    out = np.full(len(times), "X", dtype="<U1")
    mt = np.array([m[0] for m in marks]); ml = np.array([m[1] for m in marks])
    for a, b in periods:
        k = (times >= a) & (times < b)
        i = np.searchsorted(mt, times[k], side="right") - 1
        out[k] = np.where(i >= 0, ml[np.maximum(i, 0)], "X")
    return out


def load_pff(match_id, camera=False):
    """PFF FC World Cup 2022 match `match_id` (64 matches, registration-gated, specs marked "PFFA Acquisition LLC
    proprietary information"): Sportlogiq broadcast tracking merged with PFF's event data, at 29.97 fps,
    resampled here to 25 fps by nearest frame. Players are slotted by team and shirt number; their raw positions
    are kept even where the broadcast didn't see them (69% of player positions are ESTIMATED from context).
    Raw, not PFF's Kalman-smoothed tracks: those have no identity jumps but slow everyone down (90th-percentile
    speed 3.45 m/s against 4.4 raw), so the model called 51% dead ball against 41% true. The
    ball is the raw track where there is one, else PFF's event-interpolated one, else a straight line between the
    two, so it is complete like IDSSE's, and bdet / pdet say so: the harness then hides the ball the way our
    tracker loses it. camera=True gives what the broadcast actually saw instead (VISIBLE), for profiling.
    Labels: pff_labels, from the latest event files.
    Only the two halves: in extra time the teams come out swapped (labels agree with the nearest player 4-11%
    in 10506 and 10517, against 92% elsewhere), and our match has none."""
    import bz2
    path = os.path.join(CACHE, "pff_{}.npz".format(match_id))
    if not os.path.isfile(path):
        D = os.path.join(EXT, "pff")
        events = json.load(open(os.path.join(D, "Event Data", "{}.json".format(match_id))))
        stadium = next((e.get("stadiumMetadata") for e in events if e.get("stadiumMetadata")), None) or {}
        L, W = float(stadium.get("pitchLength") or 105.0), float(stadium.get("pitchWidth") or 68.0)
        bounds = {}
        for e in events:
            g = e.get("gameEvents") or {}
            p, typ = g.get("period"), g.get("gameEventType")
            if p in (1, 2) and typ:             # extra time has home and away swapped in at least 2 matches
                lo, hi = bounds.get(p, (np.inf, None))
                bounds[p] = (min(lo, e["startTime"]), e["startTime"] if typ == "END" else hi)
        periods_s = [bounds[p] for p in sorted(bounds) if bounds[p][1] is not None]
        t_src, frames, seen = [], [], set()
        with bz2.open(os.path.join(D, "Tracking Data", "{}.jsonl.bz2".format(match_id)), "rt") as fh:
            for line in fh:
                d = json.loads(line)
                if d["frameNum"] in seen:        # frames with several possession events are repeated
                    continue
                seen.add(d["frameNum"])
                ball = (d.get("balls") or [None])[0] or (d.get("ballsSmoothed") or [None])[0]
                vis = bool(d.get("balls")) and (d["balls"][0] or {}).get("visibility") == "VISIBLE"
                pl = [(t, p.get("jerseyNum"), p["x"], p["y"], p.get("visibility") == "VISIBLE")
                      for t, key in ((0, "homePlayers"), (1, "awayPlayers")) for p in (d.get(key) or []) if p.get("x") is not None]
                t_src.append(d["videoTimeMs"] / 1000.0)
                frames.append((pl, (ball["x"], ball["y"]) if ball and ball.get("x") is not None else (np.nan, np.nan), vis))
        t_src = np.array(t_src); order = np.argsort(t_src); t_src = t_src[order]; frames = [frames[i] for i in order]
        t0, t1 = periods_s[0][0], periods_s[-1][1]
        times = np.arange(np.floor(t0 * 25), np.ceil(t1 * 25)) / 25.0
        src = np.clip(np.searchsorted(t_src, times), 1, len(t_src) - 1)
        src -= (times - t_src[src - 1]) < (t_src[src] - times)     # nearest source frame
        slots = {}
        for pl, _, _ in frames:
            for t, jersey, *_ in pl:
                slots.setdefault((t, jersey), len(slots))
        T, P = len(times), len(slots)
        pxy = np.full((T, P, 2), np.nan, np.float32); pvis = np.zeros((T, P), bool)
        bxy = np.full((T, 2), np.nan, np.float32); bvis = np.zeros(T, bool)
        for k, j in enumerate(src):
            if abs(t_src[j] - times[k]) > 0.1:   # no tracking within 0.1 s (half-time, video gaps)
                continue
            pl, ball, vis = frames[j]
            for t, jersey, x, y, v in pl:
                i = slots[(t, jersey)]
                pxy[k, i] = x + L / 2, y + W / 2; pvis[k, i] = v
            bxy[k] = ball[0] + L / 2, ball[1] + W / 2; bvis[k] = vis
        team = np.array([t for t, _ in slots])
        periods = np.array([(int(np.searchsorted(times, a)), int(np.searchsorted(times, b))) for a, b in periods_s])
        for a, b in periods:                     # where neither ball track has a position, join the ends linearly
            ok = np.flatnonzero(~np.isnan(bxy[a:b, 0]))
            if len(ok) > 1:
                for c in (0, 1):
                    bxy[a:b, c] = np.interp(np.arange(b - a), ok, bxy[a:b, c][ok])
        labels = pff_labels(events, times, periods_s)
        os.makedirs(CACHE, exist_ok=True)
        np.savez(path, pxy=pxy, pvis=pvis, bxy=bxy, bvis=bvis, team=team, periods=periods, labels=labels, L=L, W=W)
    z = np.load(path)
    pxy, bxy = z["pxy"].astype(float), z["bxy"].astype(float)
    bdet, pdet = (z["bvis"], z["pvis"]) if camera else (~np.isnan(bxy[:, 0]), ~np.isnan(pxy[..., 0]))
    return Match("PFF WC2022 {}".format(match_id), "PFF FC / Sportlogiq broadcast tracking (players partly estimated)", 25.0,
                 float(z["L"]), float(z["W"]), pxy, z["team"], bxy, bdet, pdet,
                 [tuple(int(v) for v in p) for p in z["periods"]], z["labels"], "derived from PFF game events (control, out, restarts)")


def load_skillcorner(match_id):
    d = os.path.join(EXT, "skillcorner", "data", "matches", str(match_id))
    meta = json.load(open(os.path.join(d, "{}_match.json".format(match_id))))
    home_id = meta["home_team"]["id"]
    slot = {p["id"]: i for i, p in enumerate(meta["players"])}  # tracking's player_id is the player's id
    team = np.array([0 if p["team_id"] == home_id else 1 for p in meta["players"]])
    rows = [json.loads(l) for l in open(os.path.join(d, "{}_tracking_extrapolated.jsonl".format(match_id)))]
    T, P = len(rows), len(slot)
    pxy = np.full((T, P, 2), np.nan); pdet = np.zeros((T, P), bool)
    bxy = np.full((T, 2), np.nan); bdet = np.zeros(T, bool)
    lab = np.full(T, "X", dtype="<U1"); per = np.zeros(T, int)
    L, W = float(meta["pitch_length"]), float(meta["pitch_width"])
    for t, r in enumerate(rows):
        per[t] = r["period"] or 0
        b = r["ball_data"]
        if b["x"] is not None:
            bxy[t] = b["x"] + L / 2, b["y"] + W / 2; bdet[t] = bool(b["is_detected"])
        for p in r["player_data"]:
            i = slot.get(p["player_id"])
            if i is not None:
                pxy[t, i] = p["x"] + L / 2, p["y"] + W / 2; pdet[t, i] = bool(p["is_detected"])
        g = r["possession"]["group"]
        lab[t] = "H" if g == "home team" else "A" if g == "away team" else "X"
    periods = [(int(np.flatnonzero(per == k)[0]), int(np.flatnonzero(per == k)[-1]) + 1) for k in (1, 2)]
    return Match("SkillCorner {}".format(match_id), "SkillCorner, broadcast video", 10.0, L, W, pxy, team, bxy,
                 bdet, pdet, periods, lab, "possession group (H/A only; no dead-ball flag, blank = X)")


# ---------------------------------------------------------------- profile

def runs(mask):
    m = np.r_[False, mask, False]
    e = np.flatnonzero(m[1:] != m[:-1])
    return list(zip(e[::2], e[1::2]))


def auc(pos, neg):
    """P(random positive scores above random negative), ties counted half: the Mann-Whitney U."""
    pos, neg = pos[~np.isnan(pos)], neg[~np.isnan(neg)]
    if not len(pos) or not len(neg):
        return float("nan")
    allv = np.r_[pos, neg]
    order = allv.argsort(kind="mergesort")
    ranks = np.empty(len(allv)); ranks[order] = np.arange(1, len(allv) + 1)
    _, inv, cnt = np.unique(allv, return_inverse=True, return_counts=True)
    sums = np.bincount(inv, ranks); ranks = (sums / cnt)[inv]  # average ranks for ties
    return float((ranks[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def centred_mean(x, w):
    ok = ~np.isnan(x)
    c = np.convolve(np.where(ok, x, 0), np.ones(w), "same")
    n = np.convolve(ok.astype(float), np.ones(w), "same")
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(n > 0, c / n, np.nan)


def profile(m: Match):
    fps, T = m.fps, len(m.bxy)
    graded = np.zeros(T, bool)
    for a, b in m.periods:
        graded[a:b] = True
    minutes = graded.sum() / fps / 60
    out = {"name": m.name, "source": m.source, "fps": fps, "pitch_m": [round(m.length, 1), round(m.width, 1)],
           "graded_minutes": round(minutes, 1), "player_slots": int(m.pxy.shape[1]), "labels": m.label_note}

    n_tracked = (~np.isnan(m.pxy[..., 0])).sum(1)[graded]
    n_det = m.pdet.sum(1)[graded]
    out["players_present_mean"] = round(float(n_tracked.mean()), 1)
    out["players_detected_mean"] = round(float(n_det.mean()), 1)
    out["frames_over_22_present"] = round(float((n_tracked > 22).mean()), 3)

    bdet = m.bdet & graded
    out["ball_present_share"] = round(float((~np.isnan(m.bxy[:, 0]))[graded].mean()), 3)
    out["ball_detected_share"] = round(float(m.bdet[graded].mean()), 3)
    gaps = np.array([(b - a) / fps for a, b in runs(~m.bdet & graded)])
    miss = (~m.bdet & graded).sum() / fps
    out["ball_gaps_per_90"] = round(len(gaps) / minutes * 90, 1)
    out["ball_gap_median_s"] = round(float(np.median(gaps)), 2) if len(gaps) else 0
    out["missing_share_in_gaps_over_2s"] = round(float(gaps[gaps > 2].sum() / miss), 3) if miss else 0
    out["missing_share_in_gaps_over_10s"] = round(float(gaps[gaps > 10].sum() / miss), 3) if miss else 0

    step = max(1, int(round(fps / 5)))  # 0.2 s displacement: same physical window at 10 and 25 fps
    dt = step / fps
    bv = np.full(T, np.nan); bv[step:] = np.linalg.norm(m.bxy[step:] - m.bxy[:-step], axis=1) / dt
    both = m.bdet.copy(); both[step:] &= m.bdet[:-step]; both[:step] = False
    bvs = bv[both & graded]
    out["ball_speed_p50_p99"] = [round(float(np.nanpercentile(bvs, q)), 1) for q in (50, 99)] if len(bvs) else None
    out["ball_speed_over_40_share"] = round(float(np.mean(bvs > 40)), 4) if len(bvs) else None

    pv = np.full(m.pxy.shape[:2], np.nan)
    pv[step:] = np.linalg.norm(m.pxy[step:] - m.pxy[:-step], axis=2) / dt
    pvs = pv[graded]; pvs = pvs[~np.isnan(pvs)]
    out["player_speed_p50_p90_p99"] = [round(float(np.percentile(pvs, q)), 2) for q in (50, 90, 99)]
    with np.errstate(invalid="ignore"):
        speed_all = np.nanmean(np.where(pv < 12, pv, np.nan), axis=1)
    speed_2s = centred_mean(speed_all, int(2 * fps) + 1)
    with np.errstate(invalid="ignore"):
        slow = speed_2s < 1.5
    out["slow_share_ball_missing"] = round(float(np.nanmean(slow[~m.bdet & graded])), 3)
    out["slow_share_ball_seen"] = round(float(np.nanmean(slow[m.bdet & graded])), 3)

    # nearest player of each team to the ball
    d = np.linalg.norm(m.pxy - m.bxy[:, None, :], axis=2)
    dh = np.nanmin(np.where(m.team == 0, d, np.inf), axis=1)
    da = np.nanmin(np.where(m.team == 1, d, np.inf), axis=1)
    dh[np.isinf(dh)] = np.nan; da[np.isinf(da)] = np.nan
    margin = da - dh  # > 0: a home player is nearer

    if m.labels is not None:
        L = (m.labels != "X") & graded
        out["labelled_minutes"] = round(L.sum() / fps / 60, 1)
        out["label_share_HAD"] = {k: round(float((m.labels[L] == k).mean()), 3) for k in "HAD"}
        spells, dead = [], []
        for a, b in m.periods:
            lab = m.labels[a:b]
            for x, y in runs(np.isin(lab, ["H", "A"])):
                # a maximal run of one team: split where H and A meet
                seg = lab[x:y]
                cuts = np.r_[0, np.flatnonzero(seg[1:] != seg[:-1]) + 1, len(seg)]
                spells += list(np.diff(cuts) / fps)
            dead += [(y - x) / fps for x, y in runs(lab == "D")]
        spells, dead = np.array(spells), np.array(dead)
        lm = L.sum() / fps / 60
        out["possessions_per_90"] = round(len(spells) / lm * 90, 1)
        out["possession_mean_median_s"] = [round(float(spells.mean()), 1), round(float(np.median(spells)), 1)] if len(spells) else None
        out["possessions_under_2s_share"] = round(float((spells < 2).mean()), 3) if len(spells) else None
        out["dead_spells_per_90"] = round(len(dead) / lm * 90, 1)
        out["dead_spell_median_s"] = round(float(np.median(dead)), 1) if len(dead) else None
        isD = L & (m.labels == "D"); alive = L & np.isin(m.labels, ["H", "A"])
        out["auc_dead_slow_players"] = round(auc(-speed_2s[isD], -speed_2s[alive]), 3) if isD.any() else None
        out["auc_dead_ball_missing"] = round(auc((~m.bdet).astype(float)[isD], (~m.bdet).astype(float)[alive]), 3) if isD.any() else None
        hseen = alive & m.bdet
        out["auc_home_nearest_margin"] = round(auc(margin[hseen & (m.labels == "H")], margin[hseen & (m.labels == "A")]), 3)
        near = np.where(margin > 0, "H", "A")
        out["nearest_team_accuracy_alive_ball_seen"] = round(float((near[hseen] == m.labels[hseen]).mean()), 3)
        if isD.any():
            out["speed2s_median_dead_alive"] = [round(float(np.nanmedian(speed_2s[isD])), 2), round(float(np.nanmedian(speed_2s[alive])), 2)]
            # the speed threshold that best separates dead from alive (balanced accuracy): does it transfer?
            grid = np.arange(0.6, 3.01, 0.05)
            bal = [(np.nanmean(speed_2s[isD] < g) + np.nanmean(speed_2s[alive] >= g)) / 2 for g in grid]
            out["best_dead_speed_threshold"] = [round(float(grid[int(np.argmax(bal))]), 2), round(float(max(bal)), 3)]
        gapL = L & ~m.bdet
        out["labels_when_ball_missing"] = {k: round(float((m.labels[gapL] == k).mean()), 3) for k in "HAD"} if gapL.any() else None
        out["ball_seen_share_in_labelled"] = round(float(m.bdet[L].mean()), 3)
        out["ball_seen_share_when_dead"] = round(float(m.bdet[isD].mean()), 3) if isD.any() else None
    return out


def all_matches():
    yield load_ours
    for mid in sorted({f.split("_")[-1][:-4] for f in os.listdir(os.path.join(EXT, "idsse")) if f.startswith("DFL_04_03")}):
        yield lambda mid=mid: load_idsse(mid)
    for g in (1, 2):
        yield lambda g=g: load_metrica(g)
    for mid in sorted(os.listdir(os.path.join(EXT, "skillcorner", "data", "matches"))):
        f = os.path.join(EXT, "skillcorner", "data", "matches", mid, "{}_tracking_extrapolated.jsonl".format(mid))
        if os.path.getsize(f) > 1000:  # skip LFS pointers we did not fetch
            yield lambda mid=mid: load_skillcorner(mid)


if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")
    results = []
    for loader in all_matches():
        m = loader()
        p = profile(m)
        results.append(p)
        print(json.dumps(p))
        sys.stdout.flush()
    os.makedirs(os.path.join(HERE, "out"), exist_ok=True)
    json.dump(results, open(os.path.join(HERE, "out", "dataset_profiles.json"), "w"), indent=1)
