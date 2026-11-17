"""Tactical ATD and Smart Tagging -> per-frame numpy arrays and candidate features.

Runs locally on the match data. Everything it writes goes to CACHE_DIR, which is gitignored:
a cache is a frame-level extract of the match, so it may not be committed to a public repo or
shown to a cloud LLM (see OVERVIEW.md, "Use of Large Language Models").

    from preprocess import load_tracking, frame_features
    trk = load_tracking()                 # parses once (a minute or two), then reads the cache
    feats = frame_features(trk)           # one row per video frame

Conventions, all inherited from the challenge repo:
  * index i of every array is video frame i, i.e. submission row i (ATD frame_id - 1)
  * positions are in metres, pitch GAMES[game]["pitch"] = 105.3 x 68.0, origin at a corner
  * NaN means "not observed", for players and ball alike
"""

import hashlib
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(HERE, "cache")
FPS = 25


def find_challenge_repo():
    """The metrica-sports/more-than-a-hack-2027 checkout: MTAH_REPO, or next to this repo."""
    candidates = [
        os.environ.get("MTAH_REPO"),
        os.path.join(HERE, "..", "..", "more-than-a-hack-2027"),
        os.path.join(HERE, "..", "..", "metrica-sports", "more-than-a-hack-2027"),
        os.path.expanduser("~/more-than-a-hack-2027"),
    ]
    for path in candidates:
        if path and os.path.isfile(os.path.join(path, "games.py")):
            return os.path.abspath(path)
    raise FileNotFoundError(
        "challenge repo not found. Clone https://github.com/metrica-sports/more-than-a-hack-2027 "
        "and set MTAH_REPO to its path.")


REPO = find_challenge_repo()
for _path in (REPO, os.path.join(REPO, "starter"), os.path.join(REPO, "evaluation")):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from games import GAME, GAMES, half_video_blocks, require, source_frames  # noqa: E402
import baseline  # noqa: E402  starter/baseline.py
import load_smart_tagging  # noqa: E402  starter/load_smart_tagging.py

PITCH = GAMES[GAME]["pitch"]
HOME, AWAY = baseline.HOME, baseline.AWAY


# -- tracking --------------------------------------------------------------------------------

def parse_atd(raw_path, n_slots, n_frames):
    """Raw ATD text -> (players[n_frames, n_slots, 3], ball[n_frames, 2]), normalised units.

    Same line format baseline.py reads: "<frame_id>:<x,y,s>;<x,y,s>;...:<ball x,y,...>". Only the
    first two ball fields are kept: DATA.md documents ball_x and ball_y and nothing else.
    """
    players = np.full((n_frames, n_slots, 3), np.nan)
    ball = np.full((n_frames, 2), np.nan)
    bad = 0
    with open(raw_path) as handle:
        for line in handle:
            parts = line.rstrip("\n").split(":")
            if len(parts) != 3:
                bad += 1
                continue
            frame = int(parts[0]) - 1
            values = np.fromstring(parts[1].replace(";", ","), sep=",")
            if values.size != n_slots * 3:
                bad += 1
                continue
            players[frame] = values.reshape(n_slots, 3)
            if parts[2][:1] not in ("N", ""):
                ball[frame] = np.fromstring(parts[2], sep=",")[:2]
    if bad:
        print("parse_atd: skipped {} malformed lines".format(bad))
    return players, ball


def load_tracking(data_dir=None, game=GAME, cache=True):
    """Tracking in metres, cached as one .npz. Keys:

    xy[f, slot, 2]  player positions        speed[f, slot]  player speed, m/s as delivered
    ball[f, 2]      ball position           team[slot]      0 = home (H), 1 = away (A)
    """
    meta_path, raw_path = require(["atd_meta", "atd_raw"], game, data_dir)
    # Keyed by source path: the fake and real match share a game name, and unzipped files keep old mtimes.
    source = hashlib.sha1(os.path.abspath(raw_path).encode()).hexdigest()[:8]
    cache_path = os.path.join(CACHE_DIR, "{}_{}_tracking.npz".format(game, source))
    if cache and os.path.isfile(cache_path) and \
            os.path.getmtime(cache_path) > os.path.getmtime(raw_path):
        with np.load(cache_path) as npz:
            return {key: npz[key] for key in npz.files}

    team = baseline.team_of_each_track(meta_path)
    players, ball = parse_atd(raw_path, len(team), source_frames("tactical", game))
    # float64 on purpose: nearest_player_codes must reproduce baseline.py to the last tie.
    scale = np.array(PITCH)
    trk = {"xy": players[..., :2] * scale, "speed": players[..., 2].astype(np.float32),
           "ball": ball * scale, "team": team}
    if cache:
        os.makedirs(CACHE_DIR, exist_ok=True)
        np.savez(cache_path, **trk)
    return trk


def graded_mask(n_frames, game=GAME):
    """True on the frames the scorer grades: the two halves, published in DATA.md."""
    mask = np.zeros(n_frames, dtype=bool)
    for _, first, last in half_video_blocks(game):
        mask[first:last + 1] = True
    return mask


def half_of_frame(n_frames, game=GAME):
    """1 or 2 inside a half, 0 elsewhere."""
    half = np.zeros(n_frames, dtype=np.int8)
    for number, first, last in half_video_blocks(game):
        half[first:last + 1] = number
    return half


def run_lengths(mask):
    """Length of the True-run each frame belongs to (0 where False)."""
    mask = np.asarray(mask, dtype=bool)
    edges = np.diff(np.concatenate(([0], mask.astype(np.int8), [0])))
    starts, stops = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)
    out = np.zeros(len(mask), dtype=np.int32)
    for start, stop in zip(starts, stops):
        out[start:stop] = stop - start
    return out


def runs(mask):
    """[(start, stop)] half-open True-runs."""
    edges = np.diff(np.concatenate(([0], np.asarray(mask, dtype=np.int8), [0])))
    return list(zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)))


def centred_mean(values, window):
    """NaN-aware moving average over +/- window frames."""
    values = np.asarray(values, dtype=np.float64)
    ok = ~np.isnan(values)
    total = np.concatenate(([0.0], np.cumsum(np.where(ok, values, 0.0))))
    count = np.concatenate(([0], np.cumsum(ok)))
    index = np.arange(len(values))
    low, high = np.maximum(0, index - window), np.minimum(len(values), index + window + 1)
    n = count[high] - count[low]
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(n > 0, (total[high] - total[low]) / n, np.nan)


def frame_features(trk):
    """One row per video frame of candidate features. Every column is documented in the notebook.

    Distances in metres, speeds in m/s. NaN where the inputs are not observed.
    """
    xy, speed, ball, team = trk["xy"], trk["speed"], trk["ball"], trk["team"]
    n = len(ball)
    length, width = PITCH
    seen = ~np.isnan(xy[..., 0])
    ball_seen = ~np.isnan(ball[:, 0])
    f = {}

    # Ball: where it is, whether it is on the pitch, how fast it moves, how long it has been gone
    f["ball_seen"] = ball_seen
    f["ball_x"], f["ball_y"] = ball[:, 0], ball[:, 1]
    outside_x = np.maximum(0, np.maximum(-ball[:, 0], ball[:, 0] - length))
    outside_y = np.maximum(0, np.maximum(-ball[:, 1], ball[:, 1] - width))
    f["ball_out_dist"] = np.where(ball_seen, np.hypot(outside_x, outside_y), np.nan)
    step = np.full(n, np.nan)
    step[1:] = np.hypot(*(ball[1:] - ball[:-1]).T) * FPS
    f["ball_speed"] = step
    f["ball_speed_1s"] = centred_mean(step, 12)
    gap = ~ball_seen
    f["ball_gap_len"] = run_lengths(gap)
    last_seen = np.where(ball_seen, np.arange(n), -1)
    last_seen = np.maximum.accumulate(last_seen)
    f["frames_since_ball"] = np.where(last_seen >= 0, np.arange(n) - last_seen, np.nan)

    # Players: how many are tracked, how fast they move as a group, where each block sits
    for code, name in ((HOME, "h"), (AWAY, "a")):
        cols = team == code
        f["n_" + name] = seen[:, cols].sum(axis=1)
        with np.errstate(invalid="ignore"), _quiet():
            f["speed_" + name] = np.nanmean(speed[:, cols], axis=1)
            f["cx_" + name] = np.nanmean(xy[:, cols, 0], axis=1)
            f["cy_" + name] = np.nanmean(xy[:, cols, 1], axis=1)
            f["spread_x_" + name] = np.nanstd(xy[:, cols, 0], axis=1)
    with _quiet():
        f["speed_all"] = np.nanmean(speed, axis=1)
    f["speed_all_2s"] = centred_mean(f["speed_all"], 25)
    churn = np.zeros(n, dtype=np.int16)
    churn[1:] = (seen[1:] != seen[:-1]).sum(axis=1)
    f["slot_churn"] = churn

    # Ball <-> players: who is near it, and how contested it is
    d = np.hypot(xy[..., 0] - ball[:, None, 0], xy[..., 1] - ball[:, None, 1])
    with _quiet():
        d_h = np.nanmin(np.where(team == HOME, d, np.nan), axis=1)
        d_a = np.nanmin(np.where(team == AWAY, d, np.nan), axis=1)
    f["d_h"], f["d_a"] = d_h, d_a
    f["nearest_team"] = np.where(np.isnan(d_h) & np.isnan(d_a), -1,
                                 np.where(np.nan_to_num(d_h, nan=np.inf) <=
                                          np.nan_to_num(d_a, nan=np.inf), HOME, AWAY))
    f["d_margin"] = d_a - d_h  # > 0: home is closer
    f["n_within5_h"] = ((d < 5) & (team == HOME)).sum(axis=1)
    f["n_within5_a"] = ((d < 5) & (team == AWAY)).sum(axis=1)

    return pd.DataFrame(f)


class _quiet(object):
    """Silence numpy's 'mean of empty slice' warnings: an all-NaN frame is normal here."""

    def __enter__(self):
        import warnings
        self._catch = warnings.catch_warnings()
        self._catch.__enter__()
        warnings.simplefilter("ignore", category=RuntimeWarning)

    def __exit__(self, *exc):
        return self._catch.__exit__(*exc)


# -- Smart Tagging ---------------------------------------------------------------------------

TEAM_TAG = {"barcelona": "H", "atletic club": "A", "athletic club": "A"}


def load_tags(data_dir=None, game=GAME):
    """Smart Tagging as a DataFrame, one row per instance, with the Team tag mapped to H / A / N/A."""
    events = load_smart_tagging.load(game, data_dir)
    rows = []
    for event in events:
        tags = event["tags"]
        team = (tags.get("Team") or "").strip().lower()
        rows.append({
            "code": (event["code"] or "").strip().upper(),
            "start_frame": event["start_frame"],
            "end_frame": event["end_frame"],
            "duration_s": event["end_s"] - event["start_s"],
            "side": TEAM_TAG.get(team, "N/A"),
            "team_tag": tags.get("Team"),
            "half": tags.get("Half"),
            "type": tags.get("Type"),
        })
    return pd.DataFrame(rows)


# The Team tag is "the team performing the event" (OVERVIEW.md): the team with the ball for the
# attacking codes, the team WITHOUT it for the defending ones.
ATTACKING_CODES = ("BUILD UP", "PROGRESSION", "CREATING CHANCES", "ATTACKING TRANSITION",
                   "LONG BALL", "BALL IN FINAL THIRD", "BALL IN THE BOX", "PLAYERS IN THE BOX")
DEFENDING_CODES = ("DEFENDING IN ATTACKING THIRD", "DEFENDING IN MIDDLE THIRD",
                   "DEFENDING IN DEFENSIVE THIRD", "DEFENSIVE TRANSITION")


def tag_possession_vote(tags, n_frames):
    """Per frame: (# tags saying H has the ball) - (# saying A has it), from phase tags only."""
    vote = np.zeros(n_frames, dtype=np.int16)
    for codes, sign in ((ATTACKING_CODES, 1), (DEFENDING_CODES, -1)):
        chosen = tags[tags["code"].isin(codes)]
        for side, team_sign in (("H", 1), ("A", -1)):
            vote += sign * team_sign * tag_coverage(chosen[chosen["side"] == side], n_frames)
    return vote


def tag_coverage(tags, n_frames, code=None, side=None):
    """Boolean per frame: inside at least one instance matching code / side."""
    mask = np.zeros(n_frames, dtype=bool)
    chosen = tags
    if code is not None:
        chosen = chosen[chosen["code"] == code]
    if side is not None:
        chosen = chosen[chosen["side"] == side]
    for start, stop in zip(chosen["start_frame"], chosen["end_frame"]):
        mask[max(0, start):min(n_frames, stop)] = True
    return mask


# -- labels ----------------------------------------------------------------------------------

def read_labels(path):
    """A frame,label CSV as an array of one-character strings."""
    return pd.read_csv(path, dtype={"label": str})["label"].str.strip().str.upper().to_numpy()


def nearest_player_codes(trk):
    """baseline.nearest_player_labels, recomputed from the cached arrays.

    Used as a correctness check: smoothed with baseline.smooth it must reproduce the published
    baseline files exactly, which proves the parser and the frame alignment are right.
    """
    xy, ball, team = trk["xy"], trk["ball"], trk["team"]
    d2 = (xy[..., 0] - ball[:, None, 0]) ** 2 + (xy[..., 1] - ball[:, None, 1]) ** 2
    codes = np.full(len(ball), baseline.NOBODY, dtype=np.int8)
    ok = ~np.isnan(ball[:, 0]) & (~np.isnan(d2)).any(axis=1)
    codes[ok] = team[np.nanargmin(d2[ok], axis=1)]
    return codes
