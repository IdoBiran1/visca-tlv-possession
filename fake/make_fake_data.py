"""A synthetic BAR-ATH match in the exact formats of the tactical bundle, for testing code.

Nothing here comes from the real match. It exists so that code can be written and debugged
(by people and by a cloud LLM) without touching the gated data, which may not be shown to one.

    python fake/make_fake_data.py                 # writes fake/data/, ~150 MB, ~1 minute
    export MTAH_DATA_DIR=$PWD/fake/data           # then run anything as if it were the real data

Writes, mirroring DATA.md:
    fake/data/tactical/BAR-ATH_tactical_atd.xml            team of each track slot
    fake/data/tactical/BAR-ATH_tactical_atd.txt            "<frame_id>:<x,y,s>;...:<ball x,y,z>"
    fake/data/tactical/BAR-ATH_tactical_smart_tagging.xml  SportsCode instances
    fake/data/BAR-ATH_possession_gt_fake.csv               full-match H/A/D/X labels

The simulation is crude on purpose: possessions and stoppages drawn from simple distributions,
players in two moving blocks, one carrier glued to the ball. What it does reproduce is the
noise DATA.md describes, because that is what code has to survive: the ball missing on ~40% of
graded frames in bursts, 46 slots for 22 players with tracklets breaking and moving slot,
dropouts, duplicates, the odd player in the wrong team's slot, and Smart Tagging that is
jittered, incomplete and has false positives, with Team "N/A" on every throw-in.
"""

import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from preprocess import FPS, GAME, PITCH, half_video_blocks, source_frames  # noqa: E402

SECOND_HALF_KICKOFF = 113742  # FAQ.md: graded from 110808, kicks off here
SLOTS_PER_TEAM = 23
HOME_NAME, AWAY_NAME = "FC Barcelona", "Athletic Club"
TAG_TEAM = {"H": "Barcelona", "A": "Atletic Club"}  # spelling as DATA.md reports the real tags
RESTARTS = (("Throw-in", 0.45), ("Free Kick", 0.2), ("Goal Kick", 0.15), ("Corner Kick", 0.15),
            ("Kick Off (After Goal)", 0.05))
FORMATION = np.array([  # (depth, width) offsets from the block centre, attacking towards +x
    (-14, -20), (-14, -7), (-14, 7), (-14, 20),
    (0, -22), (0, -7), (0, 7), (0, 22),
    (12, -8), (12, 8)])


# -- the match: who has the ball, and where it is --------------------------------------------

def simulate_segments(rng, n_frames):
    """[(start, stop, label, restart_type)]: possession spells and stoppages, plus labels."""
    labels = np.full(n_frames, "X", dtype="<U1")
    segments = []
    for half, first, last in half_video_blocks(GAME):
        stop_half = last + 1
        start = first
        if half == 2:
            segments.append((first, SECOND_HALF_KICKOFF, "D", "Kick Off (Start)"))
            start = SECOND_HALF_KICKOFF
        team = "H" if half == 1 else "A"
        while start < stop_half:
            short = rng.random() < 0.15
            seconds = rng.uniform(0.5, 2.5) if short else np.clip(rng.lognormal(2.3, 0.7), 2, 90)
            stop = min(stop_half, start + int(seconds * FPS))
            segments.append((start, stop, team, None))
            start = stop
            if start >= stop_half:
                break
            if rng.random() < 0.22:
                names, weights = zip(*RESTARTS)
                kind = rng.choice(names, p=np.array(weights) / sum(weights))
                stop = min(stop_half, start + int(np.clip(rng.lognormal(2.6, 0.5), 3, 60) * FPS))
                segments.append((start, stop, "D", str(kind)))
                start = stop
                team = team if rng.random() < 0.35 else ("A" if team == "H" else "H")
            else:
                team = "A" if team == "H" else "H"
    for start, stop, label, _ in segments:
        labels[start:stop] = label
    return segments, labels


def attack_sign(label, frame):
    """+1 if this side attacks towards x = 105 at this frame. Sides swap at halftime."""
    first_half = frame < SECOND_HALF_KICKOFF - 3000
    home_up = first_half
    return 1.0 if (label == "H") == home_up else -1.0


def ball_path(rng, segments, n_frames):
    """Ball position for every frame of every segment, from waypoints."""
    length, width = PITCH
    keys_f, keys_x, keys_y = [], [], []
    x, y = length / 2, width / 2
    for start, stop, label, kind in segments:
        if label in "HA":
            sign = attack_sign(label, start)
            frame = start
            while frame < stop:
                keys_f.append(frame); keys_x.append(x); keys_y.append(y)
                frame += int(rng.uniform(1.0, 4.0) * FPS)
                x = float(np.clip(x + sign * rng.normal(4, 8), 2, length - 2))
                y = float(np.clip(y + rng.normal(0, 10), 2, width - 2))
        else:
            keys_f.append(start); keys_x.append(x); keys_y.append(y)
            if kind == "Throw-in":
                y = -1.0 if y < width / 2 else width + 1.0
            elif kind in ("Goal Kick", "Corner Kick"):
                x = -1.5 if x < length / 2 else length + 1.5
            out_frame = min(stop - 1, start + int(1.5 * FPS))
            keys_f.append(out_frame); keys_x.append(x); keys_y.append(y)
            if kind == "Corner Kick":
                y = 0.5 if y < width / 2 else width - 0.5
                x = 0.5 if x < length / 2 else length - 0.5
            elif kind == "Goal Kick":
                x, y = (5.5 if x < length / 2 else length - 5.5), width / 2
            elif kind == "Throw-in":
                y = -0.3 if y < width / 2 else width + 0.3
            elif kind and kind.startswith("Kick Off"):
                x, y = length / 2, width / 2
            settle = min(stop - 1, out_frame + int(rng.uniform(2, 6) * FPS))
            keys_f.append(settle); keys_x.append(x); keys_y.append(y)
            keys_f.append(stop - 1); keys_x.append(x); keys_y.append(y)
    order = np.argsort(keys_f, kind="stable")
    frames = np.arange(n_frames)
    keys_f = np.asarray(keys_f)[order]
    ball = np.stack([np.interp(frames, keys_f, np.asarray(keys_x)[order]),
                     np.interp(frames, keys_f, np.asarray(keys_y)[order])], axis=1)
    return ball


def smooth(values, window):
    """Moving average over `window` frames, edge-padded, O(n) whatever the window."""
    padded = np.pad(np.asarray(values, dtype=np.float64),
                    (window // 2, window - window // 2 - 1), mode="edge")
    total = np.concatenate(([0.0], np.cumsum(padded)))
    return (total[window:] - total[:-window]) / window


def player_paths(rng, labels, ball, n_frames):
    """True positions of 22 players [n_frames, 22, 2]; players 0-10 home, 11-21 away."""
    length, width = PITCH
    frames = np.arange(n_frames)
    pos = np.zeros((n_frames, 22, 2))
    owner = smooth(np.where(labels == "H", 1.0, np.where(labels == "A", -1.0, 0.0)), 50)
    bx, by = smooth(ball[:, 0], 75), smooth(ball[:, 1], 75)
    dead = labels == "D"
    for side, label in enumerate("HA"):
        sign = np.array([attack_sign(label, f) for f in (0, n_frames - 1)])
        sign = np.where(frames < SECOND_HALF_KICKOFF - 3000, sign[0], sign[1])
        attacking = owner * (1 if label == "H" else -1)  # +1 attacking, -1 defending
        centre_x = np.clip(bx - sign * (12 - 6 * attacking), 15, length - 15)
        for k in range(11):
            p = side * 11 + k
            walk = rng.normal(0, 1, (n_frames, 2))
            walk[dead] *= 0.3
            drift = np.stack([smooth(np.cumsum(walk[:, i]) * 0.15, 125) for i in range(2)], 1)
            drift -= smooth(drift[:, 0], 2500)[:, None] * [1, 0] + \
                smooth(drift[:, 1], 2500)[:, None] * [0, 1]
            if k == 0:  # goalkeeper
                goal_x = np.where(sign > 0, 4.0, length - 4.0)
                pos[:, p, 0] = goal_x + sign * 0.08 * np.abs(bx - goal_x)
                pos[:, p, 1] = width / 2 + 0.15 * (by - width / 2)
            else:
                depth, lateral = FORMATION[k - 1]
                pos[:, p, 0] = centre_x + sign * depth * (1.1 + 0.2 * attacking)
                pos[:, p, 1] = width / 2 + lateral * 0.9 + 0.3 * (by - width / 2)
            pos[:, p] += drift
    # The team in possession has someone on the ball; the nearest opponent closes in.
    for label, carriers, opponents in (("H", slice(1, 11), slice(12, 22)),
                                       ("A", slice(12, 22), slice(1, 11))):
        idx = np.flatnonzero(labels == label)
        d = np.linalg.norm(pos[idx, carriers] - ball[idx, None], axis=2)
        near = d.argmin(axis=1) + carriers.start
        pos[idx, near] = ball[idx] + rng.normal(0, 0.5, (len(idx), 2))
        d = np.linalg.norm(pos[idx, opponents] - ball[idx, None], axis=2)
        near = d.argmin(axis=1) + opponents.start
        away = pos[idx, near] - ball[idx]
        away /= np.maximum(np.linalg.norm(away, axis=1, keepdims=True), 1e-6)
        pos[idx, near] = ball[idx] + away * rng.uniform(1.5, 4.0, (len(idx), 1))
    for p in range(22):  # carrier snaps become short runs rather than teleports
        pos[:, p, 0], pos[:, p, 1] = smooth(pos[:, p, 0], 9), smooth(pos[:, p, 1], 9)
    return np.clip(pos, [-2, -2], [length + 2, width + 2])


# -- the tracker: what the provided data would make of it ------------------------------------

def ball_observed(rng, ball, labels, pos):
    """Ball visibility in bursts, missing more often when crowded and around stoppages."""
    n = len(ball)
    crowd = (np.linalg.norm(pos - ball[:, None], axis=2) < 5).sum(axis=1)
    p_miss = np.where(crowd >= 3, 0.6, 0.33)
    p_miss = np.where(labels == "D", 0.45, p_miss)
    p_miss = np.where(labels == "X", 0.97, p_miss)
    seen = np.zeros(n, dtype=bool)
    frame, visible = 0, True
    while frame < n:
        p = p_miss[frame]
        mean = 30.0 * (1 - p) / p if visible else 30.0
        run = max(1, int(rng.geometric(1.0 / max(mean, 1.0))))
        seen[frame:frame + run] = visible
        frame += run
        visible = not visible
    return seen


def assign_slots(rng, pos, labels):
    """Observed slot positions [n_frames, 46, 2] from true player positions.

    Tracklets break about once a minute per player, with a short dropout, and come back in a
    free slot. Now and then a slot duplicates a player, or a player shows up in the other
    team's slots for a few seconds.
    """
    n = len(pos)
    obs = np.full((n, 2 * SLOTS_PER_TEAM, 2), np.nan)
    in_play = labels != "X"
    for side in (0, 1):
        base = side * SLOTS_PER_TEAM
        slot = list(range(11))
        busy_until = np.zeros(SLOTS_PER_TEAM, dtype=np.int64)
        dropout_until = np.zeros(11, dtype=np.int64)
        breaks = rng.random((n, 11)) < 1.0 / (60 * FPS)
        for f in np.flatnonzero(in_play):
            for k in range(11):
                if breaks[f, k]:
                    free = [s for s in range(SLOTS_PER_TEAM)
                            if s not in slot and busy_until[s] < f]
                    if free:
                        busy_until[slot[k]] = f + 10 * FPS
                        slot[k] = free[rng.integers(len(free))]
                        dropout_until[k] = f + rng.integers(0, 25)
                if f >= dropout_until[k]:
                    obs[f, base + slot[k]] = pos[f, side * 11 + k]
    # duplicates and cross-team switches: short bursts copied into a spare slot
    for _ in range(400):
        f = rng.integers(0, n)
        if not in_play[f]:
            continue
        p = rng.integers(0, 22)
        same_team = rng.random() < 0.8
        team = p // 11 if same_team else 1 - p // 11
        target = team * SLOTS_PER_TEAM + SLOTS_PER_TEAM - 1 - rng.integers(0, 3)
        span = slice(f, min(n, f + rng.integers(25, 125)))
        offset = rng.normal(0, 1.0, 2) if same_team else 0.0
        obs[span, target] = pos[span, p] + offset
        if not same_team:
            obs[span, (p // 11) * SLOTS_PER_TEAM:(p // 11 + 1) * SLOTS_PER_TEAM][
                np.isclose(obs[span, (p // 11) * SLOTS_PER_TEAM:(p // 11 + 1) * SLOTS_PER_TEAM, 0],
                           pos[span, p, None, 0])] = np.nan
    return obs


def slot_speeds(obs):
    """Speed per slot from smoothed positions, m/s, NaN where the slot is empty."""
    speed = np.full(obs.shape[:2], np.nan)
    step = np.linalg.norm(obs[1:] - obs[:-1], axis=2) * FPS
    speed[1:] = step
    out = np.empty_like(speed)
    for s in range(obs.shape[1]):
        ok = ~np.isnan(speed[:, s])
        out[:, s] = np.where(ok, smooth(np.where(ok, speed[:, s], 0.0), 9), np.nan)
    return np.clip(out, 0, 11)


# -- writers ---------------------------------------------------------------------------------

def write_meta(path):
    lines = ['<?xml version="1.0" encoding="utf-8"?>', "<main>", "  <Metadata>",
             "    <GlobalConfig><FrameRate>25</FrameRate>"
             "<FieldSize><Width>{}</Width><Height>{}</Height></FieldSize></GlobalConfig>"
             .format(*PITCH), "    <Teams>"]
    for team_id, name in (("FIFATMA", HOME_NAME), ("FIFATMB", AWAY_NAME)):
        lines.append('      <Team id="{}"><Name>{}</Name></Team>'.format(team_id, name))
    lines.append("    </Teams>")
    lines.append("    <Players>")
    for s in range(2 * SLOTS_PER_TEAM):
        team_id = "FIFATMA" if s < SLOTS_PER_TEAM else "FIFATMB"
        lines.append('      <Player id="P{0}" teamId="{1}"><Name>Track {0}</Name>'
                     '<ShirtNumber>{0}</ShirtNumber></Player>'.format(s + 1, team_id))
    lines += ["    </Players>", "  </Metadata>", "</main>", ""]
    with open(path, "w") as handle:
        handle.write("\n".join(lines))


def fmt(values, digits):
    text = np.char.mod("%.{}f".format(digits), values)
    return np.where(np.isnan(values), "NaN", text)


def write_raw(path, obs, speed, ball, ball_seen):
    length, width = PITCH
    with open(path, "w") as handle:
        for lo in range(0, len(obs), 20000):  # in chunks: the formatted strings are large
            hi = min(len(obs), lo + 20000)
            px, py = fmt(obs[lo:hi, :, 0] / length, 5), fmt(obs[lo:hi, :, 1] / width, 5)
            ps = fmt(speed[lo:hi], 2)
            bx = np.where(ball_seen[lo:hi], fmt(ball[lo:hi, 0] / length, 5), "NaN")
            by = np.where(ball_seen[lo:hi], fmt(ball[lo:hi, 1] / width, 5), "NaN")
            for i in range(hi - lo):
                players = ";".join(",".join(group) for group in zip(px[i], py[i], ps[i]))
                handle.write("{}:{}:{},{},{}\n".format(
                    lo + i + 1, players, bx[i], by[i], "0.00" if ball_seen[lo + i] else "NaN"))


def fake_tags(rng, segments, ball):
    """Smart Tagging-like instances derived from the true segments, then corrupted."""
    length = PITCH[0]
    tags = []
    half_of = lambda f: "1st Half" if f < SECOND_HALF_KICKOFF - 3000 else "2nd Half"  # noqa: E731
    previous = None
    for i, (start, stop, label, kind) in enumerate(segments):
        if label == "D" and kind and kind != "Kick Off (Start)":  # those two are added below
            restart_team = segments[i + 1][2] if i + 1 < len(segments) else "H"
            team = "N/A" if kind == "Throw-in" else TAG_TEAM.get(restart_team, "N/A")
            tags.append((stop - 3 * FPS, stop + 2 * FPS, "SET PIECES",
                         {"Team": team, "Type": kind, "Half": half_of(stop)}))
            if kind == "Kick Off (After Goal)":
                tags.append((start - 2 * FPS, start + FPS, "GOALS",
                             {"Team": TAG_TEAM["A" if restart_team == "H" else "H"],
                              "Half": half_of(start)}))
        if label in "HA":
            if previous in ("H", "A") and previous != label and rng.random() < 0.4:
                tags.append((start, start + 6 * FPS, "ATTACKING TRANSITION",
                             {"Team": TAG_TEAM[label], "Half": half_of(start)}))
                tags.append((start, start + 6 * FPS, "DEFENSIVE TRANSITION",
                             {"Team": TAG_TEAM[previous], "Half": half_of(start)}))
            if stop - start >= 6 * FPS:
                x = ball[(start + stop) // 2, 0]
                depth = x if attack_sign(label, start) > 0 else length - x
                third = int(np.clip(depth // (length / 3), 0, 2))
                code = ("BUILD UP", "PROGRESSION", "BALL IN FINAL THIRD")[third]
                defending = ("DEFENDING IN DEFENSIVE THIRD", "DEFENDING IN MIDDLE THIRD",
                             "DEFENDING IN ATTACKING THIRD")[2 - third]
                other = "A" if label == "H" else "H"
                tags.append((start, stop, code, {"Team": TAG_TEAM[label], "Half": half_of(start)}))
                tags.append((start, stop, defending,
                             {"Team": TAG_TEAM[other], "Half": half_of(start)}))
        previous = label
    kept = [t for t in tags if t[2] == "SET PIECES" or rng.random() > 0.15]  # misses
    for _ in range(int(0.05 * len(kept))):  # false positives
        start = int(rng.choice([s for s, _, _, _ in segments]))
        kept.append((start, start + int(rng.uniform(3, 15) * FPS), "PROGRESSION",
                     {"Team": TAG_TEAM["H" if rng.random() < 0.5 else "A"],
                      "Half": half_of(start)}))
    jitter = lambda f: max(0, f + int(rng.integers(-FPS, FPS + 1)))  # noqa: E731
    out = [(jitter(a), max(jitter(a) + FPS, jitter(b)), c, t) for a, b, c, t in kept]
    for half, kickoff, team in (("1st Half", 11185, "H"), ("2nd Half", SECOND_HALF_KICKOFF, "A")):
        out.append((kickoff - 2 * FPS, kickoff + 3 * FPS, "SET PIECES",
                    {"Team": TAG_TEAM[team], "Type": "Kick Off (Start)", "Half": half}))
    return sorted(out, key=lambda t: t[0])


def write_tags(path, tags):
    lines = ['<?xml version="1.0" encoding="utf-8"?>', "<file>", "  <ALL_INSTANCES>"]
    for i, (start, stop, code, labels) in enumerate(tags):
        lines.append("    <instance>")
        lines.append("      <ID>{}</ID>".format(i + 1))
        lines.append("      <start>{:.2f}</start>".format(start / FPS))
        lines.append("      <end>{:.2f}</end>".format(stop / FPS))
        lines.append("      <code>{}</code>".format(code))
        for group, text in labels.items():
            lines.append("      <label><group>{}</group><text>{}</text></label>".format(group, text))
        lines.append("    </instance>")
    lines += ["  </ALL_INSTANCES>", "</file>", ""]
    with open(path, "w") as handle:
        handle.write("\n".join(lines))


def write_labels(path, labels):
    with open(path, "w") as handle:
        handle.write("frame,label\n")
        handle.writelines("{},{}\n".format(f, label) for f, label in enumerate(labels))


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", default=os.path.join(HERE, "data"))
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    n = source_frames("tactical", GAME)
    segments, labels = simulate_segments(rng, n)
    ball = ball_path(rng, segments, n)
    pos = player_paths(rng, labels, ball, n)
    ball_seen = ball_observed(rng, ball, labels, pos)
    obs = assign_slots(rng, pos, labels)
    speed = slot_speeds(obs)  # from clean positions, like a tracker's filtered speed
    obs += rng.normal(0, 0.08, obs.shape)  # tracking jitter, metres

    folder = os.path.join(args.out, "tactical")
    os.makedirs(folder, exist_ok=True)
    stem = os.path.join(folder, GAME + "_tactical")
    write_meta(stem + "_atd.xml")
    write_raw(stem + "_atd.txt", obs, speed, ball, ball_seen)
    write_tags(stem + "_smart_tagging.xml", fake_tags(rng, segments, ball))
    write_labels(os.path.join(args.out, GAME + "_possession_gt_fake.csv"), labels)

    graded = labels != "X"
    spells = sum(1 for _, _, label, _ in segments if label in "HA")
    print("fake match written to {}".format(args.out))
    print("  {} possessions, {} stoppages, ball seen on {:.1f}% of graded frames".format(
        spells, len(segments) - spells, 100.0 * ball_seen[graded].mean()))


if __name__ == "__main__":
    main()
