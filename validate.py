"""Validation harness: the datasets, the folds, one scoreboard (the official scorer) and label-free checks.

    MTAH_REPO=... python validate.py          # scores the built-in predictors, writes out/scoreboard.json

The sets (see docs/2026-10-21-preprocessing-deep-dive.html and the plan's validation section):
  idsse_*    7 Bundesliga matches, ball masked like our tracker, bounces under BOUNCE_S removed.
             Leave-one-match-out: predict(c, train) gets the other six as `train`.
  metrica*   3 matches from the same tracker family, labels derived from events; trained on since v6 (pool=True).
  ours       our match, scored on the public 5 minutes only; plus label-free checks on all of it.
A predictor is any function predict(c, train) -> array of "H"/"A"/"D"/"X", one per frame of c.
"""
import copy
import dataclasses
import json
import re
import os
import sys
import warnings
import zlib

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, "external"))
warnings.filterwarnings("ignore")
import clean
import features as fe
import phase3
import preprocess as pp          # puts the challenge repo's starter/ and evaluation/ on the path
import possession_metrics as pm
import profile_datasets as P

BOUNCE_S = 0                                     # the providers' labels as they are (v5 to v8: 1.0, matched to the public window's rate; v4: 2.5)
# The bounce threshold for the labels the judges score against (IDSSE, SoccerTrack, PFF); BOUNCE_S stays the training
# labels'. None: the same as BOUNCE_S. 2.5 s deletes every brief mid-play possession, which the target keeps
# (docs/2026-10-24-brief-possessions.md). 0 since 2 November: NEUTRAL against 1.0 and simpler, with no number taken
# from the public window (docs/2026-11-02-brief-convention.md). Keep it fixed when trying another BOUNCE_S.
JUDGE_BOUNCE_S = 0
OURS_PRE_KICKOFF = np.arange(110808, 113742)     # graded, all D (FAQ.md)
IDSSE_BALL_SIGMA_S = 0      # a blur like our tracker's (clean.smooth_ball) costs Metrica 0.05 spellF1: off
METRICA_BALL_SIGMA_S = 0    # Metrica's ball is as sharp as IDSSE's; blurring it too would validate a blurred IDSSE
PHASE_CODES = ("BUILD UP", "PROGRESSION")        # the Smart Tagging codes that track possession (deep dive, step 8)


# ---------------------------------------------------------------------------- sharing the machine

MAX_JOBS = int(os.environ.get("MTAH_JOBS", 2))   # heavy runs at once, across every worktree: one peaks near 9 GB of 24
SLOT_DIR = os.path.join(os.path.expanduser("~"), ".cache", "mtah2027")
_SLOT = []                                       # the held lock file; the OS frees it when the process ends, crashes included


def heavy_slot():
    """Wait for one of MAX_JOBS machine-wide slots, then cap this process's threads at its share of the cores.

    Uncapped, every job starts a thread per core: three at once swapped, and 46% of a profiled run's CPU time
    went to the kernel. Returns the slot number."""
    import fcntl
    import time
    os.makedirs(SLOT_DIR, exist_ok=True)
    waited = False
    while True:
        for i in range(MAX_JOBS):
            fh = open(os.path.join(SLOT_DIR, "slot{}.lock".format(i)), "w")
            try:
                fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                fh.close()
                continue
            _SLOT.append(fh)
            threads = max(1, (os.cpu_count() or 1) // MAX_JOBS)
            os.environ["OMP_NUM_THREADS"] = str(threads)   # libraries loaded later (torch) read it at start-up
            from threadpoolctl import threadpool_limits
            threadpool_limits(threads)                     # the ones already loaded (sklearn's OpenMP)
            print("heavy-run slot {} of {}, {} threads".format(i, MAX_JOBS, threads), flush=True)
            return i
        if not waited:
            print("waiting: {} heavy runs already going (MTAH_JOBS={})".format(MAX_JOBS, MAX_JOBS), flush=True)
            waited = True
        time.sleep(20)


# ---------------------------------------------------------------------------- datasets

def our_gap_model(m):
    """Our tracker's ball gaps, used to mask IDSSE: gap lengths (s), missing share, slow share of the missing."""
    g = np.zeros(len(m.bxy), bool)
    for a, b in m.periods:
        g[a:b] = True
    g[OURS_PRE_KICKOFF] = False
    slow = clean.mean_player_speed(m.pxy, m.fps) < 1.5
    miss = ~m.bdet & g
    gaps = np.array([(b - a) / m.fps for a, b in clean.runs(miss)])
    return gaps, float(miss.sum() / g.sum()), float(slow[miss].mean())


PLAYER_DROPOUT = True     # hide external players the way our tracker loses them (clean.degrade_players); baseline v2, with phase4_a05
TEAM_SWAP = 0.0           # share of external player-time put in an opponent's slot as our tracker does (clean.swap_teams)
EXTERNAL_BALL_MASK = True  # False: IDSSE's and SoccerTrack's ball stays fully visible (players still degraded), the ball-ceiling test
POSS_ALPHA = None         # set: every predictor trains with this possession weighting (phase3.POSS_ALPHA) instead of its own
BALL_GAP_CONTEXT = False  # mask the IDSSE ball with gap edges and live-gap places measured on ours (clean.degrade_ball_context)
# Our match uses the ball recovered from the tactical video (neta/merge.py: SoccerTrack-trained net, SoccerTrack's
# filters), pinned in shared/ by checksum; IDSSE and SoccerTrack are masked like it and read like its video frames
# (clean.video_ball, net A's SoccerTrack error). Implies BALL_GAP_CONTEXT. Lane B's ball (ours_ball_recovered_v5) is out: 3 Nov rules.
# "" turns it off.
RECOVERED_BALL = os.path.join(HERE, "out", "ours_ball_neta_v2.npz")
RECOVERED_BALL_SHA256 = "7da940c5cb176c65013cc275d220e518fa07517c0bd0610c1c57dd5b706e02eb"
# Our slots on the other team's shirt (vision/shirts.py: shirt colours in the tactical video), moved to that team by
# clean.move_slots, e.g. os.path.join(HERE, "shared", "ours_team_fix_v1.npz"). "" leaves our tracking as it is.
OURS_TEAM_FIX = ""
CLEAN = {}                # the same sets with every player tracked, filled by load_sets when PLAYER_DROPOUT is on
AUG_DRAWS = 2             # training matches (IDSSE, Metrica) also seen under this many other corruption draws (phase3.AUG); 2 since baseline v8
CORRUPT_SEED = 0          # offset on every corruption seed; load_sets sets it per extra draw
TRAIN_ALT = {}            # the same sets under other corruption settings, trained on instead (validate.py run ... train:KEY=VALUE)
# Ball v2's sources on the other games (docs/2026-11-10-ball-source-damage.md): our file's live frames are net A (sharp),
# the tracker alone (blurred) or video. "" (v12): every tracker frame sharp. "blocks": the tracker-seen frames of each
# masked game get net / tracker-alone runs cut in blocks from ours (clean.tracker_alone), the tracker-alone ones
# Gaussian-blurred, before hiding. "fitted": the test generator, lognormal runs and a box kernel of the same width.
# Metrica isn't masked: its tracker-alone frames are blurred the same way and nothing is hidden or moved.
BALL_SOURCE = ""
BALL_SOURCE_SIGMA_S = (0.2, 0.4)    # blur sd drawn per corruption draw (ours measured 0.2-0.28)
BALL_SOURCE_JITTER = (0.75, 1.25)   # tracker-alone and video shares, times this, per draw (one match measured them)
BALL_SOURCE_FLAG = False  # per-frame source as model input (features: src_track, src_video), ours from its file


def load_sets(idsse=True, soccertrack=True, pff=True, metrica=True):
    """name -> cleaned match (clean.clean_match). Deterministic: the IDSSE, SoccerTrack and player masks use fixed seeds.

    SoccerTrack v2's eight labelled matches (st_<id>) and a fixed 16 of PFF's 64 World Cup 2022 matches (pff_<id>)
    are held out like Metrica; their ball is complete (SoccerTrack's rebuilt from events, PFF's filled in from
    events), so it is masked the way our tracker loses the ball, as IDSSE's is.

    Our tracker often has fewer than 22 players; IDSSE and Metrica always have 22.
    With PLAYER_DROPOUT, their players are hidden like ours, for training and validation alike."""
    global CORRUPT_SEED
    ours = P.load_ours()
    tracked = ours.bdet
    if RECOVERED_BALL:   # the tracker's frames, net A where it confirms them, plus net A's video ball in the gaps
        if os.path.basename(RECOVERED_BALL) == "ours_ball_neta_v2.npz":   # the pinned file, or one rebuilt by neta/merge.py
            import hashlib
            got = hashlib.sha256(open(RECOVERED_BALL, "rb").read()).hexdigest()
            assert got == RECOVERED_BALL_SHA256, "recovered ball changed: {} (expected {})".format(got, RECOVERED_BALL_SHA256)
        z = np.load(RECOVERED_BALL)
        ours = dataclasses.replace(ours, bxy=np.c_[z["x_tidy"], z["y_tidy"]], bdet=~np.isnan(z["x_tidy"]))
        ours.ball_source = np.select([z["source"] == s for s in ("net", "track", "video")], [1, 2, 3], 0)
    assert RECOVERED_BALL or not BALL_SOURCE, "BALL_SOURCE needs RECOVERED_BALL's sources"
    if OURS_TEAM_FIX:    # stretches where our slot follows a player of the other team, moved to that team
        g = np.load(OURS_TEAM_FIX)
        pxy, team = clean.move_slots(ours.pxy, ours.team, zip(g["slot"], g["start"], g["stop"]))
        ours = dataclasses.replace(ours, pxy=pxy, team=team)
    sets = {"ours": clean.clean_match(ours, dead_frames=OURS_PRE_KICKOFF)}
    if BALL_SOURCE_FLAG:
        sets["ours"].ball_source = np.where(sets["ours"].bdet, ours.ball_source, 0).astype(np.int8)
    model = clean.player_loss_model(sets["ours"]) if PLAYER_DROPOUT else None
    CLEAN.clear()

    def add(name, m, seed, ball=None, **kw):
        pxy = m.pxy
        if model is not None:
            m.pxy = clean.degrade_players(m.pxy, m.team, m.length, m.width, model, seed=100 + seed + CORRUPT_SEED)
        if TEAM_SWAP:
            m.pxy = clean.swap_teams(m.pxy, m.team, m.fps, TEAM_SWAP, seed=200 + seed + CORRUPT_SEED)
        if ball is not None:   # the ball mask needs the players as our tracker would see them
            kw["ball_visible"] = ball(m)
        if model is not None:
            CLEAN[name] = clean.clean_match(dataclasses.replace(m, pxy=pxy), **kw)
        sets[name] = clean.clean_match(m, **kw)
        if BALL_SOURCE_FLAG:   # 0 missing, 1 net A (or a clean ball), 2 the tracker alone, 3 video
            sets[name].ball_source = np.where(sets[name].bdet, getattr(m, "ball_source", 1), 0).astype(np.int8)
        if "bounce_s" in kw and JUDGE_BOUNCE_S is not None and JUDGE_BOUNCE_S != kw["bounce_s"]:
            c = sets[name]
            c.judge = clean.drop_bounces(m.labels, JUDGE_BOUNCE_S, m.fps) if JUDGE_BOUNCE_S else np.array(m.labels, copy=True)
            c.judge[~c.graded] = "X"

    if idsse or soccertrack or pff or BALL_SOURCE:   # every masked set needs the gap models
        gaps, missing, slow_share = our_gap_model(ours)
        if BALL_GAP_CONTEXT or RECOVERED_BALL:
            c = sets["ours"]
            in_play = c.graded & ~c.forced_dead
            stopped = clean.stopped_play(clean.mean_player_speed(c.pxy, c.fps) < 1.5, in_play, c.fps)
            ctx = clean.ball_context(c.bxy, c.pxy, c.length, c.width)
            gap_model = clean.ball_gap_model(c.bdet, stopped, ctx, in_play, c.fps)
            track_model = clean.ball_gap_model(tracked, stopped, ctx, in_play, c.fps)   # the tracker alone
            video_share = float(np.mean((c.bdet & ~tracked)[in_play]))
            if BALL_SOURCE:   # ours' tracker-seen in-play frames in order: True where the tracker alone
                s = ours.ball_source[in_play]
                template = s[np.isin(s, (1, 2))] == 2

    def simulate(m, i, hide=True):
        """The ball of a clean game as ball v2 would read it: tracker-alone stretches blurred (BALL_SOURCE), video
        stretches with net A's error model; m.ball_source set. Returns the frames left visible (all, hide False)."""
        in_play = np.zeros(len(m.bxy), bool)
        for a, b in m.periods:
            in_play[a:b] = True
        stopped = clean.stopped_play(clean.mean_player_speed(m.pxy, m.fps) < 1.5, in_play, m.fps)
        ctx = clean.ball_context(m.bxy, m.pxy, m.length, m.width)
        seen = clean.degrade_ball_context(stopped, ctx, m.periods, m.fps, gap_model, seed=i + CORRUPT_SEED)
        lost = ~clean.degrade_ball_context(stopped, ctx, m.periods, m.fps, track_model, seed=1000 + i + CORRUPT_SEED)
        alone, share = np.zeros(len(m.bxy), bool), video_share
        if BALL_SOURCE:   # blur first, then hide
            rng = np.random.default_rng(3000 + i + CORRUPT_SEED)
            sigma, jit, share = rng.uniform(*BALL_SOURCE_SIGMA_S), rng.uniform(*BALL_SOURCE_JITTER), video_share * rng.uniform(*BALL_SOURCE_JITTER)
            trk = seen & ~lost & in_play
            alone[trk] = clean.tracker_alone(int(trk.sum()), template, template.mean() * jit, rng, how=BALL_SOURCE)
            blurred = clean.smooth_ball(m.bxy, m.periods, m.fps, sigma, kernel="box" if BALL_SOURCE == "fitted" else "gauss")
            m.bxy = np.where(alone[:, None], blurred, m.bxy)
        kept = np.zeros(len(m.bxy), bool)
        if hide:   # frames the tracker would lose and the video finds: read like the video ball
            m.bxy, kept = clean.video_ball(m.bxy, m.pxy, seen & lost, m.periods, m.width, m.fps, share, seed=i + CORRUPT_SEED)
        m.ball_source = np.select([alone, kept], [2, 3], 1)
        return seen if hide else np.ones(len(m.bxy), bool)

    for k, m in ((1, P.load_metrica(1)), (2, P.load_metrica(2)), (3, P.load_metrica3())) if metrica else ():
        if METRICA_BALL_SIGMA_S:
            m.bxy = clean.smooth_ball(m.bxy, m.periods, m.fps, METRICA_BALL_SIGMA_S)
        add("metrica{}".format(k), m, k, ball=(lambda m, k=k: simulate(m, 90 + k, hide=False)) if BALL_SOURCE else None)

    def add_masked(name, m, i):
        """add() with a clean ball (IDSSE's, or SoccerTrack's rebuilt one) hidden the way our tracker loses it."""
        if not EXTERNAL_BALL_MASK:
            add(name, m, 10 + i, bounce_s=BOUNCE_S)
            return
        if BALL_GAP_CONTEXT or RECOVERED_BALL:
            def ball(m, i=i):
                if RECOVERED_BALL:
                    return simulate(m, i)
                in_play = np.zeros(len(m.bxy), bool)
                for a, b in m.periods:
                    in_play[a:b] = True
                stopped = clean.stopped_play(clean.mean_player_speed(m.pxy, m.fps) < 1.5, in_play, m.fps)
                ctx = clean.ball_context(m.bxy, m.pxy, m.length, m.width)
                return clean.degrade_ball_context(stopped, ctx, m.periods, m.fps, gap_model, seed=i + CORRUPT_SEED)
            add(name, m, 10 + i, ball=ball, bounce_s=BOUNCE_S)
            return
        slow = clean.mean_player_speed(m.pxy, m.fps) < 1.5
        visible = clean.degrade_ball(slow, m.periods, m.fps, gaps, missing, slow_share, seed=i + CORRUPT_SEED)
        add(name, m, 10 + i, bounce_s=BOUNCE_S, ball_visible=visible)

    if idsse:
        ids = sorted({f.split("_")[-1][:-4] for f in os.listdir(os.path.join(P.EXT, "idsse")) if f.startswith("DFL_04_03")})
        for i, mid in enumerate(ids):
            m = P.load_idsse(mid)
            if IDSSE_BALL_SIGMA_S:
                m.bxy = clean.smooth_ball(m.bxy, m.periods, m.fps, IDSSE_BALL_SIGMA_S)
            add_masked("idsse_" + mid[-6:], m, i)
    if soccertrack:   # SoccerTrack v2's labelled matches, held out like Metrica, masked like IDSSE
        for j, mid in enumerate(P.SOCCERTRACK_LABELLED):
            add_masked("st_" + mid, P.load_soccertrack(mid), 50 + j)
    if pff:           # PFF World Cup 2022, a fixed quarter of its 64 matches, held out, masked like IDSSE
        for j, mid in enumerate(P.PFF_HELD_OUT):
            add_masked("pff_" + mid, P.load_pff(mid), 70 + j)
    if AUG_DRAWS and CORRUPT_SEED == 0:   # the extra draws, each loaded like these with every corruption seed moved
        import phase3
        kept = dict(CLEAN)
        phase3.AUG.clear()
        for d in range(1, AUG_DRAWS + 1):
            CORRUPT_SEED = 10000 * d
            try:
                extra = load_sets(idsse=idsse, soccertrack=False, pff=False, metrica=metrica)
            finally:
                CORRUPT_SEED = 0
            for n, c in extra.items():
                if n != "ours":
                    phase3.AUG.setdefault(n, []).append(c)
        CLEAN.clear(); CLEAN.update(kept)
    return sets


def training_table(sets, names):
    """(features DataFrame, labels, match name per row) for the labelled graded frames of `names`."""
    import pandas as pd
    parts, labs, who = [], [], []
    for n in names:
        c = sets[n]
        keep = c.graded & np.isin(c.labels, ["H", "A", "D"])
        parts.append(fe.frame_features(c)[keep])
        labs.append(c.labels[keep]); who += [n] * int(keep.sum())
    return pd.concat(parts, ignore_index=True), np.concatenate(labs), np.array(who)


# ---------------------------------------------------------------------------- scoring

def truth(c):
    """The labels a set is judged on: its own, or JUDGE_BOUNCE_S's where that differs from the training labels."""
    return getattr(c, "judge", c.labels)


def score(gt, pred, fps=25):
    """The official metrics (evaluation/possession_metrics.py) for one match, plus counts a coach reads."""
    g, p = pm.label_codes(gt), pm.label_codes(pred)
    runs = pm.valid_runs(g)
    filled = pm.fill_abstentions(p, runs)
    spell = pm.spell_scores(g, filled, runs)
    graded = g != pm.X
    minutes = graded.sum() / fps / 60
    dur = lambda codes: [b - a for a, b, _ in pm.possession_spells(codes, runs)]
    dp, dg = dur(filled), dur(g)
    return {"spellF1": round(spell["mean_f1"], 4), "macroF1": round(pm.macro_f1(pm.scored_confusion(pm.confusion(g, p))), 4),
            "poss_per90_pred": round(spell["n_pred"] / minutes * 90), "poss_per90_true": round(spell["n_gt"] / minutes * 90),
            "median_s_pred": round(float(np.median(dp)) / fps, 1) if dp else None, "median_s_true": round(float(np.median(dg)) / fps, 1) if dg else None,
            "D_pred": round(float(np.mean(filled[graded] == pm.D)), 3), "D_true": round(float(np.mean(g[graded] == pm.D)), 3)}


def label_free_checks(c, pred, tags):
    """Our full match, no labels: plausibility of the prediction and agreement with independent signals."""
    pred = np.asarray(pred)
    filled = pm.fill_abstentions(pm.label_codes(pred), [(a, b) for a, b in c.periods])
    lab = np.array(["H", "A", "D"])[np.minimum(filled, 2)]
    g = c.graded.copy()
    minutes = g.sum() / c.fps / 60
    spells = [(b - a) / c.fps for a, b, _ in pm.possession_spells(filled, [(a, b) for a, b in c.periods])]
    alive = g & np.isin(lab, ["H", "A"])
    # nearest team where it is trustworthy: ball seen, someone within 3 m, play looks live
    d = np.linalg.norm(c.pxy - c.bxy[:, None, :], axis=2)
    dh = np.nanmin(np.where(c.team == 0, d, np.inf), 1); da = np.nanmin(np.where(c.team == 1, d, np.inf), 1)
    live = clean.mean_player_speed(c.pxy, c.fps) > 1.5
    proxy = alive & c.bdet & (np.minimum(dh, da) < 3) & live
    near = np.where(dh < da, "H", "A")
    phase = np.zeros(len(lab), "<U1")
    for code in PHASE_CODES:
        for side in "HA":
            phase[pp.tag_coverage(tags[(tags.code == code) & (tags.side == side)], len(lab))] = side
    tagged = alive & (phase != "")
    return {"poss_per90": round(len(spells) / minutes * 90), "median_s": round(float(np.median(spells)), 1) if spells else None,
            "D_share": round(float(np.mean(lab[g] == "D")), 3),
            "pre_kickoff_all_D": bool(np.all(lab[OURS_PRE_KICKOFF] == "D")),
            "agree_nearest_team": round(float(np.mean(lab[proxy] == near[proxy])), 3),
            "agree_phase_tags": round(float(np.mean(lab[tagged] == phase[tagged])), 3)}


DECODER_IN = {}            # set -> the decoder's input (make_phase3 leaves it in "last"); log_run saves it for external/replay.py
LOGP_DTYPE = np.float32    # float16 halves logp.npz (25 MB a run) but flips 5 frames of one v4 PFF match: replays not exact


def keep_decoder_input(n, c):
    if "last" in DECODER_IN:
        logp, d = DECODER_IN.pop("last")
        DECODER_IN[n] = {"logp": logp.astype(LOGP_DTYPE), "decoder": json.dumps(d), "labels": pm.label_codes(truth(c)),
                         "periods": np.array(c.periods), "forced_dead": c.forced_dead, "fps": c.fps}
        p4 = sys.modules.get("phase4")
        if p4 is not None and p4.LAST_MIRROR is not None:   # phase4.SetAvg.MIRROR_DUMP: the mirrored prediction, for replay --mirror
            DECODER_IN[n]["logp_mirror"], p4.LAST_MIRROR = p4.LAST_MIRROR.astype(LOGP_DTYPE), None


def scoreboard(predict, sets, tags):
    """Run `predict` on every set: IDSSE leave-one-match-out, Metrica, SoccerTrack, ours (public window + label-free).

    A predictor's `pool` names the sets it trains on (pool_prefixes), always leaving out the match it predicts.
    Returns (scores per set, predicted labels per set)."""
    out, preds = {}, {}
    DECODER_IN.clear()
    idsse = sorted(n for n in sets if n.startswith("idsse_"))
    metrica = sorted(k for k in sets if k.startswith("metrica"))
    soccertrack = sorted(k for k in sets if k.startswith("st_"))
    pff = sorted(k for k in sets if k.startswith("pff_"))
    pool = [k for k in idsse + metrica + soccertrack + pff if k.startswith(pool_prefixes(predict))]
    for n in idsse + metrica + soccertrack + pff:
        train = {k: TRAIN_ALT.get(k, sets[k]) for k in pool if k != n}
        preds[n] = predict(sets[n], train)
        keep_decoder_input(n, sets[n])
        out[n] = score(truth(sets[n]), preds[n])
    preds["ours"] = predict(sets["ours"], {k: TRAIN_ALT.get(k, sets[k]) for k in pool})
    keep_decoder_input("ours", sets["ours"])
    out["ours_public"] = score(sets["ours"].labels, preds["ours"])
    out["ours_label_free"] = label_free_checks(sets["ours"], preds["ours"], tags)
    return out, preds


# ---------------------------------------------------------------------------- run ledger

RUNS = os.path.join(HERE, "out", "runs")
LEDGER = os.path.join(RUNS, "ledger.csv")
LEDGER_COLS = ["run_id", "time", "commit", "dirty", "predictor", "params", "bounce_s",
               "spellF1_idsse", "spellF1_metrica", "spellF1_ours_public", "macroF1_idsse", "macroF1_metrica",
               "macroF1_ours_public", "ours_poss_per90", "ours_median_s", "ours_D_share", "ours_pre_kickoff_all_D", "note"]


def git_state():
    """(short commit, True if the code in mtah2027/ has uncommitted changes)."""
    import subprocess
    run = lambda *a: subprocess.run(["git", *a], cwd=HERE, capture_output=True, text=True).stdout.strip()
    return run("rev-parse", "--short", "HEAD"), bool(run("status", "--porcelain", "--", "."))


GIT_AT_START = git_state()   # the code this process loaded; HEAD may move while a long run is going


def log_run(name, predict, results, preds, note=""):
    """Write out/runs/<run_id>/ (config, scores, per-frame predictions) and append a ledger line. Returns run_id."""
    import csv
    import time
    commit, dirty = GIT_AT_START
    run_id = "{}_{}{}_{}".format(time.strftime("%Y%m%d-%H%M%S"), commit, "-dirty" if dirty else "", name)
    d = os.path.join(RUNS, run_id)
    os.makedirs(d, exist_ok=True)
    params = getattr(predict, "params", {})
    config = {"run_id": run_id, "time": time.strftime("%Y-%m-%d %H:%M:%S"), "commit": commit, "dirty": dirty,
              "predictor": name, "params": params, "bounce_s": BOUNCE_S, "idsse_mask_seed": "index of the match (0-6)", "note": note,
              "judge_bounce_s": BOUNCE_S if JUDGE_BOUNCE_S is None else JUDGE_BOUNCE_S,   # the threshold the judges' labels used
              "tcn_device": sys.modules["phase4"].DEVICE if "phase4" in sys.modules else None}   # MPS and CPU runs don't compare
    json.dump(config, open(os.path.join(d, "config.json"), "w"), indent=1)
    json.dump(results, open(os.path.join(d, "scores.json"), "w"), indent=1)
    np.savez_compressed(os.path.join(d, "predictions.npz"), **{k: pm.label_codes(v) for k, v in preds.items()})
    if DECODER_IN:   # per set: <set>.logp, .decoder (its tuned settings), .labels, .periods, .forced_dead, .fps
        path = os.path.join(d, "logp.npz")
        np.savez_compressed(path, **{"{}.{}".format(n, k): v for n, x in DECODER_IN.items() for k, v in x.items()})
        print("decoder input for {} sets: {} ({:.0f} MB)".format(len(DECODER_IN), path, os.path.getsize(path) / 1e6), flush=True)
    mean =lambda pre, key: round(float(np.mean([v[key] for k, v in results.items() if k.startswith(pre)])), 4)
    lf = results["ours_label_free"]
    row = {"run_id": run_id, "time": config["time"], "commit": commit, "dirty": dirty, "predictor": name,
           "params": json.dumps(params, sort_keys=True), "bounce_s": BOUNCE_S,
           "spellF1_idsse": mean("idsse_", "spellF1"), "spellF1_metrica": mean("metrica", "spellF1"),
           "spellF1_ours_public": results["ours_public"]["spellF1"], "macroF1_idsse": mean("idsse_", "macroF1"),
           "macroF1_metrica": mean("metrica", "macroF1"), "macroF1_ours_public": results["ours_public"]["macroF1"],
           "ours_poss_per90": lf["poss_per90"], "ours_median_s": lf["median_s"], "ours_D_share": lf["D_share"],
           "ours_pre_kickoff_all_D": lf["pre_kickoff_all_D"], "note": note}
    new = not os.path.isfile(LEDGER)
    with open(LEDGER, "a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=LEDGER_COLS)
        if new:
            w.writeheader()
        w.writerow(row)
    return run_id


def find_run(key):
    """The newest run whose id contains `key` (a full id, a time prefix, or a predictor name); a full id is taken as is."""
    if os.path.isdir(os.path.join(RUNS, key)):   # "<run>" must not pick its rescored copy "<run>_judge1.0"
        return key
    runs = sorted(r for r in os.listdir(RUNS) if os.path.isdir(os.path.join(RUNS, r)) and key in r)
    runs = [r for r in runs if r.endswith("_" + key)] or runs   # "phase3_robust" must not pick phase3_robust_blend
    if not runs:
        raise SystemExit("no run matches {!r} in {}".format(key, RUNS))
    return runs[-1]


def load_predictions(run_id):
    z = np.load(os.path.join(RUNS, run_id, "predictions.npz"))
    return {k: np.array(pm.LABELS)[z[k]] for k in z.files}


def clock(frame, fps=25):
    s = frame / fps
    return "{:d}:{:04.1f}".format(int(s // 60), s % 60)


# ---------------------------------------------------------------------------- investigation tools

def compare_runs(a, b, sets=None, top=5):
    """Where two runs differ, set by set: score change, frames that differ, and which run is right there."""
    a, b = find_run(a), find_run(b)
    pa, pb = load_predictions(a), load_predictions(b)
    sets = sets or load_sets()
    report = {"a": a, "b": b, "sets": {}}
    for n in sorted(set(pa) & set(pb)):
        c = sets[n]
        sa, sb = score(truth(c), pa[n]), score(truth(c), pb[n])
        diff = (pa[n] != pb[n]) & c.graded
        lab = np.isin(truth(c), ["H", "A", "D"]) & diff
        ranges = sorted(clean.runs(diff), key=lambda r: r[0] - r[1])[:top]
        report["sets"][n] = {"spellF1_a": sa["spellF1"], "spellF1_b": sb["spellF1"], "delta_spellF1": round(sb["spellF1"] - sa["spellF1"], 4),
                             "frames_differ": int(diff.sum()),
                             "where_labelled_a_right": round(float(np.mean(pa[n][lab] == truth(c)[lab])), 3) if lab.any() else None,
                             "where_labelled_b_right": round(float(np.mean(pb[n][lab] == truth(c)[lab])), 3) if lab.any() else None,
                             "longest_differences": ["{}-{} ({}, {:.1f} s): {} -> {}".format(x, y, clock(x, c.fps), (y - x) / c.fps,
                                                     "".join(sorted(set(pa[n][x:y]))), "".join(sorted(set(pb[n][x:y])))) for x, y in ranges]}
    return report


POSS90_RANGE, D_RANGE = (250, 320), (0.41, 0.52)   # plausible for our full match: IDSSE's true ranges
POSS90_TOL = 10                                     # retraining alone moves possessions per 90 by about this much


VERDICT_GROUPS = ("idsse_", "metrica", "st_", "pff_")   # held-out groups, weighted equally in verdict
RESEED_MARGIN = 0.015   # a reseed of v5 alone moved held-out spellF1 by +0.007 (above its 2se): a smaller KEEP needs a second seed


def verdict(new, base):
    """Keep or revert run `new` against run `base`, by one fixed rule.

    Primary: spellF1 on the held-out matches, paired by match, with each group in VERDICT_GROUPS weighted equally:
    IDSSE (7), Metrica (3, another tracker), SoccerTrack (8, a fixed full-pitch camera like ours) and PFF (16,
    broadcast tracking with the best labels), so no single source's cues can win alone; a real gain is a change above twice its standard error. A group missing
    from either run (older runs have no SoccerTrack) is left out. Gate: on our full match, possessions per 90 and the D share
    must not move further outside POSS90_RANGE / D_RANGE than the base, and pre-kick-off must stay all D.
    Sets whose true labels differ between the runs (a relabelling) are left out of the primary.
    Our public window (17 possessions) is reported, never decisive. The 2se counts match-to-match spread, not
    retraining spread: a KEEP below RESEED_MARGIN is flagged second_seed_needed (docs/2026-10-24-pooled-metrica.md)."""
    a, b = find_run(base), find_run(new)
    sa, sb = (json.load(open(os.path.join(RUNS, r, "scores.json"))) for r in (a, b))
    return dict({"base": a, "new": b}, **judge(sa, sb))


def judge(sa, sb):
    """verdict's rule on two runs' scores (scores.json), new `sb` against base `sa`; external/replay.py uses it too."""
    held =sorted(k for k in sa if k.startswith(VERDICT_GROUPS) and k in sb)
    relabelled = [k for k in held if sa[k]["poss_per90_true"] != sb[k]["poss_per90_true"]]   # not comparable
    held = [k for k in held if k not in relabelled]
    d = np.array([sb[k]["spellF1"] - sa[k]["spellF1"] for k in held])
    by = {p: d[[k.startswith(p) for k in held]] for p in VERDICT_GROUPS}
    grp = [g for g in by.values() if len(g)]
    change = float(np.mean([g.mean() for g in grp]))
    se = float(np.sqrt(sum(g.var(ddof=1) / len(g) for g in grp)) / len(grp))
    off = lambda v, lo_hi: max(lo_hi[0] - v, v - lo_hi[1], 0)
    la, lb = sa["ours_label_free"], sb["ours_label_free"]
    gate = {"poss_per90": [la["poss_per90"], lb["poss_per90"], off(lb["poss_per90"], POSS90_RANGE) <= off(la["poss_per90"], POSS90_RANGE) + POSS90_TOL],
            "D_share": [la["D_share"], lb["D_share"], off(lb["D_share"], D_RANGE) <= off(la["D_share"], D_RANGE) + 0.005],
            "pre_kickoff_all_D": [la["pre_kickoff_all_D"], lb["pre_kickoff_all_D"], bool(lb["pre_kickoff_all_D"])]}
    gain, loss = change > 2 * se, change < -2 * se
    ok = all(g[2] for g in gate.values())
    call = "KEEP" if gain and ok else "REVERT" if loss or not ok else "NEUTRAL (keep only if simpler)"
    return {"verdict": call, "second_seed_needed": call == "KEEP" and change < RESEED_MARGIN,
            "held_out_spellF1": {"change": round(change, 4), "2se": round(2 * se, 4), "per_group": {p: round(float(g.mean()), 4) for p, g in by.items() if len(g)},
                                 "left_out_relabelled": relabelled,
                                 "better_on": int((d > 0).sum()), "of": len(d)},
            "gate_base_new_ok": gate,
            "ours_public_spellF1_not_decisive": [sa["ours_public"]["spellF1"], sb["ours_public"]["spellF1"]]}


def submit(name, device="cpu", settings=()):
    """out/submission_<name>.csv (+ .json): PREDICTORS[name] trained on every labelled set it may use, then the
    challenge's validator. X outside the graded frames. The TCN trains on `device`: CPU, the reproducible build.
    settings: the (KEY, VALUE) pairs already applied; they go into the file name, so a candidate never replaces it."""
    import subprocess
    import phase4
    phase4.DEVICE = device   # the TCNs; setnet keeps its own DEVICE (MPS on a Mac: repeats exactly, docs/2026-11-10-compliance.md)
    predict = PREDICTORS[name]
    sets = load_sets(soccertrack="st_" in pool_prefixes(predict), pff="pff_" in pool_prefixes(predict)); LOADED.update(sets)
    pool = [k for k in sets if k != "ours" and k.startswith(pool_prefixes(predict))]
    lab = predict(sets["ours"], {k: sets[k] for k in pool})
    lab[~sets["ours"].graded] = "X"
    tag = "_".join([name] + [re.sub(r"[^\w.=-]", "", "=".join(kv)) for kv in settings])   # phase3.TUNE_ON=idsse_
    path = os.path.join(HERE, "out", "submission_{}.csv".format(tag))
    pp.baseline.write_submission(path, lab)
    json.dump({"predictor": name, "settings": dict(settings), "commit": GIT_AT_START[0], "dirty": GIT_AT_START[1], "trained_on": sorted(pool),
               "params": getattr(predict, "params", {})},   # no public-window score: nothing in the build uses the answer key
              open(path[:-4] + ".json", "w"), indent=1, default=str)
    out = subprocess.run([sys.executable, os.path.join(pp.REPO, "evaluation", "validate_submission.py"), path], capture_output=True, text=True)
    return path, out.stdout.strip().splitlines()[-2:]


def error_segments(gt, pred, fps=25, iou=0.5):
    """Every true and predicted possession, sorted by time: matched (with timing offsets), missed or spurious.

    Matching is the scorer's: same team, one-to-one, greedy by IoU, here at one threshold. For a missed
    possession, the prediction inside it says why (split into pieces, wrong team, called dead); for a
    spurious one, the truth inside it says what it really was.
    """
    import pandas as pd
    g, p = pm.label_codes(gt), pm.label_codes(pred)
    runs = pm.valid_runs(g)
    filled = pm.fill_abstentions(p, runs)
    G, Pd = pm.possession_spells(g, runs), pm.possession_spells(filled, runs)
    mg, mp = {}, {}
    for v, i, j in pm.spell_overlaps(G, Pd):
        if v < iou:
            break
        if i in mg or j in mp:
            continue
        mg[i], mp[j] = (j, v), i
    team = lambda k: "H" if k == pm.H else "A"
    mix = lambda codes, k: (float(np.mean(codes == k)), float(np.mean((codes != k) & (codes != pm.D) & (codes != pm.X))), float(np.mean(codes == pm.D)))
    rows = []
    for i, (a, b, k) in enumerate(G):
        same, other, dead = mix(filled[a:b], k)
        pieces = sum(1 for (x, y, kk) in Pd if kk == k and x < b and y > a)
        row = {"kind": "matched" if i in mg else "missed", "team": team(k), "start": a, "end": b, "clock": clock(a, fps),
               "duration_s": round((b - a) / fps, 2), "pred_same_team": round(same, 2), "pred_other_team": round(other, 2),
               "pred_dead": round(dead, 2), "pred_pieces": pieces}
        if i in mg:
            j, v = mg[i]
            row.update(iou=round(v, 3), start_offset_s=round((Pd[j][0] - a) / fps, 2), end_offset_s=round((Pd[j][1] - b) / fps, 2))
        rows.append(row)
    for j, (a, b, k) in enumerate(Pd):
        if j not in mp:
            seg = g[a:b]
            same, other, dead = mix(seg, k)
            rows.append({"kind": "spurious", "team": team(k), "start": a, "end": b, "clock": clock(a, fps),
                         "duration_s": round((b - a) / fps, 2), "truth_same_team": round(same, 2),
                         "truth_other_team": round(other, 2), "truth_dead": round(dead, 2)})
    return pd.DataFrame(rows).sort_values("start", ignore_index=True)


def run_errors(run, set_name, sets=None):
    """error_segments for one set of a logged run; writes out/runs/<run>/errors_<set>.csv and returns a summary."""
    run = find_run(run)
    sets = sets or load_sets()
    df = error_segments(truth(sets[set_name]), load_predictions(run)[set_name], sets[set_name].fps)
    df.to_csv(os.path.join(RUNS, run, "errors_{}.csv".format(set_name)), index=False)
    m, miss, spur = df[df.kind == "matched"], df[df.kind == "missed"], df[df.kind == "spurious"]
    return {"run": run, "set": set_name, "true": int((df.kind != "spurious").sum()), "matched_iou_0.5": len(m), "missed": len(miss),
            "spurious": len(spur), "median_abs_start_offset_s": round(float(m.start_offset_s.abs().median()), 2) if len(m) else None,
            "missed_because_split": int((miss.pred_pieces > 1).sum()), "missed_wrong_team_mostly": int((miss.pred_other_team > 0.5).sum()),
            "missed_called_dead_mostly": int((miss.pred_dead > 0.5).sum()),
            "spurious_inside_dead_ball": int((spur.truth_dead > 0.5).sum()) if len(spur) else 0,
            "spurious_inside_other_team": int((spur.truth_other_team > 0.5).sum()) if len(spur) else 0,
            "csv": os.path.join(RUNS, run, "errors_{}.csv".format(set_name))}


def feature_audit(sets, step=5):
    """Per feature and dataset group: coverage, median, and how well it separates dead/alive and H/A (AUC).

    A feature that separates well on IDSSE but not on Metrica or our window is a transfer risk.
    Every `step`-th labelled frame, to keep it quick. Writes out/feature_audit.csv.
    """
    import pandas as pd
    groups = {"idsse": [n for n in sets if n.startswith("idsse_")], "metrica": [n for n in sets if n.startswith("metrica")], "ours_public": ["ours"]}
    rows = {}
    for gname, names in groups.items():
        X, y, _ = training_table(sets, names)
        X, y = X.iloc[::step], y[::step]
        dead, alive = y == "D", y != "D"
        for col in X.columns:
            v = X[col].to_numpy(float)
            r = rows.setdefault(col, {"feature": col})
            r["coverage_" + gname] = round(float(np.mean(~np.isnan(v))), 3)
            r["median_" + gname] = round(float(np.nanmedian(v)), 3) if (~np.isnan(v)).any() else None
            r["auc_dead_" + gname] = round(P.auc(v[dead], v[alive]), 3) if dead.any() else None
            r["auc_home_" + gname] = round(P.auc(v[alive & (y == "H")], v[alive & (y == "A")]), 3)
    df = pd.DataFrame(list(rows.values()))
    strength = lambda s: (s - 0.5).abs()
    for task in ("dead", "home"):
        cols = [c for c in df.columns if c.startswith("auc_{}_".format(task))]
        df["transfer_gap_" + task] = (df[cols].max(axis=1) - df[cols].min(axis=1)).round(3)
        df["strength_" + task + "_idsse"] = strength(df["auc_{}_idsse".format(task)]).round(3)
    df = df.sort_values("strength_dead_idsse", ascending=False)
    df.to_csv(os.path.join(HERE, "out", "feature_audit.csv"), index=False)
    return df


def summary(res):
    pick = lambda pre: [v for k, v in res.items() if k.startswith(pre)]
    rng = lambda vals, key: "{:.3f} ({:.3f}-{:.3f})".format(np.mean([v[key] for v in vals]), min(v[key] for v in vals), max(v[key] for v in vals)) if vals else "-"
    lines = ["  {}  IDSSE {}   Metrica {}   SoccerTrack {}   PFF {}   ours public {:.3f}".format(
                 m, rng(pick("idsse_"), m), rng(pick("metrica"), m), rng(pick("st_"), m), rng(pick("pff_"), m), res["ours_public"][m])
             for m in ("spellF1", "macroF1")]
    lines += ["  ours, no labels: {}".format(res["ours_label_free"])]
    return "\n".join(lines)


# ---------------------------------------------------------------------------- predictors

def nearest_codes(c):
    """Team of the nearest player when the ball is seen; NOBODY otherwise (the published baseline's input)."""
    d = np.linalg.norm(c.pxy - c.bxy[:, None, :], axis=2)
    d = np.where(np.isnan(d), np.inf, d)
    best = np.argmin(d, axis=1)
    codes = np.where(c.team[best] == 0, pp.HOME, pp.AWAY)
    return np.where(np.isfinite(d.min(1)), codes, pp.baseline.NOBODY)


def baseline_smooth50(c, train=None):
    """The published floor, on cleaned data: nearest player, D where the ball is missing, vote over +/-50."""
    return pp.baseline.smooth(nearest_codes(c), baseline_smooth50.params["window"])


baseline_smooth50.params = {"window": 50}


def speed_rule_naive(c, train=None):
    """Deliberately naive (no decoder): D whenever players are slow. Shows how D flicker fragments possessions."""
    codes = nearest_codes(c)
    lab = np.where(codes == pp.HOME, "H", np.where(codes == pp.AWAY, "A", "X")).astype("<U1")
    p = speed_rule_naive.params
    lab = np.array(pp.baseline.smooth(np.where(lab == "X", pp.baseline.NOBODY, codes), p["window"]))
    lab[(lab == "D")] = "X"                                   # missing ball: carry the current team
    lab[clean.mean_player_speed(c.pxy, c.fps) < p["slow_mps"]] = "D"
    lab[c.forced_dead] = "D"
    return lab


speed_rule_naive.params = {"window": 25, "slow_mps": 1.5}


def phase1_default(c, train=None):
    """Phase 1 rules + duration-aware Viterbi decoder, default settings (phase1.DEFAULTS)."""
    import phase1
    return phase1.predict(c, phase1_default.params)


phase1_default.params = {}


_FITS = {}


def phase1_tuned(c, train):
    """Phase 1 with settings tuned on `train` only (the six other IDSSE matches, or all seven for Metrica and ours)."""
    import phase1
    key = tuple(sorted(train))
    if key not in _FITS:
        _FITS[key] = phase1.fit(train, score)
        print("  fitted on {} matches: train spellF1 {:.4f}".format(len(train), _FITS[key][1]), flush=True)
    phase1_tuned.params = {"fitted_on_all_idsse": _FITS[key][0]} if len(train) == 7 else phase1_tuned.params
    return phase1.predict(c, _FITS[key][0])


phase1_tuned.params = {}


LOADED = {}   # the sets of the current run (load_sets fills it), so a predictor can tune on Metrica too


def _backend(name):
    import importlib
    mod, _, attr = name.partition(".")
    m = importlib.import_module(mod)
    return getattr(m, attr) if attr else m


def pool_prefixes(predict):
    """The sets a predictor trains on: pool False -> IDSSE, True -> IDSSE + Metrica, or a tuple of set prefixes."""
    pool = getattr(predict, "pool", False)
    return ("idsse_",) if not pool else ("idsse_", "metrica") if pool is True else tuple(pool)


CARRIED = {}              # id of a training match -> its copy for carried(), so every fold of a fit reuses it


def carried(n, c, draw=0):
    """A copy of training match n, or of its extra draw, with a ball that moves in stoppages (clean.carry_ball).
    The match itself stays as it is: it can also be a held-out test match, and those keep their true ball."""
    if id(c) not in CARRIED:
        x = copy.copy(c)
        x.bxy = c.bxy.copy()
        clean.carry_ball(x, seed=zlib.crc32(n.encode()) + draw)
        CARRIED[id(c)] = x
    return CARRIED[id(c)]


def make_phase3(name, drop=(), use_blend=False, pool=False, backend=None, poss_alpha=0.0, train_clean=False, tune_on=None, carry=False):
    """Phase 3 predictor. drop: feature prefixes to train without (e.g. phase3.BALL_CUES). use_blend: mix in Phase 1's speed-based
    dead-ball evidence, with the weight and decoder tuned on out-of-fold IDSSE plus Metrica (so Metrica is
    then a tuning set, not a held-out one). tune_on: phase3.TUNE_ON while this predictor's decoder is tuned, e.g.
    ("idsse_",) to tune a pooled predictor on IDSSE only; None leaves phase3.TUNE_ON as it is.
    carry: train on and tune on carried() copies; the match predicted is untouched."""
    fits = {}

    def predict(c, train):
        if train_clean and CLEAN:   # train on every player tracked, predict on the degraded match
            train = {k: CLEAN[k] for k in train}
        be = _backend(backend) if isinstance(backend, str) else backend   # "phase4": torch is imported only when used
        phase3.POSS_ALPHA = poss_alpha if POSS_ALPHA is None else POSS_ALPHA   # every predictor sets its own
        key = tuple(sorted(train))
        if key not in fits:
            extra = {k: v for k, v in LOADED.items() if k.startswith("metrica")} if use_blend else None
            saved, phase3.TUNE_ON = phase3.TUNE_ON, phase3.TUNE_ON if tune_on is None else tune_on
            draws = dict(phase3.AUG)
            if carry:
                phase3.AUG.update({k: [carried(k, x, i + 1) for i, x in enumerate(draws.get(k, ()))] for k in train})
            try:
                fits[key] = phase3.fit({k: carried(k, x) for k, x in train.items()} if carry else train, score, drop=drop, use_blend=use_blend, extra=extra, log=print, backend=be)
            finally:
                phase3.TUNE_ON = saved
                phase3.AUG.clear()
                phase3.AUG.update(draws)
        if len(train) == len(pool_names()):
            predict.params = {"model": phase3.MODEL, "drop": list(drop), "blend": use_blend, "poss_alpha": phase3.POSS_ALPHA, "carry": carry,
                              "tune_on": list(phase3.TUNE_ON if tune_on is None else tune_on),
                              "trained_on": sorted(train), "decoder_tuned_on_all" + ("_and_metrica" if use_blend else ""): fits[key][1]}
        mod = be or phase3                                                 # phase3.predict, keeping the decoder's input
        F = getattr(mod, "features", fe.frame_features)(c)
        logp, d = mod.log_probs(fits[key][0], F, drop), fits[key][1]
        DECODER_IN["last"] = logp, d
        return post_decode(c, phase3.decode(c, logp, F, d))

    pool_names = lambda: [k for k in LOADED if k.startswith(pool_prefixes(predict))]
    predict.__name__ = name
    predict.params = {}
    predict.pool = pool
    return predict


def post_decode(c, lab):
    """After the decoder, before the whistle: boundary.shift (our match only, off by default). replay.py calls it too."""
    import boundary
    return boundary.shift(c, lab)


WHISTLE = False  # the whistle snap (whistle.py) is not in this repository: its constants came from our match


def with_whistle(base, name):
    """`base`, then on our match only, dead-ball boundaries moved onto the referee's whistle (whistle.snap)."""
    def predict(c, train):
        lab = base(c, train)
        predict.params = dict(base.params, whistle_snap=WHISTLE)
        if c.name != "BAR-ATH (ours)" or not WHISTLE:
            return lab
        import whistle
        return whistle.snap(c, lab)
    predict.__name__, predict.pool, predict.params = name, getattr(base, "pool", False), {}
    return predict


PREDICTORS = {"baseline_smooth50": baseline_smooth50, "speed_rule_naive": speed_rule_naive,
              "phase1_default": phase1_default, "phase1_tuned": phase1_tuned,
              "phase3_gbm": make_phase3("phase3_gbm"), "phase3_robust": make_phase3("phase3_robust", drop=phase3.BALL_CUES),
              "phase3_keep_out": make_phase3("phase3_keep_out", drop=("ball_speed", "ball_fast")),
              "phase3_no_touch": make_phase3("phase3_no_touch", drop=phase3.BALL_CUES + ("last_touch_team",)),
              "phase3_pooled": make_phase3("phase3_pooled", drop=phase3.BALL_CUES, pool=True),
              "phase4_tcn": make_phase3("phase4_tcn", drop=phase3.BALL_CUES, backend="phase4"),
              "phase4_ensemble": make_phase3("phase4_ensemble", drop=phase3.BALL_CUES, backend="phase4.Ensemble"),
              "phase4_a05": make_phase3("phase4_a05", drop=phase3.BALL_CUES, backend="phase4.Ensemble", poss_alpha=0.5),
              "phase4_raw3": make_phase3("phase4_raw3", drop=phase3.BALL_CUES, backend="phase4.Raw3", poss_alpha=0.5),   # + raw TCN
              "phase4_a1": make_phase3("phase4_a1", drop=phase3.BALL_CUES, backend="phase4.Ensemble", poss_alpha=1.0),
              "phase4_trained_clean": make_phase3("phase4_trained_clean", drop=phase3.BALL_CUES, backend="phase4.Ensemble", train_clean=True),
              "phase4_a05_st": make_phase3("phase4_a05_st", drop=phase3.BALL_CUES, backend="phase4.Ensemble", poss_alpha=0.5,
                                           pool=("idsse_", "st_")),   # also trains on SoccerTrack, each match left out
              "phase4_pooled": make_phase3("phase4_pooled", drop=phase3.BALL_CUES, pool=True, backend="phase4.Ensemble"),
              "phase4_raw3_pooled": make_phase3("phase4_raw3_pooled", drop=phase3.BALL_CUES, backend="phase4.Raw3", poss_alpha=0.5,
                                                pool=True),   # phase4_raw3 trained on IDSSE + Metrica, each Metrica game left out
              "phase4_avg": make_phase3("phase4_avg", drop=phase3.BALL_CUES, backend="phase4.PooledAvg", poss_alpha=0.5,
                                        pool=True),   # phase4_raw3 and phase4_raw3_pooled averaged, the decoder tuned on the average
              "phase4_avg_idsse": make_phase3("phase4_avg_idsse", drop=phase3.BALL_CUES, backend="phase4.PooledAvg", poss_alpha=0.5,
                                              pool=True, tune_on=("idsse_",)),   # phase4_avg, its decoder tuned on IDSSE only
              "phase3_blend": make_phase3("phase3_blend", use_blend=True),
              "phase3_robust_blend": make_phase3("phase3_robust_blend", drop=phase3.BALL_CUES, use_blend=True)}
PREDICTORS["phase4_whistle"] = with_whistle(PREDICTORS["phase4_ensemble"], "phase4_whistle")
PREDICTORS["phase4_v2"] = with_whistle(PREDICTORS["phase4_a05"], "phase4_v2")   # baseline v2: dropout on, alpha 0.5, whistle snap
PREDICTORS["phase4_v3"] = with_whistle(PREDICTORS["phase4_a05"], "phase4_v3")   # baseline v3: v2 + our recovered ball (RECOVERED_BALL)
PREDICTORS["phase4_v4"] = with_whistle(PREDICTORS["phase4_raw3"], "phase4_v4")   # baseline v4: v3 + the raw ball-centric TCN (lane D); build with BOUNCE_S=2.5 features.INTERCEPT=False AUG_DRAWS=0
PREDICTORS["phase4_v5"] = with_whistle(PREDICTORS["phase4_raw3"], "phase4_v5")   # baseline v5: v4 trained on labels that keep brief possessions (BOUNCE_S 1.0); build with features.INTERCEPT=False AUG_DRAWS=0
PREDICTORS["phase4_avg_whistle"] = with_whistle(PREDICTORS["phase4_avg"], "phase4_avg_whistle")
PREDICTORS["phase4_v6"] = with_whistle(PREDICTORS["phase4_avg_idsse"], "phase4_v6")   # baseline v6: v5 averaged with v5 trained on IDSSE + Metrica, decoder tuned on IDSSE; build with features.INTERCEPT=False AUG_DRAWS=0
PREDICTORS["phase4_v7"] = with_whistle(PREDICTORS["phase4_avg_idsse"], "phase4_v7")   # baseline v7: v6 + intercept features (features.INTERCEPT); build with AUG_DRAWS=0
PREDICTORS["phase4_v8"] = with_whistle(PREDICTORS["phase4_avg_idsse"], "phase4_v8")   # baseline v8: v7 trained on two more corruption draws of each match (AUG_DRAWS)
PREDICTORS["phase4_v9"] = with_whistle(make_phase3("phase4_avg_idsse_carry", drop=phase3.BALL_CUES, backend="phase4.PooledAvg", poss_alpha=0.5,
                                                   pool=True, tune_on=("idsse_",), carry=True), "phase4_v9")   # candidate v9: v8 trained on a ball that moves in stoppages (clean.carry_ball); its two rates come from our match, see Q2
PREDICTORS["phase4_v10_set"] = with_whistle(make_phase3("phase4_avg_idsse_set", drop=phase3.BALL_CUES, backend="phase4.SetAvg", poss_alpha=0.5,
                                                        pool=True, tune_on=("idsse_",)), "phase4_v10_set")   # candidate: v8 with setnet as a third member (docs/2026-11-02-setnet.md)
PREDICTORS["phase4_v13_seeds"] = with_whistle(make_phase3("phase4_avg_idsse_seeds", drop=phase3.BALL_CUES, backend="phase4.Seeds", poss_alpha=0.5,
                                                          pool=True, tune_on=("idsse_",)), "phase4_v13_seeds")   # candidate: phase4.Seeds.K whole v12 models averaged, decoder tuned once on the average
PREDICTORS["phase4_metrica_only"] = with_whistle(make_phase3("phase4_raw3_metrica", drop=phase3.BALL_CUES, backend="phase4.Raw3", poss_alpha=0.5,
                                                             pool=("metrica",), tune_on=("metrica",)), "phase4_metrica_only")   # trained and tuned on Metrica alone, our match's provider (PooledAvg needs IDSSE)

# The candidate final recipe: `python validate.py submit final`. phase4.Seeds.K is 3 by default; the compliance values
# (clean.video_ball long_per_min 0, features.TOUCH_WINDOW_S 0.4, ...) are the code's defaults. FINAL_BALL_SOURCE is
# the one switch for the ball-source damage (docs/2026-11-10-ball-source-damage.md): "" turns it off. In `submit` every
# external set loaded is a training set (IDSSE and Metrica, with their extra draws), so the setting damages all of them.
# To score the same recipe: python validate.py run phase4_v13_seeds WHISTLE=False train:BALL_SOURCE='blocks'
# (train: applies it to the sets trained on only; add BALL_SOURCE='fitted' to damage the test sets with the other generator).
FINAL_BALL_SOURCE = "blocks"
FINAL = ("phase4_v13_seeds", [("WHISTLE", "False")] + ([("BALL_SOURCE", repr(FINAL_BALL_SOURCE))] if FINAL_BALL_SOURCE else []))


def check_scorer():
    """Rebuild the published baseline exactly (raw tracking, no cleaning) and score it on the public window."""
    trk = pp.load_tracking()
    pub = pp.baseline.smooth(pp.nearest_player_codes(trk), 50)
    gt = pp.read_labels(os.path.join(pp.REPO, "evaluation", "gt", "BAR-ATH_possession_gt_public.csv"))
    s = score(gt, pub)
    assert (s["spellF1"], s["macroF1"]) == (0.3256, 0.6207), s   # EVALUATION.md's published public scores
    return s


def apply_settings(pairs):
    """(KEY, VALUE) pairs from the command line onto settings, e.g. ("BALL_GAP_CONTEXT", "True"); "phase3.DEC_GRID" sets another module's, "phase4.Raw3.W_RAW" a class's."""
    import ast
    import importlib
    for k, v in pairs:
        mod, _, name = k.rpartition(".")
        try:
            obj = importlib.import_module(mod) if mod else sys.modules[__name__]   # imports e.g. phase4 if run hasn't yet
        except ModuleNotFoundError:   # a class's setting, e.g. phase4.Raw3.W_RAW
            m, _, cls = mod.rpartition(".")
            obj = getattr(importlib.import_module(m), cls, None) if m else None
        if not hasattr(obj, name):
            raise SystemExit("no setting " + k)
        setattr(obj, name, ast.literal_eval(v))


USAGE = """python validate.py                      score every predictor in PREDICTORS, log each run
python validate.py run P.. KEY=VAL train:KEY=VAL   with settings changed (e.g. BALL_GAP_CONTEXT=True); train: only for the sets trained on
python validate.py compare A B            where runs A and B differ (A, B: run id, time prefix or predictor name)
python validate.py submit PREDICTOR       out/submission_<PREDICTOR>.csv from every allowed training set, validated
python validate.py submit P KEY=VAL       a candidate with settings changed: out/submission_<P>_<KEY=VAL>.csv
python validate.py submit final           the candidate final recipe (FINAL): phase4_v13_seeds WHISTLE=False BALL_SOURCE='blocks'
python validate.py verdict NEW BASE       keep / revert NEW against BASE by the fixed rule (held-out spellF1, full-match gate)
python validate.py errors RUN SET         missed / spurious / late possessions of RUN on SET (e.g. ours, metrica1)
python validate.py audit                  per-feature coverage and separation, per dataset group"""

if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "run"
    if cmd in ("run", "submit", "audit"):
        heavy_slot()
    if cmd == "run":
        print("scorer check (published baseline_smooth50 on the public window):", check_scorer())
        settings = [a.split("=", 1) for a in sys.argv[2:] if "=" in a]
        train = [(k[6:], v) for k, v in settings if k.startswith("train:")]
        if train:   # build the sets trained on under their settings first, then put the settings back
            before = [(k, repr(globals()[k])) for k, _ in train]
            apply_settings(train)
            TRAIN_ALT.update(load_sets(soccertrack=False, pff=False))   # SoccerTrack and PFF are never trained on
            import phase3
            train_aug = dict(phase3.AUG)   # the extra draws too: the load below rebuilds them under the test settings
            apply_settings(before)
        apply_settings([kv for kv in settings if not kv[0].startswith("train:")])
        sets = load_sets()
        if train:
            phase3.AUG.clear(); phase3.AUG.update(train_aug)
        LOADED.update(sets)
        tags = pp.load_tags()
        results = {}
        chosen = [a for a in sys.argv[2:] if "=" not in a] or list(PREDICTORS)            # python validate.py run phase1_tuned
        for name, predict in ((n, PREDICTORS[n]) for n in chosen):
            results[name], preds = scoreboard(predict, sets, tags)
            print(name, "->", log_run(name, predict, results[name], preds, note=" ".join("=".join(kv) for kv in settings)))
            print(summary(results[name]))
        json.dump(results, open(os.path.join(HERE, "out", "scoreboard.json"), "w"), indent=1)
    elif cmd == "compare" and len(sys.argv) == 4:
        print(json.dumps(compare_runs(sys.argv[2], sys.argv[3]), indent=1))
    elif cmd == "submit" and len(sys.argv) >= 3:   # submit phase4_v4 boundary.SHIFT_S=0.6: a candidate, its own file
        name, settings = (FINAL[0], list(FINAL[1])) if sys.argv[2] == "final" else (sys.argv[2], [])
        settings += [a.split("=", 1) for a in sys.argv[3:]]
        apply_settings(settings)
        print(*submit(name, settings=settings), sep="\n")
    elif cmd == "verdict" and len(sys.argv) == 4:
        print(json.dumps(verdict(sys.argv[2], sys.argv[3]), indent=1))
    elif cmd == "errors" and len(sys.argv) == 4:
        print(json.dumps(run_errors(sys.argv[2], sys.argv[3]), indent=1))
    elif cmd == "audit":
        df = feature_audit(load_sets())
        cols = ["feature"] + [c for c in df.columns if c.startswith(("auc_dead", "auc_home", "coverage_idsse", "transfer_gap"))]
        print(df[cols].to_string(index=False))
    else:
        print(USAGE)
