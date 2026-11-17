"""Our tracking parser against kloppy's independent EPTS reader, on our own match.

    MTAH_REPO=... python external/test_parsers.py

If both read every player and ball value identically, (1) preprocess.parse_atd is right beyond the
baseline check, and (2) kloppy can be trusted for Metrica game 3, whose EPTS layout parse_atd can't read.
"""
import os
import sys
import warnings

import numpy as np

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import preprocess as pp
from kloppy import metrica

meta, raw = pp.require(["atd_meta", "atd_raw"])
n_frames = pp.source_frames("tactical", pp.GAME)
team = pp.baseline.team_of_each_track(meta)
ours_players, ours_ball = pp.parse_atd(raw, len(team), n_frames)          # file units, row = video frame

ds = metrica.load_tracking_epts(meta_data=meta, raw_data=raw, coordinates="metrica")
df = ds.to_df(engine="pandas")
assert len(df) == n_frames, (len(df), n_frames)
rows = df["frame_id"].to_numpy() - 1                                      # frame_id 1 = video frame 0
players = sorted((p for t in ds.metadata.teams for p in t.players), key=lambda p: p.jersey_no)
assert [p.jersey_no for p in players] == list(range(1, len(team) + 1)), "slot = ShirtNumber"

worst = 0.0
for slot, p in enumerate(players):
    k = np.c_[df["{}_x".format(p.player_id)].to_numpy(float), df["{}_y".format(p.player_id)].to_numpy(float)]
    o = ours_players[rows, slot, :2]
    assert (np.isnan(k) == np.isnan(o)).all(), "missing-value pattern differs for slot {}".format(slot + 1)
    ok = ~np.isnan(o)
    worst = max(worst, float(np.abs(k[ok] - o[ok]).max()) if ok.any() else 0.0)
    assert (p.team.ground.name.lower() == "home") == (team[slot] == pp.HOME), "team of slot {}".format(slot + 1)
kb = np.c_[df["ball_x"].to_numpy(float), df["ball_y"].to_numpy(float)]
ob = ours_ball[rows]
assert (np.isnan(kb) == np.isnan(ob)).all(), "ball missing-value pattern differs"
okb = ~np.isnan(ob)
worst_ball = float(np.abs(kb[okb] - ob[okb]).max())
print("players: {} slots x {} frames, max difference {:.2e}; ball: max difference {:.2e}; teams agree".format(
    len(players), n_frames, worst, worst_ball))
assert worst < 1e-6 and worst_ball < 1e-6
print("ok")
