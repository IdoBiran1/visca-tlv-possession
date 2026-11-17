"""python external/test_profile.py -- checks the logic the profiles and loaders rest on."""
import numpy as np
from profile_datasets import auc, dequantize, metrica_labels, pff_labels, soccertrack_labels

# AUC: perfect, reversed, chance, and ties counted half
assert auc(np.array([3., 4.]), np.array([1., 2.])) == 1.0
assert auc(np.array([1., 2.]), np.array([3., 4.])) == 0.0
assert auc(np.array([1., 1.]), np.array([1., 1.])) == 0.5

# Metrica events -> labels: Home passes, the ball goes out, Away throws in
ev = lambda team, typ, start, end="", sub="": {"Team": team, "Type": typ, "Subtype": sub, "Start Frame": str(start), "End Frame": str(end)}
lab = metrica_labels(12, [ev("Home", "SET PIECE", 1, sub="KICK OFF"), ev("Home", "PASS", 3, 4),
                          ev("Home", "BALL OUT", 6, 6), ev("Away", "SET PIECE", 9, sub="THROW IN"), ev("Away", "PASS", 10, 11)])
assert "".join(lab) == "HHHHHDDDAAAA", "".join(lab)
# a shot off target stays dead until the goal kick, which Metrica types as a PASS, not a SET PIECE
lab = metrica_labels(12, [ev("Home", "SET PIECE", 1, sub="KICK OFF"), ev("Home", "SHOT", 3, 4, sub="OFF TARGET-OUT"),
                          ev("Away", "RECOVERY", 6), ev("Away", "PASS", 8, 9, sub="GOAL KICK"), ev("Home", "RECOVERY", 11)])
assert "".join(lab) == "HHHDDDDAAAHH", "".join(lab)
# fouls typed only as a CHALLENGE ...-FAULT-..., unless advantage is played; at fps 1, a BALL LOST 2 s into a stoppage
# is part of it, but one 5 s after a ball out (a keeper's kick Metrica didn't type) restarts play
lab = metrica_labels(12, [ev("Home", "PASS", 1, 2), ev("Away", "CHALLENGE", 3, sub="TACKLE-FAULT-WON"),
                          ev("Away", "BALL LOST", 5, 6, sub="FORCED"), ev("Away", "SET PIECE", 9, sub="FREE KICK")], fps=1)
assert "".join(lab) == "HHDDDDDDAAAA", "".join(lab)
lab = metrica_labels(8, [ev("Home", "PASS", 1, 2), ev("Away", "CHALLENGE", 3, sub="GROUND-ADVANTAGE-WON"), ev("Home", "PASS", 5, 6)], fps=1)
assert "".join(lab) == "HHHHHHHH", "".join(lab)
lab = metrica_labels(12, [ev("Home", "PASS", 1, 2), ev("Home", "BALL OUT", 3, 4), ev("Away", "RECOVERY", 6),
                          ev("Away", "BALL LOST", 9, 10, sub="INTERCEPTION"), ev("Home", "RECOVERY", 11)], fps=1)
assert "".join(lab) == "HHHDDDDDAAHH", "".join(lab)
# a set piece with no stoppage recorded: dead from the end of the last event; the end of a half is dead to its end
lab = metrica_labels(12, [ev("Home", "PASS", 1, 3), ev("Away", "CARD", 5, sub="YELLOW"), ev("Home", "SET PIECE", 8, sub="FREE KICK"),
                          ev("Home", "BALL LOST", 9, 10, sub="END HALF")], fps=1)
assert "".join(lab) == "HHDDDDDHHDDD", "".join(lab)
# SoccerTrack ballStatus -> labels: NEUTRAL stays with the last team, but not across halves; INPLAY isn't a label
st = ["NEUTRAL", "HOME", "NEUTRAL", "BALLOUT", "AWAY", "NEUTRAL", "NEUTRAL", "HOME", "INPLAY"]
assert "".join(soccertrack_labels(st, [0, 6])) == "XHHDAAXHX", "".join(soccertrack_labels(st, [0, 6]))

# dequantize: a straight run rounded to a 1 m grid comes back close to the line, gaps stay gaps
truth = np.c_[np.linspace(10, 20, 200), np.full(200, 30.0)]
rounded = np.round(truth)
rounded[100:110] = np.nan
smooth = dequantize(rounded, [(0, 200)], 25.0, 0.3)
assert np.isnan(smooth[100:110]).all()
assert np.abs(rounded - truth)[20:90].max() > 0.4 and np.abs(smooth - truth)[20:90].max() < 0.25
# PFF events -> labels: home controls, a loose ball keeps home, out (D) until away's throw-in, a foul (no OUT)
# makes D from the end of away's control to home's free kick; outside the period is X
ge = lambda i, t0, t1, typ, sp=None, home=None: {"gameEventId": i, "startTime": t0, "endTime": t1,
                                                "gameEvents": {"gameEventType": typ, "setpieceType": sp, "homeTeam": home}}
evs = [ge(1, 0, 1, "FIRSTKICKOFF", "K", True), ge(2, 2, 3, "OTB", "O", True), ge(3, 5, None, "OUT"),
       ge(4, 7, 8, "OTB", "T", False), ge(5, 9, 10, "OTB", "F", True), ge(5, 9, 10, "OTB", "F", True)]
assert "".join(pff_labels(evs, np.arange(0, 12.0), [(0, 11)])) == "HHHHHDDADHHX", "".join(pff_labels(evs, np.arange(0, 12.0), [(0, 11)]))
print("ok")
