# MTAH 2027: ball possession

Team Visca TLV's code for [More than a Hack 2027](https://github.com/metrica-sports/more-than-a-hack-2027): label
every frame of Barcelona v Athletic as H (home possession), A (away possession) or D (dead ball).

Platform the file was built and checked on: macOS 26.5 on an Apple M4 Pro (arm64), Python 3.12.13, ffmpeg 8.1.1,
the pinned requirements.txt. The same file is expected on that platform; other platforms may differ in the last bits.

```bash
git clone https://github.com/metrica-sports/more-than-a-hack-2027 ../more-than-a-hack-2027
git -C ../more-than-a-hack-2027 checkout 3de6b60
# the competition data bundle goes in ../more-than-a-hack-2027/data/ (tactical/, broadcast/)
export MTAH_REPO=$PWD/../more-than-a-hack-2027 MTAH_EXT=$PWD/../external_data
python3.12 -m venv .venv && .venv/bin/pip install -r requirements.txt   # pinned
.venv/bin/python data.py fetch                       # the public training data (IDSSE, Metrica samples)

# 1. our match's ball from the tactical video (ffmpeg on the PATH): net A, trained on SoccerTrack only
#    (weights/net_a.pt; training code in neta/train/). About 12.5 h on the Mac's GPU (MPS); it resumes if stopped.
.venv/bin/python neta/detect_all.py weights/net_a.pt out/neta_peaks
.venv/bin/python neta/merge.py out/neta_peaks out/ours_ball_neta_v2.npz       # about 1 min
shasum -a 256 out/ours_ball_neta_v2.npz              # 7da940c5cb176c65013cc275d220e518fa07517c0bd0610c1c57dd5b706e02eb (validate.py checks it too)

# 2. the possession file: trained on the CPU from the public data, then the challenge's validator
.venv/bin/python validate.py submit final   # out/submission_phase4_v13_seeds_WHISTLE=False_BALL_SOURCE=blocks.csv
```

Pipeline: `clean.py` (one cleaning pipeline for every dataset) -> `features.py` (identity-free per-frame features,
including which team can reach the ball first) -> three members averaged: `phase3.py` (gradient-boosted trees) with
`phase4.py`'s two temporal convolutional networks (one on the frame features, one on the raw positions of the ball and
the players nearest it), each trained on IDSSE and on IDSSE plus the Metrica samples, and `setnet.py` (the ball and the
players nearest it, at full rate) -> three such models from three seeds averaged (`phase4.Seeds`) -> `phase1.py`'s
duration-aware Viterbi decoder, tuned on IDSSE. Every training match is seen under three draws of a corruption that
makes it look like our tracking (players and ball hidden the way our tracker loses them). Where our tracking has no
ball, `neta/` finds it in the tactical video with net A, a U-Net trained on SoccerTrack, filtered and linked with
settings chosen on SoccerTrack, and merged with the tracker's ball. Nothing is trained or tuned on our match.
`validate.py` holds the evaluation harness. Each `test_*.py` is a self-check.

Notes. `requirements.txt` pins the direct dependencies; `requirements.lock` lists every package version of the
environment that built the submitted file. Regenerating the file doesn't need net A's training: `neta/train/job.sh`
is the record of how the weights were made (on a rented Linux GPU machine, from SoccerTrack v2 only) and installs its
own packages. Comments that cite dated notes under `docs/` refer to the team's working notes, which are not part of
this repository; the technical document submitted with the entry summarises them. The match data is not here and must
not be added: `out/` and `cache/` are ignored.
