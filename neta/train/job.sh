#!/bin/bash
# The Nebius run, one stage at a time so the speed check can stop it: ./job.sh setup|fetch|crops|train|run
# SoccerTrack only: this folder holds no match code or data (vision/frames.py and preprocess.py are stubs).
set -euo pipefail
cd ~/job
export MTAH_EXT=$HOME/ext MTAH_BALL_DEVICE=cuda OMP_NUM_THREADS=3 OPENCV_FFMPEG_CAPTURE_OPTIONS="threads;4"
TRAIN="117092 117093 118575 118576 118577 118578 128057 128058 132831"
[ -f ~/v/bin/activate ] && . ~/v/bin/activate
case "$1" in
setup)
  python3 -m venv ~/v; . ~/v/bin/activate
  pip -q install torch torchvision opencv-python-headless ultralytics huggingface_hub scipy numpy
  nvidia-smi -L
  python -c "import torch; assert torch.cuda.is_available(); print(torch.__version__, torch.cuda.get_device_name())" ;;
fetch)
  python nb_fetch.py $TRAIN 132877 ;;
crops)
  for m in $TRAIN; do for h in 1 2; do echo "$m $h"; done; done |
    xargs -P 12 -L 1 sh -c 'python nb_crops.py "$0" "$1" > "log_crops_$0_$1.txt" 2>&1 && tail -1 "log_crops_$0_$1.txt"' ;;
train)   # A, B, C share the GPU; same seed and crops, so only frames and fake camera movement differ
  G="$MTAH_EXT/_cache/touch_spot/ball_crops_*.npz"
  python train_ext.py net_a.pt epochs=20 c=64 aug=2 rgb=1 crops="$G" frames=3 pan=0 > log_train_a.txt 2>&1 &
  python train_ext.py net_b.pt epochs=20 c=64 aug=2 rgb=1 crops="$G" frames=5 pan=1 > log_train_b.txt 2>&1 &
  python train_ext.py net_c.pt epochs=20 c=64 aug=2 rgb=1 crops="$G" frames=9 pan=1 > log_train_c.txt 2>&1 &
  wait ;;
run)
  NET=net_a.pt FRAMES=3 TAG=_a python nb_run.py 132877 1 > log_run_a.txt 2>&1 &
  NET=net_b.pt FRAMES=5 TAG=_b python nb_run.py 132877 1 > log_run_b.txt 2>&1 &
  NET=net_c.pt FRAMES=9 TAG=_c python nb_run.py 132877 1 > log_run_c.txt 2>&1 &
  wait ;;
esac
