"""Net A over every graded frame of our match (docs/2026-11-03-net-a-ball-plan.md, step 2).

Imports lane B's pipeline (vision/ball.py) from the code worktree read-only and only swaps the weights. The search mask is
lane B's: near the players, plus around the tracker's ball where it has one, else around the last and next tracked spots.
Top K_PEAKS per frame are saved in 2,000-frame pieces, so a stopped run resumes where it left off.

    python neta/detect_all.py NET.pt OUT_DIR [--check]    # --check: first 100 frames of chunk 24432, players only, vs our12
"""
import os, sys
import numpy as np, torch
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CODE = os.environ.get("MTAH_CODE") or (os.path.join(ROOT, "mtah2027") if os.path.isdir(os.path.join(ROOT, "mtah2027")) else ROOT)   # private repo / public export
sys.path[:0] = [CODE, CODE + "/vision"]
os.chdir(CODE)
import ball as BL
from frames import load_homography, to_pixels
from preprocess import GAME, half_video_blocks, load_tracking

PIECE = 2000


def run(net, trk, Hm, a, b, anchors=True):
    """[b - a, K, 3] peaks (u, v, p) for frames a..b-1."""
    ball = trk["ball"]; has = np.isfinite(ball[:, 0]); known = np.flatnonzero(has)
    out = np.zeros((b - a, BL.K_PEAKS, 3), np.float32)
    with torch.no_grad():
        for f, trip in BL.triples(a, b - 1):
            pl = to_pixels(Hm, f, trk["xy"][f]); pl = pl[BL.in_view(pl)]
            if not anchors:
                anc = []
            elif has[f]:
                anc = [(*to_pixels(Hm, f, ball[f]), BL.R0)]
            else:
                anc = BL.anchors_for(f, ball, known, Hm)
            out[f - a] = BL.peaks(net, trip, BL.search_mask(pl, anc, Hm[f]))
    return out


if __name__ == "__main__":
    path, out_dir = sys.argv[1], sys.argv[2]
    net = BL.Net(64, 3).to(BL.DEV); net.load_state_dict(torch.load(path, map_location=BL.DEV)); net.eval()
    trk, Hm = load_tracking(), load_homography()
    if "--check" in sys.argv:
        ref = np.load(os.path.join(os.path.dirname(path), "net_a.npz"))["peaks"][0, :100]
        got = run(net, trk, Hm, 24432, 24532, anchors=False)
        print("max |diff| vs our12:", float(np.abs(got - ref).max()))
        sys.exit()
    os.makedirs(out_dir, exist_ok=True)
    import time; t0 = time.time()
    for _, first, last in half_video_blocks(GAME):
        for a in range(first, last + 1, PIECE):
            b = min(a + PIECE, last + 1); f = os.path.join(out_dir, "peaks_%06d_%06d.npy" % (a, b))
            if os.path.exists(f):
                continue
            np.save(f + ".tmp.npy", run(net, trk, Hm, a, b)); os.replace(f + ".tmp.npy", f)
            print(a, b, "%.0f s" % (time.time() - t0), flush=True)
