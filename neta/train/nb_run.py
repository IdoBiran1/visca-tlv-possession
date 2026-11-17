"""A net (NET env, default net.pt) on every frame of a SoccerTrack half, then the ball path as touch_spot.track makes it (static white things
and off-pitch peaks dropped, lane B's linker). touch_spot.detect / track with our net (RGB) instead of ball_net_MID.pt.
python st_track.py MID H [LIMIT]  -> ~/Github/external_data/_cache/ball_review/MID_H/{peaks.npz, track.npy}"""
import numpy as np, os, sys, time, torch
import torch.nn.functional as F
sys.path[:0] = [os.getcwd(), os.getcwd() + "/external", os.getcwd() + "/vision", os.path.dirname(os.path.abspath(__file__))]
import touch_spot as T
MID, HALF = sys.argv[1], int(sys.argv[2])
TAG, N = os.environ.get("TAG", ""), int(os.environ.get("FRAMES", 3))
OUT = os.path.join(os.path.dirname(T.CACHE), "ball_review", f"{MID}_{HALF}{TAG}")
os.makedirs(OUT, exist_ok=True)
b = T.lane_b()
P = T.player_heights(T.players(MID, HALF), T.heights(MID))
n = int(P["frame"].max()) if len(sys.argv) < 4 else int(sys.argv[3])
pk_path = os.path.join(OUT, "peaks.npz")
if not os.path.isfile(pk_path) or len(sys.argv) > 3:
    net = b.Net(64, N).to(b.DEV); net.load_state_dict(torch.load(os.environ.get("NET", "net.pt"), map_location=b.DEV)); net.eval().half()
    Pf, s = T.by_frame(P, n)
    top = np.zeros((n + 1, T.K, 3), np.float32)
    canvas = torch.zeros(T.H, T.W, device=b.DEV, dtype=torch.float16)
    t0 = time.time()
    with torch.no_grad():
        for f, trip in T.frames(T.video_path(MID, HALF), range(2, n), N):
            sl = slice(s[f], s[f + 1]); ts = T.tiles_near(Pf["u"][sl], Pf["v"][sl], Pf["bh"][sl])
            if not ts:
                continue
            x = np.stack([trip[:, y0:y0 + T.TILE, x0:x0 + T.TILE] for y0, x0 in ts])[..., ::-1]   # BGR -> RGB
            hm = net(b.to_input(np.ascontiguousarray(x)).half()).sigmoid()
            canvas.zero_()
            for (y0, x0), t in zip(ts, hm):
                c = canvas[y0:y0 + T.TILE, x0:x0 + T.TILE]; torch.maximum(c, t, out=c)
            p = canvas.float()
            local = p == F.max_pool2d(p[None, None], 9, stride=1, padding=4)[0, 0]
            val, idx = (p * local).flatten().topk(T.K)
            idx, val = idx.cpu().numpy(), val.cpu().numpy()
            top[f] = np.c_[idx % T.W, idx // T.W, val]
            if f % 2500 == 0:
                print(f, f"{(time.time() - t0) / f:.3f} s/frame", flush=True)
                np.savez(pk_path, peaks=top)
    np.savez(pk_path, peaks=top)
    print("peaks done", f"{time.time() - t0:.0f} s", flush=True)
pk = np.load(pk_path)["peaks"]
to_img, to_pitch = T.pitch_map(T.players(MID, HALF))
xy = to_pitch(pk[..., 0], pk[..., 1])
p = np.where((pk[..., 2] > 0.05) & ~T.static(pk) & T.on_pitch(pk, to_img), np.minimum(pk[..., 2] / T.P_TOP, 1), 0)
xy[p <= 0] = np.nan
out = np.full((len(pk), 5), np.nan)
for a in range(0, len(pk), 5000):
    sl = slice(a, min(a + 5000, len(pk)))
    ch = b.link(xy[sl], p[sl], fps=T.FPS)
    seen = ch < T.K
    i = np.arange(sl.start, sl.stop)[seen]
    out[i, :3] = pk[i, ch[seen]]; out[i, 3:] = xy[i, ch[seen]]
np.save(os.path.join(OUT, "track.npy"), out)
print("track: ball seen on", f"{np.isfinite(out[:, 0]).mean():.0%}", "of frames")
