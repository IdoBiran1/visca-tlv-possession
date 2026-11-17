"""Steps 3-5 of docs/2026-11-03-net-a-ball-plan.md: net A's peaks -> linked ground path -> despike + 1-frame Gaussian
(neta/st_spike.py, chosen on SoccerTrack 132877) -> merged with the organisers' tracker ball -> a RECOVERED_BALL file.

Per frame: the tracker has the ball and net A's ball is within R_CONFIRM m of it -> net A's (sharp; source "net");
the tracker has it otherwise -> the tracker's (a ball in the air projects far from the ground point; "track");
no tracker ball and net A has one -> net A's ("video"); else none. R_CONFIRM is set a priori, not tuned.

    python neta/merge.py PEAKS_DIR OUT.npz      # pieces from neta/detect_all.py
    python neta/merge.py --our12                # the 12 chunks (players-only search), label-free stats only
"""
import glob, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import detect_all as DA                     # puts the code worktree on the path
import ball as BL
sys.path.insert(0, DA.CODE + "/external")
import touch_spot as DA_T
from frames import load_homography, to_pitch, to_pixels
from preprocess import load_tracking
from st_spike import despike, smooth

R_CONFIRM, SIG, WINDOW, FPS = 2.0, 2.0, 5000, 25.0
HOLD, RAMP, GAP = 12, 9, 5        # frames: switch hysteresis, cross-fade, net gaps filled

# SoccerTrack's candidate filters, unchanged (touch_spot.static and on_pitch, review/st_relink.py's long_static for
# off-pitch peaks, the p floor; set on SoccerTrack 117093 and 132877: the chain that scored track_L6s2). They are in
# image pixels, and our tactical camera's scale is SoccerTrack's (px/m along / across the pitch, ours vs 132877:
# far 19.8/6.4 vs 17.9/12.7, centre 26.4/11.1 vs 24.1/11.9, near 40.5/25.5 vs 41.3/38), so they run on our pixel peaks
# as they are. Our camera pans, so the touchlines for on_pitch are projected per frame.
# ponytail: a spare ball is masked only while the camera holds still for the 5 s blocks; panning leaves it in.
P_FLOOR, LONG_BLOCKS, LONG_FRAC, LONG_CELL = 0.05, 6, 0.25, 12


def long_static(pk, blocks, frac):
    """review/st_relink.py's long_static, unchanged: peaks at spots (LONG_CELL cells, with their 8 neighbours)
    holding a peak on frac of the frames of a window of blocks * 125 frames around them."""
    B = DA_T.STATIC_BLOCK
    cell = (pk[..., 0] // LONG_CELL).astype(np.int64) * 10000 + (pk[..., 1] // LONG_CELL).astype(np.int64)
    cell[pk[..., 2] <= 0] = -1
    cnt = []
    for a in range(0, len(pk), B):
        u, n = np.unique(np.concatenate([np.unique(r) for r in cell[a:a + B]]), return_counts=True)
        cnt.append(dict(zip(u[u >= 0], n[u >= 0])))
    ring = (-10001, -10000, -9999, -1, 0, 1, 9999, 10000, 10001)
    out = np.zeros(cell.shape, bool)
    h = blocks // 2
    for i in range(len(cnt)):
        tot = {}
        for c in cnt[max(0, i - h):i + h + 1]:
            for k, n in c.items():
                tot[k] = tot.get(k, 0) + n
        need = frac * B * len(cnt[max(0, i - h):i + h + 1])
        bad = [k for k in cnt[i] if sum(tot.get(k + d, 0) for d in ring) >= need]
        if bad:
            sl = slice(i * B, (i + 1) * B)
            out[sl] = np.isin(cell[sl], bad)
    return out


def keep(pk, xy, on, off):
    """(T, K) candidates that pass SoccerTrack's filters. pk image peaks (u, v, p), xy their pitch metres, on: between
    the touchlines plus touch_spot's pixel margins, off: outside the touchlines or goal lines (no margin)."""
    drop = DA_T.static(pk) | ~on | (long_static(pk, LONG_BLOCKS, LONG_FRAC) & off)
    return (pk[..., 2] > P_FLOOR) & np.isfinite(xy[..., 0]) & ~drop


def link_path(pk, xy, on, off):
    """[T, 2] net A's linked ground path from candidates, filtered as on SoccerTrack, then despike + Gaussian 0.04 s."""
    ok = keep(pk, xy, on, off); p = np.where(ok, np.minimum(pk[..., 2] / DA_T.P_TOP, 1), 0)
    xy = np.where(ok[..., None], xy, np.nan); n = len(xy); out = np.full((n, 2), np.nan)
    for a in range(0, n, WINDOW):
        b = min(a + WINDOW, n); ch = BL.link(xy[a:b], p[a:b], sig=SIG); s = ch < xy.shape[1]
        out[a:b][s] = xy[a:b][np.arange(b - a)[s], ch[s]]
    return smooth(despike(out, fps=FPS), "gauss", 0.04)


def net_path(pk, first, Hm, length=105.3, width=68.0):
    """[n, 2] net A's path on our match for frames first..first+n-1 (pitch 0..length x 0..width; near touchline y = width)."""
    fr = np.arange(first, first + len(pk)); pk = pk.astype(float)
    xy = to_pitch(Hm, fr, pk[:, :, :2]); xy[pk[:, :, 2] <= 0] = np.nan
    x = np.linspace(-10, length + 10, 64); on = np.zeros(pk.shape[:2], bool)
    for i, f in enumerate(fr):   # touch_spot.on_pitch with this frame's touchlines
        near = to_pixels(Hm, f, np.c_[x, np.full_like(x, width)]); far = to_pixels(Hm, f, np.c_[x, np.zeros_like(x)])
        nv = np.interp(pk[i, :, 0], *near[np.argsort(near[:, 0])].T); fv = np.interp(pk[i, :, 0], *far[np.argsort(far[:, 0])].T)
        on[i] = (pk[i, :, 1] <= nv + DA_T.NEAR_MARGIN) & (pk[i, :, 1] >= fv - DA_T.FAR_MARGIN)
    off = (xy[..., 0] < 0) | (xy[..., 0] > length) | (xy[..., 1] < 0) | (xy[..., 1] > width)
    return link_path(pk, xy, on, off)


def fill(x, max_gap):
    """Short gaps inside a path, linearly interpolated."""
    x = x.copy(); ok = np.isfinite(x[:, 0]); k = np.flatnonzero(ok)
    for i, j in zip(k[:-1], k[1:]):
        if 1 < j - i <= max_gap + 1:
            x[i + 1:j] = x[i] + (x[j] - x[i]) * (np.arange(1, j - i) / (j - i))[:, None]
    return x


def merge(net, trk):
    """Net A where it confirms the tracker over a run (gaps under HOLD frames closed, runs under HOLD dropped), with a
    RAMP-frame cross-fade at each switch so the tracker's offset from the net doesn't make a jump."""
    from scipy.ndimage import binary_closing, uniform_filter1d
    net = fill(net, GAP)
    has_t, has_n = np.isfinite(trk[:, 0]), np.isfinite(net[:, 0])
    both = has_t & has_n
    near = both & (np.linalg.norm(np.nan_to_num(net - trk, nan=1e9), axis=1) <= R_CONFIRM)
    near = binary_closing(near, np.ones(HOLD, bool)) & both
    e = np.flatnonzero(np.diff(np.r_[0, near.astype(int), 0]))
    for a, b in zip(e[::2], e[1::2]):
        if b - a < HOLD:
            near[a:b] = False
    w = np.where(both, uniform_filter1d(near.astype(float), RAMP), 0.0)[:, None]
    xy = np.where(has_t[:, None], w * np.nan_to_num(net) + (1 - w) * trk, net)
    src = np.where(near, "net", np.where(has_t, "track", np.where(has_n, "video", "none")))
    return xy, src


def stats(xy, live=None):
    ok = np.isfinite(xy[:, 0]) if live is None else np.isfinite(xy[:, 0]) & live
    h = 5; t = np.arange(h, len(xy) - h); t = t[ok[t] & ok[t - h] & ok[t + h]]
    dv = np.linalg.norm((xy[t + h] - xy[t]) - (xy[t] - xy[t - h]), axis=1) * FPS / h
    k = np.flatnonzero(ok); adj = np.diff(k) == 1
    sp = np.linalg.norm(np.diff(xy[k], axis=0), axis=1)[adj] * FPS
    return "dv95 %.1f  spikes/min %.2f" % (np.percentile(dv, 95), (sp > 40).sum() / (len(xy) / FPS / 60))


if __name__ == "__main__":
    trk, Hm = load_tracking(), load_homography(); ball = trk["ball"].astype(float)
    if sys.argv[1] == "--our12":
        z = np.load(os.path.expanduser("~/Github/external_data/_cache/ball_ext/our12/net_a.npz"))
        nets, trks, srcs, merged = [], [], [], []
        for k, a in enumerate(z["starts"]):
            pk = z["peaks"][k]; L = len(pk)
            net = net_path(pk, a, Hm); t = ball[a:a + L]; xy, src = merge(net, t)
            nets.append(net); trks.append(t); srcs.append(src); merged.append(xy)
            nets.append(np.full((10, 2), np.nan)); trks.append(np.full((10, 2), np.nan)); merged.append(np.full((10, 2), np.nan))
        src = np.concatenate(srcs)
        print("sources:", {s: round(float((src == s).mean()), 3) for s in np.unique(src)})
        for name, x in (("tracker", trks), ("net A path", nets), ("merged", merged)):
            print("%-11s" % name, stats(np.concatenate(x)))
        sys.exit()
    pieces = sorted(glob.glob(os.path.join(sys.argv[1], "peaks_*.npy")))
    n = len(ball); xy, src = ball.copy(), np.where(np.isfinite(ball[:, 0]), "track", "none")
    for p in pieces:
        a, b = map(int, os.path.basename(p)[6:-4].split("_"))
        net = net_path(np.load(p), a, Hm); xy[a:b], src[a:b] = merge(net, ball[a:b])
    os.makedirs(os.path.dirname(os.path.abspath(sys.argv[2])), exist_ok=True)
    np.savez(sys.argv[2], frame=np.arange(n), x=xy[:, 0], y=xy[:, 1], x_tidy=xy[:, 0], y_tidy=xy[:, 1], source=src,
             confidence=np.ones(n, np.float32), airborne=np.zeros(n, bool), plausible=np.isfinite(xy[:, 0]))
    print("wrote", sys.argv[2], {s: round(float((src == s).mean()), 3) for s in np.unique(src)})
