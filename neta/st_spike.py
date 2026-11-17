"""Step 3 of docs/2026-11-03-net-a-ball-plan.md: which spike filter on net A's linked ball, chosen on SoccerTrack 132877
(held out of net A's training), never on our match. Truth is st_rescore's: YOLO else SoccerTrack, overridden by the reviewed
labels; right = within TOL px. Also spikes (> VMAX) per minute and dv95 (95th-pct velocity change across +-0.2 s).
    python neta/st_spike.py [HALF_DIR ...]      default: 132877_1_a 132877_2_a (half 2 has no truth: spikes and dv95 only)"""
import glob, json, os, sys
import numpy as np
from scipy.ndimage import gaussian_filter1d, median_filter
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CODE = os.path.join(ROOT, "mtah2027") if os.path.isdir(os.path.join(ROOT, "mtah2027")) else ROOT   # private repo / public export
sys.path[:0] = [CODE, CODE + "/external"]
import touch_spot as T
REV = os.path.expanduser("~/Github/external_data/_cache/ball_review")
FPS, TOL, VMAX = T.FPS, 10, 40.0


def runs(ok):
    e = np.flatnonzero(np.diff(np.r_[0, ok.astype(int), 0])); return zip(e[::2], e[1::2])


def despike(xy, vmax=VMAX, fps=FPS):
    """Drop isolated points that sit too fast from both kept neighbours, repeatedly; then cut the path where a jump stays."""
    xy = xy.copy()
    for _ in range(5):
        k = np.flatnonzero(np.isfinite(xy[:, 0]))
        if len(k) < 3:
            break
        d = np.linalg.norm(np.diff(xy[k], axis=0), axis=1) * fps / np.diff(k)
        bad = np.r_[False, (d[:-1] > vmax) & (d[1:] > vmax), False]
        if not bad.any():
            break
        xy[k[bad]] = np.nan
    return xy


def smooth(xy, kind, w):
    out = xy.copy()
    for a, b in runs(np.isfinite(xy[:, 0])):
        if kind == "median" and b - a >= w:
            out[a:b] = median_filter(xy[a:b], size=(w, 1), mode="nearest")
        if kind == "gauss":
            out[a:b] = gaussian_filter1d(xy[a:b], w * FPS, axis=0, mode="nearest")
    return out


FILTERS = {"raw": lambda xy: xy, "despike": despike,
           "despike+median3": lambda xy: smooth(despike(xy), "median", 3),
           "despike+median5": lambda xy: smooth(despike(xy), "median", 5),
           "despike+gauss0.04": lambda xy: smooth(despike(xy), "gauss", 0.04),
           "despike+gauss0.08": lambda xy: smooth(despike(xy), "gauss", 0.08),
           "gauss0.28 (tracker-like)": lambda xy: smooth(xy, "gauss", 0.28),
           "gauss0.5 (lane B tidy)": lambda xy: smooth(xy, "gauss", 0.5)}


def stats(xy):
    k = np.flatnonzero(np.isfinite(xy[:, 0]))
    adj = np.diff(k) == 1
    sp = np.linalg.norm(np.diff(xy[k], axis=0), axis=1)[adj] * FPS
    h = round(0.2 * FPS); ok = np.isfinite(xy[:, 0]); t = np.arange(h, len(xy) - h); t = t[ok[t] & ok[t - h] & ok[t + h]]
    dv = np.linalg.norm((xy[t + h] - xy[t]) - (xy[t] - xy[t - h]), axis=1) * FPS / h
    return (sp > VMAX).sum() / (len(xy) / FPS / 60), np.percentile(dv, 95), ok.mean()


def truth(D):
    z = np.load(os.path.join(D, "review_track.npz"))
    lab = np.where(np.isfinite(z["yolo"][:, :1]), z["yolo"][:, :2], z["st"])
    ans = {}
    for p in sorted(glob.glob(os.path.join(D, "labels_*.json"))):
        ans.update(json.load(open(p))["answers"])
    uns = {l.split()[0] for l in open(os.path.join(D, "UNSURE.txt")) if l.strip()} - set(ans)
    rows = []
    for i, f in enumerate(map(str, z["frame"])):
        a = ans.get(f, "keep")
        if f in uns or (z["st_zero"][i] and a == "keep" and not np.isfinite(z["yolo"][i, 0])) or len(a) == 3:
            continue
        rows.append((int(f), None if a == "none" else (lab[i] if a == "keep" else np.array(a[:2], float))))
    return rows


if __name__ == "__main__":
    for name in sys.argv[1:] or ["132877_1_a", "132877_2_a"]:
        D = os.path.join(REV, name); mid, h = name.split("_")[:2]
        tr = np.load(os.path.join(D, "track_L6s2.npy")); xy0 = tr[:, 3:5]
        to_img, to_pitch = T.pitch_map(T.players(mid, int(h)))
        rows = truth(D) if os.path.isfile(os.path.join(D, "review_track.npz")) and glob.glob(os.path.join(D, "labels_*.json")) else []
        print(f"== {name}: {len(xy0)} frames, truth on {len(rows)}")
        if rows:   # st_rescore's own score, in pixels, to check the truth is built the same way
            px = [np.isfinite(tr[f, 0]) and np.hypot(*(tr[f, :2] - L)) <= TOL if L is not None else not np.isfinite(tr[f, 0]) for f, L in rows]
            print(f"   raw path in pixels: right {np.mean(px):.1%} (st_rescore's measure)")
            rows = [(f, None if L is None else to_pitch(L[:1], L[1:])[0]) for f, L in rows]
        for fn, f in FILTERS.items():
            xy = f(xy0); spm, dv95, seen = stats(xy)
            line = f"{fn:26s} spikes/min {spm:5.2f}  dv95 {dv95:5.1f}  seen {seen:.3f}"
            if rows:
                ok = []
                for fr, L in rows:
                    p = xy[fr]
                    if L is None:
                        ok.append(not np.isfinite(p[0]))
                    else:
                        ok.append(np.hypot(*(p - L)) if np.isfinite(p[0]) else np.inf)
                ok = np.array(ok, float); bl = np.array([L is None for _, L in rows])
                r1 = np.where(bl, ~np.isfinite(ok) , ok <= 1.0).mean(); r2 = np.where(bl, ~np.isfinite(ok), ok <= 2.0).mean()
                e = ok[~bl & np.isfinite(ok)]
                line += f"  within 1 m {r1:.1%}  2 m {r2:.1%}  median err {np.median(e):.2f} m"
            print(line, flush=True)
