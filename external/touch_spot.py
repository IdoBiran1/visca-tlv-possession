"""Ball touches from video: the frame each touch happens, who touched it and their team, on SoccerTrack v2.

Ball from the video (lane B's 3-frame heatmap net, retrained on the panorama), players from the game-state annotations
(as our match has tracking), touches where the ball's path bends or stops at a player. Judged against the ball-action
events, which name the frame, the actor and the team. Train / tune on the 1st half, test on the 2nd.
docs/2026-10-25-ball-touches.md. Steps, cached under $MTAH_EXT/_cache/touch_spot/:

    python external/touch_spot.py players MID        # game-state players per frame, both halves      players_MID_H.npz
    python external/touch_spot.py crops MID          # ball labels (event-anchored ball, snapped) -> crops, 1st half
    python external/touch_spot.py train MID          # the ball net                                   ball_net.pt
    python external/touch_spot.py detect MID H       # ball peaks on every frame of half H             peaks_MID_H.npz
    python external/touch_spot.py touches MID        # link, touches, score against the events         touches_MID.json
    python external/touch_spot.py video MID H        # review clips: hits, misses, false touches       out/touches_MID_H*.mp4
    python external/touch_spot.py test               # self-checks on synthetic tracks

Needs torch and opencv, which aren't in requirements.txt: run it from its own venv, e.g.
    uv venv /tmp/vvenv --python 3.12 && uv pip install --python /tmp/vvenv/bin/python numpy scipy torch opencv-python-headless
Needs the match's SoccerTrack videos/ and gsr/ folders (not fetched by data.py).
"""
import json, os, sys
import numpy as np

EXT = os.path.join(os.environ.get("MTAH_EXT", os.path.expanduser("~/Github/external_data")), "soccertrack")
CACHE = os.path.join(os.path.dirname(EXT), "_cache", "touch_spot")
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "out")
W, H = 4096, 1080
YOLO_DEV = os.environ.get("MTAH_BALL_DEVICE") or "mps"
FPS = 25
NOT_TOUCH = {"OUT", "GOAL"}          # ball-action labels that aren't a touch


def half_name(h):
    return "1st" if h == 1 else "2nd"


def video_path(mid, h):
    return os.path.join(EXT, "videos", mid, f"{mid}_panorama_{half_name(h)}_half.mp4")


def events(mid, half=None):
    """Touch events: dicts with half, frame (1-based video frame), pid, team ('left' / 'right'), label."""
    d = json.load(open(os.path.join(EXT, "bas", mid, mid + "_12_class_events.json")))
    ev = [dict(half=int(a["gameTime"][0]), frame=int(a["frame"]), pid=int(a.get("player_id") or 0), team=a["team"],
               label=a["label"]) for a in d["actions"] if a["label"] not in NOT_TOUCH]
    return [e for e in ev if half is None or e["half"] == half]


# -- players ---------------------------------------------------------------------------------------

def players(mid, h):
    """Game-state players of half h as flat arrays: frame (1-based), pid, team (0 left, 1 right), u, v (image foot
    point), bh (box height, px), x, y (pitch m). Streamed from the pretty-printed JSON a line at a time."""
    path = os.path.join(CACHE, f"players_{mid}_{h}.npz")
    if os.path.isfile(path):
        return dict(np.load(path))
    rows, cur, inann = [], {}, False
    for line in open(os.path.join(EXT, "gsr", mid, f"{mid}_{half_name(h)}.json")):
        s = line.strip()
        if not inann:
            inann = s.startswith('"annotations"')
            continue
        key = s.split('"')[1] if s.startswith('"') else ""
        if key == "image_id":
            cur = {"frame": int(s.split('"')[3][-6:])}
        elif key in ("role", "team"):
            cur[key] = s.split('"')[3]
        elif key == "player_id":
            cur["pid"] = int(s.split(":")[1].strip().rstrip(","))
        elif key in ("x_center", "y", "h") and "bx" not in cur:
            cur[key] = float(s.split(":")[1].strip().rstrip(","))
            if key == "h":
                cur["bx"] = True
        elif key == "x_bottom_middle" and "px" not in cur:
            cur["px"] = float(s.split(":")[1].strip().rstrip(","))
        elif key == "y_bottom_middle" and "py" not in cur and "px" in cur:
            if cur.get("role") in ("player", "goalkeeper") and "pid" in cur and "bx" in cur:
                rows.append((cur["frame"], cur["pid"], cur["team"] == "right", cur["x_center"], cur["y"] + cur["h"],
                             cur["h"], cur["px"], float(s.split(":")[1].strip().rstrip(","))))
            cur = {"frame": cur["frame"]}
    r = np.array(rows)
    out = dict(frame=r[:, 0].astype(np.int32), pid=r[:, 1].astype(np.int64), team=r[:, 2].astype(np.int8),
               u=r[:, 3].astype(np.float32), v=r[:, 4].astype(np.float32), bh=r[:, 5].astype(np.float32),
               x=r[:, 6].astype(np.float32), y=r[:, 7].astype(np.float32))
    os.makedirs(CACHE, exist_ok=True)
    np.savez(path, **out)
    return out


def by_frame(P, n):
    """Row slices per 1-based frame: P sorted by frame, so frame f's rows are P[s[f]:s[f + 1]]."""
    o = np.argsort(P["frame"], kind="stable")
    P = {k: v[o] for k, v in P.items()}
    return P, np.searchsorted(P["frame"], np.arange(n + 2))


def _lookup(a, b, src, dst, cell):
    """Interpolator src -> dst from the mean dst per cell of src (a, b), linear between cells, nearest outside."""
    from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator
    k = np.round(np.c_[a, b] / cell).astype(np.int64)
    keys, inv, cnt = np.unique(k, axis=0, return_inverse=True, return_counts=True)
    inv = inv.ravel()
    mean = np.stack([np.bincount(inv, d, len(keys)) / cnt for d in dst.T], 1)
    ok = cnt >= 3
    pts = keys[ok] * cell
    lin, near = LinearNDInterpolator(pts, mean[ok]), NearestNDInterpolator(pts, mean[ok])

    def f(x, y):
        q = np.c_[np.ravel(x), np.ravel(y)].astype(float)
        out = lin(q)
        bad = ~np.isfinite(out[:, 0]) & np.isfinite(q).all(1)      # a NaN query stays NaN
        out[bad] = near(q[bad])
        return out.reshape(np.shape(x) + (2,))
    return f


def pitch_map(P):
    """Maps pitch (x, y) -> image (u, v) and back, from the players' feet over the half: the camera is fixed, but the
    panorama is stitched and bent, so a lookup of mean positions (1 m / 8 px cells) rather than a formula."""
    to_img = _lookup(P["x"], P["y"], None, np.c_[P["u"], P["v"]].astype(float), 1.0)
    to_pitch = _lookup(P["u"], P["v"], None, np.c_[P["x"], P["y"]].astype(float), 8.0)
    return to_img, to_pitch


def heights(mid, h=1, n=40):
    """Player height (px) by image position: the game-state boxes are a fixed ~43 px, so YOLO person boxes on n frames,
    each matched to the nearest game-state foot within 40 px, give a lookup (64 px cells) from foot (u, v) to height."""
    path = os.path.join(CACHE, f"heights_{mid}.npz")
    if not os.path.isfile(path):
        from ultralytics import YOLO
        model = YOLO(os.path.join(CACHE, "yolo", "yolo11x.pt"))
        P = players(mid, h)
        P, s = by_frame(P, int(P["frame"].max()))
        want = np.linspace(500, len(s) - 600, n).astype(int)
        rows = []
        for f, trip in frames(video_path(mid, h), want):
            img = trip[1]
            for x0 in range(0, W - 1024 + 1, 768):
                r = model.predict(img[:, x0:x0 + 1024], classes=[0], conf=0.3, verbose=False, device=YOLO_DEV, imgsz=1088)[0]
                for bx0, by0, bx1, by1 in r.boxes.xyxy.cpu().numpy():
                    foot = np.array([x0 + (bx0 + bx1) / 2, by1])
                    sl = slice(s[f], s[f + 1])
                    d = np.hypot(P["u"][sl] - foot[0], P["v"][sl] - foot[1])
                    if len(d) and d.min() < 40:
                        j = sl.start + d.argmin()
                        rows.append((P["u"][j], P["v"][j], by1 - by0))
        np.savez(path, r=np.array(rows))
    r = np.load(path)["r"]
    return _lookup(r[:, 0], r[:, 1], None, np.c_[r[:, 2], r[:, 2]], 64.0)


def player_heights(P, f_h):
    """P with bh replaced by the measured height at each player's foot point."""
    return {**P, "bh": f_h(P["u"], P["v"])[..., 0].astype(np.float32)}


# -- ball labels and the net -----------------------------------------------------------------------

CROP = 128                    # training crop side (px)


def frames(path, want, n=3):
    """Yield (1-based frame, (n, H, W, 3) frames f-n//2..f+n//2) for each wanted frame, decoding the video once in order."""
    import cv2
    want = sorted(set(want))
    cap, buf, k, i, h = cv2.VideoCapture(path), [], 0, 0, n // 2
    while i < len(want):
        ok, img = cap.read()
        if not ok:
            break
        k += 1
        buf = (buf + [img])[-n:]
        while i < len(want) and want[i] < k - h:
            i += 1
        if i < len(want) and want[i] == k - h and len(buf) == n:
            yield k - h, np.stack(buf)
            i += 1


YOLO_OFFS = range(-6, 7)      # frames around each event that YOLO labels
N_FRAMES = 3                  # frames kept per training crop
YOLO_BOX = 320                # px cut around the projected ball, upscaled 2x for YOLO
YOLO_CONF = 0.15


def yolo_ball(model, img, uv):
    """(u, v, conf) of the COCO 'sports ball' nearest the projected ball uv in a YOLO_BOX cut around it, or None.
    The event-anchored ball is too rough to snap to (white socks, shirts and lines win half the time); YOLO's ball
    boxes are rarely wrong but miss about half the balls, so they label and the net learns from them."""
    import cv2
    x0 = int(np.clip(uv[0] - YOLO_BOX / 2, 0, W - YOLO_BOX))
    y0 = int(np.clip(uv[1] - YOLO_BOX * 0.6, 0, H - YOLO_BOX))
    c = cv2.resize(img[y0:y0 + YOLO_BOX, x0:x0 + YOLO_BOX], (2 * YOLO_BOX, 2 * YOLO_BOX), interpolation=cv2.INTER_CUBIC)
    b = model.predict(c, classes=[32], conf=YOLO_CONF, verbose=False, device=YOLO_DEV)[0].boxes
    if not len(b):
        return None
    xy = b.xywh[:, :2].cpu().numpy() / 2 + [x0, y0]
    k = np.linalg.norm(xy - uv, axis=1).argmin()
    return np.array([*xy[k], float(b.conf[k])], np.float32)


def crops(mid, h=1, seed=0):
    """Training crops from half h: 3 frames x CROP x CROP around each YOLO-labelled ball (with jitter), plus one crop
    on a random player away from the ball per label (label NaN)."""
    from ultralytics import YOLO
    model = YOLO(os.path.join(CACHE, "yolo", "yolo11x.pt"))
    rng = np.random.default_rng(seed)
    P = players(mid, h)
    to_img, _ = pitch_map(P)
    z = np.load(os.path.join(EXT, "ball", f"{mid}_{half_name(h)}_ball.npz"))
    uvb = to_img(z["x"], z["y"])
    uvb[(z["x"] == 0) & (z["y"] == 0)] = np.nan        # dead-ball placeholder, not a position
    P, s = by_frame(P, len(z["x"]))
    want = sorted({e["frame"] + o for e in events(mid, h) for o in YOLO_OFFS} - {0, 1})
    X, lab, fr, score = [], [], [], []
    for f, trip in frames(video_path(mid, h), want, N_FRAMES):
        if not np.isfinite(uvb[f - 1, 0]):
            continue
        r = yolo_ball(model, trip[N_FRAMES // 2], uvb[f - 1])
        score.append(0 if r is None else r[2])
        if r is None:
            continue
        ball = r[:2]
        pl = np.c_[P["u"][s[f]:s[f + 1]], P["v"][s[f]:s[f + 1]]]
        pl = pl[np.abs(pl - ball).max(1) > CROP]
        cs = [(ball + rng.integers(-48, 49, 2), True)] + ([(pl[rng.integers(len(pl))] - [0, 20], False)] if len(pl) else [])
        for c, is_ball in cs:
            tl = np.clip(np.round(c).astype(int) - CROP // 2, 0, [W - CROP, H - CROP])
            X.append(trip[:, tl[1]:tl[1] + CROP, tl[0]:tl[0] + CROP].copy())
            rel = ball - tl
            lab.append(rel if is_ball and ((rel >= 0) & (rel < CROP)).all() else [np.nan, np.nan])
            fr.append(f)
    print(f"{len(want)} frames wanted, {len(score)} with a ball, {int((np.array(score) > 0).sum())} labelled by YOLO; "
          f"{len(X)} crops", flush=True)
    np.savez(os.path.join(CACHE, f"ball_crops_{mid}_{h}.npz"), X=np.stack(X), lab=np.array(lab, np.float32), frame=np.array(fr))


def lane_b():
    """vision/ball.py (the net, its loss, the linker), imported with this file's device."""
    os.environ.setdefault("MTAH_REPO", os.path.expanduser("~/Github/more-than-a-hack-2027"))
    sys.path[:0] = [os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), d) for d in ("vision", ".")]
    import ball
    return ball


def train(mid, epochs=20, bs=32, seed=0):
    """lane B's net and recipe (focal loss on a Gaussian heatmap, mirror and time-reverse augmentation) on crops."""
    import torch
    b = lane_b()
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    d = np.load(os.path.join(CACHE, f"ball_crops_{mid}_1.npz"))
    X, lab = d["X"], d["lab"]
    blk = (d["frame"] // 1500) % 5 == 2              # 60 s blocks held out for the crop check
    net = b.Net().to(b.DEV)
    opt = torch.optim.AdamW(net.parameters(), 1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, 2e-3, total_steps=epochs * (int((~blk).sum()) // bs))
    for epoch in range(epochs):
        net.train()
        idx = rng.permutation(np.flatnonzero(~blk))
        tot = 0.0
        for k in range(len(idx) // bs):
            i = np.sort(idx[k * bs:(k + 1) * bs])
            x, y = X[i], lab[i].copy()
            if rng.random() < 0.5:
                x, y[:, 0] = x[:, :, :, ::-1], CROP - 1 - y[:, 0]
            if rng.random() < 0.5:
                x = x[:, ::-1]
            xin = b.to_input(np.ascontiguousarray(x)) * torch.empty(len(i), 1, 1, 1, device=b.DEV).uniform_(0.8, 1.2)
            loss = b.focal(net(xin), b.heatmap_target(y, CROP))
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            tot += loss.item()
        print("epoch", epoch, "train", round(tot / (k + 1), 4), "held-out crops", round(b.evaluate_crops(net, X[blk], lab[blk]), 3), flush=True)
    torch.save(net.state_dict(), os.path.join(CACHE, f"ball_net_{mid}.pt"))


K = 10                        # ball candidates kept per frame
NEAR_PX = 3.0                 # search within NEAR_PX player heights of a player's feet
TILE, STRIDE = 256, 224       # the net runs on overlapping tiles that touch a player's search disc
TX = np.r_[np.arange(0, W - TILE, STRIDE), W - TILE]
TY = np.r_[np.arange(0, H - TILE, STRIDE), H - TILE]


def tiles_near(u, v, bh):
    """(row, col) top-left corners of the tiles any player's search disc (radius NEAR_PX heights) reaches."""
    r = NEAR_PX * np.maximum(bh, 30)
    cy = v - bh / 2
    out = set()
    for x0 in TX:
        for y0 in TY:
            dx = np.maximum(np.maximum(x0 - u, u - x0 - TILE), 0)
            dy = np.maximum(np.maximum(y0 - cy, cy - y0 - TILE), 0)
            if (dx * dx + dy * dy < r * r).any():
                out.add((y0, x0))
    return sorted(out)


def detect(mid, h, limit=None):
    """Top-K ball peaks (u, v, p) on every frame of half h: the net in half precision on the tiles near players, the
    tiles' heatmaps max-merged onto the panorama, local maxima 9 px apart."""
    import torch
    import torch.nn.functional as F
    b = lane_b()
    net = b.Net().to(b.DEV)
    net.load_state_dict(torch.load(os.path.join(CACHE, f"ball_net_{mid}.pt"), map_location=b.DEV))
    net.eval().half()
    P = player_heights(players(mid, h), heights(mid))
    n = int(P["frame"].max()) if limit is None else limit
    P, s = by_frame(P, n)
    top = np.zeros((n + 1, K, 3), np.float32)
    canvas = torch.zeros(H, W, device=b.DEV, dtype=torch.float16)
    with torch.no_grad():
        for f, trip in frames(video_path(mid, h), range(2, n)):
            sl = slice(s[f], s[f + 1])
            ts = tiles_near(P["u"][sl], P["v"][sl], P["bh"][sl])
            if not ts:
                continue
            x = np.stack([trip[:, y0:y0 + TILE, x0:x0 + TILE] for y0, x0 in ts])
            hm = net(b.to_input(x).half()).sigmoid()
            canvas.zero_()
            for (y0, x0), t in zip(ts, hm):
                c = canvas[y0:y0 + TILE, x0:x0 + TILE]
                torch.maximum(c, t, out=c)
            p = canvas.float()
            local = p == F.max_pool2d(p[None, None], 9, stride=1, padding=4)[0, 0]
            val, idx = (p * local).flatten().topk(K)
            idx, val = idx.cpu().numpy(), val.cpu().numpy()
            top[f] = np.c_[idx % W, idx // W, val]
            if f % 5000 == 0:                  # checkpoint: a stopped run keeps what it has (frames after f are zero)
                print(f, top[f, 0], flush=True)
                np.savez(os.path.join(CACHE, f"peaks_{mid}_{h}.npz"), peaks=top)
    np.savez(os.path.join(CACHE, f"peaks_{mid}_{h}.npz"), peaks=top)


# -- one ball, touches, scoring ----------------------------------------------------------------------

P_TOP = 0.5                              # the net's peaks top out near 0.5: rescaled to 1 for the linker's costs
STATIC_CELL, STATIC_BLOCK = 6, 125       # px cells; 5 s blocks (60 s and 20 s let half-static things through)
STATIC_IN = 0.6                          # static in a block: in one cell on 60% of its frames


def static(pk):
    """(T, K) mask of peaks on static white things (spare balls and cones on or by the pitch, markers past the far
    touchline), block by block: a cell occupied on STATIC_IN of a block's frames and of a neighbouring block's is
    masked in that block. Things are left out and picked up during the half, so one map for the half misses them; a
    match ball rests in one place for two minutes' worth only at a long stoppage."""
    cell = (pk[..., 0] // STATIC_CELL).astype(np.int64) * 10000 + (pk[..., 1] // STATIC_CELL).astype(np.int64)
    cell[pk[..., 2] <= 0] = -1
    per = []
    for a in range(0, len(pk), STATIC_BLOCK):
        c = cell[a:a + STATIC_BLOCK]
        u, n = np.unique(np.concatenate([np.unique(r) for r in c]), return_counts=True)
        per.append(set(u[(n >= STATIC_IN * len(c)) & (u >= 0)]))
    out = np.zeros(cell.shape, bool)
    ring = (-10001, -10000, -9999, -1, 0, 1, 9999, 10000, 10001)
    for i, cells in enumerate(per):
        nb = (per[i - 1] if i else set()) | (per[i + 1] if i + 1 < len(per) else set())
        bad = [k for k in cells if any(k + d in nb for d in ring)]
        if bad:
            sl = slice(i * STATIC_BLOCK, (i + 1) * STATIC_BLOCK)
            out[sl] = np.isin(cell[sl], [k + d for k in bad for d in ring])
    return out


NEAR_MARGIN, FAR_MARGIN = 40, 120       # px a candidate may sit below the near / above the far touchline (in the air)


def on_pitch(pk, to_img):
    """(T, K) mask of peaks between the touchlines (plus margins): the painted lettering below the near touchline and
    the spare balls past the far one are white blobs too."""
    x = np.linspace(-56, 56, 113)
    near, far = to_img(x, np.full_like(x, 34.0)), to_img(x, np.full_like(x, -34.0))
    nv = np.interp(pk[..., 0], *near[np.argsort(near[:, 0])].T)
    fv = np.interp(pk[..., 0], *far[np.argsort(far[:, 0])].T)
    return (pk[..., 1] <= nv + NEAR_MARGIN) & (pk[..., 1] >= fv - FAR_MARGIN)


def track(mid, h):
    """One ball per frame from the peaks, by lane B's linker in pitch metres: (n + 1, 5) of u, v, p, x, y, NaN
    where unseen. Row f is 1-based frame f."""
    path = os.path.join(CACHE, f"track_{mid}_{h}.npy")
    if os.path.isfile(path):
        return np.load(path)
    b = lane_b()
    pk = np.load(os.path.join(CACHE, f"peaks_{mid}_{h}.npz"))["peaks"]
    to_img, to_pitch = pitch_map(players(mid, h))
    xy = to_pitch(pk[..., 0], pk[..., 1])
    p = np.where((pk[..., 2] > 0.05) & ~static(pk) & on_pitch(pk, to_img), np.minimum(pk[..., 2] / P_TOP, 1), 0)
    xy[p <= 0] = np.nan
    out = np.full((len(pk), 5), np.nan)
    for a in range(0, len(pk), 5000):                 # chunks keep the linker's memory small; seams cost a restart
        sl = slice(a, min(a + 5000, len(pk)))
        ch = b.link(xy[sl], p[sl], fps=FPS)
        seen = ch < K
        i = np.arange(sl.start, sl.stop)[seen]
        out[i, :3] = pk[i, ch[seen]]
        out[i, 3:] = xy[i, ch[seen]]
    np.save(path, out)
    return out


def fill(a, gap):
    """Linear fill of NaN runs up to gap frames long, per column."""
    a = a.copy()
    for c in range(a.shape[1]):
        ok = np.isfinite(a[:, c])
        if ok.sum() < 2:
            continue
        idx = np.flatnonzero(ok)
        prev = np.maximum.accumulate(np.where(ok, np.arange(len(a)), -1))
        nxt = np.minimum.accumulate(np.where(ok, np.arange(len(a)), len(a))[::-1])[::-1]
        short = ~ok & (prev >= 0) & (nxt < len(a)) & (nxt - prev <= gap)
        a[short, c] = np.interp(np.flatnonzero(short), idx, a[idx, c])
    return a


TOUCH = dict(k=3, thr=6.0, reach=0.7, nms=6, gap=4)


def touches(trk, P, s, k=3, thr=6.0, reach=0.7, nms=6, gap=4):
    """Touches from one ball: frames where its velocity changes by more than thr m/s (over k frames either side) while
    it is within reach player heights of a player's body (the segment from feet to head in the image), the strongest
    in each +-nms frames. The toucher is the nearest such player. Returns dicts frame, pid, team, dv, d."""
    t = fill(trk, gap)
    xy, uv = t[:, 3:5], t[:, 0:2]
    vin = (xy - np.roll(xy, k, 0)) * FPS / k
    vout = (np.roll(xy, -k, 0) - xy) * FPS / k
    dv = np.linalg.norm(vout - vin, axis=1)
    dv[:k], dv[-k:] = np.nan, np.nan
    near, who = np.full(len(t), np.inf), np.zeros(len(t), int)
    for f in np.flatnonzero(np.nan_to_num(dv) > thr):
        sl = slice(s[f], s[f + 1])
        if sl.start == sl.stop:
            continue
        top = np.clip(uv[f, 1], P["v"][sl] - P["bh"][sl], P["v"][sl])
        d = np.hypot(uv[f, 0] - P["u"][sl], uv[f, 1] - top) / np.maximum(P["bh"][sl], 20)
        j = d.argmin()
        near[f], who[f] = d[j], sl.start + j
    score = np.where(near < reach, np.nan_to_num(dv), 0)
    out = []
    for f in np.flatnonzero(score > 0):
        lo, hi = max(f - nms, 0), f + nms + 1
        if score[f] == score[lo:hi].max() and f == lo + int(score[lo:hi].argmax()):
            j = who[f]
            out.append(dict(frame=int(f), pid=int(P["pid"][j]), team="right" if P["team"][j] else "left",
                            dv=round(float(dv[f]), 1), d=round(float(near[f]), 2)))
    return out


def match(truth, pred, tol):
    """Greedy one-to-one pairs (i_truth, i_pred) by frame distance, closest first, within tol frames."""
    pairs = sorted((abs(a["frame"] - b["frame"]), i, j) for i, a in enumerate(truth) for j, b in enumerate(pred)
                   if abs(a["frame"] - b["frame"]) <= tol)
    ti, pj, out = set(), set(), []
    for _, i, j in pairs:
        if i not in ti and j not in pj:
            ti.add(i), pj.add(j), out.append((i, j))
    return out


def score(truth, pred, tol=12):
    """Recall, precision, frame errors and who-touched accuracy of pred against truth, matched within tol frames."""
    m = match(truth, pred, tol)
    err = np.array([pred[j]["frame"] - truth[i]["frame"] for i, j in m])
    pid = np.mean([pred[j]["pid"] == truth[i]["pid"] for i, j in m]) if m else np.nan
    team = np.mean([pred[j]["team"] == truth[i]["team"] for i, j in m]) if m else np.nan
    r = dict(n_truth=len(truth), n_pred=len(pred), matched=len(m), recall=len(m) / max(len(truth), 1),
             precision=len(m) / max(len(pred), 1), player=pid, team=team)
    for w in (0, 1, 2, 3, 5):
        r[f"within_{w}"] = float(np.mean(np.abs(err) <= w)) if len(err) else np.nan
    r["median_err"] = float(np.median(err)) if len(err) else np.nan
    r["exact_player_2f"] = sum(abs(pred[j]["frame"] - truth[i]["frame"]) <= 2 and pred[j]["pid"] == truth[i]["pid"]
                               for i, j in m) / max(len(truth), 1)
    return r


def fmt(r):
    return (f"recall {r['recall']:.2f}, precision {r['precision']:.2f} ({r['n_pred']} touches, {r['n_truth']} events); "
            f"of matched: frame error median {r['median_err']:+.0f}, within 1/2/3 frames {r['within_1']:.2f}/"
            f"{r['within_2']:.2f}/{r['within_3']:.2f}, player {r['player']:.2f}, team {r['team']:.2f}; "
            f"right player within 2 frames {r['exact_player_2f']:.2f} of events")


def run(mid, h, **kw):
    P = player_heights(players(mid, h), heights(mid))
    trk = track(mid, h)
    P, s = by_frame(P, len(trk))
    return touches(trk, P, s, **{**TOUCH, **kw})


# -- learned spotter: a per-frame classifier on the ball's path and the players around it --------------

LABEL_W = 1                   # a frame is positive within LABEL_W frames of an event
NMS = 6                       # one touch per +-NMS frames


def nearest(uv, P, s):
    """Per frame: normalised distance of the ball to the nearest and second-nearest player's body (the image segment
    from feet to head, in player heights), their rows in P, and the nearest one's height. inf / -1 where unknown."""
    n = len(uv)
    d1, d2, j1, j2, h1 = np.full(n, np.inf), np.full(n, np.inf), np.full(n, -1), np.full(n, -1), np.full(n, np.nan)
    for f in np.flatnonzero(np.isfinite(uv[:, 0])):
        a, b = s[min(f, len(s) - 2)], s[min(f + 1, len(s) - 1)]
        if b - a < 2:
            continue
        top = np.clip(uv[f, 1], P["v"][a:b] - P["bh"][a:b], P["v"][a:b])
        d = np.hypot(uv[f, 0] - P["u"][a:b], uv[f, 1] - top) / np.maximum(P["bh"][a:b], 20)
        o = np.argsort(d)[:2]
        d1[f], d2[f], j1[f], j2[f], h1[f] = d[o[0]], d[o[1]], a + o[0], a + o[1], P["bh"][a + o[0]]
    return d1, d2, j1, j2, h1


def shift(a, k):
    """a moved k frames later (a[f - k] at f), NaN-padded."""
    out = np.full_like(a, np.nan, dtype=float)
    if k >= 0:
        out[k:] = a[:len(a) - k]
    else:
        out[:k] = a[-k:]
    return out


def features(trk, P, s, gap=4):
    """(n, F) per-frame features and the nearest player's row in P per frame."""
    t = fill(trk, gap)
    uv, xy = t[:, 0:2], t[:, 3:5]
    seen = np.isfinite(trk[:, 0]).astype(float)
    d1, d2, j1, j2, h1 = nearest(uv, P, s)
    h = np.where(np.isfinite(h1), h1, 50.0)[:, None]
    cols = {}
    for k in (1, 2, 4, 8):
        vi = (uv - np.stack([shift(uv[:, 0], k), shift(uv[:, 1], k)], 1)) / k / h * FPS      # heights per second
        vo = (np.stack([shift(uv[:, 0], -k), shift(uv[:, 1], -k)], 1) - uv) / k / h * FPS
        cols[f"img_dv{k}"] = np.linalg.norm(vo - vi, axis=1)
        cols[f"img_in{k}"], cols[f"img_out{k}"] = np.linalg.norm(vi, axis=1), np.linalg.norm(vo, axis=1)
        cols[f"img_cos{k}"] = (vi * vo).sum(1) / (np.linalg.norm(vi, axis=1) * np.linalg.norm(vo, axis=1) + 1e-6)
        if k > 1:
            pi = (xy - np.stack([shift(xy[:, 0], k), shift(xy[:, 1], k)], 1)) / k * FPS              # m/s
            po = (np.stack([shift(xy[:, 0], -k), shift(xy[:, 1], -k)], 1) - xy) / k * FPS
            cols[f"pitch_dv{k}"] = np.linalg.norm(po - pi, axis=1)
            cols[f"pitch_in{k}"], cols[f"pitch_out{k}"] = np.linalg.norm(pi, axis=1), np.linalg.norm(po, axis=1)
    dv2 = np.nan_to_num(cols["img_dv2"])
    peak = np.lib.stride_tricks.sliding_window_view(np.pad(dv2, NMS, mode="edge"), 2 * NMS + 1).max(1)
    cols["img_dv2_rel"] = dv2 / (peak + 1e-6)
    d1c = np.minimum(d1, 5.0)
    cols["d1"], cols["d2"] = d1c, np.minimum(d2, 5.0)
    for k in (-6, -3, 3, 6):
        cols[f"d1_{k}"] = shift(d1c, k)
    cols["d1_min3"] = np.lib.stride_tricks.sliding_window_view(np.pad(d1c, 3, mode="edge"), 7).min(1)
    pid1 = np.where(j1 >= 0, P["pid"][np.maximum(j1, 0)], -1).astype(float)
    team1 = np.where(j1 >= 0, P["team"][np.maximum(j1, 0)], -1).astype(float)
    team2 = np.where(j2 >= 0, P["team"][np.maximum(j2, 0)], -1).astype(float)
    cols["pair_same_team"] = (team1 == team2).astype(float)
    cols["owner_change6"] = (shift(pid1, 6) != shift(pid1, -6)).astype(float)
    cols["team_change6"] = (shift(team1, 6) != shift(team1, -6)).astype(float)
    foot = np.where(j1 >= 0, P["v"][np.maximum(j1, 0)], np.nan)
    cols["above_feet"] = (foot - uv[:, 1]) / h[:, 0]
    cols["seen"] = seen
    cols["seen6"] = np.convolve(seen, np.ones(13) / 13, "same")
    cols["filled"] = (np.isfinite(t[:, 0]) & (seen == 0)).astype(float)
    return np.stack(list(cols.values()), 1).astype(np.float32), j1, list(cols)


def labels(n, ev, w=LABEL_W):
    y = np.zeros(n, np.int8)
    for e in ev:
        y[max(e["frame"] - w, 0):e["frame"] + w + 1] = 1
    return y


def peaks_of(prob, P, j1, thr, nms=NMS):
    """Touches at the local maxima of prob above thr, one per +-nms frames; the toucher is the nearest player."""
    out = []
    for f in np.flatnonzero(prob > thr):
        lo, hi = max(f - nms, 0), f + nms + 1
        if f == lo + int(prob[lo:hi].argmax()) and j1[f] >= 0:
            j = j1[f]
            out.append(dict(frame=int(f), pid=int(P["pid"][j]), team="right" if P["team"][j] else "left",
                            p=round(float(prob[f]), 3)))
    return out


def model():
    from sklearn.ensemble import HistGradientBoostingClassifier
    return HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_leaf_nodes=31, min_samples_leaf=40,
                                          l2_regularization=1.0, random_state=0)


def spot(mid, folds=5, tol=3):
    """Train on half 1 (out-of-fold over contiguous blocks there, to pick the threshold), test on half 2. Saves both
    halves' touches (half 1 out of fold) and prints the scores."""
    data = {}
    for h in (1, 2):
        P = player_heights(players(mid, h), heights(mid))
        trk = track(mid, h)
        P, s = by_frame(P, len(trk))
        X, j1, names = features(trk, P, s)
        data[h] = (X, j1, P, events(mid, h))
    X, j1, P, ev = data[1]
    y = labels(len(X), ev)
    oof = np.zeros(len(X))
    blk = np.arange(len(X)) * folds // len(X)
    for k in range(folds):
        oof[blk == k] = model().fit(X[blk != k], y[blk != k]).predict_proba(X[blk == k])[:, 1]
    f1 = {}
    for th in np.arange(0.05, 0.9, 0.025):
        r = score(ev, peaks_of(oof, P, j1, th), tol)
        f1[th] = 2 * r["recall"] * r["precision"] / max(r["recall"] + r["precision"], 1e-9)
    th = max(f1, key=f1.get)
    pred1 = peaks_of(oof, P, j1, th)
    print(f"half 1 (out of fold), threshold {th:.3f}: " + fmt(score(ev, pred1)))
    m = model().fit(X, y)
    X2, j2, P2, ev2 = data[2]
    prob2 = m.predict_proba(X2)[:, 1]
    pred2 = peaks_of(prob2, P2, j2, th)
    print("half 2 (test): " + fmt(score(ev2, pred2)))
    for h, pred, prob in ((1, pred1, oof), (2, pred2, prob2)):
        json.dump(pred, open(os.path.join(CACHE, f"touches_{mid}_{h}.json"), "w"))
        np.save(os.path.join(CACHE, f"prob_{mid}_{h}.npy"), prob)
    return th


# -- review video ----------------------------------------------------------------------------------

VW, VH = 1280, 720            # review window cut from the panorama, native pixels (no scaling)
PRE, POST = 50, 38            # clip frames before / after the event (2 s, 1.5 s)
GREEN, RED, ORANGE, YELLOW, MAGENTA = (80, 220, 80), (60, 60, 255), (0, 160, 255), (0, 255, 255), (255, 80, 255)


def cases(mid, h, tol=12):
    """Every event and predicted touch of half h, as review cases: kind (hit, wrong player, miss, false), frame, and
    the event and touch involved."""
    truth, pred = events(mid, h), json.load(open(os.path.join(CACHE, f"touches_{mid}_{h}.json")))
    m = match(truth, pred, tol)
    out = [dict(kind="hit" if pred[j]["pid"] == truth[i]["pid"] else "wrong player", frame=truth[i]["frame"],
                truth=truth[i], pred=pred[j]) for i, j in m]
    ti, pj = {i for i, _ in m}, {j for _, j in m}
    out += [dict(kind="miss", frame=e["frame"], truth=e, pred=None) for i, e in enumerate(truth) if i not in ti]
    out += [dict(kind="false", frame=p["frame"], truth=None, pred=p) for j, p in enumerate(pred) if j not in pj]
    return sorted(out, key=lambda c: c["frame"]), truth, pred


def video(mid, h, kinds=("miss", "false", "wrong player"), slow=2, limit=None):
    """A reel of review clips for half h (default: every error), each PRE..POST frames around the event, played slow x
    slower, with all players (team colour, id), the tracked ball, the events (green) and the predicted touches (red),
    and a header saying what the clip is. Writes out/touches_<mid>_<h>_<kinds>.mp4 and a CSV index of the clips."""
    import cv2, subprocess, csv
    cs, truth, pred = cases(mid, h)
    cs = [c for c in cs if c["kind"] in kinds][:limit]
    P = player_heights(players(mid, h), heights(mid))
    trk = track(mid, h)
    P, s = by_frame(P, len(trk))
    ev_at = {}
    for e in truth:
        for f in range(e["frame"] - 2, e["frame"] + 13):
            ev_at.setdefault(f, []).append(e)
    pr_at = {}
    for p in pred:
        for f in range(p["frame"] - 2, p["frame"] + 13):
            pr_at.setdefault(f, []).append(p)
    os.makedirs(OUT, exist_ok=True)
    tag = "errors" if set(kinds) == {"miss", "false", "wrong player"} else "_".join(k.replace(" ", "-") for k in kinds)
    path = os.path.join(OUT, f"touches_{mid}_{h}_{tag}.mp4")
    pipe = subprocess.Popen(["ffmpeg", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{VW}x{VH}",
                             "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-crf", "20", "-preset", "veryfast",
                             "-pix_fmt", "yuv420p", path], stdin=subprocess.PIPE)
    cap = cv2.VideoCapture(video_path(mid, h))
    index, t_out = [], 0.0
    for n, c in enumerate(cs):
        e, p = c["truth"], c["pred"]
        who = e or p
        f0 = c["frame"]
        sl = slice(s[f0], s[f0 + 1])
        j = np.flatnonzero(P["pid"][sl] == who["pid"])
        cu, cv_ = (P["u"][sl][j[0]], P["v"][sl][j[0]]) if len(j) else (np.nan, np.nan)
        if not np.isfinite(cu):
            cu, cv_ = np.nanmean(trk[max(f0 - 5, 0):f0 + 5, :2], 0)
        x0 = int(np.clip(cu - VW / 2, 0, W - VW))
        y0 = int(np.clip(cv_ - VH * 0.6, 0, H - VH))
        mm, ss = divmod((f0 - 1) / FPS, 60)
        head = {"miss": "MISSED: no touch found", "false": "FALSE TOUCH: no event here",
                "wrong player": "WRONG PLAYER", "hit": "HIT"}[c["kind"]]
        what = (f"event {e['label']} by #{e['pid']} ({e['team']}) at frame {e['frame']}" if e else "")
        conf = f"confidence {p['p']:.2f}" if p and "p" in p else f"dv {p['dv']} m/s" if p else ""
        got = f"found touch by #{p['pid']} ({p['team']}) at frame {p['frame']}, {conf}" if p else ""
        index.append(dict(clip=n + 1, reel_time=f"{int(t_out // 60)}:{t_out % 60:04.1f}", kind=c["kind"],
                          half=h, match_time=f"{int(mm)}:{ss:04.1f}", frame=f0, event=what, touch=got))
        first = max(f0 - PRE, 1)
        cap.set(cv2.CAP_PROP_POS_FRAMES, first - 1)
        for f in range(first, f0 + POST):
            ok, img = cap.read()
            if not ok:
                break
            fs = slice(s[f], s[f + 1]) if 0 <= f < len(s) - 1 else slice(0, 0)
            for u, v, bh, pid, tm in zip(P["u"][fs], P["v"][fs], P["bh"][fs], P["pid"][fs], P["team"][fs]):
                col = MAGENTA if tm else ORANGE
                cv2.circle(img, (int(u), int(v)), 3, col, -1)
                cv2.putText(img, str(pid)[-3:], (int(u) + 5, int(v) + 12), 0, 0.35, col, 1)
            if 0 <= f < len(trk) and np.isfinite(trk[f, 0]):
                cv2.circle(img, (int(trk[f, 0]), int(trk[f, 1])), 7, YELLOW, 1)
            for lst, col, dy, txt in ((ev_at.get(f, []), GREEN, 0, "EVENT {label} #{pid}"),
                                      (pr_at.get(f, []), RED, 1, "TOUCH #{pid}")):
                for q in lst:
                    k = np.flatnonzero(P["pid"][fs] == q["pid"])
                    if len(k):
                        u, v, bh = P["u"][fs][k[0]], P["v"][fs][k[0]], P["bh"][fs][k[0]]
                        th = 3 if f == q["frame"] else 1
                        cv2.rectangle(img, (int(u - bh / 2 - 4 * dy), int(v - bh - 4 * dy)),
                                      (int(u + bh / 2 + 4 * dy), int(v + 4 + 4 * dy)), col, th)
                        cv2.putText(img, txt.format(**{"label": "", **q}), (int(u - bh / 2), int(v - bh - 8 - 14 * dy)),
                                    0, 0.45, col, 1 + (f == q["frame"]))
            out = img[y0:y0 + VH, x0:x0 + VW].copy()
            cv2.rectangle(out, (0, 0), (VW, 74), (0, 0, 0), -1)
            col = {"miss": RED, "false": ORANGE, "wrong player": MAGENTA, "hit": GREEN}[c["kind"]]
            cv2.putText(out, f"#{n + 1}  {head}   half {h} {int(mm)}:{ss:04.1f}", (10, 22), 0, 0.65, col, 2)
            cv2.putText(out, what, (10, 44), 0, 0.5, GREEN, 1)
            cv2.putText(out, got, (10, 64), 0, 0.5, RED, 1)
            d = f - f0
            cv2.putText(out, f"frame {f} ({d:+d})", (VW - 190, 22), 0, 0.55, YELLOW if d == 0 else (200, 200, 200), 1 + (d == 0))
            cv2.putText(out, "dots: left team", (VW - 300, 44), 0, 0.45, ORANGE, 1)
            cv2.putText(out, "right team", (VW - 165, 44), 0, 0.45, MAGENTA, 1)
            cv2.putText(out, "o ball  green = event  red = found", (VW - 300, 64), 0, 0.45, YELLOW, 1)
            for _ in range(slow):
                pipe.stdin.write(out.tobytes())
            t_out += slow / FPS
    pipe.stdin.close()
    pipe.wait()
    with open(path[:-4] + ".csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(index[0]))
        w.writeheader()
        w.writerows(index)
    print(f"{len(cs)} clips -> {path}")


def test():
    """A ball rolls at 10 m/s along x, bounces back off player 7 (left) at frame 50, then rolls past player 9 without a
    change: one touch, at 50, by 7. Image = pitch x 20 px; players 40 px tall."""
    n = 100
    x = np.where(np.arange(n + 1) <= 50, np.arange(n + 1) * 0.4, 20 - (np.arange(n + 1) - 50) * 0.4)
    trk = np.c_[x * 20, np.full(n + 1, 500.0), np.ones(n + 1), x, np.zeros(n + 1)]
    trk[20:23] = np.nan                                      # a short gap, filled
    fr = np.repeat(np.arange(n + 1), 2)
    P = dict(frame=fr, pid=np.tile([7, 9], n + 1), team=np.tile([0, 1], n + 1).astype(np.int8),
             u=np.tile([20 * 20.5, 20 * 12.0], n + 1), v=np.full(2 * (n + 1), 505.0), bh=np.full(2 * (n + 1), 40.0),
             x=np.tile([20.5, 12.0], n + 1), y=np.zeros(2 * (n + 1)))
    P, s = by_frame(P, n)
    t = touches(trk, P, s)
    assert [(e["frame"], e["pid"], e["team"]) for e in t] == [(50, 7, "left")], t
    truth = [dict(frame=50, pid=7, team="left"), dict(frame=80, pid=9, team="right")]
    r = score(truth, t + [dict(frame=10, pid=9, team="right")])
    assert (r["matched"], r["recall"], r["player"], r["within_0"]) == (1, 0.5, 1.0, 1.0), r
    assert match([dict(frame=10), dict(frame=14)], [dict(frame=13)], 5) == [(1, 0)]
    print("ok")


if __name__ == "__main__":
    cmd, args = sys.argv[1], sys.argv[2:]
    if cmd == "test":
        test()
    elif cmd == "players":
        for h in (1, 2):
            P = players(args[0], h)
            to_img, to_pitch = pitch_map(P)
            err = np.abs(to_img(P["x"], P["y"]) - np.c_[P["u"], P["v"]])
            back = np.hypot(*(to_pitch(P["u"], P["v"]) - np.c_[P["x"], P["y"]]).T)
            print(f"half {h}: {len(P['frame'])} rows, {len(np.unique(P['frame']))} frames, box h median {np.median(P['bh']):.0f} px; "
                  f"pitch->image error median {np.median(err, 0).round(1)} px, 95% {np.percentile(err, 95, 0).round(1)}; "
                  f"image->pitch median {np.median(back):.2f} m, 95% {np.percentile(back, 95):.2f}")
    elif cmd == "crops":
        crops(args[0])
    elif cmd == "train":
        train(args[0])
    elif cmd == "detect":
        detect(args[0], int(args[1]), int(args[2]) if len(args) > 2 else None)
    elif cmd == "spot":
        spot(args[0])
    elif cmd == "touches":
        mid = args[0]
        for h in (1, 2):
            if os.path.isfile(os.path.join(CACHE, f"peaks_{mid}_{h}.npz")):
                pred = run(mid, h)
                print(f"half {h}: " + fmt(score(events(mid, h), pred)))
                json.dump(pred, open(os.path.join(CACHE, f"touches_{mid}_{h}.json"), "w"))
    elif cmd == "video":                 # video MID H [kind ...] [LIMIT]   kinds: miss, false, wrong-player, hit
        limit = int(args.pop()) if args[-1].isdigit() and len(args) > 2 else None
        kinds = tuple(k.replace("-", " ") for k in args[2:]) or ("miss", "false", "wrong player")
        video(args[0], int(args[1]), kinds, limit=limit)
    elif cmd == "sheet":              # eye check: 48 random labelled crops, label marked
        import cv2
        d = np.load(os.path.join(CACHE, f"ball_crops_{args[0]}_1.npz"))
        i = np.random.default_rng(1).choice(np.flatnonzero(np.isfinite(d["lab"][:, 0])), 48, replace=False)
        t = []
        for k in i:
            c = cv2.resize(d["X"][k, 1], (256, 256), interpolation=cv2.INTER_NEAREST)
            u, v = (d["lab"][k] * 2 + 1).astype(int)
            cv2.circle(c, (u, v), 14, (0, 0, 255), 1)
            t.append(c)
        cv2.imwrite(args[1], np.vstack([np.hstack(t[r:r + 8]) for r in range(0, 48, 8)]))
