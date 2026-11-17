"""Ball recovery from the tactical video (lane B).

    python vision/ball.py data      # snapped labels -> training crops        out/ball_crops.npz
    python vision/ball.py mine      # its false peaks on training frames       out/ball_hard.npz
    python vision/ball.py train     # 3-frame heatmap net on MPS (+ the hard)  out/ball_net.pt
    python vision/ball.py detect    # held-out + missing graded frames         out/ball_detections.npz
    python vision/ball.py redo      # rerun only detections the mask now rules out
    python vision/ball.py write     # airborne flag, merge with tracking and   out/ours_ball_recovered.npz
                                    # the players fallback (vision/radar.py)

Labels are free: the tracked ball projected through the homography. It is only roughly on the
ball in the image (a few px off, and well below it while the ball is in the air, because the tracking
reports the ground point under the ball), so each label is snapped to the strongest white
blob in a window around it, and kept only where the snapped track is smooth over three frames.

Held out for evaluation: every fifth 60 s block (BLOCK, HELD_OUT), never used for training.
"""
import os
import sys
import warnings
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import ndimage as ndi

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from frames import FPS, H_PX, W, camera, fit_track, load_homography, stream, to_pitch, to_pixels  # noqa: E402
from preprocess import GAME, PITCH, graded_mask, half_video_blocks, load_tracking  # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "out")
BLOCK, HELD_OUT = 1500, 2           # 60 s blocks; block % 5 == HELD_OUT is evaluation only
CROP, MIN_BLOB = 128, 50.0          # training crop side (px); DoG score to trust a snapped label
RP, R0, RV = 250, 60, 20            # search radius (px) around players / a known ball, + px per frame
MARGIN = 2.0                        # m beyond the lines still searched (balls just out, or in the air)
OVERLAYS = [(0, 0, 480, 90), (1620, 0, 1919, 150)]   # score bug and logos, fixed in the image (px)
POSTS = [(x, PITCH[1] / 2 + s * 3.66) for x in (0, PITCH[0]) for s in (-1, 1)]
POST_R = 1.0                        # m: a post's white base looks like a ball. A geometric rule
                                    # added after lane B's first run on our match, radius not fitted: docs/2026-11-10-compliance.md
WARM_UP = (110808, 113742)          # graded but before the 2nd-half kick-off (FAQ.md): warm-up balls
K_PEAKS = 5                         # candidates kept per frame for linking over time
# MPS by default; MTAH_BALL_DEVICE=cpu runs it on the CPU: about 25x slower, the same peak positions, heights
# within 1e-6. Each device repeats its own peaks byte for byte (checked on an M4 Pro).
DEV = os.environ.get("MTAH_BALL_DEVICE") or ("mps" if torch.backends.mps.is_available() else "cpu")


def held_out(frames):
    return (np.asarray(frames) // BLOCK) % 5 == HELD_OUT


def in_view(uv, margin=8):
    return (np.isfinite(uv[..., 0]) & (uv[..., 0] >= margin) & (uv[..., 0] < W - margin)
            & (uv[..., 1] >= margin) & (uv[..., 1] < H_PX - margin))


# -- labels ------------------------------------------------------------------------------------

def blob(img):
    """Response to a bright 5-9 px disc: difference of Gaussians on min(R, G, B), i.e. whiteness."""
    w = img.min(-1).astype(np.float32)
    return ndi.gaussian_filter(w, 1.5) - ndi.gaussian_filter(w, 4.5)


def snap(img, uv, left=50, up=90, down=30):
    """(u, v, score): the strongest white blob near a projected tracked ball, mostly above it."""
    u, v = int(round(uv[0])), int(round(uv[1]))
    u0, v0 = max(u - left, 0), max(v - up, 0)
    d = blob(img[v0:v + down, u0:u + left])
    j, i = np.unravel_index(d.argmax(), d.shape)
    return np.array([u0 + i, v0 + j, d[j, i]], np.float32)


def smooth(r0, r1, r2):
    """Three consecutive snapped labels look like one ball: strong, close, near-constant velocity."""
    return (min(r0[2], r1[2], r2[2]) >= MIN_BLOB and np.abs(r1[:2] - (r0[:2] + r2[:2]) / 2).max() <= 4
            and np.abs(r1[:2] - r0[:2]).max() <= 40 and np.abs(r2[:2] - r1[:2]).max() <= 40)


def triples(first, last, n=3):
    """Yield (f, (n, 1080, 1920, 3) frames f-n//2..f+n//2) for f in first..last, decoding once."""
    buf, h = deque(maxlen=n), n // 2
    for k, img in enumerate(stream(first - h, last - first + n), first - h):
        buf.append(img)
        if len(buf) == n:
            yield k - h, np.stack(buf)


def build(step=4, seed=0):
    """Training crops from the non-held-out blocks: 3 frames x CROP x CROP, label in crop pixels."""
    trk, Hm = load_tracking(), load_homography()
    n = len(trk["ball"])
    uv = to_pixels(Hm, np.arange(n), trk["ball"])
    ok = in_view(uv, 60) & graded_mask(n) & ~held_out(np.arange(n))
    rng = np.random.default_rng(seed)
    X, lab, fr = [], [], []
    for _, first, last in half_video_blocks(GAME):
        imgs, refs = deque(maxlen=3), deque(maxlen=3)
        for k, img in enumerate(stream(first - 1, last - first + 3), first - 1):
            imgs.append(img)
            refs.append(snap(img, uv[k]) if ok[k] else np.zeros(3, np.float32))
            f = k - 1
            if len(refs) < 3 or f % step or not smooth(*refs):
                continue
            ball = refs[1][:2]
            centres = [ball + rng.integers(-48, 49, 2)]
            if len(fr) % 3 == 0:                         # one hard negative per three: a player
                p = to_pixels(Hm, f, trk["xy"][f])
                p = p[in_view(p, 64) & (np.abs(p - ball).max(1) > CROP)]
                centres.append(p[rng.integers(len(p))] if len(p) else rng.uniform([64, 300], [W - 64, H_PX - 64]))
            for c in centres:
                tl = np.clip(np.round(c).astype(int) - CROP // 2, 0, [W - CROP, H_PX - CROP])
                X.append(np.stack(imgs)[:, tl[1]:tl[1] + CROP, tl[0]:tl[0] + CROP].copy())
                rel = ball - tl
                lab.append(rel if ((rel >= 0) & (rel < CROP)).all() else [np.nan, np.nan])
                fr.append(f)
        print("half done", first, len(fr), flush=True)
    np.savez(os.path.join(OUT, "ball_crops.npz"), X=np.stack(X), lab=np.array(lab, np.float32), frame=np.array(fr))


def mine(step=10, min_p=0.3, seed=0):
    """Hard negatives: on every step-th tracked frame of the training blocks, searched near the
    players only, the detector's peaks of min_p or more whose crop holds no ball (the tracked point,
    or up to 90 px above it where a ball in the air shows). Socks, boots, line marks. Crops as in
    build(), label NaN -> out/ball_hard.npz, which train() adds."""
    trk, Hm, net = load_tracking(), load_homography(), load_net()
    n = len(trk["ball"])
    f_all = np.arange(n)
    uv = to_pixels(Hm, f_all, trk["ball"])
    ok = in_view(uv) & graded_mask(n) & ~held_out(f_all) & (f_all % step == 0)
    ok[WARM_UP[0]:WARM_UP[1]] = False                  # several balls on the pitch before the 2nd-half kick-off
    rng = np.random.default_rng(seed)
    up = np.arange(0, 91, 10)
    X, fr = [], []
    for _, first, last in half_video_blocks(GAME):
        for f, trip in triples(first, last):
            if not ok[f]:
                continue
            players = to_pixels(Hm, f, trk["xy"][f])
            for u, v, p in peaks(net, trip, search_mask(players[in_view(players)], [], Hm[f])):
                tl = np.clip(np.array([u, v], int) + rng.integers(-20, 21, 2) - CROP // 2, 0, [W - CROP, H_PX - CROP])
                ball_in = ((uv[f, 0] >= tl[0] - 30) & (uv[f, 0] < tl[0] + CROP + 30) &
                           (uv[f, 1] - up >= tl[1] - 30) & (uv[f, 1] - up < tl[1] + CROP + 30)).any()
                if p >= min_p and not ball_in:
                    X.append(trip[:, tl[1]:tl[1] + CROP, tl[0]:tl[0] + CROP].copy())
                    fr.append(f)
        print("half done", first, len(fr), flush=True)
    np.savez(os.path.join(OUT, "ball_hard.npz"), X=np.stack(X), lab=np.full((len(X), 2), np.nan, np.float32), frame=np.array(fr))


# -- model ---------------------------------------------------------------------------------------

def _block(i, o):
    return nn.Sequential(nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(inplace=True),
                         nn.Conv2d(o, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(inplace=True))


class Net(nn.Module):
    """TrackNet-style U-Net: t frames around the middle one (3t channels) -> ball heatmap logits, same size."""

    def __init__(self, c=16, t=3):
        super().__init__()
        self.e1 = nn.Sequential(nn.Conv2d(3 * t, c, 3, padding=1), nn.BatchNorm2d(c), nn.ReLU(inplace=True))
        self.e2, self.e3, self.e4 = _block(c, 2 * c), _block(2 * c, 4 * c), _block(4 * c, 4 * c)
        self.d3, self.d2 = _block(8 * c, 2 * c), _block(4 * c, c)
        self.d1 = nn.Sequential(nn.Conv2d(2 * c, c, 3, padding=1), nn.ReLU(inplace=True), nn.Conv2d(c, 1, 1))

    def forward(self, x):
        e1 = self.e1(x)
        e2 = self.e2(F.max_pool2d(e1, 2))
        e3 = self.e3(F.max_pool2d(e2, 2))
        e4 = self.e4(F.max_pool2d(e3, 2))
        d3 = self.d3(torch.cat([F.interpolate(e4, scale_factor=2), e3], 1))
        d2 = self.d2(torch.cat([F.interpolate(d3, scale_factor=2), e2], 1))
        return self.d1(torch.cat([F.interpolate(d2, scale_factor=2), e1], 1))[:, 0]


def to_input(x):
    """uint8 (B, t, h, w, 3) on any device -> float (B, 3t, h, w) on DEV."""
    x = torch.as_tensor(x).to(DEV)
    return x.permute(0, 1, 4, 2, 3).reshape(x.shape[0], -1, x.shape[2], x.shape[3]).float() / 255 - 0.5


def heatmap_target(lab, size, sigma=2.0):
    """Gaussian peaks of height 1 at integer label pixels; all zero where the label is NaN."""
    yy, xx = np.mgrid[:size, :size]
    t = np.zeros((len(lab), size, size), np.float32)
    for i, (u, v) in enumerate(lab):
        if np.isfinite(u):
            t[i] = np.exp(-((xx - round(u)) ** 2 + (yy - round(v)) ** 2) / (2 * sigma ** 2))
    return torch.from_numpy(t).to(DEV)


def focal(logit, target):
    """CenterNet focal loss on a Gaussian target heatmap, per positive."""
    p = logit.sigmoid().clamp(1e-4, 1 - 1e-4)
    pos = target > 0.999
    loss = torch.where(pos, (1 - p) ** 2 * torch.log(p), (1 - target) ** 4 * p ** 2 * torch.log(1 - p))
    return -loss.sum() / pos.sum().clamp(min=1)


def train(epochs=15, bs=32, seed=0):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    d = np.load(os.path.join(OUT, "ball_crops.npz"))
    X, lab = d["X"], d["lab"]
    hard = os.path.join(OUT, "ball_hard.npz")
    if os.path.isfile(hard):                           # mined false positives (mine())
        h = np.load(hard)
        X, lab = np.concatenate([X, h["X"]]), np.concatenate([lab, h["lab"]])
    val = rng.random(len(X)) < 0.05
    net = Net().to(DEV)
    opt = torch.optim.AdamW(net.parameters(), 1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, 2e-3, total_steps=epochs * (int((~val).sum()) // bs))
    for epoch in range(epochs):
        net.train()
        idx = rng.permutation(np.flatnonzero(~val))
        tot = 0.0
        for b in range(len(idx) // bs):
            i = np.sort(idx[b * bs:(b + 1) * bs])
            x, y = X[i], lab[i].copy()
            if rng.random() < 0.5:                       # mirror left-right
                x, y[:, 0] = x[:, :, :, ::-1], CROP - 1 - y[:, 0]
            if rng.random() < 0.5:                       # play backwards
                x = x[:, ::-1]
            xin = to_input(np.ascontiguousarray(x)) * torch.empty(len(i), 1, 1, 1, device=DEV).uniform_(0.8, 1.2)
            loss = focal(net(xin), heatmap_target(y, CROP))
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            tot += loss.item()
        print("epoch", epoch, "train", round(tot / (b + 1), 4), "val", round(evaluate_crops(net, X[val], lab[val]), 3), flush=True)
    torch.save(net.state_dict(), os.path.join(OUT, "ball_net.pt"))


@torch.no_grad()
def evaluate_crops(net, X, lab, bs=64):
    """Share of crops whose peak lands within 3 px of the label (positives) or stays < 0.5 (negatives)."""
    net.eval()
    hits = []
    for b in range(0, len(X), bs):
        p = net(to_input(X[b:b + bs])).sigmoid().flatten(1)
        conf, k = p.max(1)
        uv = torch.stack([k % CROP, k // CROP], 1).cpu().numpy()
        y = lab[b:b + bs]
        pos = np.isfinite(y[:, 0])
        hits += list(np.where(pos, np.abs(uv - np.nan_to_num(y)).max(1) <= 3, conf.cpu().numpy() < 0.5))
    return float(np.mean(hits))


def load_net():
    net = Net().to(DEV)
    net.load_state_dict(torch.load(os.path.join(OUT, "ball_net.pt"), map_location=DEV))
    return net.eval()


# -- detection ------------------------------------------------------------------------------------

def search_mask(players, anchors, M, step=8):
    """Where the ball may be: within RP of a player or within r of an anchor (u, v, r), on the pitch
    or within MARGIN m of it, not at a goal-post base, and off the broadcast overlays. M is the
    frame's homography."""
    yy, xx = np.mgrid[step // 2:H_PX:step, step // 2:W:step]
    m = np.zeros(xx.shape, bool)
    for u, v, r in [(u, v, RP) for u, v in players] + list(anchors):
        m |= (xx - u) ** 2 + (yy - v) ** 2 < r * r
    g = to_pitch(M[None], 0, np.stack([xx, yy], -1).astype(float))
    m &= ((g >= -MARGIN) & (g <= np.array(PITCH) + MARGIN)).all(-1)
    m &= np.min([np.linalg.norm(g - q, axis=-1) for q in POSTS], 0) >= POST_R
    for u0, v0, u1, v1 in OVERLAYS:
        m[v0 // step:v1 // step + 1, u0 // step:u1 // step + 1] = False
    return np.kron(m, np.ones((step, step), bool))


@torch.no_grad()
def peaks(net, trip, mask, k=K_PEAKS, nms=15):
    """(k, 3) array of (u, v, p): the k most ball-like local maxima inside the mask, at least nms px
    apart, strongest first; p = 0 where there are fewer. Greedy over the 50 strongest local maxima,
    so a flat top (ties) still yields one peak per blob."""
    p = net(to_input(trip[None]))[0].sigmoid() * torch.from_numpy(mask).to(DEV)
    local = p == F.max_pool2d(p[None, None], nms, stride=1, padding=nms // 2)[0, 0]
    val, idx = (p * local).flatten().topk(50)
    idx, val = idx.cpu().numpy(), val.cpu().numpy()
    uv = np.stack([idx % W, idx // W], 1)
    out = np.zeros((k, 3), np.float32)
    j = 0
    for i in range(len(idx)):
        if j == k or val[i] <= 0:
            break
        if j == 0 or np.abs(out[:j, :2] - uv[i]).max(1).min() >= nms:
            out[j] = uv[i, 0], uv[i, 1], val[i]
            j += 1
    return out


def anchors_for(f, ball, known, Hm):
    """The nearest tracked ball before and after f, projected into frame f, with a radius that
    grows with the time since (or until) it was seen."""
    i = np.searchsorted(known, f)
    return [(*to_pixels(Hm, f, ball[g]), R0 + RV * abs(f - g)) for g in known[max(i - 1, 0):i + 1]
            if abs(f - g) < 250]


def detect(redo=False):
    """Run the net on every graded frame with no tracked ball, and, for evaluation, on held-out
    frames that have one: 2 s on, 4 s off, so flight fits get runs, searched near the players only
    as if the ball were lost. redo: rerun only the frames whose saved detection fails the current
    search mask (after a mask change), and update the file."""
    trk, Hm, net = load_tracking(), load_homography(), load_net()
    n = len(trk["ball"])
    f_all = np.arange(n)
    uv = to_pixels(Hm, f_all, trk["ball"])
    has = np.isfinite(trk["ball"][:, 0])
    graded = graded_mask(n)
    need = graded & (~has | (held_out(f_all) & in_view(uv) & (f_all // 50 % 3 == 0)))
    known = np.flatnonzero(has)
    if DEV != "cpu":                                   # the GPU runs the net: leave the other cores to ffmpeg
        torch.set_num_threads(3)
    res = np.full((n, 3), np.nan, np.float32)
    top = np.zeros((n, K_PEAKS, 3), np.float32)
    path = os.path.join(OUT, "ball_detections.npz")
    if redo:
        d = np.load(path)
        res[d["frame"]] = np.stack([d["u"], d["v"], d["p"]], 1)
        g = to_pitch(Hm, d["frame"], np.stack([d["u"], d["v"]], 1).astype(float))
        again = np.zeros(n, bool)
        again[d["frame"]] = np.min([np.linalg.norm(g - q, axis=-1) for q in POSTS], 0) < POST_R
        print("redo", again.sum(), flush=True)
    for _, first, last in half_video_blocks(GAME):
        for f, trip in triples(first, last):
            if not (again[f] if redo else need[f]):
                continue
            players = to_pixels(Hm, f, trk["xy"][f])
            players = players[in_view(players)]
            anchors = [] if has[f] else anchors_for(f, trk["ball"], known, Hm)   # eval: players only
            top[f] = peaks(net, trip, search_mask(players, anchors, Hm[f]))
            res[f] = top[f, 0]
            if f % 5000 == 0:
                print(f, res[f], flush=True)
    np.savez(path, frame=np.flatnonzero(need), u=res[need, 0], v=res[need, 1], p=res[need, 2])
    if not redo:
        np.savez(os.path.join(OUT, "ball_peaks.npz"), frame=np.flatnonzero(need), peaks=top[need])


# -- airborne flag and output ---------------------------------------------------------------------

def flight(fr, uv, Hm, xy, half=12):
    """Airborne flag per detection, and the ground point under the ball where airborne: a ballistic
    fit over the 25 frames around it beats a ground roll by 2x in pixel error and puts the ball
    0.5 m or more up. fr must be sorted; windows need 25 consecutive frames. Looser rules or 15
    frames made held-out errors worse: one camera barely separates height from depth."""
    air, xy = np.zeros(len(fr), bool), xy.copy()
    Ps = np.array([camera(Hm[f]) for f in fr])
    for i in np.flatnonzero(fr[2 * half:] - fr[:-2 * half] == 2 * half) + half:
        w = slice(i - half, i + half + 1)
        t = (fr[w] - fr[i]) / FPS
        X, ef = fit_track(Ps[w], uv[w], t, True)
        _, eg = fit_track(Ps[w], uv[w], t, False)
        if X[half, 2] >= 0.5 and 2 * ef < eg:
            air[i], xy[i] = True, X[half, :2]
    return air, xy


def link(xy, p, start=None, end=None, fps=FPS, sig=1.0, p0=0.3, c_break=8.0, max_skip=25, vmax=40.0, slack_m=2.0, extra=None):
    """One ball through T frames of K candidates each (pitch metres xy (T, K, 2), peak p (T, K), p = 0
    for none). Shortest path over chosen candidates, where a frame may be skipped (ball not seen):
    a chosen candidate costs -log p, a skipped frame -log p0, and a step of g frames between chosen
    candidates d^2 / (2 sig^2 g), impossible beyond vmax g / fps + slack_m or g > max_skip. So the
    ball is picked up again near where it was lost. Restarting anywhere costs c_break (free at an
    end with no anchor). start / end: the tracked ball just before / after, if any. extra (T, K): added to
    a candidate's cost.
    Returns the chosen candidate per frame, K for "not seen"."""
    T, K = p.shape
    X = np.full((T + 2, K, 2), np.nan)                 # frame 0 and T + 1: the anchors
    X[1:-1] = xy
    U = np.full((T + 2, K), np.inf)
    U[1:-1] = np.where(p > 0, -np.log(np.clip(p, 1e-6, 1)), np.inf) + (0 if extra is None else extra)
    U[0, 0] = U[-1, 0] = 0
    if start is not None:
        X[0, 0] = start
    if end is not None:
        X[-1, 0] = end
    u0 = -np.log(p0)
    best = np.full((T + 2, K), np.inf)
    best[0, 0] = 0
    prev = np.full((T + 2, K, 2), -1)
    run, run_at = np.inf, (0, 0)                       # min over s >= 1 of best[s] - s u0, for restarts
    for t in range(1, T + 2):
        free_end = t == T + 1 and end is None
        cand = [((c_break if start is not None and not free_end else 0) + (t - 1) * u0, (0, 0)),
                ((0 if free_end else c_break) + run + (t - 1) * u0, run_at)]
        val, at = min(cand, key=lambda c: c[0])
        val, at = np.full(K, val), np.tile(at, (K, 1))
        for g in range(1, min(max_skip, t) + 1):
            d = np.linalg.norm(X[t - g][:, None] - X[t][None], axis=2)
            with np.errstate(invalid="ignore"):
                c = best[t - g][:, None] + np.where(d <= vmax * g / fps + slack_m, d ** 2 / (2 * sig ** 2 * g), np.inf) + (g - 1) * u0
            i = c.argmin(0)
            better = c[i, np.arange(K)] < val
            val[better], at[better] = c[i, np.arange(K)][better], np.stack([np.full(K, t - g), i], 1)[better]
        best[t] = U[t] + val
        prev[t] = at
        if 1 <= t <= T and best[t].min() - t * u0 < run:
            run, run_at = best[t].min() - t * u0, (t, int(best[t].argmin()))
    path = np.full(T + 2, K)
    t, i = T + 1, 0
    while t > 0:
        path[t] = i
        t, i = prev[t, i]
    return path[1:-1]


def linked(Hm, ball, **kw):
    """Detections chosen by link() from out/ball_peaks.npz, as (frame, u, v, p) like ball_detections.npz
    (p = 0 where the path says "not seen"). Each run of consecutive frames is one path: runs inside
    a tracker gap are anchored on the tracked ball either side; held-out evaluation runs (which
    have a tracked ball) are linked without it."""
    d = np.load(os.path.join(OUT, "ball_peaks.npz"))
    fr, pk = d["frame"], d["peaks"]
    xy = to_pitch(Hm, fr, pk[..., :2].astype(float))
    xy[pk[..., 2] <= 0] = np.nan
    has = np.isfinite(ball[:, 0])
    out = pk[:, 0].copy()
    cut = np.flatnonzero((np.diff(fr) > 1) | (has[fr[1:]] != has[fr[:-1]])) + 1
    for a, b in zip(np.r_[0, cut], np.r_[cut, len(fr)]):
        f0, f1 = fr[a], fr[b - 1]
        anchored = not has[f0]
        path = link(xy[a:b], pk[a:b, :, 2],
                    start=ball[f0 - 1] if anchored and has[f0 - 1] else None,
                    end=ball[f1 + 1] if anchored and f1 + 1 < len(ball) and has[f1 + 1] else None, **kw)
        seen = path < pk.shape[1]
        out[a:b] = 0
        out[np.arange(a, b)[seen]] = pk[np.arange(a, b)[seen], path[seen]]
    return fr, out[:, 0], out[:, 1], out[:, 2]


def plausible(xy, trusted, pxy, live, periods, fps=FPS, vmax=40.0, slack_m=2.0, far_m=15.0):
    """Which frames of a ball path the ball could really be at: each untrusted frame must be reachable
    at vmax (plus slack_m for the video's ground-point error) from the last accepted position before it
    and from the next after it, chained from the trusted frames inward; and in live play it must be
    within far_m of some player. Trusted frames are always kept."""
    seen = np.isfinite(xy[:, 0])
    ok = seen & ~trusted
    for step in (1, -1):                                   # forwards, then backwards
        reach = np.zeros(len(xy), bool)
        for a, b in periods:
            last, t_last = None, None
            for f in range(a, b)[::step]:
                if not seen[f]:
                    continue
                if trusted[f] or last is None or \
                        np.linalg.norm(xy[f] - last) <= slack_m + vmax * abs(f - t_last) / fps:
                    reach[f] = True
                    last, t_last = xy[f], f
        ok &= reach
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        near = np.nanmin(np.linalg.norm(pxy - xy[:, None], axis=2), 1) <= far_m
    return trusted | (ok & (near | ~live))


def tidy(xy, trusted, periods, fps=FPS, out_m=2.0, bridge_s=0.4, sigma_s=0.5, ramp_s=1.0, split=True):
    """A ball path the features can use: drop untrusted frames more than out_m from the median of
    their 4 neighbours (or with no neighbour within 2 frames), bridge gaps of bridge_s or less by
    straight lines, blur (clean.smooth_ball), then shift each untrusted run so its ends meet the
    trusted ball beside it (the video's ground point and the tracker's differ by ~1 m), the shift
    fading over ramp_s where only one end meets it. Trusted frames keep their values.

    sigma_s 0.5, more than our tracker's blur: the video is noisier inside its runs. split: the blur doesn't
    cross a kick out of stillness (features.restarts on a 9-frame median of the path), and stretches
    still for 1 s or more are held exactly still, so restarts survive."""
    import clean
    xy = xy.copy()
    nb = np.full((4,) + xy.shape, np.nan)
    for i, k in enumerate((-2, -1, 1, 2)):
        nb[i, max(k, 0):len(xy) + min(k, 0)] = xy[max(-k, 0):len(xy) - max(k, 0)]
    with warnings.catch_warnings():                                         # all-NaN neighbours are expected
        warnings.simplefilter("ignore", RuntimeWarning)
        far = ~(np.linalg.norm(xy - np.nanmedian(nb, 0), axis=1) <= out_m)       # NaN median counts as far
    xy[~trusted & far] = np.nan
    for a, b in periods:
        seen = np.flatnonzero(np.isfinite(xy[a:b, 0])) + a
        for s0, s1 in zip(seen[:-1], seen[1:]):
            if 1 < s1 - s0 <= round(bridge_s * fps) + 1:
                t = np.linspace(0, 1, s1 - s0 + 1)[1:-1, None]
                xy[s0 + 1:s1] = xy[s0] + t * (xy[s1] - xy[s0])
    if split:                                          # blur each piece between kicks out of stillness on its own
        from scipy.ndimage import gaussian_filter1d, median_filter
        import features
        pre = np.where(np.isfinite(xy), median_filter(np.nan_to_num(xy), size=(9, 1), mode="nearest"), np.nan)
        kick = features.restarts(clean.ball_speed_clean(pre, fps)[0], fps)
        out = xy.copy()
        for a, b in periods:
            for x, y in clean.runs(np.isfinite(xy[a:b, 0])):
                cuts = [a + x] + list(np.flatnonzero(kick[a + x:a + y]) + a + x) + [a + y]
                for c0, c1 in zip(cuts[:-1], cuts[1:]):
                    if c1 > c0:
                        out[c0:c1] = gaussian_filter1d(xy[c0:c1], sigma_s * fps, axis=0, mode="nearest")
        sp = clean.ball_speed_clean(pre, fps)[0]
        for x, y in clean.runs(np.nan_to_num(sp, nan=99) < features.STILL_SPEED):
            if y - x >= fps:                           # still for 1 s or more: hold it exactly still
                out[x:y] = np.nanmedian(xy[x:y], axis=0)
    else:
        out = clean.smooth_ball(xy, periods, fps, sigma_s)
    out[trusted] = xy[trusted]
    k = round(ramp_s * fps)
    for a, b in clean.runs(np.isfinite(out[:, 0]) & ~trusted):
        d0 = out[a - 1] - out[a] if a > 0 and trusted[a - 1] else None
        d1 = out[b] - out[b - 1] if b < len(out) and trusted[b] else None
        i = np.arange(b - a)[:, None]
        if d0 is not None and d1 is not None:
            out[a:b] += d0 + (d1 - d0) * i / max(b - a - 1, 1)
        elif d0 is not None:
            out[a:b] += d0 * np.clip(1 - i / k, 0, 1)
        elif d1 is not None:
            out[a:b] += d1 * np.clip(1 - (b - a - 1 - i) / k, 0, 1)
    return out


def write(p_fit=0.3, video_min=0.3, tidy_min=0.5):
    """Evaluate on the held-out tracked frames, then write the recovered ball for every video frame:
    the tracked ball where there is one (source "track"), else the video detection if its peak is
    at least video_min ("video"), else the ball-from-players estimate of radar.py ("players").
    plausible: video frames that pass plausible() (reachable, near a player in live play).
    x_tidy, y_tidy: the tracked ball plus plausible video at tidy_min or more, through tidy(); NaN elsewhere."""
    trk, Hm = load_tracking(), load_homography()
    if os.path.isfile(os.path.join(OUT, "ball_peaks.npz")):
        fr, u, v, p = linked(Hm, trk["ball"])
    else:
        d = np.load(os.path.join(OUT, "ball_detections.npz"))
        fr, u, v, p = d["frame"], d["u"], d["v"], d["p"]
    uv = np.stack([u, v], 1).astype(float)
    ground = to_pitch(Hm, fr, uv)
    air, xy = np.zeros(len(fr), bool), ground.copy()
    ok = p >= p_fit
    air[ok], xy[ok] = flight(fr[ok], uv[ok], Hm, ground[ok])

    n, ball = len(trk["ball"]), trk["ball"]
    ev = np.isfinite(ball[fr, 0])
    err = np.linalg.norm(xy[ev] - ball[fr[ev]], axis=1)
    err_ground = np.linalg.norm(ground[ev] - ball[fr[ev]], axis=1)
    px = np.linalg.norm(uv[ev] - to_pixels(Hm, fr[ev], ball[fr[ev]]), axis=1)
    print("held-out tracked frames", ev.sum(), "| missing graded frames", (~ev).sum())
    print("thr  recovered  median_m  <1m  <2m  <5m  median_px | missing covered")
    for thr in (0.1, 0.3, 0.5, 0.7, 0.9):
        m = p[ev] >= thr
        print("{:.1f}  {:8.3f}  {:7.2f}  {:.3f} {:.3f} {:.3f} {:8.1f} | {:.3f} ({})".format(
            thr, m.mean(), np.median(err[m]), *[(err[m] < r).mean() for r in (1, 2, 5)], np.median(px[m]),
            (p[~ev] >= thr).mean(), int((p[~ev] >= thr).sum())))
    a = air[ev]
    print("airborne: {:.3f} of held-out detections; their error ground-plane {:.2f} m -> flight fit {:.2f} m"
          .format(a.mean(), np.median(err_ground[a]), np.median(err[a])))
    print("airborne: {:.3f} of missing-frame detections".format(air[~ev].mean()))

    x, y = ball[:, 0].copy(), ball[:, 1].copy()
    conf = np.where(np.isfinite(x), 1.0, 0.0).astype(np.float32)
    source = np.where(np.isfinite(x), "track", "none").astype("<U7")
    airborne = np.zeros(n, bool)
    miss = fr[~ev]
    radar = os.path.join(OUT, "radar_ours.npz")
    if os.path.isfile(radar):
        r = np.load(radar)
        fall = (source == "none") & graded_mask(n)
        x[fall], y[fall], conf[fall], source[fall] = r["x"][fall], r["y"][fall], r["confidence"][fall], "players"
    seen = p[~ev] >= video_min
    x[miss[seen]], y[miss[seen]] = xy[~ev][seen, 0], xy[~ev][seen, 1]
    conf[miss[seen]], source[miss[seen]], airborne[miss[seen]] = p[~ev][seen], "video", air[~ev][seen]
    print("missing graded frames by source:", {k: int((source[graded_mask(n) & ~np.isfinite(ball[:, 0])] == k).sum())
                                               for k in ("video", "players", "none")})
    periods = [(a, b + 1) for _, a, b in half_video_blocks(GAME)]
    track = source == "track"
    cand = np.where((track | (source == "video"))[:, None], np.stack([x, y], 1), np.nan)
    import clean
    live = graded_mask(n) & (clean.mean_player_speed(trk["xy"], FPS) > 1.5)
    ok = plausible(cand, track, trk["xy"], live, periods) & (source == "video")
    keep = track | (ok & (conf >= tidy_min))
    t = tidy(np.where(keep[:, None], np.stack([x, y], 1), np.nan), track, periods)
    np.savez(os.path.join(OUT, "ours_ball_recovered.npz"), frame=np.arange(n), x=x, y=y,
             confidence=conf, source=source, airborne=airborne, plausible=ok, x_tidy=t[:, 0], y_tidy=t[:, 1])


if __name__ == "__main__":
    {"data": build, "mine": mine, "train": train, "detect": detect, "redo": lambda: detect(redo=True), "write": write}[sys.argv[1]]()
