"""Ball net trained on OTHER games only (SoccerTrack 117093 1st-half crops, labelled by COCO YOLO11x: no person, not our
match). Augmentations aimed at the look of our tactical video. Held-out 60 s blocks of crops give an external score.

    python train_ext.py OUT.pt [epochs=20] [c=16] [aug=0|1|2] [seed=0] [rgb=0|1] [hard=PATH] [halves=1|2] [crops=GLOB] [frames=3] [pan=0|1]

The crops come from cv2 (BGR); our match is decoded by ffmpeg as RGB. rgb=1 flips the crops to RGB."""
import glob, numpy as np, os, sys, time, torch
import torch.nn.functional as F
sys.path[:0] = [os.getcwd(), os.getcwd() + "/vision"]
import ball as BL

CROPS = os.path.expanduser("~/Github/external_data/_cache/touch_spot/ball_crops_117093_1.npz")


def pan_frames(x, rng, vmax=8.0):
    """A moving camera on SoccerTrack's fixed one: half the crops get a steady shift of up to vmax px per frame in a random
    direction (our tactical camera: 90% of frames under 4, 99% under 8). The middle frame, which carries the label, stays put."""
    B, C, h, w = x.shape
    t = C // 3
    v = rng.uniform(0, vmax, B) * (rng.random(B) < 0.5)
    a = rng.uniform(0, 2 * np.pi, B)
    k = torch.arange(t, device=x.device, dtype=torch.float32) - t // 2
    dx = torch.as_tensor(v * np.cos(a), device=x.device, dtype=torch.float32)[:, None] * k * 2 / w
    dy = torch.as_tensor(v * np.sin(a), device=x.device, dtype=torch.float32)[:, None] * k * 2 / h
    theta = torch.zeros(B * t, 2, 3, device=x.device)
    theta[:, 0, 0] = theta[:, 1, 1] = 1
    theta[:, 0, 2], theta[:, 1, 2] = dx.flatten(), dy.flatten()
    v3 = x.view(B * t, 3, h, w)
    return F.grid_sample(v3, F.affine_grid(theta, v3.shape, align_corners=False), padding_mode="border", align_corners=False).view(B, C, h, w)


def augment(x, y, rng, level):
    """x: float (B, 3t, h, w) in [-0.5, 0.5] on DEV, y: (B, 2) labels (NaN: no ball). Level 1: colour, contrast,
    saturation, blur, noise. Level 2: also scale 0.7-1.5x about the crop centre (labels follow)."""
    B, C, h, w = x.shape
    t = C // 3
    dev = x.device
    if level >= 1:
        v = (x + 0.5).view(B, t, 3, h, w)                                  # (B, time, rgb, h, w)
        grey = v.mean(2, keepdim=True)
        sat = torch.empty(B, 1, 1, 1, 1, device=dev).uniform_(0.4, 1.6)
        v = grey + (v - grey) * sat
        gain = torch.empty(B, 1, 3, 1, 1, device=dev).uniform_(0.85, 1.15)  # white balance
        v = v * gain
        mean = v.mean((1, 2, 3, 4), keepdim=True)
        con = torch.empty(B, 1, 1, 1, 1, device=dev).uniform_(0.6, 1.4)
        bri = torch.empty(B, 1, 1, 1, 1, device=dev).uniform_(-0.2, 0.2)
        v = (v - mean) * con + mean + bri
        x = v.reshape(B, C, h, w)
        s = float(rng.uniform(0, 1.3))                                     # blur: one sigma per batch
        if s > 0.3:
            k = torch.arange(-3, 4, device=dev, dtype=torch.float32)
            g = torch.exp(-k ** 2 / (2 * s * s)); g = g / g.sum()
            x = F.conv2d(F.pad(x, (3, 3, 0, 0), mode="replicate"), g.view(1, 1, 1, 7).repeat(C, 1, 1, 1), groups=C)
            x = F.conv2d(F.pad(x, (0, 0, 3, 3), mode="replicate"), g.view(1, 1, 7, 1).repeat(C, 1, 1, 1), groups=C)
        x = x + torch.randn_like(x) * float(rng.uniform(0, 0.04))
        x = x.clamp(0, 1) - 0.5
    if level >= 2:
        z = float(rng.uniform(0.7, 1.5))
        n = int(round(h * z))
        r = F.interpolate(x, size=(n, n), mode="bilinear", align_corners=False)
        if n >= h:
            o = (n - h) // 2; x = r[:, :, o:o + h, o:o + w]; y = y * z - o
        else:
            o = (h - n) // 2; x = F.pad(r, (o, h - n - o, o, h - n - o), mode="replicate"); y = y * z + o
        y = np.where((y >= 0).all(1, keepdims=True) & (y < h).all(1, keepdims=True), y, np.nan)
    return x, y


def main(out, epochs=20, c=16, aug=1, seed=0, rgb=0, hard="", halves=1, crops="", frames=3, pan=0):
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    d = [np.load(p) for p in sorted(glob.glob(crops))] if crops else [np.load(CROPS.replace("_1.npz", f"_{h}.npz")) for h in range(1, halves + 1)]
    X, lab, fr = (np.concatenate([e[k] for e in d]) for k in ("X", "lab", "frame"))
    if hard:
        h = np.load(hard); X, lab, fr = np.concatenate([X, h["X"]]), np.concatenate([lab, h["lab"]]), np.concatenate([fr, h["frame"]])
    m = X.shape[1] // 2
    X = X[:, m - frames // 2:m + frames // 2 + 1]
    if rgb:
        X = X[..., ::-1]
    blk = (fr // 1500) % 5 == 2
    net = BL.Net(c, frames).to(BL.DEV)
    bs = 32
    opt = torch.optim.AdamW(net.parameters(), 1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, 2e-3, total_steps=epochs * (int((~blk).sum()) // bs))
    t0 = time.time()
    for epoch in range(epochs):
        net.train(); idx = rng.permutation(np.flatnonzero(~blk)); tot = 0.0
        for k in range(len(idx) // bs):
            i = np.sort(idx[k * bs:(k + 1) * bs]); x, y = np.ascontiguousarray(X[i]), lab[i].copy()
            if rng.random() < 0.5:
                x, y[:, 0] = x[:, :, :, ::-1], BL.CROP - 1 - y[:, 0]
            if rng.random() < 0.5:
                x = x[:, ::-1]
            xin = BL.to_input(np.ascontiguousarray(x))
            xin, y = augment(pan_frames(xin, rng) if pan else xin, y, rng, aug)
            loss = BL.focal(net(xin), BL.heatmap_target(y, BL.CROP))
            opt.zero_grad(); loss.backward(); opt.step(); sched.step(); tot += loss.item()
        print("epoch", epoch, "loss", round(tot / (k + 1), 4), "held-out crops", round(BL.evaluate_crops(net, np.ascontiguousarray(X[blk]), lab[blk]), 3),
              round(time.time() - t0), "s", flush=True)
        torch.save(net.state_dict(), out)


if __name__ == "__main__":
    a = dict(kv.split("=") for kv in sys.argv[2:])
    main(sys.argv[1], int(a.get("epochs", 20)), int(a.get("c", 16)), int(a.get("aug", 1)), int(a.get("seed", 0)),
         int(a.get("rgb", 0)), a.get("hard", ""), int(a.get("halves", 1)), a.get("crops", ""), int(a.get("frames", 3)), int(a.get("pan", 0)))
