"""Ball from players (the Ball Radar idea, arXiv 2306.08206; our own code): where the ball is, from
where the players are and how they move, as a fallback where the video can't see it.

    python vision/radar.py train     # on IDSSE (last match held out)        out/radar_net.pt
    python vision/radar.py ours      # test on our tracked frames, predict all  out/radar_ours.npz

Per player: position, velocity over 0.2 s, team, all mirrored at random. A set network (shared
per-player layers, attention pooling) points at players, weights their positions, adds an
offset, and predicts a spread; confidence is the chance the
ball is within 2 m under that spread. IDSSE players are hidden at random, as ours go missing.
"""
import os
import sys

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ball import DEV, OUT  # noqa: E402
from preprocess import PITCH, graded_mask, load_tracking  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "external"))
LAG = 5                       # frames for the velocity (0.2 s at 25 fps)
SLOTS = 46                    # our tracker's slot count


def player_table(pxy, team, length, width):
    """(T, P, 6) features [x, y, vx, vy, team, 1] scaled to about +-1, zero where the player is
    missing, and the (T, P) presence mask."""
    v = np.full_like(pxy, np.nan)
    v[LAG:] = (pxy[LAG:] - pxy[:-LAG]) * 25 / LAG
    f = np.concatenate([pxy / [length, width] * 2 - 1, v / 10,
                        np.broadcast_to(team[None, :, None], pxy.shape[:2] + (1,)), np.ones(pxy.shape[:2] + (1,))], -1)
    m = np.isfinite(f).all(-1)
    return np.where(m[..., None], f, 0).astype(np.float32), m


class Radar(nn.Module):
    def __init__(self, h=128, heads=4):
        super().__init__()
        self.phi = nn.Sequential(nn.Linear(6, h), nn.ReLU(), nn.Linear(h, h), nn.ReLU())
        self.att = nn.Linear(h, heads)
        self.rho = nn.Sequential(nn.Linear(heads * h, h), nn.ReLU(), nn.Linear(h, 3))
        self.ptr = nn.Sequential(nn.Linear(h + heads * h, h), nn.ReLU(), nn.Linear(h, 1))

    def forward(self, f, m):
        h = self.phi(f)
        a = self.att(h).masked_fill(~m[..., None], -1e9).softmax(1)
        g = torch.einsum("bpk,bph->bkh", a, h).flatten(1)
        out = self.rho(g)
        w = self.ptr(torch.cat([h, g[:, None].expand(-1, h.shape[1], -1)], -1))[..., 0]
        w = w.masked_fill(~m, -1e9).softmax(1)     # which player has it, given the whole picture
        return (w[..., None] * f[..., :2]).sum(1) + 0.1 * out[:, :2], out[:, 2]   # ball (scaled), log spread


def mirror(f, y, rng):
    """Flip length and/or width at random (positions, velocities and the ball)."""
    for axis in (0, 1):
        flip = rng.random(len(f)) < 0.5
        f[flip, :, axis] *= -1
        f[flip, :, axis + 2] *= -1
        y[flip, axis] *= -1
    return f, y


def nll(xy, logs, y):
    """Gaussian negative log-likelihood, isotropic spread exp(logs)."""
    return (((xy - y) ** 2).sum(1) / (2 * torch.exp(2 * logs)) + 2 * logs).mean()


def idsse_frames(step=5):
    import profile_datasets as pdx
    ids = sorted({f.split("_")[-1][:-4] for f in os.listdir(os.path.join(pdx.EXT, "idsse")) if f.startswith("DFL_04_03")})
    out = []
    for mid in ids:
        m = pdx.load_idsse(mid)
        live = np.zeros(len(m.bxy), bool)
        for a, b in m.periods:
            live[a + LAG:b] = True
        keep = np.flatnonzero(live & m.bdet & (m.labels != "D"))[::step]
        f, mask = player_table(m.pxy, m.team, m.length, m.width)
        pad = SLOTS - f.shape[1]                    # every match to our 46 slots, the extra ones empty
        out.append((np.pad(f[keep], ((0, 0), (0, pad), (0, 0))), np.pad(mask[keep], ((0, 0), (0, pad))),
                    (m.bxy[keep] / [m.length, m.width] * 2 - 1).astype(np.float32)))
    return out


def train(epochs=8, bs=512, drop=0.07, seed=0):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    sets = idsse_frames()
    F = np.concatenate([s[0] for s in sets[:-1]])
    M = np.concatenate([s[1] for s in sets[:-1]])
    Y = np.concatenate([s[2] for s in sets[:-1]])
    net = Radar().to(DEV)
    opt = torch.optim.AdamW(net.parameters(), 1e-3, weight_decay=1e-4)
    for epoch in range(epochs):
        net.train()
        idx = rng.permutation(len(F))
        for b in range(len(idx) // bs):
            i = idx[b * bs:(b + 1) * bs]
            f, y = mirror(F[i].copy(), Y[i].copy(), rng)
            m = M[i] & (rng.random(M[i].shape) > drop)
            xy, logs = net(torch.from_numpy(f * m[..., None]).to(DEV), torch.from_numpy(m).to(DEV))
            loss = nll(xy, logs, torch.from_numpy(y).to(DEV))
            opt.zero_grad()
            loss.backward()
            opt.step()
        f, m, y = sets[-1]
        err = np.linalg.norm((predict(net, f, m)[0] - y) / 2 * [105, 68], axis=1)
        print("epoch", epoch, "loss", round(loss.item(), 3), "held-out IDSSE median error m", round(float(np.median(err)), 2), flush=True)
    torch.save(net.state_dict(), os.path.join(OUT, "radar_net.pt"))


@torch.no_grad()
def predict(net, f, m, bs=4096):
    net.eval()
    xy, s = [], []
    for b in range(0, len(f), bs):
        a, c = net(torch.from_numpy(f[b:b + bs]).to(DEV), torch.from_numpy(m[b:b + bs]).to(DEV))
        xy.append(a.cpu().numpy())
        s.append(c.cpu().numpy())
    return np.concatenate(xy), np.exp(np.concatenate(s))


def ours():
    trk = load_tracking()
    net = Radar().to(DEV)
    net.load_state_dict(torch.load(os.path.join(OUT, "radar_net.pt"), map_location=DEV))
    f, m = player_table(trk["xy"], trk["team"].astype(int), *PITCH)
    xy, s = predict(net, f, m)
    xy = (xy + 1) / 2 * PITCH
    spread = s / 2 * np.mean(PITCH)                                  # metres, roughly
    conf = 1 - np.exp(-2.0 ** 2 / (2 * spread ** 2))                 # P(within 2 m)
    ball = trk["ball"]
    ev = graded_mask(len(ball)) & np.isfinite(ball[:, 0]) & m.any(1)
    err = np.linalg.norm(xy[ev] - ball[ev], axis=1)
    print("our tracked graded frames: median error {:.2f} m, <2 m {:.3f}, <5 m {:.3f}, <10 m {:.3f}".format(
        np.median(err), *[(err < r).mean() for r in (2, 5, 10)]))
    for thr in (0.1, 0.3, 0.5):
        k = conf[ev] >= thr
        print("  conf >= {}: {:.3f} of frames, median error {:.2f} m".format(thr, k.mean(), np.median(err[k]) if k.any() else np.nan))
    np.savez(os.path.join(OUT, "radar_ours.npz"), x=xy[:, 0], y=xy[:, 1], confidence=conf.astype(np.float32))


if __name__ == "__main__":
    {"train": train, "ours": ours}[sys.argv[1]]()
