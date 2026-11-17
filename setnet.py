"""An all-players model: the ball and the players nearest it, every frame, attending to each other (docs/2026-11-02-setnet.md).

v8's members see hand-made per-team summaries, or the 5 nearest players averaged over 0.2 s steps. A human watching
reads duels: who is goal-side, who is closing, who comes out with the ball. Here each frame is a set of tokens (the
ball and the K players nearest it), one attention layer lets them interact, and a dilated convolution over full-rate
frames (about ±2.5 s) reads how the contest unfolds. Trained on the same frames as v8 (IDSSE and Metrica, masked like
our tracker, with the extra corruption draws), each ball but ours first blurred like our tracker's (change.SMOOTH_S).

    X = tokens(c)                                  # [T, K + 1, F] float16, once per match
    G, Gm = frame_inputs(c)                        # v8's frame features (and their mirror), for the long memory
    model = fit({name: [(c, X, G, Gm), ...]})      # the match, then its extra corruption draws; G, Gm may be None
    logp = log_probs(model, X, G)                  # [T, 3]: log P(H), P(A), P(D)

With frame inputs (the default since the first screen, where tokens alone called the team worse than v8 in live
play: they can't remember who had the ball through a gap), each frame's embedding is the set's plus v8's frame
features, and the convolution reaches about ±10 s.
    python setnet.py test
"""
import numpy as np
import torch
from torch import nn

import clean
import change
import features as fe
import phase3

K = 10                     # players per frame, nearest the ball first
F = 10                     # features per token
POS_M, VEL_MS = 10.0, 5.0  # scales
HIDDEN, D_TOK = 64, 32
DILATIONS = (1, 2, 4, 8, 16, 32, 64, 128)
CHUNK, BATCH, EPOCHS, LR = 750, 16, 10, 2e-3
DROPOUT, WD = 0.1, 1e-3
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
SEED = 0
# mirror: home and away swapped, the pitch turned 180 degrees. Token columns:
# dx dy vx vy closing d side opp_d present is_ball (players); x y vx vy seen since_seen 0 0 1 1 (ball)
FLIP_PLAYER = np.array([-1, -1, -1, -1, 1, 1, -1, 1, 1, 1], np.float32)
FLIP_BALL = np.array([-1, -1, -1, -1, 1, 1, 1, 1, 1, 1], np.float32)


def _vel(xy, fps):
    h = max(1, round(0.1 * fps))
    v = np.full(xy.shape, np.nan)
    v[h:-h] = (xy[2 * h:] - xy[:-2 * h]) * fps / (2 * h)
    return v


def tokens(c):
    """[T, K + 1, F] float16: K player tokens (nearest the ball first; absent ones all zero), then the ball token."""
    T, fps = len(c.bxy), c.fps
    b = c.bxy if str(c.name).startswith(change.OURS) else clean.smooth_ball(c.bxy, c.periods, fps, change.SMOOTH_S)
    seen = ~np.isnan(b[:, 0])
    last = np.maximum.accumulate(np.where(seen, np.arange(T), -1))
    bc = np.where(last[:, None] >= 0, b[np.maximum(last, 0)], [c.length / 2, c.width / 2])   # last seen place, carried
    bv = np.nan_to_num(_vel(b, fps))
    pv = _vel(c.pxy, fps)
    rel = c.pxy - bc[:, None]
    d = np.linalg.norm(rel, axis=2)
    side = np.where(c.team == 0, 1.0, -1.0)
    # each player's nearest opponent
    opp = np.full(d.shape, np.nan)
    for t in (0, 1):
        me, them = np.flatnonzero(c.team == t), np.flatnonzero(c.team != t)
        dd = np.linalg.norm(c.pxy[:, me, None] - c.pxy[:, None, them], axis=3)
        with np.errstate(invalid="ignore"), np.testing.suppress_warnings() as sup:
            sup.filter(RuntimeWarning)
            opp[:, me] = np.nanmin(dd, axis=2)
    order = np.argsort(np.where(np.isnan(d), np.inf, d), axis=1)[:, :K]
    rows = np.arange(T)[:, None]
    r, v, dn = rel[rows, order], pv[rows, order], d[rows, order]
    rv = v - bv[:, None]
    with np.errstate(invalid="ignore", divide="ignore"):
        closing = -np.sum(r * rv, axis=2) / np.maximum(dn, 0.1)
    present = np.isfinite(dn)
    P = np.stack([r[..., 0] / POS_M, r[..., 1] / POS_M, v[..., 0] / VEL_MS, v[..., 1] / VEL_MS, closing / VEL_MS,
                  dn / POS_M, np.broadcast_to(side[order], dn.shape), opp[rows, order] / POS_M,
                  present.astype(float), np.zeros(dn.shape)], axis=2)
    P = np.where(present[..., None], np.nan_to_num(P), 0.0)
    since = np.minimum((np.arange(T) - np.maximum(last, 0)) / fps, 5.0)
    B = np.stack([(bc[:, 0] - c.length / 2) / (c.length / 2), (bc[:, 1] - c.width / 2) / (c.width / 2), bv[:, 0] / VEL_MS,
                  bv[:, 1] / VEL_MS, seen.astype(float), np.where(last >= 0, since, 5.0), np.zeros(T), np.zeros(T),
                  np.ones(T), np.ones(T)], axis=1)
    return np.concatenate([P, B[:, None]], axis=1).astype(np.float16)


def frame_inputs(c):
    """(v8's frame inputs, the same for the mirrored match) as float16 [T, C]: phase3.with_context without the ball cues."""
    F = fe.frame_features(c)
    return tuple(phase3.with_context(x, drop=phase3.BALL_CUES).to_numpy(np.float16) for x in (F, phase3.mirror(F, c.length, c.width)))


class Net(nn.Module):
    def __init__(self, n_frame=0):
        super().__init__()
        self.tok = nn.Sequential(nn.Linear(F, D_TOK), nn.GELU(), nn.Linear(D_TOK, D_TOK))
        self.att = nn.TransformerEncoderLayer(D_TOK, nhead=2, dim_feedforward=2 * D_TOK, dropout=0.0, batch_first=True)
        self.frame = nn.Linear(2 * D_TOK + n_frame, HIDDEN)
        self.blocks = nn.ModuleList(nn.Sequential(nn.Conv1d(HIDDEN, HIDDEN, 3, padding=d, dilation=d), nn.GELU(),
                                                  nn.Dropout(DROPOUT), nn.Conv1d(HIDDEN, HIDDEN, 1)) for d in DILATIONS)
        self.out = nn.Conv1d(HIDDEN, 3, 1)

    def forward(self, x, g=None):                           # x: [B, n, K + 1, F]; g: [B, n, C] frame inputs, normalised
        Bn, n = x.shape[:2]
        z = x.reshape(Bn * n, K + 1, F)
        absent = z[..., 8] == 0
        absent[:, -1] = False                               # the ball token is always there
        h = self.att(self.tok(z), src_key_padding_mask=absent)
        players = (h[:, :-1] * (~absent[:, :-1, None])).sum(1) / (~absent[:, :-1]).sum(1, keepdim=True).clamp(min=1)
        e = torch.cat([h[:, -1], players] + ([g.reshape(Bn * n, -1)] if g is not None else []), 1)
        e = self.frame(e).reshape(Bn, n, HIDDEN).transpose(1, 2)
        for b in self.blocks:
            e = e + b(e)
        return self.out(e)                                  # [B, 3, n]


def _chunks(c, X, G, Gm):
    lab = np.where(c.graded & np.isin(c.labels, phase3.CLASSES), c.labels, "X")
    y = np.array([{"H": 0, "A": 1, "D": 2}.get(v, -1) for v in lab])
    w = phase3.spell_weights(c.labels, c.fps, 0.5).astype(np.float32)
    cut = lambda A, a: None if A is None else A[a:a + CHUNK]
    return [(X[a:a + CHUNK], y[a:a + CHUNK], w[a:a + CHUNK], cut(G, a), cut(Gm, a))
            for a in range(0, len(X) - CHUNK + 1, CHUNK // 2) if (y[a:a + CHUNK] >= 0).any()]


def fit(train, log=None):
    """train: {name: [(c, X, G, Gm), ...]}, the match then its extra draws (G, Gm None: tokens only). The fitted net (CPU)."""
    chunks = [ch for draws in train.values() for d in draws for ch in _chunks(*d)]
    first = next(iter(train.values()))[0]
    frames = first[2] is not None
    if frames:
        allG = np.concatenate([d[2][::25] for draws in train.values() for d in draws]).astype(np.float32)
        mean, std = np.nanmean(allG, 0), np.nanstd(allG, 0) + 1e-3
    norm = lambda A: torch.from_numpy(np.nan_to_num((np.stack(A).astype(np.float32) - mean) / std)).to(DEVICE)
    torch.manual_seed(SEED); rng = np.random.default_rng(SEED)
    net = Net(first[2].shape[1] if frames else 0).to(DEVICE)
    opt = torch.optim.AdamW(net.parameters(), lr=LR, weight_decay=WD)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, LR, total_steps=EPOCHS * (len(chunks) // BATCH + 1))
    lossf = nn.CrossEntropyLoss(ignore_index=-1, reduction="none")
    flip = torch.from_numpy(np.r_[np.tile(FLIP_PLAYER, (K, 1)), FLIP_BALL[None]]).to(DEVICE)
    net.train()
    for ep in range(EPOCHS):
        order, total = rng.permutation(len(chunks)), 0.0
        for i in range(0, len(order), BATCH):
            b = [chunks[j] for j in order[i:i + BATCH]]
            x = torch.from_numpy(np.stack([c[0] for c in b]).astype(np.float32)).to(DEVICE)
            y = torch.from_numpy(np.stack([c[1] for c in b])).long().to(DEVICE)
            w = torch.from_numpy(np.stack([c[2] for c in b])).to(DEVICE)
            mm = rng.random(len(b)) < 0.5                                       # mirror half the batch
            m = torch.from_numpy(mm).to(DEVICE)
            x = torch.where(m[:, None, None, None], x * flip, x)
            y = torch.where(m[:, None] & (y >= 0) & (y < 2), 1 - y, y)
            g = norm([c[4] if f else c[3] for c, f in zip(b, mm)]) if frames else None
            opt.zero_grad()
            loss = lossf(net(x, g), y)
            wb = w * (y >= 0)
            loss = (loss * wb).sum() / wb.sum().clamp(min=1e-6)
            loss.backward(); opt.step(); sched.step()
            total += float(loss)
        if log:
            log("  setnet epoch {} loss {:.4f}".format(ep, total / (len(order) / BATCH)))
    net = net.cpu().eval()
    net.norm = (mean, std) if frames else None
    return net


def log_probs(net, X, G=None, piece=20000):
    """[T, 3] log-probabilities, in overlapping pieces so the convolution sees context at each piece's edges."""
    pad = 2 * sum(DILATIONS)
    out = np.zeros((len(X), 3), np.float32)
    with torch.no_grad():
        for a in range(0, len(X), piece):
            lo, hi = max(0, a - pad), min(len(X), a + piece + pad)
            x = torch.from_numpy(X[lo:hi].astype(np.float32))[None]
            g = None if net.norm is None else torch.from_numpy(np.nan_to_num((G[lo:hi].astype(np.float32) - net.norm[0]) / net.norm[1]))[None]
            lp = torch.log_softmax(net(x, g)[0], 0).T.numpy()
            out[a:min(len(X), a + piece)] = lp[a - lo:a - lo + min(piece, len(X) - a)]
    return np.maximum(out, np.log(1e-4))


def test():
    from types import SimpleNamespace
    T = 1200
    t = np.arange(T) / 25.0
    pxy = np.zeros((T, 22, 2)); pxy[:, :, 0] = np.linspace(10, 90, 22); pxy[:, :, 1] = 30
    bxy = np.c_[50 + 10 * np.sin(t), np.full(T, 30.0)]
    bxy[100:150] = np.nan
    c = SimpleNamespace(name="BAR-ATH test", fps=25.0, length=105.0, width=68.0, pxy=pxy, team=np.repeat([0, 1], 11),
                        bxy=bxy, periods=[(0, T)], graded=np.ones(T, bool),
                        labels=np.where(bxy[:, 0] > 50, "H", "A").astype("<U1"))
    X = tokens(c)
    assert X.shape == (T, K + 1, F) and np.isfinite(X.astype(float)).all()
    assert X[120, -1, 4] == 0 and abs(float(X[120, -1, 5]) - 0.8) < 0.05       # ball hidden 20 frames in
    flipped = X.astype(np.float32) * np.r_[np.tile(FLIP_PLAYER, (K, 1)), FLIP_BALL[None]]
    assert np.allclose(flipped[:, :-1, 6], -X[:, :-1, 6])
    global EPOCHS, DEVICE
    EPOCHS, DEVICE = 2, "cpu"
    net = fit({"m": [(c, X, None, None)]})
    lp = log_probs(net, X, piece=300)
    assert lp.shape == (T, 3) and np.allclose(np.exp(lp).sum(1), 1, atol=1e-3)
    G = np.random.default_rng(0).normal(size=(T, 7)).astype(np.float16)
    net = fit({"m": [(c, X, G, -G)]})
    lp = log_probs(net, X, G, piece=300)
    assert lp.shape == (T, 3) and np.allclose(np.exp(lp).sum(1), 1, atol=1e-3)
    print("ok")


if __name__ == "__main__":
    test()
