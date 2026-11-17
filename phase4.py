"""Phase 4: a temporal convolutional network (TCN) over the frame features, in place of Phase 3's per-frame trees.

    model = fit_model(train, feats, drop)          # same interface as phase3.fit_model / log_probs,
    logp = log_probs(model, F, drop)               # so phase3.fit tunes Phase 1's decoder on top of it

The per-frame trees see a frame plus centred 1 s / 4 s summaries. A TCN sees the sequence: steps of 0.2 s
(5-frame means, the decoder's own step), dilated convolutions reaching about 6 s either side, so it can learn
how a possession starts and ends. Trained on the same frames (teams-swapped mirror copies included), with the
same IDSSE-only ball cues dropped.

raw=True fits the same TCN on features.raw_features instead (the ball and the 5 players nearest it, Sanford 2020
style): weaker alone, but a third ensemble member (Raw3) that the screen in docs/2026-10-23-lane-d-model-screen.md
found adds what the other two miss.
"""
import numpy as np
import torch
from torch import nn

import features as fe
import phase3

STEP = 5                 # frames per step (0.2 s at 25 fps)
CHUNK = 300              # steps per training chunk (60 s)
HIDDEN, DILATIONS, DROPOUT = 64, (1, 2, 4, 8, 16), 0.1
EPOCHS, BATCH, LR = 12, 32, 2e-3
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"   # training only: ~40x faster on the Mac's GPU, but its floats
                                                                  # differ from CPU's, so validate.submit sets "cpu" (the reproducible build)
SEED = 0                 # other values only for the seed-agreement check (external/generalisation.py)
N_SEEDS = 1              # >1: train that many nets (seeds SEED, SEED+1, ...) and average their log-probabilities (P3, flicker)
LAST_MIRROR = None       # SetAvg.MIRROR_DUMP: the last log_probs call's prediction on the mirrored match (H, A swapped back), for logp.npz
_MODELS = {}


def _steps(X):
    """[T, C] frame values -> [T // STEP, C] step means."""
    n = len(X) // STEP
    return X[:n * STEP].reshape(n, STEP, -1).mean(1)


def _inputs(F, drop, cols=None, raw=False):
    X = F[[c for c in F.columns if c.startswith("raw_")]] if raw else phase3.with_context(F, drop=drop)
    return X if cols is None else X[cols]


class TCN(nn.Module):
    def __init__(self, c_in):
        super().__init__()
        self.inp = nn.Conv1d(c_in, HIDDEN, 1)
        self.blocks = nn.ModuleList(nn.Sequential(nn.Conv1d(HIDDEN, HIDDEN, 3, padding=d, dilation=d), nn.GELU(),
                                                  nn.Dropout(DROPOUT), nn.Conv1d(HIDDEN, HIDDEN, 1)) for d in DILATIONS)
        self.out = nn.Conv1d(HIDDEN, 3, 1)

    def forward(self, x):                      # x: [B, C, n]
        h = self.inp(x)
        for b in self.blocks:
            h = h + b(h)
        return self.out(h)                     # [B, 3, n] logits for H, A, D


def fit_model(train, feats, drop=(), raw=False):
    key = (tuple(drop), tuple(sorted((n, id(c)) for n, c in train.items())), phase3.POSS_ALPHA, raw, SEED, N_SEEDS)
    if key in _MODELS:
        return _MODELS[key]
    cols = list(_inputs(feats[sorted(train)[0]], drop, raw=raw).columns)
    seqs = []
    for c, F0 in (x for n, c0 in train.items() for x in phase3.with_aug(n, c0, feats[n])):
        lab = np.where(c.graded & np.isin(c.labels, phase3.CLASSES), c.labels, "X")
        wt = phase3.spell_weights(c.labels, c.fps, phase3.POSS_ALPHA)[STEP // 2::STEP]
        swap = np.where(lab == "H", "A", np.where(lab == "A", "H", lab))
        for F, y in ((F0, lab), (phase3.mirror(F0, c.length, c.width), swap)):
            X = _steps(_inputs(F, drop, cols, raw).to_numpy(np.float32))
            ys = y[STEP // 2::STEP][:len(X)]                     # the label at each step's middle frame
            seqs.append((X, np.array([{"H": 0, "A": 1, "D": 2}.get(v, -1) for v in ys]), wt[:len(X)]))
    allX = np.concatenate([s[0] for s in seqs])
    wmean = np.concatenate([s[2][s[1] >= 0] for s in seqs]).mean()
    mean, std = np.nanmean(allX, 0), np.nanstd(allX, 0) + 1e-6
    norm = lambda X: np.nan_to_num((X - mean) / std).astype(np.float32)
    chunks = [(norm(X[a:a + CHUNK]), y[a:a + CHUNK], (wt[a:a + CHUNK] / wmean).astype(np.float32))
              for X, y, wt in seqs for a in range(0, len(X) - CHUNK + 1, CHUNK // 2)]
    nets = [_train(chunks, len(cols), s) for s in range(SEED, SEED + N_SEEDS)]
    model = {"nets": nets, "cols": cols, "mean": mean, "std": std, "raw": raw}
    _MODELS[key] = model
    return model


def _train(chunks, c_in, seed):
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    net = TCN(c_in).to(DEVICE)   # initialised on CPU first, so both devices start from the same weights
    opt = torch.optim.AdamW(net.parameters(), lr=LR, weight_decay=1e-3)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, LR, total_steps=EPOCHS * (len(chunks) // BATCH + 1))
    lossf = nn.CrossEntropyLoss(ignore_index=-1, reduction="none" if phase3.POSS_ALPHA else "mean")
    net.train()
    for _ in range(EPOCHS):
        order = rng.permutation(len(chunks))
        for i in range(0, len(order), BATCH):
            b = [chunks[j] for j in order[i:i + BATCH]]
            x = torch.from_numpy(np.stack([c[0] for c in b])).transpose(1, 2).to(DEVICE)
            y = torch.from_numpy(np.stack([c[1] for c in b])).long().to(DEVICE)
            opt.zero_grad(); loss = lossf(net(x), y)
            if phase3.POSS_ALPHA:   # weighted only when asked: the unweighted mean keeps the submission byte-identical
                wb = torch.from_numpy(np.stack([c[2] for c in b])).to(DEVICE) * (y >= 0)
                loss = (loss * wb).sum() / wb.sum()
            loss.backward(); opt.step(); sched.step()
    return net.cpu().eval()           # predictions always on CPU


def log_probs(model, F, drop=()):
    """[T, 3] log P(H), P(A), P(D) per frame (each step's value repeated over its frames)."""
    X = _steps(_inputs(F, drop, model["cols"], model["raw"]).to_numpy(np.float32))
    x = torch.from_numpy(np.nan_to_num((X - model["mean"]) / model["std"]).astype(np.float32)).T[None]
    with torch.no_grad():
        lp = np.mean([torch.log_softmax(net(x)[0], 0).T.numpy() for net in model["nets"]], 0)
    out = np.repeat(lp, STEP, axis=0)
    if len(out) < len(F):
        out = np.r_[out, np.repeat(out[-1:], len(F) - len(out), axis=0)]
    return np.maximum(out, np.log(1e-4))


class Ensemble:
    """Weighted mean of the trees' and the TCN's log-probabilities (W_TCN on the TCN), plus the raw TCN's (W_RAW)."""
    W_TCN, W_RAW = 0.5, 0.0

    @classmethod
    def features(cls, c):
        F = fe.frame_features(c)
        return F.join(fe.raw_features(c)) if cls.W_RAW else F

    @classmethod
    def fit_model(cls, train, feats, drop=()):
        raw = fit_model(train, feats, drop, raw=True) if cls.W_RAW else None
        return phase3.fit_model(train, feats, drop), fit_model(train, feats, drop), raw

    @classmethod
    def log_probs(cls, model, F, drop=()):
        lp = (1 - cls.W_TCN - cls.W_RAW) * phase3.log_probs(model[0], F, drop) + cls.W_TCN * log_probs(model[1], F, drop)
        return lp + cls.W_RAW * log_probs(model[2], F, drop) if cls.W_RAW else lp


class Raw3(Ensemble):
    """Trees, TCN and raw TCN, a third each."""
    W_TCN = W_RAW = 1 / 3


class PooledAvg(Raw3):
    """Raw3 on the training set's IDSSE matches (the v5 recipe) and Raw3 on all of it (IDSSE and Metrica: the pooled
    recipe), their log-probabilities averaged (docs/2026-10-24-model-averaging.md). validate's phase4_avg trains it on
    IDSSE and Metrica, so phase3.fit tunes the decoder on the averaged out-of-fold probabilities."""

    @classmethod
    def fit_model(cls, train, feats, drop=()):
        idsse = {n: c for n, c in train.items() if n.startswith("idsse_")}   # validate's set names
        return Raw3.fit_model(idsse, feats, drop), Raw3.fit_model(train, feats, drop)

    @classmethod
    def log_probs(cls, model, F, drop=()):
        return (Raw3.log_probs(model[0], F, drop) + Raw3.log_probs(model[1], F, drop)) / 2


class SetAvg(PooledAvg):
    """PooledAvg and setnet (the ball and the players nearest it, plus the frame features, at full rate), their
    log-probabilities mixed W_SET to setnet (docs/2026-11-02-setnet.md). setnet trains on every training match with its
    extra corruption draws. Its inputs for a frame table sit in _BY_TABLE until that table is freed (not in DataFrame.attrs,
    which pandas deep-copies on every column read)."""
    W_SET = 1 / 3
    MIRROR = False        # True: average each prediction with the prediction on the mirrored match, H and A swapped back
    MIRROR_DUMP = False   # True: also keep that mirrored prediction in LAST_MIRROR (validate saves it as <set>.logp_mirror)
    SETNET = {"EPOCHS": 5, "DROPOUT": 0.3, "WD": 0.05}   # the regularised screen's settings
    _IN = {}         # id(match) -> setnet inputs, for the extra draws
    _BY_TABLE = {}   # id(frame table from features()) -> setnet inputs
    _LW = {}         # id(frame table from features()) -> the pitch's (length, width), for the mirror

    @classmethod
    def _inputs(cls, c, F):
        import setnet
        G = phase3.with_context(F, drop=phase3.BALL_CUES).to_numpy(np.float16)
        Gm = phase3.with_context(phase3.mirror(F, c.length, c.width), drop=phase3.BALL_CUES).to_numpy(np.float16)
        return setnet.tokens(c), G, Gm

    @classmethod
    def features(cls, c):
        import weakref
        F = super().features(c)
        cls._BY_TABLE[id(F)] = cls._inputs(c, F)
        cls._LW[id(F)] = c.length, c.width
        weakref.finalize(F, cls._BY_TABLE.pop, id(F), None)
        weakref.finalize(F, cls._LW.pop, id(F), None)
        return F

    @classmethod
    def fit_model(cls, train, feats, drop=()):
        import setnet
        for k, v in dict(cls.SETNET, SEED=SEED).items():
            setattr(setnet, k, v)
        data = {}
        for n, c0 in train.items():
            data[n] = []
            for c, F in phase3.with_aug(n, c0, feats[n]):
                if id(F) not in cls._BY_TABLE and id(c) not in cls._IN:
                    cls._IN[id(c)] = cls._inputs(c, F)
                data[n].append((c,) + cls._BY_TABLE.get(id(F), cls._IN.get(id(c))))
        return super().fit_model(train, feats, drop), setnet.fit(data)

    @classmethod
    def log_probs(cls, model, F, drop=()):
        import setnet
        global LAST_MIRROR
        X, G, Gm = cls._BY_TABLE[id(F)]
        lp = (1 - cls.W_SET) * super().log_probs(model[0], F, drop) + cls.W_SET * setnet.log_probs(model[1], X, G)
        if not (cls.MIRROR or cls.MIRROR_DUMP):
            return lp
        flip = np.r_[np.tile(setnet.FLIP_PLAYER, (setnet.K, 1)), setnet.FLIP_BALL[None]].astype(np.float16)
        Fm = phase3.mirror(F, *cls._LW[id(F)])
        lm = (1 - cls.W_SET) * super().log_probs(model[0], Fm, drop) + cls.W_SET * setnet.log_probs(model[1], X * flip, Gm)
        lm = lm[:, [1, 0, 2]]                                    # the mirror's H is our A
        if cls.MIRROR_DUMP:
            LAST_MIRROR = lm
        return (lp + lm) / 2 if cls.MIRROR else lp


class Seeds(SetAvg):
    """K whole SetAvg models at seeds SEED .. SEED + K - 1 (the TCNs and setnet; the trees are deterministic and fitted
    once), their log-probabilities averaged. phase3.fit tunes the decoder once on the averaged out-of-fold ones."""
    K = 3
    PROB = False          # True: average probabilities instead of log-probabilities

    @classmethod
    def fit_model(cls, train, feats, drop=()):
        global SEED
        base = SEED
        try:
            out = []
            for k in range(cls.K):
                SEED = base + k
                out.append(SetAvg.fit_model.__func__(cls, train, feats, drop))
            return out
        finally:
            SEED = base

    @classmethod
    def log_probs(cls, model, F, drop=()):
        lps = np.stack([SetAvg.log_probs.__func__(cls, m, F, drop) for m in model])
        if cls.PROB:
            return np.log(np.exp(lps).mean(0))
        return lps.mean(0)
