"""After the decoder, on our match only: every live turnover earlier, the competition's convention (decoder lab P4).

    labels = shift(c, labels)       # c: a cleaned match; labels: "H" / "A" / "D" / "X" per frame, after the decoder

The Conventions audit (docs/2026-10-24-label-conventions.md) found the competition's truth hands the ball over a fraction of a
second before every convention we train and judge on. shift() moves every live turnover (H <-> A, no D between) on our
match SHIFT_S earlier, never past the first frame of the possession it cuts into. Dead-ball boundaries stay. The
judges can't measure it, by design. SHIFT_S 0 is off, the default; the audit recommends 0.6.

Snapping boundaries onto touches, control onsets, ball-outs or restarts was screened and dropped
(docs/2026-10-23-boundary-snap.md; the code is in git history at e6412c8).
"""
import numpy as np

SHIFT_S = 0.0        # seconds; 0: off
OURS = "BAR-ATH (ours)"


def shift(c, labels):
    if not SHIFT_S or c.name != OURS:
        return labels
    k = int(round(SHIFT_S * c.fps))
    out = labels.copy()
    edges = np.flatnonzero(labels[1:] != labels[:-1]) + 1
    starts = np.r_[0, edges]                                            # run i is labels[starts[i]:edges[i]]
    for i, f in enumerate(edges):
        if {labels[f - 1], labels[f]} == {"H", "A"}:
            out[max(starts[i] + 1, f - k):f] = labels[f]
    return out
