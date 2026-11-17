"""Tactical video frames and the pitch <-> pixel mapping.

    from vision.frames import read, to_pixels, to_pitch
    clip = read(20000, 3)             # (3, 1080, 1920, 3) uint8 RGB, video frames 20000..20002
    u, v = to_pixels(H, frames, xy)   # pitch metres -> pixels, per frame

Frame i of the video is submission row i (keyframes every 2 s, no B-frames, pts from 0, so an
input seek is frame-accurate). The homography maps normalised pitch (0-1) to pixels, so metres are
divided by PITCH first, the same convention as preprocess.load_tracking.
"""
import os
import subprocess
import sys

import numpy as np
from scipy.optimize import least_squares

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from preprocess import PITCH, REPO, FPS  # noqa: E402

VIDEO = os.path.join(REPO, "data", "tactical", "BAR-ATH_tactical.mp4")
W, H_PX = 1920, 1080


def stream(first, n):
    """Yield frames first..first+n-1 as (1080, 1920, 3) uint8 RGB, decoded once, in order."""
    cmd = ["ffmpeg", "-v", "error", "-threads", "3", "-ss", "{:.3f}".format(first / FPS), "-i", VIDEO,
           "-frames:v", str(n), "-f", "rawvideo", "-pix_fmt", "rgb24", "-"]
    size = W * H_PX * 3
    with subprocess.Popen(cmd, stdout=subprocess.PIPE, bufsize=size * 4) as proc:
        for _ in range(n):
            buf = proc.stdout.read(size)
            if len(buf) < size:
                break
            yield np.frombuffer(buf, np.uint8).reshape(H_PX, W, 3)
        proc.stdout.close()


def read(first, n=1):
    return np.stack(list(stream(first, n)))


def load_homography():
    """(n, 3, 3) matrices, pitch normalised -> pixels. Cached as .npy next to the tracking cache."""
    cache = os.path.join(os.path.dirname(HERE), "cache", "BAR-ATH_tactical_homography.npy")
    if not os.path.isfile(cache):
        sys.path.insert(0, os.path.join(REPO, "starter"))
        import load_homography
        os.makedirs(os.path.dirname(cache), exist_ok=True)
        np.save(cache, load_homography.load()[0])
    return np.load(cache)


def _apply(M, x, y):
    d = M[..., 2, 0] * x + M[..., 2, 1] * y + M[..., 2, 2]
    return ((M[..., 0, 0] * x + M[..., 0, 1] * y + M[..., 0, 2]) / d,
            (M[..., 1, 0] * x + M[..., 1, 1] * y + M[..., 1, 2]) / d)


def to_pixels(H, frames, xy):
    """Pitch metres (..., 2) at video frames (broadcastable) -> pixels (..., 2)."""
    M = H[frames]
    while M.ndim - 2 < xy.ndim - 1:
        M = M[..., None, :, :]
    return np.stack(_apply(M, xy[..., 0] / PITCH[0], xy[..., 1] / PITCH[1]), -1)


def to_pitch(H, frames, uv):
    """Pixels (..., 2) at video frames -> pitch metres (..., 2), on the ground plane."""
    M = np.linalg.inv(H[frames])
    while M.ndim - 2 < uv.ndim - 1:
        M = M[..., None, :, :]
    x, y = _apply(M, uv[..., 0], uv[..., 1])
    return np.stack([x * PITCH[0], y * PITCH[1]], -1)


def camera(Hf):
    """3x4 projection, pitch metres (x, y, z up) -> pixels, from one frame's homography.

    Assumes square pixels and the principal point at the image centre; the focal length then
    follows from r1 . r2 = 0 and |r1| = |r2| (Zhang's constraints), both in a least-squares fit.
    """
    Hm = Hf @ np.diag([1 / PITCH[0], 1 / PITCH[1], 1.0])
    a = np.array([[1, 0, -W / 2], [0, 1, -H_PX / 2], [0, 0, 1]]) @ Hm
    (x1, y1, z1), (x2, y2, z2) = a[:, 0], a[:, 1]
    A = np.array([x1 * x2 + y1 * y2, x1 * x1 + y1 * y1 - x2 * x2 - y2 * y2])
    B = -np.array([z1 * z2, z1 * z1 - z2 * z2])
    f2 = A @ A / (A @ B)                          # A / f^2 = B, least squares in 1/f^2
    # ponytail: a few frames give f^2 <= 0; they take a typical focal length for this camera. Upgrade: carry the focal length over from the neighbouring frames.
    f = np.sqrt(f2) if f2 > 0 else 2450.0
    K = np.array([[f, 0, W / 2], [0, f, H_PX / 2], [0, 0, 1]])
    b = np.linalg.inv(K) @ Hm
    lam = (np.linalg.norm(b[:, 0]) + np.linalg.norm(b[:, 1])) / 2
    r1, r2, t = b[:, 0] / lam, b[:, 1] / lam, b[:, 2] / lam
    P = K @ np.column_stack([r1, r2, np.cross(r1, r2), t]) * lam
    centre = P @ [PITCH[0] / 2, PITCH[1] / 2, 0, 1], P @ [PITCH[0] / 2, PITCH[1] / 2, 1, 1]
    if centre[1][1] / centre[1][2] > centre[0][1] / centre[0][2]:   # +z must move up the image
        P[:, 2] *= -1
    return P


def project3d(P, xyz):
    """Pitch metres (..., 3) -> pixels (..., 2) through a 3x4 camera."""
    h = np.asarray(xyz) @ P[:, :3].T + P[:, 3]
    return h[..., :2] / h[..., 2:]



def fit_track(Ps, uv, t, flight):
    """Fit ground roll at constant velocity (flight=False) or ballistic flight under gravity
    (flight=True) to pixel observations uv (n, 2) at times t (s), cameras Ps (n, 3, 4).

    The roll is linear least squares (u (p3 . X) = p1 . X per frame). The flight starts from it and
    minimises the pixel error itself (the algebraic version collapses onto the camera centre),
    bounded to above the ground and near the pitch (unbounded, it runs off along the ray).
    Returns the fitted positions (n, 3) in pitch metres and the RMS pixel error.
    """
    n = len(t)
    M = np.zeros((n, 4, 6))                       # homogeneous X_t = M_t (x0, y0, z0, vx, vy, vz) + c_t
    M[:, 0, 0] = M[:, 1, 1] = M[:, 2, 2] = 1
    M[:, 0, 3] = M[:, 1, 4] = M[:, 2, 5] = t
    c = np.zeros((n, 4))
    c[:, 3] = 1

    def project(theta):
        h = np.einsum("nij,nj->ni", Ps, M @ theta + c)
        return h[:, :2] / h[:, 2:]

    rows = np.concatenate([uv[:, k, None] * Ps[:, 2] - Ps[:, k] for k in (0, 1)])
    A = np.einsum("ij,ijk->ik", rows, np.concatenate([M, M])[:, :, [0, 1, 3, 4]])
    g = np.linalg.lstsq(A, -rows @ c[0], rcond=None)[0]
    theta = np.array([g[0], g[1], 0, g[2], g[3], 0])
    if flight:
        c[:, 2] = -9.81 * t ** 2 / 2
        lo, hi = [-20, -20, 0, -40, -40, -40], [PITCH[0] + 20, PITCH[1] + 20, 50, 40, 40, 40]
        fits = [least_squares(lambda th: (project(th) - uv).ravel(),
                              np.clip(theta + [0, 0, z, 0, 0, 0], lo, hi), bounds=(lo, hi)) for z in (0.5, 3.0)]
        theta = min(fits, key=lambda r: r.cost).x
    return (M @ theta + c)[:, :3], float(np.sqrt(np.mean((project(theta) - uv) ** 2)))
