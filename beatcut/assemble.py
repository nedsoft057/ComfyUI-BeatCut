"""Beat-synced assembler: conform generated clips onto the shot plan and add
hit-synced FX (punch zoom, flash, RGB split, shake). Pure numpy (+ OpenCV
for fast warps, scipy fallback). Frames are float32 RGB in [0, 1], (H, W, 3).
"""
from __future__ import annotations

import math

import numpy as np

try:
    import cv2
except Exception:  # pragma: no cover
    cv2 = None


# ---------------------------------------------------------------- utilities
def fit_frames(frames: np.ndarray, width: int, height: int) -> np.ndarray:
    """Cover-resize (centre crop) a clip to width x height."""
    n, h, w, _ = frames.shape
    if (w, h) == (width, height):
        return frames
    s = max(width / w, height / h)
    nw, nh = max(width, int(round(w * s))), max(height, int(round(h * s)))
    out = np.empty((n, height, width, 3), np.float32)
    for i in range(n):
        f = _resize(frames[i], nw, nh)
        y0, x0 = (nh - height) // 2, (nw - width) // 2
        out[i] = f[y0:y0 + height, x0:x0 + width]
    return out


def _resize(img, w, h):
    if cv2 is not None:
        return cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA if w < img.shape[1] else cv2.INTER_LINEAR)
    from scipy.ndimage import zoom
    return zoom(img, (h / img.shape[0], w / img.shape[1], 1), order=1)


def _warp(img, scale, dx, dy, angle=0.0):
    """Zoom about centre by `scale`, translate by (dx, dy) px, rotate deg."""
    h, w = img.shape[:2]
    if cv2 is not None:
        m = cv2.getRotationMatrix2D((w / 2, h / 2), angle, scale)
        m[0, 2] += dx
        m[1, 2] += dy
        return cv2.warpAffine(img, m, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    from scipy.ndimage import affine_transform
    c = np.array([h / 2, w / 2])
    mat = np.eye(2) / scale
    off = c - mat @ (c + np.array([dy, dx]))
    return np.stack([affine_transform(img[..., k], mat, offset=off, order=1, mode="reflect") for k in range(3)], -1)


# ------------------------------------------------------------- time remap
def remap_indices(n_in: int, n_out: int, mode: str) -> np.ndarray:
    """Fractional source index for every output frame."""
    if n_out <= 1:
        return np.zeros(max(1, n_out))
    u = np.linspace(0.0, 1.0, n_out)
    if mode == "ramp":
        # fast-in / slow-out speed ramp: rush through the motion on the hit,
        # then settle - the classic music-video ramp
        v = 1.0 - (1.0 - u) ** 2.2
        span = n_in - 1
    elif mode == "slowmo":
        v = u
        span = min(n_in - 1, max(1, int(round(n_out * 0.6))))
    else:
        v = u
        span = min(n_in - 1, n_out - 1) if n_in >= n_out else n_in - 1
    return v * span


def sample(frames: np.ndarray, idx: np.ndarray) -> np.ndarray:
    i0 = np.floor(idx).astype(int)
    i1 = np.minimum(i0 + 1, len(frames) - 1)
    f = (idx - i0).astype(np.float32)[:, None, None, None]
    return frames[i0] * (1 - f) + frames[i1] * f


# ------------------------------------------------------------------- FX
def apply_fx(video: np.ndarray, fps: float, plan: dict, beat_map: dict, intensity: float = 1.0,
             start_offset: float = 0.0) -> np.ndarray:
    """In-place hit-synced FX over the full edit."""
    if intensity <= 0:
        return video
    n = len(video)
    t = np.arange(n) / fps + start_offset
    zoom = np.ones(n)
    flash = np.zeros(n)
    split = np.zeros(n)
    shake = np.zeros(n)

    sec_at = np.empty(n, dtype=object)
    for s in plan["shots"]:
        sec_at[s["frame_start"]: s["frame_start"] + s["frame_count"]] = s["section"]
    sec_at[sec_at == None] = "mid"  # noqa: E711

    def decay(t0, tau):
        d = t - t0
        return np.where(d >= 0, np.exp(-np.maximum(d, 0) / tau), 0.0)

    strength = {"low": 0.0, "mid": 0.35, "build": 0.6, "high": 0.8, "drop": 1.0}
    for bt in beat_map["beats"]:
        k = int(round((bt - start_offset) * fps))
        if 0 <= k < n:
            a = strength.get(sec_at[k], 0.4) * intensity
            zoom += 0.035 * a * decay(bt, 0.12)
    for ht in beat_map["hits"]:
        k = int(round((ht - start_offset) * fps))
        if 0 <= k < n:
            a = strength.get(sec_at[k], 0.4) * intensity
            zoom += 0.04 * a * decay(ht, 0.10)
            shake += 1.0 * a * decay(ht, 0.12)
            if sec_at[k] in ("drop", "high"):
                flash += 0.35 * a * decay(ht, 0.07)
    for dt in beat_map["drops"]:
        flash += 0.9 * intensity * decay(dt, 0.18)
        split += 1.0 * intensity * decay(dt, 0.45)
        shake += 2.0 * intensity * decay(dt, 0.35)
        zoom += 0.12 * intensity * decay(dt, 0.25)
    # drop/high: punch-in jump cuts on every beat inside a shot (editor's
    # "coverage" trick: one generated clip yields several on-beat cuts)
    beats_arr = np.array(beat_map["beats"]) - start_offset
    jump_levels = [1.0, 1.28, 1.12, 1.4]
    for s in plan["shots"]:
        if s["section"] not in ("drop", "high"):
            continue
        a, b = s["frame_start"], s["frame_start"] + s["frame_count"]
        inner = [int(round(bt * fps)) for bt in beats_arr if a + 2 < bt * fps < b - 2]
        lvl = 0
        for k0, k1 in zip(inner, inner[1:] + [b]):
            lvl = (lvl + 1) % len(jump_levels)
            zoom[k0:k1] *= 1.0 + (jump_levels[lvl] - 1.0) * min(1.0, intensity)
    # build: creeping zoom toward the drop
    for s in plan["shots"]:
        if s["section"] == "build":
            a, b = s["frame_start"], s["frame_start"] + s["frame_count"]
            zoom[a:b] += np.linspace(0, 0.06 * intensity, b - a)

    rng = np.random.default_rng(7)
    h, w = video.shape[1:3]
    for i in range(n):
        if zoom[i] > 1.001 or shake[i] > 0.01:
            amp = shake[i] * 0.012 * w
            dx, dy = rng.uniform(-amp, amp), rng.uniform(-amp, amp)
            video[i] = _warp(video[i], float(zoom[i]), dx, dy, angle=float(rng.uniform(-1, 1) * shake[i] * 0.6))
        if split[i] > 0.02:
            px = int(round(split[i] * 0.012 * w))
            if px:
                r = np.roll(video[i][..., 0], px, axis=1)
                b = np.roll(video[i][..., 2], -px, axis=1)
                video[i][..., 0], video[i][..., 2] = r, b
        if flash[i] > 0.01:
            video[i] = video[i] + (1.0 - video[i]) * min(1.0, flash[i])
    np.clip(video, 0, 1, out=video)
    return video


# --------------------------------------------------------------- assemble
def assemble(clips: list, plan: dict, beat_map: dict, intensity: float = 1.0,
             letterbox: float = 0.0, grain: float = 0.0) -> np.ndarray:
    W, H, fps = plan["width"], plan["height"], plan["fps"]
    shots = plan["shots"]
    if len(clips) < len(shots):
        raise ValueError(f"BeatCut: plan has {len(shots)} shots but got {len(clips)} clips")
    out = np.zeros((plan["total_frames"], H, W, 3), np.float32)
    for s, clip in zip(shots, clips):
        clip = np.asarray(clip, dtype=np.float32)
        if clip.ndim == 3:
            clip = clip[None]
        clip = fit_frames(clip[..., :3], W, H)
        idx = remap_indices(len(clip), s["frame_count"], s["remap"])
        a = s["frame_start"]
        seg = sample(clip, idx)
        out[a:a + len(seg)] = seg[: max(0, len(out) - a)]
    apply_fx(out, fps, plan, beat_map, intensity)
    if grain > 0:
        rng = np.random.default_rng(3)
        for i in range(len(out)):
            out[i] += rng.normal(0, grain * 0.04, (H, W, 1)).astype(np.float32)
        np.clip(out, 0, 1, out=out)
    if letterbox > 0:
        bar = int(round(H * letterbox / 2))
        if bar:
            out[:, :bar] = 0
            out[:, H - bar:] = 0
    return out


def slice_audio(waveform: np.ndarray, sample_rate: int, start: float, duration: float) -> np.ndarray:
    a = int(round(start * sample_rate))
    b = a + int(round(duration * sample_rate))
    return waveform[..., a:b]
