"""Beat, bar, energy and drop analysis in pure numpy/scipy.

No librosa dependency: onset envelope via log-spectral flux, tempo via
prior-weighted autocorrelation, beats via Ellis-style dynamic programming,
downbeats via bass-onset phase voting, sections via bar-energy novelty.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field, asdict

import numpy as np
from scipy.signal import resample_poly, get_window
from scipy.ndimage import uniform_filter1d

SR = 22050
N_FFT = 2048
HOP = 512


@dataclass
class BeatMap:
    duration: float
    start: float
    tempo: float
    beats: list
    downbeats: list
    beat_energy: list          # 0..1 per beat (RMS, percentile-normalised)
    beat_strength: list        # 0..1 onset strength at each beat
    bars: list                 # [{start, end, energy, label}]
    drops: list                # times (s) of detected drops
    hits: list                 # accent times (s) for FX
    env_t: list = field(default_factory=list)   # downsampled timeline for viz
    env_rms: list = field(default_factory=list)
    env_onset: list = field(default_factory=list)

    def to_dict(self):
        return asdict(self)

    def summary(self) -> str:
        labels = [b["label"] for b in self.bars]
        return (f"tempo {self.tempo:.1f} BPM | {len(self.beats)} beats | {len(self.bars)} bars | "
                f"{len(self.drops)} drop(s) at {', '.join(f'{d:.2f}s' for d in self.drops) or '-'} | "
                f"{len(self.hits)} hits | sections: {' '.join(_rle(labels))}")


def _rle(xs):
    out, prev, n = [], None, 0
    for x in xs + [None]:
        if x == prev:
            n += 1
        else:
            if prev is not None:
                out.append(f"{prev}x{n}")
            prev, n = x, 1
    return out


def to_mono(waveform: np.ndarray) -> np.ndarray:
    w = np.asarray(waveform, dtype=np.float32)
    while w.ndim > 2:
        w = w[0]
    if w.ndim == 2:
        w = w.mean(axis=0)
    return w


def _resample(y, sr):
    if sr == SR:
        return y
    g = math.gcd(int(sr), SR)
    return resample_poly(y, SR // g, int(sr) // g).astype(np.float32)


def _stft_mag(y):
    win = get_window("hann", N_FFT).astype(np.float32)
    pad = N_FFT // 2
    y = np.pad(y, (pad, pad), mode="reflect")
    n = 1 + (len(y) - N_FFT) // HOP
    idx = np.arange(N_FFT)[None, :] + HOP * np.arange(n)[:, None]
    frames = y[idx] * win[None, :]
    return np.abs(np.fft.rfft(frames, axis=1)).astype(np.float32)  # (T, F)


def _norm01(x, lo_q=5, hi_q=95):
    x = np.asarray(x, dtype=np.float64)
    if x.size == 0:
        return x
    lo, hi = np.percentile(x, lo_q), np.percentile(x, hi_q)
    if hi - lo < 1e-9:
        return np.zeros_like(x)
    return np.clip((x - lo) / (hi - lo), 0, 1)


def onset_envelope(mag):
    logm = np.log1p(100.0 * mag)
    flux = np.maximum(0.0, np.diff(logm, axis=0, prepend=logm[:1]))
    env = flux.mean(axis=1)
    # bass-band flux (≈ < 150 Hz) for downbeat phase
    freqs = np.fft.rfftfreq(N_FFT, 1.0 / SR)
    bass = flux[:, freqs < 150].mean(axis=1)
    # remove slow trend, half-wave rectify
    env = np.maximum(0, env - uniform_filter1d(env, 16))
    bass = np.maximum(0, bass - uniform_filter1d(bass, 16))
    return env / (env.std() + 1e-9), bass / (bass.std() + 1e-9)


def estimate_tempo(env, fps, bpm_min=70, bpm_max=180, prior_bpm=120.0, hint=0.0):
    env = env - env.mean()
    ac = np.correlate(env, env, mode="full")[len(env) - 1:]
    ac /= ac[0] + 1e-9
    lags = np.arange(len(ac))
    with np.errstate(divide="ignore"):
        bpm = 60.0 * fps / np.maximum(lags, 1e-9)
    valid = (bpm >= bpm_min) & (bpm <= bpm_max)
    centre = hint if hint > 0 else prior_bpm
    width = 0.25 if hint > 0 else 1.0
    prior = np.exp(-0.5 * (np.log2(np.maximum(bpm, 1e-9) / centre) / width) ** 2)
    # reward lags whose double also correlates (metrical consistency)
    score = ac.copy()
    def at(m):
        x = lags * m
        i = np.clip(np.floor(x).astype(int), 0, len(ac) - 2)
        f = x - np.floor(x)
        return ac[i] * (1 - f) + ac[i + 1] * f
    # comb over metrical relatives: double, half and quadruple periods
    score = score + 0.5 * at(2.0) + 0.5 * at(0.5) + 0.25 * at(4.0)
    score = np.where(valid, score * prior, -np.inf)
    lag = int(np.argmax(score))
    # parabolic refinement
    if 1 <= lag < len(score) - 1 and np.isfinite(score[lag - 1]) and np.isfinite(score[lag + 1]):
        a, b, c = score[lag - 1], score[lag], score[lag + 1]
        d = 0.5 * (a - c) / (a - 2 * b + c + 1e-12)
        lagf = lag + float(np.clip(d, -0.5, 0.5))
    else:
        lagf = float(lag)
    return 60.0 * fps / lagf, lagf


def track_beats(env, period, tightness=100.0):
    """Ellis (2007) dynamic-programming beat tracker on the onset envelope."""
    n = len(env)
    local = uniform_filter1d(env, max(1, int(period / 8)))
    score = local.astype(np.float64).copy()
    back = -np.ones(n, dtype=np.int64)
    lo, hi = int(round(period / 2)), int(round(period * 2))
    offs = np.arange(-hi, -lo + 1)
    penalty = -tightness * (np.log(-offs / period)) ** 2
    for t in range(n):
        cand = t + offs
        ok = cand >= 0
        if not ok.any():
            continue
        c = cand[ok]
        vals = score[c] + penalty[ok]
        j = int(np.argmax(vals))
        if vals[j] > 0:
            score[t] = local[t] + vals[j]
            back[t] = c[j]
    # start from best score within last period
    tail = max(1, int(period))
    t = n - tail + int(np.argmax(score[-tail:]))
    beats = [t]
    while back[t] >= 0:
        t = back[t]
        beats.append(t)
    beats = np.array(beats[::-1])
    # trim weak leading/trailing beats
    thr = 0.3 * np.median(local[beats]) if len(beats) else 0
    keep = np.where(local[beats] > thr)[0]
    if len(keep):
        beats = beats[keep[0]: keep[-1] + 1]
    return beats


def analyze(waveform, sample_rate, start=0.0, duration=0.0, bpm_hint=0.0,
            beats_per_bar=4, drop_sensitivity=0.5) -> BeatMap:
    y = to_mono(waveform)
    sr = int(sample_rate)
    s0 = int(max(0.0, start) * sr)
    y = y[s0:]
    if duration and duration > 0:
        y = y[: int(duration * sr)]
    if len(y) < sr:
        raise ValueError("BeatCut: need at least 1 second of audio")
    dur = len(y) / sr
    y = _resample(y, sr)
    fps = SR / HOP
    mag = _stft_mag(y)
    env, bass = onset_envelope(mag)
    rms = np.sqrt((mag ** 2).mean(axis=1))
    _f = np.fft.rfftfreq(N_FFT, 1.0 / SR)
    low_rms = np.sqrt((mag[:, _f < 160] ** 2).mean(axis=1))  # kick + sub-bass energy
    tempo, period = estimate_tempo(env, fps, hint=bpm_hint)
    # bass-weighted envelope for tracking: kicks/bass define the pulse,
    # broadband hats otherwise pull the tracker onto 8th-note offbeats
    env_track = env + 1.5 * bass
    bframes = track_beats(env_track, period)
    if len(bframes) < 4:
        bframes = np.arange(0, len(env), period).astype(int)
    # on-beat vs off-beat check: DP can lock onto the 8th-note offbeats when
    # hats dominate; kicks/bass decide which lattice is the real pulse.
    half = int(round(period / 2))
    sm = uniform_filter1d(bass, 3)
    shifted = bframes + half
    shifted = shifted[shifted < len(sm)]
    if len(shifted) > 4 and sm[shifted].mean() > 1.05 * sm[bframes[:len(shifted)]].mean():
        bframes = shifted
    beats_t = bframes / fps

    # per-beat energy & strength
    edges = np.concatenate([bframes, [len(rms)]])
    be = np.array([rms[edges[i]:max(edges[i] + 1, edges[i + 1])].mean() for i in range(len(bframes))])
    be = _norm01(np.log1p(be * 50))
    bs = _norm01(uniform_filter1d(env, 3)[bframes])

    # downbeat phase: which beat-in-bar has most bass onset + energy rise
    bb = beats_per_bar
    bass_at = uniform_filter1d(bass, 3)[bframes]
    phase_scores = [bass_at[p::bb].mean() + 0.5 * bs[p::bb].mean() for p in range(min(bb, len(bframes)))]
    phase = int(np.argmax(phase_scores))
    down_idx = list(range(phase, len(bframes), bb))
    if phase > 0:
        down_idx = [0] + down_idx  # pickup bar
    downbeats_t = beats_t[down_idx]

    # bars
    bars = []
    for k, i0 in enumerate(down_idx):
        i1 = down_idx[k + 1] if k + 1 < len(down_idx) else len(bframes)
        t0 = float(beats_t[i0])
        t1 = float(beats_t[i1]) if i1 < len(bframes) else dur
        bars.append({"start": t0, "end": t1, "energy": float(be[i0:i1].mean()) if i1 > i0 else float(be[i0]),
                     "beat_index": int(i0), "n_beats": int(i1 - i0)})
    bar_e = np.array([b["energy"] for b in bars])
    # bass energy per bar: drops are where the low end arrives, not where a
    # riser gets loud
    edges_b = [int(bframes[b["beat_index"]]) for b in bars] + [len(low_rms)]
    bar_low = np.array([low_rms[edges_b[k]:max(edges_b[k] + 1, edges_b[k + 1])].mean() for k in range(len(bars))])
    bar_low = _norm01(np.log1p(bar_low * 50), 0, 100)

    # drops: big positive jump in bar energy into a high-energy region
    drops = []
    if len(bars) >= 3:
        drive = 0.35 * bar_e + 0.65 * bar_low
        prev2 = np.array([drive[max(0, i - 2):i].mean() if i > 0 else drive[0] for i in range(len(bars))])
        prev1 = np.concatenate([[drive[0]], drive[:-1]])
        jump = 0.5 * (drive - prev1) + 0.5 * (drive - prev2)
        thr_jump = 0.10 + 0.2 * (1 - drop_sensitivity)
        hi = np.percentile(drive, 55)
        cand = sorted((i for i in range(1, len(bars)) if jump[i] > thr_jump and drive[i] >= hi),
                      key=lambda i: -jump[i])
        for i in cand:  # strongest first, keep drops >= 4 bars apart
            if all(abs(i - j) >= 4 for j in drops):
                drops.append(i)
        drops.sort()
        if not drops:  # fall back to the single biggest jump
            i = int(np.argmax(jump[1:]) + 1)
            if jump[i] > 0.08:
                drops.append(i)

    # section labels
    hiq, loq = np.percentile(bar_e, 66), np.percentile(bar_e, 33)
    drop_set = set(drops)
    in_drop = False
    for i, b in enumerate(bars):
        e = bar_e[i]
        if i in drop_set:
            in_drop = True
            b["label"] = "drop"
            continue
        if in_drop and e >= loq + 0.5 * (hiq - loq):
            b["label"] = "drop"
            continue
        in_drop = False
        nxt = [j for j in drops if j > i]
        if nxt and nxt[0] - i <= 2 and e > loq:
            b["label"] = "build"
        elif e >= hiq:
            b["label"] = "high"
        elif e <= loq:
            b["label"] = "low"
        else:
            b["label"] = "mid"

    # hits: strong on-beat accents (top quantile) plus every downbeat in drops
    q = np.percentile(bs, 75) if len(bs) else 1
    hits = set(float(beats_t[i]) for i in range(len(bframes)) if bs[i] >= q)
    for b in bars:
        if b["label"] == "drop":
            hits.add(b["start"])
    hits = sorted(hits)

    # downsampled envelopes for timeline rendering
    step = max(1, len(rms) // 1200)
    env_t = (np.arange(0, len(rms), step) / fps).round(3).tolist()
    env_rms = _norm01(rms[::step], 1, 99).round(4).tolist()
    env_on = _norm01(env[::step], 1, 99).round(4).tolist()

    return BeatMap(
        duration=float(dur), start=float(start), tempo=float(tempo),
        beats=[float(t) for t in beats_t], downbeats=[float(t) for t in downbeats_t],
        beat_energy=[float(x) for x in be], beat_strength=[float(x) for x in bs],
        bars=bars, drops=[float(bars[i]["start"]) for i in drops], hits=hits,
        env_t=env_t, env_rms=env_rms, env_onset=env_on,
    )
