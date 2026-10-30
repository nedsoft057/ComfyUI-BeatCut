import numpy as np
def synth_track(bpm=128.0, sr=44100, intro_bars=4, build_bars=2, drop_bars=4, seed=0):
    """Synthetic EDM-ish track: quiet hats intro, rising build, kick+bass drop."""
    rng = np.random.default_rng(seed)
    spb = 60.0 / bpm
    nbars = intro_bars + build_bars + drop_bars
    dur = nbars * 4 * spb + 0.5
    n = int(dur * sr); y = np.zeros(n, np.float32); t = np.arange(n) / sr
    def env(len_s, tau):
        k = np.arange(int(len_s * sr)) / sr; return np.exp(-k / tau)
    for b in range(nbars * 4):
        t0 = 0.25 + b * spb; i0 = int(t0 * sr); bar = b // 4
        sect = "intro" if bar < intro_bars else ("build" if bar < intro_bars + build_bars else "drop")
        # hat on every 8th
        for h in (0, 0.5):
            j = int((t0 + h * spb) * sr); L = int(0.03 * sr)
            amp = 0.05 if sect == "intro" else (0.08 + 0.1 * (b % 8) / 8 if sect == "build" else 0.12)
            y[j:j+L] += amp * rng.standard_normal(L).astype(np.float32) * env(0.03, 0.01)[:L]
        if sect == "intro" and b % 4 == 0:  # soft pad pluck on downbeat
            L = int(0.4 * sr); k = np.arange(L) / sr
            y[i0:i0+L] += 0.08 * np.sin(2*np.pi*220*k) * np.exp(-k/0.2)
        if sect == "build":  # snare roll + riser
            for s in range(2 if bar == intro_bars else 4):
                j = int((t0 + s * spb / (2 if bar == intro_bars else 4)) * sr); L = int(0.08 * sr)
                y[j:j+L] += 0.15 * rng.standard_normal(L).astype(np.float32) * env(0.08, 0.03)[:L]
        if sect == "drop":  # kick + bass
            L = int(0.25 * sr); k = np.arange(L) / sr
            f = 50 + 120 * np.exp(-k / 0.03)
            y[i0:i0+L] += 0.9 * np.sin(2*np.pi*np.cumsum(f)/sr) * np.exp(-k/0.12)
            L2 = int(spb * sr * 0.9); k2 = np.arange(L2) / sr
            y[i0:i0+L2] += 0.25 * np.sign(np.sin(2*np.pi*55*k2)) * np.minimum(1, k2/0.05) * np.exp(-k2/0.4)
    rs = 0.25 + intro_bars * 4 * spb
    re = rs + build_bars * 4 * spb
    m = (t >= rs) & (t < re)
    y[m] += 0.04 * ((t[m]-rs)/(re-rs)) * rng.standard_normal(m.sum()).astype(np.float32)
    y = np.clip(y, -1, 1)
    return np.stack([y, y]), sr, {"bpm": bpm, "first_beat": 0.25, "drop_time": 0.25 + (intro_bars+build_bars)*4*spb}
