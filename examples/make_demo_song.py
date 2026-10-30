"""Procedurally synthesise the demo track (original, license-free): 124 BPM,
intro -> build -> drop. `python make_demo_song.py demo_song.wav`"""
import sys, wave
import numpy as np
SR, BPM = 44100, 124.0
SPB = 60 / BPM
BARS = {"intro": 4, "build": 2, "drop": 6}
rng = np.random.default_rng(1)
n_bars = sum(BARS.values()); total = int((n_bars * 4 * SPB + 1.0) * SR)
L = np.zeros(total); R = np.zeros(total)
def note(f): return 440 * 2 ** ((f - 69) / 12)
def add(sig, t, pan=0.0, g=1.0):
    i = int(t * SR); j = min(total, i + len(sig)); s = sig[: j - i] * g
    L[i:j] += s * (1 - pan) ; R[i:j] += s * (1 + pan)
def env(n, a=0.005, d=0.3):
    k = np.arange(n) / SR; return np.minimum(1, k / a) * np.exp(-k / d)
def saw(f, dur, det=0.0):
    k = np.arange(int(dur * SR)) / SR
    return sum(2 * ((k * f * (1 + x)) % 1) - 1 for x in (-det, 0, det)) / 3
CH = [[57, 60, 64], [53, 57, 60], [48, 52, 55], [55, 59, 62]]  # Am F C G
BASS = [33, 29, 36, 31]
t = 0.0
def kick():
    n = int(0.35 * SR); k = np.arange(n) / SR; f = 45 + 110 * np.exp(-k / 0.035)
    return np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-k / 0.16)
def noise(dur, d): n = int(dur * SR); return rng.standard_normal(n) * env(n, 0.001, d)
def hp(x):  # crude high-pass
    return x - np.convolve(x, np.ones(8) / 8, "same")
for sect, nb in BARS.items():
    for b in range(nb):
        bar_t = t + b * 4 * SPB; ci = b % 4; ch = CH[ci]
        prog = b / max(1, nb - 1)
        if sect == "intro":
            pad = sum(saw(note(m + 12), 4 * SPB, 0.004) for m in ch) / 3
            n = len(pad); pad *= np.minimum(1, np.arange(n) / (0.4 * SR)) * np.minimum(1, (n - np.arange(n)) / (0.3 * SR))
            pad = np.convolve(pad, np.ones(24) / 24, "same")
            add(pad, bar_t, 0, 0.22)
            for e in range(8):
                add(hp(noise(0.04, 0.012)), bar_t + e * SPB / 2 + SPB / 2 * (e % 2 == 0) * 0, 0.3, 0.10 if e % 2 else 0.05)
            add(np.sin(2*np.pi*note(ch[0]+24)*np.arange(int(.5*SR))/SR)*env(int(.5*SR),.002,.15), bar_t, -.2, .15)
        elif sect == "build":
            div = 4 if b == 0 else 8
            for e in range(4 * div // 2 * 2 // 2 if False else div * 2):
                tt = bar_t + e * (4 * SPB) / (div * 2)
                add(noise(0.09, 0.035), tt, 0, 0.12 + 0.22 * (e / (div * 2) + b) / 2)
            riser = saw(220 * (1 + 2 * prog), 4 * SPB, 0.01) * np.linspace(0.05 + 0.4 * b, 0.45 + 0.4 * b, int(4 * SPB * SR))
            add(np.convolve(riser, np.ones(6) / 6, "same"), bar_t, 0, 0.12)
            add(hp(rng.standard_normal(int(4*SPB*SR))) * np.linspace(0.0, 0.15, int(4*SPB*SR)) * (0.5 + b), bar_t, 0, 0.5)
        else:  # drop
            for q in range(4):
                bt = bar_t + q * SPB
                add(kick(), bt, 0, 0.95)
                if q % 2 == 1:
                    add(noise(0.18, 0.06), bt, 0, 0.35)          # clap
                add(hp(noise(0.05, 0.015)), bt + SPB / 2, 0.3, 0.14)  # offbeat hat
                bs = saw(note(BASS[ci]), SPB * 0.9, 0.006) * env(int(SPB * 0.9 * SR), 0.004, 0.25)
                add(np.convolve(bs, np.ones(10) / 10, "same"), bt + SPB * 0.5 * 0, 0, 0.0)
                # side-chained offbeat bass
                bo = saw(note(BASS[ci] + 12), SPB * 0.45, 0.008) * env(int(SPB * 0.45 * SR), 0.01, 0.2)
                add(bo, bt + SPB / 2, 0, 0.32)
            for s in (0, 1.5, 3):  # chord stabs
                st = sum(saw(note(m + 12), 0.22, 0.01) for m in ch) / 3 * env(int(0.22 * SR), 0.002, 0.09)
                add(st, bar_t + s * SPB, 0.25 * (1 if s else -1), 0.3)
    t += nb * 4 * SPB
mix = np.stack([L, R]); mix /= np.abs(mix).max() * 1.05
mix = np.tanh(mix * 1.3) / np.tanh(1.3)
out = sys.argv[1] if len(sys.argv) > 1 else "demo_song.wav"
with wave.open(out, "wb") as w:
    w.setnchannels(2); w.setsampwidth(2); w.setframerate(SR)
    w.writeframes((mix.T * 32000).astype(np.int16).tobytes())
print("wrote", out, f"{total / SR:.1f}s")
