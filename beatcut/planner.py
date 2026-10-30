"""Beat-aligned shot planner: turns a BeatMap + creative brief into a shot list.

Cinematography grammar:
  * shot length follows energy: calm bars hold for a bar or two, builds cut
    every 2 beats, drops cut on every beat with a hero shot on the drop.
  * every cut lands exactly on a beat (frame-quantised globally, no drift).
  * framing / camera move / action vocabularies are chosen by section and
    never repeat back-to-back.
  * the generator only renders what it must: each shot gets a generation
    length (LTX 8n+1 frames) and a time-remap (slow-mo for calm shots,
    speed-ramp for hits) so short clips fill long holds and vice versa.
"""
from __future__ import annotations

import math
import random

FRAMINGS = {
    "low":   ["wide establishing shot", "silhouette wide shot", "slow aerial shot", "medium wide shot"],
    "mid":   ["medium shot", "over-the-shoulder shot", "low-angle medium shot", "profile shot"],
    "build": ["tight medium shot", "low-angle shot", "dutch-angle shot", "close-up"],
    "high":  ["close-up", "low-angle hero shot", "medium close-up", "tracking shot"],
    "drop":  ["extreme close-up", "low-angle hero shot", "fisheye close-up", "dynamic wide shot", "close-up"],
}
MOVES = {
    "low":   ["slow dolly in", "slow pan right", "gentle crane up", "static camera, subtle drift"],
    "mid":   ["steady push in", "smooth orbit left", "handheld follow", "slow pull back"],
    "build": ["accelerating push in", "rising crane", "slow spin", "creeping zoom"],
    "high":  ["fast orbit", "whip pan", "quick push in", "tracking alongside"],
    "drop":  ["crash zoom in", "fast 360 orbit", "whip pan", "explosive pull back", "shaky handheld rush"],
}
ACTIONS = {
    "low":   ["stands still, hair and clothes moving in the wind", "slowly turns their head", "breathes calmly, eyes closed"],
    "mid":   ["walks toward the camera", "turns to face the camera", "looks up confidently"],
    "build": ["clenches fists, tension rising", "steps forward with intent", "looks straight into the lens"],
    "high":  ["moves energetically", "spins around", "gestures boldly at the camera"],
    "drop":  ["explodes into motion", "jumps toward the camera", "strikes a powerful pose as particles burst",
              "dances with intense energy"],
}
LIGHT = {
    "low": "moody soft light, haze", "mid": "cinematic key light", "build": "rising rim light, flicker",
    "high": "bold neon light", "drop": "strobing high-contrast light, lens flares, sparks",
}
# time-remap modes the assembler understands
REMAP = {"low": "slowmo", "mid": "linear", "build": "linear", "high": "ramp", "drop": "ramp"}


def ltx_frames(n: int, lo: int = 17, hi: int = 97) -> int:
    """Round up to LTX's 8n+1 frame grid."""
    n = max(lo, min(hi, n))
    return int(math.ceil((n - 1) / 8) * 8 + 1)


def _scenes(brief: str):
    parts = [p.strip() for p in brief.replace("|", "\n").splitlines()]
    return [p for p in parts if p] or ["a striking cinematic location"]


def _pick(rng, options, last):
    opts = [o for o in options if o != last] or options
    return rng.choice(opts)


def plan(beat_map: dict, subject: str, brief: str, style: str, fps: float = 24.0,
         width: int = 768, height: int = 512, max_shots: int = 12, min_beats: int = 1,
         max_gen_seconds: float = 3.0, seed: int = 0, pace: float = 1.0, drop_look: str = "") -> dict:
    """pace > 1 cuts faster, < 1 holds longer."""
    rng = random.Random(seed)
    beats = beat_map["beats"]
    bars = beat_map["bars"]
    dur = beat_map["duration"]
    if not beats:
        raise ValueError("BeatCut: beat map has no beats")

    # 1) desired beats-per-shot for each bar
    base = {"low": 4, "mid": 4, "build": 2, "high": 2, "drop": 1}
    cuts = []  # (beat_index, label)
    for b in bars:
        lab = b["label"]
        step = max(min_beats, int(round(base[lab] / max(pace, 1e-3))))
        if lab == "low" and b["n_beats"] >= 4 and pace < 1:
            step = 8
        i0, n = b["beat_index"], b["n_beats"]
        for k in range(0, n, step):
            cuts.append((i0 + k, lab))
    # ensure the start of the edit is a cut
    if not cuts or cuts[0][0] != 0:
        cuts.insert(0, (0, bars[0]["label"] if bars else "mid"))
    drop_starts = set(beat_map.get("drops", []))

    # 2) merge to respect max_shots: repeatedly merge the shortest calm pair
    def shot_bounds(cs):
        out = []
        for j, (bi, lab) in enumerate(cs):
            t0 = 0.0 if j == 0 else beats[bi]
            t1 = beats[cs[j + 1][0]] if j + 1 < len(cs) else dur
            out.append((t0, t1, lab, bi))
        return out

    prio = {"low": 0, "mid": 1, "build": 2, "high": 3, "drop": 4}
    while len(cuts) > max_shots:
        sb = shot_bounds(cuts)
        best, best_score = None, None
        for j in range(1, len(cuts)):
            if beats[cuts[j][0]] in drop_starts:
                continue  # never remove the cut on the drop
            # merge the pair that yields the shortest combined shot; calm
            # sections merge first, high-energy cuts are protected
            combined = (sb[j][1] - sb[j][0]) + (sb[j - 1][1] - sb[j - 1][0])
            score = combined * (1.0 + 0.6 * prio[cuts[j][1]])
            if best_score is None or score < best_score:
                best, best_score = j, score
        if best is None:
            break
        cuts.pop(best)

    # 3) build shots, frame-quantised on a global grid (no drift)
    scenes = _scenes(brief)
    shots = []
    last = {"framing": None, "move": None, "action": None}
    section_idx, prev_lab = -1, None
    total_frames = int(round(dur * fps))
    for j, (t0, t1, lab, bi) in enumerate(shot_bounds(cuts)):
        if lab != prev_lab:
            section_idx += 1
            prev_lab = lab
        f0, f1 = int(round(t0 * fps)), int(round(t1 * fps)) if j + 1 < len(cuts) else total_frames
        n_out = max(1, f1 - f0)
        is_drop = any(abs(t0 - d) < 1e-3 for d in drop_starts)
        sec = "drop" if is_drop else lab
        framing = "low-angle hero shot" if is_drop else _pick(rng, FRAMINGS[sec], last["framing"])
        move = "crash zoom in" if is_drop else _pick(rng, MOVES[sec], last["move"])
        action = _pick(rng, ACTIONS[sec], last["action"])
        last.update(framing=framing, move=move, action=action)
        scene = scenes[section_idx % len(scenes)]
        remap = REMAP[sec]
        # how many source frames to generate: slow-mo holds need fewer source
        # frames per output frame; ramps want extra motion to compress.
        speed = {"slowmo": 0.6, "linear": 1.0, "ramp": 1.6}[remap]
        gen = ltx_frames(int(math.ceil(n_out * speed)) + 1, hi=ltx_frames(int(max_gen_seconds * fps)))
        energy = float(sum(beat_map["beat_energy"][bi: bi + 4]) / max(1, len(beat_map["beat_energy"][bi: bi + 4])))
        first_drop = min(drop_starts) if drop_starts else None
        transformed = bool(drop_look) and first_drop is not None and t0 >= first_drop - 1e-3
        if transformed:
            # the drop is the transformation: same identity, new form
            kf = (f"{framing} of {subject}, {drop_look}, {scene}. {LIGHT[sec]}. {style}. "
                  f"Keep the subject's face and identity from the reference image, but transform their look as described.")
            mo = (f"{move}. {subject}, {drop_look}, {action}, {scene}. {LIGHT[sec]}. {style}. "
                  f"cinematic, sharp focus, high detail, energetic motion.")
        else:
            kf = (f"{framing} of {subject}, {scene}. {LIGHT[sec]}. {style}. "
                  f"Keep the subject's identity, face, outfit and colors exactly as in the reference image.")
            mo = (f"{move}. {subject} {action}, {scene}. {LIGHT[sec]}. {style}. "
                  f"cinematic, sharp focus, high detail, smooth natural motion.")
        shots.append({
            "index": j, "section": sec, "t0": round(t0, 4), "t1": round(t1, 4),
            "frame_start": f0, "frame_count": n_out, "gen_frames": gen, "remap": remap,
            "energy": round(energy, 3), "framing": framing, "move": move, "action": action, "scene": scene,
            "keyframe_prompt": kf, "motion_prompt": mo, "seed": seed * 1000 + j * 7919 + 17,
            "is_drop": is_drop, "transformed": transformed,
        })
    return {"fps": fps, "width": width, "height": height, "duration": dur, "total_frames": total_frames,
            "subject": subject, "brief": brief, "style": style, "shots": shots}


def describe(p: dict) -> str:
    lines = [f"{len(p['shots'])} shots, {p['total_frames']} frames @ {p['fps']}fps "
             f"({p['duration']:.2f}s), gen budget {sum(s['gen_frames'] for s in p['shots'])} frames"]
    for s in p["shots"]:
        lines.append(f"#{s['index']:02d} {s['t0']:6.2f}-{s['t1']:6.2f}s {s['section']:5s} "
                     f"out {s['frame_count']:3d}f gen {s['gen_frames']:3d}f {s['remap']:6s} | "
                     f"{s['framing']} / {s['move']}" + ("  << DROP" if s["is_drop"] else "") + ("  [transformed]" if s.get("transformed") else ""))
    return "\n".join(lines)
