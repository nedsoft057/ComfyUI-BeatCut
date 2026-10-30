"""Render the beat map + shot plan as a timeline image (for previews/demos)."""
from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw, ImageFont

SECTION_COLORS = {
    "low": (70, 110, 190), "mid": (90, 160, 150), "build": (235, 170, 60),
    "high": (230, 110, 70), "drop": (240, 60, 90),
}
BG, FG, GRID = (14, 14, 18), (235, 235, 240), (60, 60, 70)


def _font(sz):
    for name in ("DejaVuSans-Bold.ttf", "DejaVuSans.ttf", "Arial.ttf"):
        try:
            return ImageFont.truetype(name, sz)
        except Exception:
            continue
    return ImageFont.load_default()


def render(beat_map: dict, plan: dict | None = None, width: int = 1600, thumbs: list | None = None) -> np.ndarray:
    has_plan = plan is not None
    H = 470 if has_plan else 300
    if thumbs:
        H += 120
    img = Image.new("RGB", (width, H), BG)
    d = ImageDraw.Draw(img)
    f_big, f_sm = _font(22), _font(13)
    dur = beat_map["duration"]
    L, R = 40, width - 20
    X = lambda t: L + (R - L) * (t / dur)  # noqa: E731

    title = f"BeatCut  ·  {beat_map['tempo']:.1f} BPM  ·  {len(beat_map['beats'])} beats  ·  {dur:.1f}s"
    if has_plan:
        title += f"  ·  {len(plan['shots'])} shots"
    d.text((L, 14), title, fill=FG, font=f_big)

    # section band
    y0 = 56
    for b in beat_map["bars"]:
        d.rectangle([X(b["start"]), y0, X(b["end"]), y0 + 16], fill=SECTION_COLORS.get(b["label"], GRID))
    d.text((L, y0 + 18), "sections", fill=(150, 150, 160), font=f_sm)

    # waveform energy + onset
    y1, hh = 100, 120
    t = np.array(beat_map["env_t"])
    rms = np.array(beat_map["env_rms"])
    on = np.array(beat_map["env_onset"])
    mid = y1 + hh / 2
    for ti, r in zip(t, rms):
        x = X(ti)
        d.line([x, mid - r * hh / 2, x, mid + r * hh / 2], fill=(90, 90, 110))
    pts = [(X(ti), y1 + hh - o * hh) for ti, o in zip(t, on)]
    if len(pts) > 1:
        d.line(pts, fill=(160, 200, 255), width=1)
    # beats / downbeats / drops
    for bt in beat_map["beats"]:
        d.line([X(bt), y1 + hh + 4, X(bt), y1 + hh + 12], fill=(170, 170, 180))
    for db in beat_map["downbeats"]:
        d.line([X(db), y1 + hh + 2, X(db), y1 + hh + 20], fill=FG, width=2)
    for dr in beat_map["drops"]:
        d.line([X(dr), y0, X(dr), y1 + hh + 24], fill=SECTION_COLORS["drop"], width=3)
        d.text((X(dr) + 4, y1 + 2), "DROP", fill=SECTION_COLORS["drop"], font=f_sm)
    for ht in beat_map["hits"]:
        d.ellipse([X(ht) - 2, y1 + hh + 24, X(ht) + 2, y1 + hh + 28], fill=(255, 220, 90))
    d.text((L, y1 + hh + 32), "beats · downbeats · hits", fill=(150, 150, 160), font=f_sm)

    if has_plan:
        y2 = y1 + hh + 60
        for s in plan["shots"]:
            xa, xb = X(s["t0"]), X(s["t1"])
            c = SECTION_COLORS.get(s["section"], GRID)
            d.rectangle([xa + 1, y2, xb - 1, y2 + 70], fill=tuple(int(v * 0.55) for v in c), outline=c, width=2)
            if xb - xa > 34:
                d.text((xa + 5, y2 + 4), f"#{s['index']}", fill=FG, font=f_sm)
                d.text((xa + 5, y2 + 22), s["remap"], fill=(220, 220, 230), font=f_sm)
                d.text((xa + 5, y2 + 40), f"{s['frame_count']}f", fill=(200, 200, 210), font=f_sm)
        d.text((L, y2 + 76), "shot plan  (cut on the beat · remap: slowmo / linear / ramp)", fill=(150, 150, 160), font=f_sm)
        if thumbs:
            y3 = y2 + 100
            for s, th in zip(plan["shots"], thumbs):
                xa, xb = int(X(s["t0"])) + 1, int(X(s["t1"])) - 1
                wbox = max(1, xb - xa)
                im = Image.fromarray((np.clip(th, 0, 1) * 255).astype(np.uint8))
                hbox = 110
                s_ = max(wbox / im.width, hbox / im.height)
                im = im.resize((max(1, int(im.width * s_)), max(1, int(im.height * s_))))
                cx, cy = (im.width - wbox) // 2, (im.height - hbox) // 2
                img.paste(im.crop((cx, cy, cx + wbox, cy + hbox)), (xa, y3))
    return np.asarray(img).astype(np.float32) / 255.0
