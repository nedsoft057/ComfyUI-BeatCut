"""CPU end-to-end: synthetic song -> analyze -> plan -> fake clips -> assemble -> mp4."""
import sys, os, subprocess, json, time
sys.path[:0] = [os.path.dirname(os.path.dirname(__file__)), os.path.dirname(__file__)]
import numpy as np
from synth import synth_track
from beatcut.audio import analyze
from beatcut.planner import plan as make_plan, describe
from beatcut.assemble import assemble, slice_audio
from beatcut.timeline import render
from PIL import Image, ImageDraw

def fake_clip(n, w, h, hue, label):
    """Moving gradient + label, stands in for a generated LTX clip."""
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    out = np.empty((n, h, w, 3), np.float32)
    for i in range(n):
        ph = i / max(1, n - 1)
        base = np.stack([0.5 + 0.5*np.sin(xx/90 + ph*3 + hue), 0.5 + 0.5*np.sin(yy/70 + hue*2), 0.5 + 0.5*np.sin((xx+yy)/120 - ph*2)], -1)
        out[i] = base * 0.8
    im = Image.fromarray((out[0]*255).astype(np.uint8)); d = ImageDraw.Draw(im); d.text((20, 20), label, fill=(255,255,255))
    lab = np.asarray(im).astype(np.float32)/255
    out[:, :60, :300] = lab[:60, :300]
    return out

def run(outdir):
    os.makedirs(outdir, exist_ok=True)
    w, sr, gt = synth_track(bpm=128.0)
    bm = analyze(w, sr).to_dict()
    p = make_plan(bm, "a hero", "neon city", "film", fps=24, width=512, height=320, max_shots=10)
    print(describe(p))
    clips = [fake_clip(s["gen_frames"], 512, 320, s["index"]*0.7, f"shot {s['index']} {s['section']}") for s in p["shots"]]
    t = time.time(); vid = assemble(clips, p, bm, intensity=1.0, letterbox=0.1); print("assemble", round(time.time()-t, 2), "s", vid.shape)
    assert vid.shape[0] == p["total_frames"]
    tl = render(bm, p, thumbs=[c[0] for c in clips])
    Image.fromarray((tl*255).astype(np.uint8)).save(f"{outdir}/timeline.png")
    import wave
    a = slice_audio(w, sr, bm["start"], bm["duration"]); pcm = (np.clip(a.T, -1, 1) * 32767).astype(np.int16)
    with wave.open(f"{outdir}/a.wav", "wb") as wf:
        wf.setnchannels(2); wf.setsampwidth(2); wf.setframerate(sr); wf.writeframes(pcm.tobytes())
    raw = (vid * 255).astype(np.uint8)
    cmd = ["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", "512x320", "-r", "24", "-i", "-",
           "-i", f"{outdir}/a.wav", "-c:v", "libx264", "-preset", "veryfast", "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest", f"{outdir}/e2e.mp4"]
    subprocess.run(cmd, input=raw.tobytes(), check=True)
    print("wrote", f"{outdir}/e2e.mp4")

if __name__ == "__main__":
    run(sys.argv[1] if len(sys.argv) > 1 else "/tmp/beatcut_test")
