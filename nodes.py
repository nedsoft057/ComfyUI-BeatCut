"""ComfyUI node definitions for BeatCut."""
from __future__ import annotations

import json

import numpy as np
import torch

from .beatcut.audio import analyze as _analyze, BeatMap
from .beatcut.planner import plan as _plan, describe as _describe
from .beatcut.assemble import assemble as _assemble, slice_audio as _slice_audio
from .beatcut.timeline import render as _render

CAT = "BeatCut"
BEATMAP = "BEATCUT_BEATMAP"
PLAN = "BEATCUT_PLAN"


def _np_img(t):
    if isinstance(t, torch.Tensor):
        return t.detach().float().cpu().numpy()
    return np.asarray(t, dtype=np.float32)


def _first(x):
    return x[0] if isinstance(x, (list, tuple)) else x


class BeatCutAnalyze:
    """Song in -> tempo, beats, downbeats, bars, sections, drops, hits."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "audio": ("AUDIO",),
            "start_seconds": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 3600.0, "step": 0.01,
                                        "tooltip": "Where in the song the edit starts."}),
            "duration_seconds": ("FLOAT", {"default": 15.0, "min": 0.0, "max": 600.0, "step": 0.01,
                                           "tooltip": "Edit length. 0 = rest of the song."}),
            "bpm_hint": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 300.0, "step": 0.1,
                                   "tooltip": "Optional tempo hint. 0 = auto."}),
            "drop_sensitivity": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.05}),
        }}

    RETURN_TYPES = (BEATMAP, "STRING")
    RETURN_NAMES = ("beat_map", "summary")
    FUNCTION = "run"
    CATEGORY = CAT

    def run(self, audio, start_seconds, duration_seconds, bpm_hint, drop_sensitivity):
        wav = audio["waveform"]
        wav = wav.detach().float().cpu().numpy() if isinstance(wav, torch.Tensor) else np.asarray(wav)
        bm = _analyze(wav, audio["sample_rate"], start=start_seconds, duration=duration_seconds,
                      bpm_hint=bpm_hint, drop_sensitivity=drop_sensitivity)
        return (bm.to_dict(), bm.summary())


class BeatCutPlanShots:
    """Beat map + brief -> beat-aligned shot list (one list item per shot)."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "beat_map": (BEATMAP,),
            "subject": ("STRING", {"default": "the person in the reference image", "multiline": False}),
            "brief": ("STRING", {"default": "neon-lit rooftop at night\nrain-soaked city street\nstadium of light",
                                 "multiline": True, "tooltip": "One scene per line; scenes advance with song sections."}),
            "style": ("STRING", {"default": "music video, anamorphic 35mm, teal and orange grade", "multiline": True}),
            "fps": ("FLOAT", {"default": 24.0, "min": 8.0, "max": 60.0, "step": 1.0}),
            "width": ("INT", {"default": 768, "min": 256, "max": 2048, "step": 32}),
            "height": ("INT", {"default": 512, "min": 256, "max": 2048, "step": 32}),
            "max_shots": ("INT", {"default": 10, "min": 1, "max": 64}),
            "pace": ("FLOAT", {"default": 1.0, "min": 0.25, "max": 4.0, "step": 0.05,
                               "tooltip": ">1 cuts faster, <1 holds longer."}),
            "max_gen_seconds": ("FLOAT", {"default": 3.0, "min": 0.5, "max": 10.0, "step": 0.1,
                                          "tooltip": "Cap on generated clip length per shot (VRAM/time budget)."}),
            "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFF}),
        }}

    RETURN_TYPES = (PLAN, "STRING", "STRING", "INT", "INT", "STRING", "INT", "INT", "FLOAT")
    RETURN_NAMES = ("plan", "keyframe_prompts", "motion_prompts", "gen_frames", "seeds", "plan_text",
                    "width", "height", "fps")
    OUTPUT_IS_LIST = (False, True, True, True, True, False, False, False, False)
    FUNCTION = "run"
    CATEGORY = CAT

    def run(self, beat_map, subject, brief, style, fps, width, height, max_shots, pace, max_gen_seconds, seed):
        p = _plan(beat_map, subject, brief, style, fps=fps, width=width, height=height, max_shots=max_shots,
                  pace=pace, max_gen_seconds=max_gen_seconds, seed=seed)
        sh = p["shots"]
        return (p, [s["keyframe_prompt"] for s in sh], [s["motion_prompt"] for s in sh],
                [s["gen_frames"] for s in sh], [s["seed"] for s in sh], _describe(p), width, height, float(fps))


class BeatCutAssemble:
    """Generated clips (one per shot) -> beat-synced edit with hit FX + trimmed audio."""

    INPUT_IS_LIST = True

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "plan": (PLAN,),
            "beat_map": (BEATMAP,),
            "clips": ("IMAGE", {"tooltip": "One clip per shot, in shot order (list output of your video model)."}),
            "audio": ("AUDIO",),
            "fx_intensity": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05}),
            "letterbox": ("FLOAT", {"default": 0.12, "min": 0.0, "max": 0.4, "step": 0.01}),
            "grain": ("FLOAT", {"default": 0.25, "min": 0.0, "max": 1.0, "step": 0.05}),
        }}

    RETURN_TYPES = ("IMAGE", "AUDIO", "FLOAT")
    RETURN_NAMES = ("frames", "audio", "fps")
    FUNCTION = "run"
    CATEGORY = CAT

    def run(self, plan, beat_map, clips, audio, fx_intensity, letterbox, grain):
        p, bm, a = _first(plan), _first(beat_map), _first(audio)
        clip_list = []
        for c in clips:
            c = _np_img(c)
            # a batched IMAGE of N frames is one clip
            clip_list.append(c)
        if len(clip_list) == 1 and len(p["shots"]) > 1:
            raise ValueError("BeatCut Assemble got a single clip; feed it the per-shot list from your video model.")
        vid = _assemble(clip_list, p, bm, intensity=_first(fx_intensity), letterbox=_first(letterbox),
                        grain=_first(grain))
        wav = a["waveform"]
        wav_np = wav.detach().float().cpu().numpy() if isinstance(wav, torch.Tensor) else np.asarray(wav)
        cut = _slice_audio(wav_np, a["sample_rate"], bm["start"], bm["duration"])
        return (torch.from_numpy(vid), {"waveform": torch.from_numpy(np.ascontiguousarray(cut, dtype=np.float32)),
                                        "sample_rate": a["sample_rate"]}, float(p["fps"]))


class BeatCutTimeline:
    """Render beat map + plan (+ optional keyframes) as a timeline image."""

    INPUT_IS_LIST = True

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"beat_map": (BEATMAP,), "width": ("INT", {"default": 1600, "min": 512, "max": 4096})},
                "optional": {"plan": (PLAN,), "keyframes": ("IMAGE",)}}

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("timeline",)
    FUNCTION = "run"
    CATEGORY = CAT

    def run(self, beat_map, width, plan=None, keyframes=None):
        thumbs = [_np_img(k)[0] for k in keyframes] if keyframes else None
        img = _render(_first(beat_map), _first(plan) if plan else None, width=_first(width), thumbs=thumbs)
        return (torch.from_numpy(img)[None],)


class BeatCutVideoToFrames:
    """VIDEO -> IMAGE frames, so any video model (LTX-2.x blueprints, MiniMax, Wan) can feed Assemble."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"video": ("VIDEO",)}}

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "run"
    CATEGORY = CAT

    def run(self, video):
        return (video.get_components().images,)


class BeatCutPlanJSON:
    """Export the plan as JSON (for editing in other tools / debugging)."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"plan": (PLAN,)}}

    RETURN_TYPES = ("STRING",)
    FUNCTION = "run"
    CATEGORY = CAT

    def run(self, plan):
        return (json.dumps(plan, indent=2),)


class BeatCutLoadKeyframes:
    """Load saved keyframes back as a per-shot list (stage-2 of a two-pass run,
    or to re-use hand-picked keyframes)."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"filename_prefix": ("STRING", {"default": "BeatCut/kf"}),
                             "count": ("INT", {"default": 0, "min": 0, "max": 256,
                                               "tooltip": "Expected number of keyframes (0 = all found)."})}}

    RETURN_TYPES = ("IMAGE",)
    OUTPUT_IS_LIST = (True,)
    FUNCTION = "run"
    CATEGORY = CAT

    @classmethod
    def IS_CHANGED(cls, **kw):
        return float("nan")

    def run(self, filename_prefix, count):
        import glob
        import os
        import folder_paths
        from PIL import Image
        base = os.path.join(folder_paths.get_output_directory(), filename_prefix)
        files = sorted(glob.glob(base + "_*.png"))
        if count:
            files = files[-count:]
        if not files:
            raise FileNotFoundError(f"BeatCut: no keyframes matching {base}_*.png")
        out = []
        for f in files:
            im = np.asarray(Image.open(f).convert("RGB")).astype(np.float32) / 255.0
            out.append(torch.from_numpy(im)[None])
        return (out,)


class BeatCutSaveMP4:
    """Frames + audio -> H.264/AAC mp4 in the output folder."""

    OUTPUT_NODE = True

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"frames": ("IMAGE",), "audio": ("AUDIO",),
                             "fps": ("FLOAT", {"default": 24.0, "min": 1.0, "max": 120.0}),
                             "filename_prefix": ("STRING", {"default": "BeatCut/edit"}),
                             "crf": ("INT", {"default": 18, "min": 0, "max": 51})}}

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("path",)
    FUNCTION = "run"
    CATEGORY = CAT

    def run(self, frames, audio, fps, filename_prefix, crf):
        import os
        import shutil
        import subprocess
        import tempfile
        import wave
        import folder_paths
        out_dir = folder_paths.get_output_directory()
        stem = os.path.join(out_dir, filename_prefix)
        os.makedirs(os.path.dirname(stem), exist_ok=True)
        k = 1
        while os.path.exists(f"{stem}_{k:05d}.mp4"):
            k += 1
        path = f"{stem}_{k:05d}.mp4"
        vid = (np.clip(_np_img(frames), 0, 1) * 255).astype(np.uint8)
        n, h, w = vid.shape[:3]
        wav = audio["waveform"]
        wav = wav.detach().float().cpu().numpy() if isinstance(wav, torch.Tensor) else np.asarray(wav)
        while wav.ndim > 2:
            wav = wav[0]
        sr = int(audio["sample_rate"])
        tmp = tempfile.mkdtemp()
        wav_path = os.path.join(tmp, "a.wav")
        pcm = (np.clip(wav.T, -1, 1) * 32767).astype(np.int16)
        with wave.open(wav_path, "wb") as wf:
            wf.setnchannels(pcm.shape[1] if pcm.ndim == 2 else 1)
            wf.setsampwidth(2)
            wf.setframerate(sr)
            wf.writeframes(pcm.tobytes())
        ff = shutil.which("ffmpeg")
        if ff:
            cmd = [ff, "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}", "-r", str(fps),
                   "-i", "-", "-i", wav_path, "-c:v", "libx264", "-preset", "veryfast", "-crf", str(crf),
                   "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k", "-shortest", "-movflags", "+faststart", path]
            subprocess.run(cmd, input=vid.tobytes(), check=True)
        else:
            import av
            from fractions import Fraction
            with av.open(path, "w") as c:
                vs = c.add_stream("libx264", rate=Fraction(fps).limit_denominator(1000))
                vs.width, vs.height, vs.pix_fmt = w, h, "yuv420p"
                vs.options = {"crf": str(crf)}
                for f in vid:
                    for pk in vs.encode(av.VideoFrame.from_ndarray(f, format="rgb24")):
                        c.mux(pk)
                for pk in vs.encode():
                    c.mux(pk)
        shutil.rmtree(tmp, ignore_errors=True)
        print(f"[BeatCut] wrote {path}")
        return {"ui": {"text": [path]}, "result": (path,)}


NODE_CLASS_MAPPINGS = {
    "BeatCutAnalyze": BeatCutAnalyze,
    "BeatCutPlanShots": BeatCutPlanShots,
    "BeatCutAssemble": BeatCutAssemble,
    "BeatCutTimeline": BeatCutTimeline,
    "BeatCutVideoToFrames": BeatCutVideoToFrames,
    "BeatCutPlanJSON": BeatCutPlanJSON,
    "BeatCutLoadKeyframes": BeatCutLoadKeyframes,
    "BeatCutSaveMP4": BeatCutSaveMP4,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "BeatCutAnalyze": "BeatCut · Analyze Song",
    "BeatCutPlanShots": "BeatCut · Plan Shots",
    "BeatCutAssemble": "BeatCut · Assemble Edit",
    "BeatCutTimeline": "BeatCut · Timeline Preview",
    "BeatCutVideoToFrames": "BeatCut · Video to Frames",
    "BeatCutPlanJSON": "BeatCut · Plan to JSON",
    "BeatCutLoadKeyframes": "BeatCut · Load Keyframes",
    "BeatCutSaveMP4": "BeatCut · Save MP4",
}
