#!/usr/bin/env python3
"""Headless BeatCut: song + subject image + brief -> finished music video.

Runs two passes against a running ComfyUI server (default 127.0.0.1:8188):
  pass 1  analyze -> plan -> FLUX.2 Klein keyframes (+ timeline preview)
  pass 2  analyze -> plan -> load keyframes -> LTX-Video clips -> assemble -> mp4
Two passes keep only one model family in memory at a time (fits a free 16 GB T4)
and let you re-roll keyframes before spending GPU time on video.

  python run_beatcut.py --audio song.mp3 --image me.png \
      --subject "a man in a black hoodie" \
      --brief "neon rooftop at night|rain-soaked street|stadium of light" \
      --start 30 --duration 15
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
import urllib.request
import uuid

NEG_VIDEO = ("worst quality, inconsistent motion, blurry, jittery, distorted, deformed face, "
             "extra limbs, watermark, text, subtitles")


def post(server, path, payload):
    req = urllib.request.Request(f"http://{server}{path}", data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req) as r:
            return json.loads(r.read())
    except urllib.error.HTTPError as e:
        body = e.read().decode(errors="replace")
        sys.exit(f"ComfyUI rejected the prompt ({e.code}):\n{body}")


def get(server, path):
    with urllib.request.urlopen(f"http://{server}{path}") as r:
        return json.loads(r.read())


def wait(server, prompt_id, label, watch_dir=None, prefix=""):
    t0 = time.time()
    last = ""
    while True:
        h = get(server, f"/history/{prompt_id}")
        if prompt_id in h:
            st = h[prompt_id].get("status", {})
            if st.get("status_str") == "error":
                for m in st.get("messages", []):
                    if m[0] == "execution_error":
                        d = m[1]
                        sys.exit(f"[{label}] node {d.get('node_id')} ({d.get('node_type')}) failed:\n"
                                 f"{d.get('exception_message')}\n{''.join(d.get('traceback', [])[-6:])}")
                sys.exit(f"[{label}] failed: {st}")
            print(f"[{label}] done in {time.time() - t0:.0f}s", flush=True)
            return h[prompt_id]
        q = get(server, "/queue")
        done = ""
        if watch_dir and os.path.isdir(watch_dir):
            done = f" | {len([f for f in os.listdir(watch_dir) if f.startswith(prefix)])} {prefix or 'files'} saved"
        msg = f"[{label}] running... {time.time() - t0:.0f}s{done}"
        if done != last or int(time.time() - t0) % 60 < 5:
            print(msg, flush=True)
            last = done
        time.sleep(5)


def base_nodes(a):
    return {
        "1": {"class_type": "LoadAudio", "inputs": {"audio": os.path.basename(a.audio)}},
        "2": {"class_type": "BeatCutAnalyze", "inputs": {
            "audio": ["1", 0], "start_seconds": a.start, "duration_seconds": a.duration,
            "bpm_hint": a.bpm, "drop_sensitivity": a.drop_sensitivity}},
        "4": {"class_type": "BeatCutPlanShots", "inputs": {
            "beat_map": ["2", 0], "subject": a.subject, "brief": a.brief.replace("|", "\n"), "style": a.style,
            "fps": a.fps, "width": a.width, "height": a.height, "max_shots": a.max_shots, "pace": a.pace,
            "max_gen_seconds": a.max_gen_seconds, "seed": a.seed}},
        "5": {"class_type": "BeatCutPlanJSON", "inputs": {"plan": ["4", 0]}},
    }


def keyframe_prompt(a, run):
    g = base_nodes(a)
    if a.image:
        g["3"] = {"class_type": "LoadImage", "inputs": {"image": os.path.basename(a.image)}}
    else:
        # no reference: generate a hero portrait first, then lock every
        # keyframe to it so the cast stays consistent across shots
        g.update({
            "50": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["11", 0], "text":
                   f"cinematic full-body portrait of {a.subject}, standing, facing camera, plain studio backdrop, "
                   f"sharp detailed face, {a.style}"}},
            "51": {"class_type": "CFGGuider", "inputs": {"model": ["10", 0], "positive": ["50", 0],
                                                           "negative": ["14", 0], "cfg": a.klein_cfg}},
            "52": {"class_type": "Flux2Scheduler", "inputs": {"steps": a.klein_steps, "width": 640, "height": 832}},
            "53": {"class_type": "EmptyFlux2LatentImage", "inputs": {"width": 640, "height": 832, "batch_size": 1}},
            "54": {"class_type": "RandomNoise", "inputs": {"noise_seed": a.seed + 4242}},
            "55": {"class_type": "SamplerCustomAdvanced", "inputs": {
                "noise": ["54", 0], "guider": ["51", 0], "sampler": ["20", 0], "sigmas": ["52", 0], "latent_image": ["53", 0]}},
            "3": {"class_type": "VAEDecode", "inputs": {"samples": ["55", 0], "vae": ["12", 0]}},
            "56": {"class_type": "SaveImage", "inputs": {"images": ["3", 0], "filename_prefix": f"BeatCut/{run}/cast"}},
        })
    g.update({
        "10": {"class_type": "UNETLoader", "inputs": {"unet_name": a.klein, "weight_dtype": "default"}},
        "11": {"class_type": "CLIPLoader", "inputs": {"clip_name": a.klein_te, "type": "flux2", "device": "default"}},
        "12": {"class_type": "VAELoader", "inputs": {"vae_name": a.flux_vae}},
        "13": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["11", 0], "text": ["4", 1]}},
        "14": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["11", 0], "text": ""}},
        "15": {"class_type": "ImageScaleToTotalPixels", "inputs": {
            "image": ["3", 0], "upscale_method": "lanczos", "megapixels": 1.0, "resolution_steps": 1}},
        "16": {"class_type": "VAEEncode", "inputs": {"pixels": ["15", 0], "vae": ["12", 0]}},
        "17": {"class_type": "ReferenceLatent", "inputs": {"conditioning": ["13", 0], "latent": ["16", 0]}},
        "18": {"class_type": "ReferenceLatent", "inputs": {"conditioning": ["14", 0], "latent": ["16", 0]}},
        "19": {"class_type": "CFGGuider", "inputs": {"model": ["10", 0], "positive": ["17", 0],
                                                       "negative": ["18", 0], "cfg": a.klein_cfg}},
        "20": {"class_type": "KSamplerSelect", "inputs": {"sampler_name": "euler"}},
        "21": {"class_type": "Flux2Scheduler", "inputs": {"steps": a.klein_steps, "width": ["4", 6], "height": ["4", 7]}},
        "22": {"class_type": "EmptyFlux2LatentImage", "inputs": {"width": ["4", 6], "height": ["4", 7], "batch_size": 1}},
        "23": {"class_type": "RandomNoise", "inputs": {"noise_seed": ["4", 4]}},
        "24": {"class_type": "SamplerCustomAdvanced", "inputs": {
            "noise": ["23", 0], "guider": ["19", 0], "sampler": ["20", 0], "sigmas": ["21", 0], "latent_image": ["22", 0]}},
        "25": {"class_type": "VAEDecode", "inputs": {"samples": ["24", 0], "vae": ["12", 0]}},
        "26": {"class_type": "SaveImage", "inputs": {"images": ["25", 0], "filename_prefix": f"BeatCut/{run}/kf"}},
        "27": {"class_type": "BeatCutTimeline", "inputs": {"beat_map": ["2", 0], "width": 1600, "plan": ["4", 0],
                                                            "keyframes": ["25", 0]}},
        "28": {"class_type": "SaveImage", "inputs": {"images": ["27", 0], "filename_prefix": f"BeatCut/{run}/timeline"}},
        "29": {"class_type": "PreviewAny", "inputs": {"source": ["4", 5]}},
    })
    return g


def video_prompt(a, run, n_shots):
    g = base_nodes(a)
    g.update({
        "25": {"class_type": "BeatCutLoadKeyframes", "inputs": {"filename_prefix": f"BeatCut/{run}/kf", "count": n_shots}},
        "30": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": a.ltx}},
        "31": {"class_type": "CLIPLoader", "inputs": {"clip_name": a.ltx_te, "type": "ltxv", "device": "default"}},
        "32": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["31", 0], "text": ["4", 2]}},
        "33": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["31", 0], "text": NEG_VIDEO}},
        "34": {"class_type": "LTXVPreprocess", "inputs": {"image": ["25", 0], "img_compression": 35}},
        "35": {"class_type": "LTXVImgToVideo", "inputs": {
            "positive": ["32", 0], "negative": ["33", 0], "vae": ["30", 2], "image": ["34", 0],
            "width": ["4", 6], "height": ["4", 7], "length": ["4", 3], "batch_size": 1, "strength": 1.0}},
        "36": {"class_type": "LTXVConditioning", "inputs": {"positive": ["35", 0], "negative": ["35", 1],
                                                             "frame_rate": ["4", 8]}},
        "37": {"class_type": "KSamplerSelect", "inputs": {"sampler_name": "euler"}},
        "38": {"class_type": "LTXVScheduler", "inputs": {"steps": a.ltx_steps, "max_shift": 2.05, "base_shift": 0.95,
                                                          "stretch": True, "terminal": 0.1, "latent": ["35", 2]}},
        "39": {"class_type": "SamplerCustom", "inputs": {
            "model": ["30", 0], "add_noise": True, "noise_seed": ["4", 4], "cfg": a.ltx_cfg,
            "positive": ["36", 0], "negative": ["36", 1], "sampler": ["37", 0], "sigmas": ["38", 0],
            "latent_image": ["35", 2]}},
        "40": {"class_type": "VAEDecode", "inputs": {"samples": ["39", 0], "vae": ["30", 2]}},
        "41": {"class_type": "BeatCutAssemble", "inputs": {
            "plan": ["4", 0], "beat_map": ["2", 0], "clips": ["40", 0], "audio": ["1", 0],
            "fx_intensity": a.fx, "letterbox": a.letterbox, "grain": a.grain}},
        "42": {"class_type": "BeatCutSaveMP4", "inputs": {"frames": ["41", 0], "audio": ["41", 1], "fps": ["41", 2],
                                                           "filename_prefix": f"BeatCut/{run}/edit", "crf": 18}},
        "43": {"class_type": "BeatCutSaveMP4", "inputs": {"frames": ["40", 0], "audio": ["1", 0], "fps": ["4", 8],
                                                           "filename_prefix": f"BeatCut/{run}/raw_clip", "crf": 20}},
    })
    if not a.save_raw_clips:
        del g["43"]
    return g


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--server", default="127.0.0.1:8188")
    ap.add_argument("--comfy-dir", default=os.environ.get("COMFY_DIR", "ComfyUI"))
    ap.add_argument("--audio", required=True)
    ap.add_argument("--image", default="", help="Subject reference. Omit to generate a cast from --subject.")
    ap.add_argument("--subject", default="the person in the reference image")
    ap.add_argument("--brief", default="neon-lit rooftop at night|rain-soaked city street|stadium of light")
    ap.add_argument("--style", default="music video, anamorphic 35mm, teal and orange grade, high contrast")
    ap.add_argument("--start", type=float, default=0.0)
    ap.add_argument("--duration", type=float, default=15.0)
    ap.add_argument("--bpm", type=float, default=0.0)
    ap.add_argument("--drop-sensitivity", type=float, default=0.5)
    ap.add_argument("--fps", type=float, default=24.0)
    ap.add_argument("--width", type=int, default=768)
    ap.add_argument("--height", type=int, default=512)
    ap.add_argument("--max-shots", type=int, default=10)
    ap.add_argument("--pace", type=float, default=1.0)
    ap.add_argument("--max-gen-seconds", type=float, default=3.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--fx", type=float, default=1.0)
    ap.add_argument("--letterbox", type=float, default=0.12)
    ap.add_argument("--grain", type=float, default=0.25)
    ap.add_argument("--klein", default="flux-2-klein-base-4b-fp8.safetensors")
    ap.add_argument("--klein-te", default="qwen_3_4b.safetensors")
    ap.add_argument("--flux-vae", default="flux2-vae.safetensors")
    ap.add_argument("--klein-steps", type=int, default=20)
    ap.add_argument("--klein-cfg", type=float, default=5.0)
    ap.add_argument("--ltx", default="ltxv-2b-0.9.8-distilled.safetensors")
    ap.add_argument("--ltx-te", default="t5xxl_fp8_e4m3fn_scaled.safetensors")
    ap.add_argument("--ltx-steps", type=int, default=8)
    ap.add_argument("--ltx-cfg", type=float, default=1.0)
    ap.add_argument("--run", default=None, help="Run id; reuse one to skip pass 1 and re-render video.")
    ap.add_argument("--skip-keyframes", action="store_true")
    ap.add_argument("--only-keyframes", action="store_true")
    ap.add_argument("--save-raw-clips", action="store_true")
    a = ap.parse_args()

    run = a.run or time.strftime("%H%M%S") + "-" + uuid.uuid4().hex[:4]
    inp = os.path.join(a.comfy_dir, "input")
    os.makedirs(inp, exist_ok=True)
    for f in (a.audio, a.image):
        if not f:
            continue
        dst = os.path.join(inp, os.path.basename(f))
        if os.path.abspath(f) != os.path.abspath(dst):
            shutil.copy(f, dst)

    print(f"BeatCut run {run}")
    if not a.skip_keyframes:
        r = post(a.server, "/prompt", {"prompt": keyframe_prompt(a, run), "client_id": run})
        h = wait(a.server, r["prompt_id"], "pass 1: beat analysis + shot plan + keyframes",
                 os.path.join(a.comfy_dir, "output", "BeatCut", run), "kf_")
        for nid, out in h.get("outputs", {}).items():
            if "text" in out:
                print("\n".join(map(str, out["text"])))
    out_dir = os.path.join(a.comfy_dir, "output", "BeatCut", run)
    n_kf = len([f for f in os.listdir(out_dir) if f.startswith("kf_")]) if os.path.isdir(out_dir) else 0
    print(f"{n_kf} keyframes in {out_dir}")
    if a.only_keyframes:
        return
    r = post(a.server, "/prompt", {"prompt": video_prompt(a, run, n_kf), "client_id": run})
    wait(a.server, r["prompt_id"], "pass 2: LTX clips + beat-synced assembly", out_dir, "edit_")
    vids = sorted(f for f in os.listdir(out_dir) if f.startswith("edit_") and f.endswith(".mp4"))
    print("FINAL:", os.path.join(out_dir, vids[-1]) if vids else "(no mp4 found)")


if __name__ == "__main__":
    main()
