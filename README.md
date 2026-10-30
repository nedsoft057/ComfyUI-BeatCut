# BeatCut — song in, music video out

**An open-source, open-weights music-video director for ComfyUI.** Entry for Comfy's `#OpenSourceHiggsfield` challenge.

You give BeatCut a song, a subject (a photo, or just a description) and a one-line brief. It listens to the track, works out the tempo, beats, bars, sections and the drop, then **directs** it like an editor would: it plans a shot list where every cut lands on a beat, calm sections hold wide and slow, builds cut every two beats and push in, and the drop hits with a hero shot and cuts on every beat. Each shot gets a FLUX.2 Klein keyframe that is locked to your subject, LTX-Video animates it, and BeatCut conforms every clip to its exact beat window with speed ramps, slow-mo and hit-synced punch zooms, flashes, shake and RGB split. You get one mp4 with the song on it.

It runs fully headless on a **free 16 GB Kaggle T4** in a single notebook cell, and the nodes also work in the normal ComfyUI graph.

```
song.mp3 ─▶ Analyze ─▶ Plan Shots ─┬─▶ FLUX.2 Klein (ref-locked keyframe per shot) ─▶ LTX-Video i2v (per shot) ─┐
                                   └──────────────────────────────────────────────────────────────────────────▶ Assemble ─▶ edit.mp4
                                                                                               (beat-conform + hit FX + audio)
```

## Why it's different

Most "AI music video" apps pick clip lengths at random and drop a song over the top. BeatCut treats the **song as the script**:

| Stage | What it does |
|---|---|
| **Analyze** | Log-spectral-flux onset envelope → tempo via a comb-weighted autocorrelation (handles 8th-note hats and triplet traps) → Ellis dynamic-programming beat tracker on a bass-weighted envelope, with an on/off-beat lattice check → downbeat phase from kick/bass onsets → per-bar energy → **drop detection from low-end arrival** (not loudness, so risers don't fool it) → section labels: low / mid / build / drop / high. Pure numpy/scipy, no librosa. |
| **Plan Shots** | Beat-quantised cuts with a cinematography grammar (framing, camera move, action and light vocabularies per section, never repeated back-to-back). A shot budget merges the calmest cuts first and never removes the cut on the drop. Frames are quantised on one global grid, so there's zero drift against the audio. Each shot gets an LTX-legal (8n+1) generation length and a time-remap mode, so the GPU only renders what the edit needs. |
| **Keyframes** | FLUX.2 Klein 4B *edit* with the subject as a ReferenceLatent, so the same person appears in every shot. With no photo, BeatCut first generates a cast portrait and locks every shot to it. |
| **Motion** | LTX-Video 2B distilled image-to-video, one clip per shot, driven by the planner's motion prompt. |
| **Assemble** | Cover-fits every clip, then time-remaps it into its beat window: *slow-mo* for calm holds, *linear* for builds, and an ease-out *speed ramp* for hits. Then hit-synced FX: punch zoom on beats, flash + shake on accents, and a big flash + RGB split + crash zoom on the drop, plus a creeping zoom through builds. Optional letterbox and grain. Muxes the song slice. |

Plus **Timeline Preview**, which renders the beat map, sections, drop, shot plan and keyframe thumbnails as one image, so you can see the edit before you spend GPU time on it.

## Nodes

| Node | In → Out |
|---|---|
| `BeatCut · Analyze Song` | AUDIO → `BEATCUT_BEATMAP`, summary |
| `BeatCut · Plan Shots` | beat map + subject/brief/style → plan, **lists** of keyframe prompts, motion prompts, gen frames and seeds (downstream nodes run once per shot) |
| `BeatCut · Assemble Edit` | plan + beat map + per-shot clips (list) + AUDIO → frames, trimmed audio, fps |
| `BeatCut · Timeline Preview` | beat map (+ plan, keyframes) → IMAGE |
| `BeatCut · Save MP4` | frames + audio → H.264/AAC mp4 |
| `BeatCut · Load Keyframes` | saved keyframes → per-shot list (two-pass runs / hand-picked keyframes) |
| `BeatCut · Video to Frames` | VIDEO → IMAGE, so any video model (LTX-2.x blueprints, Wan, MiniMax) can feed Assemble |
| `BeatCut · Plan to JSON` | plan → JSON string |

## Run it free on Kaggle (one cell)

1. New Kaggle notebook → Settings: **Accelerator: GPU T4 x2**, **Internet: On** (needs a phone-verified account).
2. Paste [`kaggle/beatcut_kaggle.py`](kaggle/beatcut_kaggle.py) into a cell, edit `SONG_URL`, `SUBJECT` and `BRIEF` at the top, and run it.
3. Download `BeatCut/<run>/edit_00001.mp4` (plus `timeline_*.png`, keyframes and raw clips) from the Output panel.

The cell installs ComfyUI and this pack, downloads the open-weights models, starts ComfyUI headless and runs two passes: **(1)** analysis, planning and keyframes, then **(2)** LTX clips and assembly. Only one model family is loaded per pass, so it fits a 16 GB card. To re-render the video with the same keyframes, set `RUN_ID`.

## Run it on your own ComfyUI

```bash
cd ComfyUI/custom_nodes && git clone https://github.com/nedsoft057/ComfyUI-BeatCut && pip install -r ComfyUI-BeatCut/requirements.txt
# start ComfyUI, then:
python custom_nodes/ComfyUI-BeatCut/scripts/run_beatcut.py --comfy-dir . \
  --audio song.mp3 --image me.png --subject "a woman in a silver jacket" \
  --brief "desert highway at dusk|neon tunnel|rooftop in the rain" --start 42 --duration 15
```

API-format graphs for both passes are in [`workflows/`](workflows/).

**Models (all open weights):** [FLUX.2 Klein base 4B fp8](https://huggingface.co/black-forest-labs/FLUX.2-klein-base-4b-fp8) + Qwen3-4B text encoder + FLUX.2 VAE; [LTX-Video 2B distilled](https://huggingface.co/Lightricks/LTX-Video) + T5-XXL fp8. No partner nodes, no closed APIs. On a bigger GPU, wire Plan Shots' lists into the built-in *Image to Video (LTX-2.5)* blueprint and route its VIDEO through `Video to Frames`.

## Tests

```bash
python tests/test_audio.py   # tempo within 4% and drop within 0.15 s on synthetic tracks at 100/128/140 BPM
python tests/test_e2e.py     # CPU-only: analyze → plan → stand-in clips → assemble → mp4 + timeline
```

`examples/make_demo_song.py` synthesises the license-free demo track (124 BPM, intro → build → drop).

## License

Apache-2.0. Model weights are under their own licenses.
