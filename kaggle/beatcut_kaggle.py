# BeatCut on a free Kaggle T4 — paste this whole file into ONE notebook cell and run.
# Notebook settings: Accelerator = GPU T4 x2 (or P100), Internet = On.
# ---------------------------------------------------------------- settings
SONG_URL  = ""          # direct link to an mp3/wav you can use. Empty = BeatCut synthesises its demo track.
IMAGE_URL = ""          # optional: direct link to a photo of your subject. Empty = BeatCut generates a cast.
SUBJECT   = "a confident young man in a bright orange bomber jacket and black jeans"
BRIEF     = "neon-lit rooftop at night | rain-soaked city street with reflections | stadium flooded with light beams"
STYLE     = "music video, anamorphic 35mm, teal and orange grade, high contrast"
START, DURATION = 0.0, 0.0   # seconds into the song, edit length (0 = whole song)
MAX_SHOTS = 12
RUN_ID    = ""          # set to a previous run id to re-render video without redoing keyframes
# -------------------------------------------------------------------------
import os, subprocess, sys, time, urllib.request, shutil, glob

W = "/kaggle/working" if os.path.isdir("/kaggle/working") else os.getcwd()
ROOT = "/tmp/bc"
COMFY = f"{ROOT}/ComfyUI"
os.makedirs(ROOT, exist_ok=True)

def stream(cmd, shell=False):
    """Run a command and stream its output into the notebook cell."""
    p = subprocess.Popen(cmd, shell=shell, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    for line in p.stdout:
        print(line, end="", flush=True)
    return p.wait()

def sh(cmd, check=True):
    print("$", cmd, flush=True)
    rc = stream(cmd, shell=True)
    if check and rc:
        raise SystemExit(f"command failed: {cmd}")

t0 = time.time()
if not os.path.isdir(COMFY):
    sh(f"git clone --depth 1 https://github.com/comfyanonymous/ComfyUI.git {COMFY}")
# keep Kaggle's CUDA torch/torchvision/torchaudio; install everything else
# (careful: torchsde etc. also start with "torch")
import re
KEEP = {"torch", "torchvision", "torchaudio"}
req = []
for l in open(f"{COMFY}/requirements.txt").read().splitlines():
    l = l.strip()
    if not l or l.startswith("#"):
        continue
    name = re.split(r"[<>=!~\[; ]", l, 1)[0].lower()
    if name not in KEEP:
        req.append(l)
open(f"{ROOT}/req.txt", "w").write("\n".join(req))
sh(f"{sys.executable} -m pip install -q -r {ROOT}/req.txt")
nodes = f"{COMFY}/custom_nodes/ComfyUI-BeatCut"
if os.path.isdir(nodes):
    sh(f"git -C {nodes} pull -q")
else:
    sh(f"git clone --depth 1 https://github.com/nedsoft057/ComfyUI-BeatCut.git {nodes}")
sh(f"{sys.executable} -m pip install -q -r {nodes}/requirements.txt huggingface_hub")

# ---------------------------------------------------------------- models
from huggingface_hub import hf_hub_download
token = None
try:
    from kaggle_secrets import UserSecretsClient
    token = UserSecretsClient().get_secret("HF_TOKEN")
except Exception:
    pass

MODELS = [  # (folder, final name, [(repo, path-in-repo), ...fallbacks])
    ("diffusion_models", "flux-2-klein-base-4b-fp8.safetensors",
     [("black-forest-labs/FLUX.2-klein-base-4b-fp8", "flux-2-klein-base-4b-fp8.safetensors")]),
    ("text_encoders", "qwen_3_4b.safetensors",
     [("Comfy-Org/z_image_turbo", "split_files/text_encoders/qwen_3_4b.safetensors")]),
    ("vae", "flux2-vae.safetensors",
     [("Comfy-Org/flux2-dev", "split_files/vae/flux2-vae.safetensors")]),
    ("checkpoints", "ltxv-2b.safetensors",
     [("Lightricks/LTX-Video", "ltxv-2b-0.9.8-distilled.safetensors"),
      ("Lightricks/LTX-Video", "ltxv-2b-0.9.6-distilled-04-25.safetensors")]),
    ("text_encoders", "t5xxl_fp8_e4m3fn_scaled.safetensors",
     [("comfyanonymous/flux_text_encoders", "t5xxl_fp8_e4m3fn_scaled.safetensors"),
      ("comfyanonymous/flux_text_encoders", "t5xxl_fp8_e4m3fn.safetensors")]),
]
for folder, name, cands in MODELS:
    dst = f"{COMFY}/models/{folder}/{name}"
    if os.path.exists(dst):
        continue
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    for repo, path in cands:
        try:
            print(f"downloading {repo}/{path}", flush=True)
            p = hf_hub_download(repo, path, token=token, cache_dir=f"{ROOT}/hf")
            os.symlink(p, dst)
            break
        except Exception as e:
            print(f"  failed: {type(e).__name__}: {str(e)[:200]}")
    else:
        raise SystemExit(f"Could not download {name}. If it is gated, add an HF_TOKEN Kaggle secret.")
print(f"setup done in {time.time() - t0:.0f}s", flush=True)

# ---------------------------------------------------------------- inputs
def fetch(url, name):
    dst = f"{ROOT}/{name}"
    if url.startswith("http"):
        urllib.request.urlretrieve(url, dst)
    else:
        shutil.copy(url, dst)
    return dst
if SONG_URL:
    song = fetch(SONG_URL, "song" + (os.path.splitext(SONG_URL.split("?")[0])[1] or ".mp3"))
else:
    song = f"{ROOT}/demo_song.wav"
    sh(f"{sys.executable} {nodes}/examples/make_demo_song.py {song}")
image = fetch(IMAGE_URL, "subject" + (os.path.splitext(IMAGE_URL.split("?")[0])[1] or ".png")) if IMAGE_URL else ""

# ---------------------------------------------------------------- server
import socket
def up():
    try:
        socket.create_connection(("127.0.0.1", 8188), 1).close(); return True
    except OSError:
        return False
if not up():
    log = open(f"{ROOT}/comfy.log", "w")
    subprocess.Popen([sys.executable, "main.py", "--listen", "127.0.0.1", "--port", "8188", "--disable-auto-launch"],
                     cwd=COMFY, stdout=log, stderr=subprocess.STDOUT)
    for _ in range(180):
        if up(): break
        time.sleep(2)
    else:
        print(open(f"{ROOT}/comfy.log").read()[-4000:]); raise SystemExit("ComfyUI did not start")
print("ComfyUI is up", flush=True)
errs = [l for l in open(f"{ROOT}/comfy.log").read().splitlines() if "BeatCut" in l and ("Error" in l or "fail" in l.lower())]
if errs: print("\n".join(errs))

# ---------------------------------------------------------------- render
args = [sys.executable, f"{nodes}/scripts/run_beatcut.py", "--comfy-dir", COMFY, "--audio", song,
        "--subject", SUBJECT, "--brief", BRIEF, "--style", STYLE, "--start", str(START), "--duration", str(DURATION),
        "--max-shots", str(MAX_SHOTS), "--ltx", "ltxv-2b.safetensors", "--save-raw-clips"]
if image: args += ["--image", image]
if RUN_ID: args += ["--run", RUN_ID, "--skip-keyframes"]
rc = stream(args)
# copy results to /kaggle/working so they show up under Output for download
for d in glob.glob(f"{COMFY}/output/BeatCut/*"):
    shutil.copytree(d, f"{W}/BeatCut/{os.path.basename(d)}", dirs_exist_ok=True)
if rc:
    print("---- last ComfyUI log lines ----"); print(open(f"{ROOT}/comfy.log").read()[-5000:])
print(f"total {time.time() - t0:.0f}s — results in {W}/BeatCut/")
for f in sorted(glob.glob(f"{W}/BeatCut/*/*.mp4")): print(f)
