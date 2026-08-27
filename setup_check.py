"""
setup_check.py

Run this after installation to verify everything is ready.
Usage:
    python setup_check.py
"""

import sys
import os

print("=" * 55)
print("  NavAssist — Environment Check")
print("=" * 55)

errors = []
warnings = []

# ── Python version ──────────────────────────────────────────
print(f"\n[1] Python version: {sys.version.split()[0]}", end="")
major, minor = sys.version_info[:2]
if major == 3 and minor >= 10:
    print("  [OK]")
else:
    print(f"  [WARN]  (Python 3.10+ recommended, you have {major}.{minor})")
    warnings.append("Python version may cause issues with some dependencies")

# ── CUDA / GPU ──────────────────────────────────────────────
print("\n[2] PyTorch + CUDA...", end="")
try:
    import torch
    cuda_ok = torch.cuda.is_available()
    print(f"  PyTorch {torch.__version__}", end="")
    if cuda_ok:
        gpu = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"\n    GPU: {gpu}  ({vram:.1f} GB VRAM)  [OK]")
    else:
        print("\n    [WARN]  CUDA not available — GPU inference will be disabled")
        warnings.append("CUDA not available")
except ImportError:
    print("  [FAIL]  Not installed")
    errors.append("PyTorch not installed")

# ── OpenAI key ──────────────────────────────────────────────
print("\n[3] OpenAI API key...", end="")
from dotenv import load_dotenv
load_dotenv()
key = os.getenv("OPENAI_API_KEY", "")
if key and key != "sk-paste-your-key-here" and key.startswith("sk-"):
    print(f"  [OK]  (sk-...{key[-4:]})")
else:
    print("  [FAIL]  NOT SET")
    errors.append("OPENAI_API_KEY missing — edit .env and paste your key")

# ── Model weights ───────────────────────────────────────────
print("\n[4] Model weights...")
midas_path = "models/midas_v21_small_256.pt"
yolo_path  = "models/yolov8x-worldv2.pt"

for label, path, min_mb in [
    ("MiDaS v2.1 small", midas_path, 80),
    ("YOLO-World x",     yolo_path,  100),
]:
    if os.path.exists(path):
        size_mb = os.path.getsize(path) / 1e6
        if size_mb > min_mb:
            print(f"    {label}: {path}  ({size_mb:.0f} MB)  [OK]")
        else:
            print(f"    {label}: {path}  ({size_mb:.0f} MB)  [WARN]  (file may be incomplete)")
            warnings.append(f"{label} weight file is small ({size_mb:.0f} MB), may be incomplete")
    else:
        print(f"    {label}: [FAIL]  NOT FOUND at {path}")
        errors.append(f"Model not found: {path}")

# ── Key packages ────────────────────────────────────────────
print("\n[5] Key packages...")
packages = [
    ("opencv-python",    "cv2",          None),
    ("ultralytics",      "ultralytics",  None),
    ("edge-tts",         "edge_tts",     None),
    ("pygame",           "pygame",       None),
    ("SpeechRecognition","speech_recognition", None),
    ("flask",            "flask",        None),
    ("loguru",           "loguru",       None),
    ("timm",             "timm",         None),
    ("einops",           "einops",       None),
    ("networkx",         "networkx",     None),
    ("openai",           "openai",       None),
    ("pyyaml",           "yaml",         None),
    ("python-dotenv",    "dotenv",       None),
]
for name, mod, _ in packages:
    try:
        m = __import__(mod)
        ver = getattr(m, "__version__", "?")
        print(f"    {name}: {ver}  [OK]")
    except ImportError:
        print(f"    {name}:  [FAIL]  NOT INSTALLED")
        errors.append(f"{name} not installed — run: pip install {name}")

# ── Phone stream URL ────────────────────────────────────────
print("\n[6] Phone stream URL...")
import yaml
with open("config/settings.yaml") as f:
    cfg = yaml.safe_load(f)
url = cfg.get("phone_stream_url", "")
if "192.168." in url or "10." in url or "172." in url:
    print(f"    {url}  [OK]")
else:
    print(f"    {url}  [WARN]  Update with your phone's actual IP")
    warnings.append("Update phone_stream_url in config/settings.yaml")

# ── Map directory ───────────────────────────────────────────
print("\n[7] Maps directory...")
import glob
maps = glob.glob("maps/*/osmag.json")
if maps:
    for m in maps:
        print(f"    Found: {m}  [OK]")
else:
    print("    No maps yet — run scan_walk.py once per building")
    warnings.append("No building maps yet — run scan_walk.py")

# ── Summary ─────────────────────────────────────────────────
print("\n" + "=" * 55)
if errors:
    print(f"  [FAIL]  {len(errors)} ERROR(S) — must fix before running:")
    for e in errors:
        print(f"     • {e}")
else:
    print("  [OK]  No blocking errors!")

if warnings:
    print(f"\n  [WARN]  {len(warnings)} WARNING(S):")
    for w in warnings:
        print(f"     • {w}")

if not errors:
    print("\n  Ready to run! Next:")
    print("  1. python scan_walk.py --building mybuilding --stream <phone-url>")
    print("  2. python main.py --building mybuilding --stream <phone-url>")
print("=" * 55)
