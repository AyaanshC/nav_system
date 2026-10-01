# 🧭 NavSystem 2.0 — AI Indoor Navigation for Visually Impaired

![Status](https://img.shields.io/badge/Status-Active-brightgreen)
![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-CUDA-red)
![Ollama](https://img.shields.io/badge/Ollama-Local_VLM-orange)
![License](https://img.shields.io/badge/License-MIT-green)

A real-time, AI-powered indoor navigation assistant for visually impaired users. Uses a laptop GPU, a phone camera (or webcam), and local/cloud Vision-Language Models to provide continuous spoken navigation guidance — no expensive sensors or ROS required.

---

## 🌟 Key Features

| Feature | Details |
|---|---|
| **Live Localization** | DINOv2 + FAISS matches current camera view to the nearest map node in real-time |
| **VLM Navigation Guidance** | Qwen2.5-VL or Gemini 2.0 Flash generates spoken instructions every ~1-4s |
| **Instant Hazard Alerts** | YOLO obstacle < 0.5m → immediate TTS "STOP!" bypassing VLM lag |
| **Depth Sensing** | Depth Anything V2 estimates left/center/right distance buckets |
| **Auto Path Planning** | BFS graph search on OSMAG map with automatic re-routing |
| **Goal Reached Detection** | Announces arrival and shows overlay when destination is reached |
| **Live Camera Support** | Webcam, phone IP camera, or video file |
| **Glassmorphism HUD** | Professional real-time UI with location, depth, bounding boxes |
| **Edge-TTS Voice** | Natural neural voice (Jenny, en-US) for all guidance |

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────┐
│                  Phase 1: Map Building               │
│  scan_walk.py                                        │
│  Video → DINOv2 keyframes → VLM room labels →        │
│  OSMAG graph (osmag.json) + FAISS index              │
└─────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────┐
│                  Phase 2: Navigation                 │
│  tests/visualizer.py                                 │
│                                                      │
│  Camera/Video → [Fast Worker Thread]                 │
│    ├─ Depth Anything V2 (depth sensing)              │
│    ├─ YOLOv8-World (obstacle detection)              │
│    └─ DINOv2 FAISS (live localization)               │
│                                                      │
│  → [Slow Worker Thread]                              │
│    ├─ Tier 1: Rule-based instant guidance (0ms)      │
│    └─ Tier 2: VLM visual guidance (~1-4s)            │
│         ├─ Gemini 2.0 Flash (cloud, ~0.3s) ⚡        │
│         └─ qwen2.5vl:3b (local, ~3-4s)              │
│                                                      │
│  → Edge-TTS spoken output + Glassmorphism HUD        │
└─────────────────────────────────────────────────────┘
```

---

## 💻 System Requirements

| Component | Minimum | Recommended |
|---|---|---|
| **OS** | Windows 10/11 | Windows 11 |
| **GPU** | NVIDIA 6GB VRAM | RTX 4060 8GB |
| **RAM** | 16 GB | 32 GB |
| **Python** | 3.10 | 3.12 |
| **CUDA** | 11.8 | 12.1+ |
| **Storage** | 15 GB free | 20 GB |

---

## 🚀 Installation

### Step 1 — Clone the Repository
```bash
git clone https://github.com/AyaanshC/nav_system.git
cd nav_system
```

### Step 2 — Create Virtual Environment
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux / Mac
source venv/bin/activate
```

### Step 3 — Install PyTorch with CUDA
> ⚠️ Install PyTorch **before** `requirements.txt` to get the CUDA version.

```bash
# CUDA 12.1 (RTX 30/40 series)
pip install torch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0 --index-url https://download.pytorch.org/whl/cu121

# CUDA 11.8 (older GPUs)
pip install torch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0 --index-url https://download.pytorch.org/whl/cu118
```

### Step 4 — Install Dependencies
```bash
pip install -r requirements.txt

# Also install Gemini SDK (for cloud VLM, optional but recommended)
pip install google-generativeai
```

### Step 5 — Install Ollama & Pull Models
Download Ollama from [https://ollama.com](https://ollama.com), install it, then:

```bash
# Start Ollama (runs as a background service)
ollama serve

# Pull required models (in a new terminal)
ollama pull qwen2.5vl:7b    # For map building (high accuracy)
ollama pull qwen2.5vl:3b    # For real-time navigation (fast)
ollama pull qwen2.5:3b      # For path description text
```

> 💡 `qwen2.5vl:7b` is ~6GB download. Run this once and it's cached.

### Step 6 — Download YOLO Model
The YOLO model auto-downloads on first run. Alternatively:
```bash
mkdir models
# Auto-downloaded to models/ on first run of visualizer.py
```

### Step 7 — Verify Setup
```bash
python setup_check.py
```
This checks GPU, CUDA, Ollama, and all modules. All items should show ✅.

---

## 🗺️ Usage Guide

### Phase 1: Build a Map (Once per building)

**Option A — From a video file** (recommended for testing):
```bash
# Record a slow, steady walkthrough of the building on your phone
# Then run:
python scan_walk.py --video path/to/walkthrough.mp4 --building mybuilding
```

**Option B — From a live phone camera**:
1. Install **IP Webcam** (Android) or **EpocCam** (iPhone) on your phone
2. Start the server on the phone and note the IP address (e.g., `192.168.1.5:8080`)
3. Run:
```bash
python scan_walk.py --video http://192.168.1.5:8080/video --building mybuilding
```

Map building takes ~30-60 minutes for a 5-minute video (VLM labels each room).
Output: `maps/mybuilding/osmag.json` + `maps/mybuilding/dino_index.faiss`

> 💡 Review labels: `python review_labels.py --building mybuilding`

---

### Phase 2: Real-Time Navigation

#### Find your goal node ID first:
```bash
python -c "
import json
data = json.load(open('maps/mybuilding/osmag.json'))
for nid, node in data['nodes'].items():
    print(nid, '-', node['name'])
"
```

#### Run the Visualizer:

**With video file (for testing/demo):**
```bash
python tests/visualizer.py --building mybuilding --goal node_163
```

**With USB webcam (live camera):**
```bash
python tests/visualizer.py --building mybuilding --goal node_163 --video 0
```

**With phone camera (IP Webcam app):**
```bash
python tests/visualizer.py --building mybuilding --goal node_163 --video "http://192.168.1.5:8080/video"
```

**With Gemini 2.0 Flash (fastest VLM, ~0.3s):**
```bash
# Get free API key at: https://aistudio.google.com/app/apikey
python tests/visualizer.py --building mybuilding --goal node_163 --gemini_key "AIzaSy..."

# Or set environment variable once:
set GEMINI_API_KEY=AIzaSy...
python tests/visualizer.py --building mybuilding --goal node_163
```

---

## ⚙️ All Command-Line Arguments

### `scan_walk.py` (Map Building)
```
--video          Path to video file or phone stream URL
--building       Name for the building map (e.g. myoffice)
--frames         Max keyframes to extract (default: 300)
```

### `tests/visualizer.py` (Navigation)
```
--video          Video file path, webcam index (0/1), or phone stream URL
                 Default: test_videos/house_tour.mp4
--building       Building map name (must exist in maps/)
                 Default: housetour
--node           Starting node ID
                 Default: node_001
--goal           Destination node ID
                 Default: node_163
--vlm_model      Local Ollama model when no --gemini_key
                 Default: qwen2.5vl:3b
--gemini_key     Google Gemini API key (enables ~0.3s cloud VLM)
--no_tts         Disable text-to-speech output
--no_localize    Disable DINOv2 live localization (uses fixed start node)
```

---

## 🎮 HUD Guide

```
┌────────────────────────────────────────────────────────────────┐
│  NavSystem 2.0          DEPTH | L: mid  C: far  R: near        │
├────────────────────────────────────────────────────────────────┤
│  LOCATION: Living Room near Kitchen  [-> Master Bedroom] LIVE● │
├────────────────────────────────────────────────────────────────┤
│                                                                 │
│              [ CAMERA FEED + BOUNDING BOXES ]                   │
│                                                                 │
│   [chair 1.2m]    [desk 2.4m]                                  │
├────────────────────────────────────────────────────────────────┤
│  ● VLM ACTIVE                                                   │
│  Move forward, the door is straight ahead.                      │
└────────────────────────────────────────────────────────────────┘
```

| HUD Element | Description |
|---|---|
| **DEPTH** row | Left/Center/Right distance (near < 1m, mid 1-3m, far > 3m) |
| **LOCATION** | Current room (DINOv2 live localization) → Goal |
| **LIVE●** | Green = localization active, Red = using fixed node |
| **Bounding boxes** | Red < 1.5m, Yellow 1.5-3m, Cyan > 3m |
| **VLM panel** | Current navigation instruction |

**Keyboard:** `ESC` to stop.

---

## 🔊 Guidance System

The system uses a **3-tier approach** to ensure you always have guidance:

| Tier | Speed | Source |
|---|---|---|
| **Tier 1** | Instant (0ms) | Rule-based from YOLO + depth |
| **Tier 2** | ~0.3s | Gemini 2.0 Flash (cloud) |
| **Tier 2** | ~3-4s | qwen2.5vl:3b (local) |
| **Emergency** | Instant | YOLO < 0.5m → "STOP!" TTS |

---

## 📁 Project Structure

```
nav_system_ollama/
├── config/
│   └── settings.yaml          # System configuration
├── maps/
│   └── <building>/
│       ├── osmag.json          # Topological map (nodes + edges)
│       ├── dino_index.faiss    # FAISS localization index
│       ├── dino_node_ids.json  # Node ID lookup
│       └── keyframes/          # Extracted keyframe images
├── models/
│   └── yolov8s-worldv2.pt     # YOLO-World model (auto-downloaded)
├── modules/
│   ├── m0_image_enhancer.py   # CLAHE + Gamma enhancement
│   ├── m1_user_interaction.py # TTS (Edge-TTS) + Speech recognition
│   ├── m2_depth_anything.py   # Depth Anything V2 estimator
│   ├── m2_dino_localizer.py   # DINOv2 + FAISS live localization
│   ├── m2_phone_stream.py     # IP Webcam phone stream reader
│   ├── m3_map_builder.py      # OSMAG graph builder
│   ├── m4_path_planner.py     # BFS path planner
│   ├── m5_vlm_guidance.py     # VLM guidance engine
│   ├── m5_yolo.py             # YOLO-World obstacle detector
│   └── m8_sonification.py     # Audio danger tones
├── test_videos/
│   └── house_tour.mp4         # Sample test video
├── tests/
│   ├── visualizer.py          # ← MAIN ENTRY POINT for navigation
│   └── evaluate_repope.py     # Evaluation script
├── scan_walk.py               # Map building script
├── setup_check.py             # Installation verifier
├── requirements.txt
└── README.md
```

---

## 🐛 Troubleshooting

| Error | Solution |
|---|---|
| `Could not open video source: 0` | Check webcam is connected. Try `--video 1` |
| `[Ollama] Connection refused` | Run `ollama serve` in a separate terminal |
| `CUDA out of memory` | Close other GPU apps. Use `--vlm_model qwen2.5:1.5b` |
| `[DINOv2] Localizer failed` | Run map build again: check `maps/<building>/dino_index.faiss` exists |
| `Map building failed` | Ensure `qwen2.5vl:7b` is pulled: `ollama pull qwen2.5vl:7b` |
| `'half' is deprecated` (warning) | Safe to ignore — cosmetic warning from YOLO internals |
| Low FAISS similarity scores | Re-scan building at slower walking speed in similar lighting |

---

## 🔑 Optional: Gemini API Key (Recommended for Speed)

Gemini 2.0 Flash gives ~**10x faster** VLM guidance than local models:

1. Go to [https://aistudio.google.com/app/apikey](https://aistudio.google.com/app/apikey)
2. Click **"Create API Key"** (free, no credit card)
3. Use: `python tests/visualizer.py --gemini_key "AIzaSy..."`

**Free tier:** Sufficient for demo usage.

---

## 👥 Team

Final Year Project — Indoor Navigation System for Visually Impaired Users

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.
