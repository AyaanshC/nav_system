# Multimodal Indoor Navigation System (MINS)

![MINS Architecture](https://img.shields.io/badge/Status-Active-brightgreen)
![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue)
![Ollama](https://img.shields.io/badge/Ollama-Local_LLM-orange)
![PyTorch](https://img.shields.io/badge/PyTorch-CUDA_12.1-red)

A real-time, vision-language-driven indoor navigation assistant designed for visually impaired users. 
By utilizing a standard smartphone camera and a local laptop GPU (e.g., RTX 4060), this system provides highly accurate, voice-guided navigation without requiring expensive depth cameras, ROS, or manual map labeling.

## 🌟 Key Features

- **Zero-Manual Mapping:** Simply walk through a building once. The system automatically detects scene changes, captures keyframes, and uses local Vision-Language Models (VLM) to semantically label rooms, corridors, and landmarks.
- **Local AI Pipeline:** 100% local inference ensuring absolute privacy and low latency. Uses **Qwen2.5-VL** (via Ollama) for spatial reasoning and **YOLO-World** + **MiDaS** for obstacle avoidance.
- **DINOv2 Localization:** Sub-meter accuracy localization using Facebook's DINOv2 vision transformer and FAISS vector search.
- **Dynamic Image Enhancement:** OpenCV-powered CLAHE, Unsharp Masking, and Gamma correction pipeline to fix low lighting and motion blur on-the-fly.
- **Caregiver Dashboard:** A live web UI (`localhost:5050`) allowing caretakers to monitor the user's location, path, and live camera feed.

---

## 🏗️ System Architecture

The system operates in two distinct phases:

1. **Offline Map Building (`scan_walk.py`)**:
   - Analyzes video stream using DINOv2 to extract keyframes only when the scene changes.
   - Applies image enhancement to fix lighting/blur.
   - Passes 3-frame sequences to `qwen2.5vl:7b` to generate rich semantic labels and landmarks.
   - Builds a bidirectional topological graph (`osmag.json`) and a FAISS localization index.

2. **Real-Time Navigation (`main.py`)**:
   - User states their destination via voice (Speech-to-Text).
   - The path planner (BFS) calculates the optimal route.
   - Live frames are processed by MiDaS (depth) and YOLO-World (obstacles).
   - Current view + obstacles + path logic are passed to `qwen2.5vl:3b` to generate clear, concise verbal instructions (Edge-TTS).

---

## 🚀 Setup & Installation

### 1. Prerequisites
- **OS:** Windows 10/11 or Linux (WSL2 supported)
- **Hardware:** NVIDIA GPU with at least 6GB VRAM (RTX 3060/4060 recommended)
- **Software:** Python 3.10.x, [Ollama](https://ollama.com/) installed and running.
- **Camera:** Android phone running the [IP Webcam](https://play.google.com/store/apps/details?id=com.pas.webcam) app.

### 2. Environment Setup
```bash
# Clone the repository
git clone https://github.com/yourusername/nav_system.git
cd nav_system

# Create and activate virtual environment
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac

# Install PyTorch with CUDA 12.1
pip install torch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0 --index-url https://download.pytorch.org/whl/cu121

# Install project dependencies
pip install -r requirements.txt
```

### 3. Download Models
```bash
# Create models directory
mkdir models

# Download MiDaS depth model
curl -L -o models/midas_v21_small_256.pt https://github.com/isl-org/MiDaS/releases/download/v2_1/midas_v21_small_256.pt

# YOLO-World will auto-download on the first run.
```

### 4. Setup Ollama
Ensure Ollama is running in the background, then pull the required Vision-Language Models:
```bash
ollama pull qwen2.5-vl:7b  # Used for high-accuracy offline map building
ollama pull qwen2.5-vl:3b  # Used for fast real-time navigation
```

### 5. Configuration
Rename `.env.example` to `.env` (if applicable).
Edit `config/settings.yaml` to include your phone's IP address (from the IP Webcam app):
```yaml
phone_stream_url: "http://192.168.1.X:8080/video"
```

---

## 🗺️ Usage Guide

### Phase 1: Scan Walk (Map Building)
Run this **once per new building**.
```bash
python scan_walk.py --building my_office --stream http://192.168.1.X:8080/video
```
*Instructions:* Walk slowly through the building. The system automatically detects scene changes and captures keyframes. Press `Ctrl+C` when finished. The system will auto-label the map and save it to `maps/my_office/osmag.json`.

*(Optional: Run `python review_labels.py --building my_office` to visually verify label accuracy).*

### Phase 2: Live Navigation
Run this every time you want to navigate.
```bash
python main.py --building my_office --stream http://192.168.1.X:8080/video
```
*Instructions:*
1. **Wait for the system to say "Ready".**
2. **Speak your destination:** (e.g., "Take me to the main conference room").
3. **Follow the voice guidance.**

**Caregiver Dashboard:** Open a browser to `http://localhost:5050` to monitor the user's location, see the active path, and adjust text-to-speech speed.

### Voice Commands
During live navigation, the user can interrupt the system by saying:
- **"Stop"**: Cancels the current route.
- **"Repeat"**: Replays the last verbal instruction.
- **"Where am I"**: Announces the current localized node.
- **"Help"**: Lists available commands.

---

## ⚙️ Configuration (`settings.yaml`)
You can tweak system behavior via `config/settings.yaml` without changing code:
- **`enable_image_enhancement`**: Toggles OpenCV preprocessing (CLAHE, Gamma, Unsharp).
- **`dino_min_similarity`**: Threshold for localization confidence.
- **`yolo_every_n` & `midas_every_n`**: Throttles inference to save compute resources.

---

## 🛡️ License
This project is licensed under the MIT License.
