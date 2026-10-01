"""
tests/evaluate_repope.py

Automated benchmark script to recreate Table 1 (Hallucination Metrics) from the RePOPE paper.
Compares Qwen2.5-VL (Baseline) against Qwen2.5-VL + YOLO-World (Proposed) on Yes/No object hallucination.

Usage:
  python tests/evaluate_repope.py --mode mini   # Runs a fast 10-image synthetic benchmark
  python tests/evaluate_repope.py --mode full --coco_dir path/to/val2014 --repope_dir path/to/repope_annotations
"""

import argparse
import json
import time
import os
import cv2
import requests
import base64
from pathlib import Path
from openai import OpenAI
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tabulate import tabulate
from loguru import logger

# All 80 standard COCO object classes — used for the benchmark
# to ensure YOLO can detect the same objects the questions ask about.
COCO_80_CLASSES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
    "truck", "boat", "traffic light", "fire hydrant", "stop sign",
    "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep",
    "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
    "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
    "sports ball", "kite", "baseball bat", "baseball glove", "skateboard",
    "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork",
    "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
    "couch", "potted plant", "bed", "dining table", "toilet", "tv",
    "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave",
    "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
    "scissors", "teddy bear", "hair drier", "toothbrush"
]

import sys
import torch
from ultralytics import YOLO as UltralyticsYOLO
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class BenchmarkYOLO:
    """
    Minimal YOLO wrapper for the RePOPE benchmark.
    
    Critically, set_classes() is called BEFORE the first GPU inference
    (warm-up), which prevents the CLIP token_embedding CPU/CUDA mismatch
    that occurs when set_classes() is called after model warm-up.
    """
    def __init__(self, model_path: str, classes: list[str], conf: float = 0.35):
        self.conf = conf
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.classes = classes
        
        logger.info(f"[YOLO] Loading {model_path} for benchmark...")
        self.model = UltralyticsYOLO(model_path)
        
        # MUST call set_classes BEFORE first inference to avoid GPU/CPU split
        self.model.set_classes(classes)
        logger.info(f"[YOLO] Set {len(classes)} COCO classes.")
        
        # Warm up AFTER classes are set
        dummy = __import__('numpy').zeros((480, 640, 3), dtype=__import__('numpy').uint8)
        self.model(dummy, verbose=False, device=self.device, 
                   half=(self.device == "cuda"), conf=self.conf)
        logger.info("[YOLO] Benchmark model warmed up.")

    def detect_object(self, frame_bgr, target_object: str) -> str:
        """
        Detect if target_object is present and return a context string.
        Returns empty string if not found.
        """
        results = self.model(
            frame_bgr, verbose=False, device=self.device,
            half=(self.device == "cuda"), conf=self.conf
        )
        for r in results:
            if r.boxes is None:
                continue
            for box in r.boxes:
                cls_idx = int(box.cls[0])
                if cls_idx < len(self.classes):
                    cls_name = self.classes[cls_idx].lower()
                    if target_object.lower() in cls_name:
                        conf_val = float(box.conf[0])
                        label = "high" if conf_val > 0.7 else "medium"
                        return f"Detected objects: {cls_name} [{label} confidence]."
        return ""

client = OpenAI(
    api_key="ollama",
    base_url="http://localhost:11434/v1"
)
MODEL_NAME = "qwen2.5vl:7b"

# Ollama Client
MINI_COCO_URLS = {
    "COCO_val2014_000000391895.jpg": "http://images.cocodataset.org/val2014/COCO_val2014_000000391895.jpg",
    "COCO_val2014_000000522418.jpg": "http://images.cocodataset.org/val2014/COCO_val2014_000000522418.jpg",
    "COCO_val2014_000000184613.jpg": "http://images.cocodataset.org/val2014/COCO_val2014_000000184613.jpg",
    "COCO_val2014_000000318219.jpg": "http://images.cocodataset.org/val2014/COCO_val2014_000000318219.jpg",
    "COCO_val2014_000000554625.jpg": "http://images.cocodataset.org/val2014/COCO_val2014_000000554625.jpg",
}

# Synthetic POPE Q&A mapping (Image -> Questions & Answers)
MINI_QA = [
    # Image 1 (Motorcycle/Person)
    {"image": "COCO_val2014_000000391895.jpg", "question": "Is there a person in the image?", "label": "yes", "split": "popular"},
    {"image": "COCO_val2014_000000391895.jpg", "question": "Is there a bicycle in the image?", "label": "no", "split": "adversarial"}, # Co-occurrence hallucination trap
    # Image 2 (Food/Dining Table)
    {"image": "COCO_val2014_000000522418.jpg", "question": "Is there a table in the image?", "label": "yes", "split": "popular"},
    {"image": "COCO_val2014_000000522418.jpg", "question": "Is there a chair in the image?", "label": "no", "split": "adversarial"},
    # Image 3 (Cat/Bed)
    {"image": "COCO_val2014_000000184613.jpg", "question": "Is there a cat in the image?", "label": "yes", "split": "random"},
    {"image": "COCO_val2014_000000184613.jpg", "question": "Is there a dog in the image?", "label": "no", "split": "adversarial"},
    # Image 4 (Traffic signs/Cars)
    {"image": "COCO_val2014_000000318219.jpg", "question": "Is there a sign in the image?", "label": "yes", "split": "popular"},
    {"image": "COCO_val2014_000000318219.jpg", "question": "Is there a person in the image?", "label": "no", "split": "random"},
    # Image 5 (Kitchen)
    {"image": "COCO_val2014_000000554625.jpg", "question": "Is there an oven in the image?", "label": "yes", "split": "random"},
    {"image": "COCO_val2014_000000554625.jpg", "question": "Is there a person in the image?", "label": "no", "split": "adversarial"},
]

def download_mini_dataset(data_dir: Path):
    """Downloads the 5 mini benchmark images from COCO."""
    img_dir = data_dir / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("Downloading Mini-Benchmark images...")
    for filename, url in MINI_COCO_URLS.items():
        filepath = img_dir / filename
        if not filepath.exists():
            r = requests.get(url, allow_redirects=True)
            with open(filepath, 'wb') as f:
                f.write(r.content)
    logger.info("Mini-Benchmark images ready.")
    return img_dir

def _encode_image(image_path: str) -> str:
    img = cv2.imread(image_path)
    h, w = img.shape[:2]
    new_h = int(h * 320 / w)
    img = cv2.resize(img, (320, new_h))
    _, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return base64.b64encode(buf.tobytes()).decode("utf-8")

def ask_vlm(image_path: str, question: str, yolo_context: str = "") -> str:
    """Asks the VLM a yes/no question about the image."""
    b64_img = _encode_image(image_path)
    
    # Paper methodology: "If object detected, add to prompt"
    # Merging sys prompt into user text because Qwen-VL system prompts can cause Ollama loop bugs
    instruction = "You are a visual assistant. Briefly describe if the object is present, then end your answer with strictly [Yes] or [No]."
    user_text = f"{instruction}\n\nQuestion: {question}"
    if yolo_context:
        user_text = f"Object Detector Context: {yolo_context}\n\n{instruction}\n\nQuestion: {question}"
        
    content = [
        {"type": "text", "text": user_text},
        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64_img}", "detail": "low"}}
    ]
    
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            max_tokens=50,
            temperature=0.7,
            messages=[
                {"role": "user", "content": content}
            ]
        )
        ans = response.choices[0].message.content.strip().lower()
        # Check for [Yes]/[No] CoT tags first, then fall back to plain yes/no
        if "[yes]" in ans: return "yes"
        if "[no]" in ans: return "no"
        if "yes" in ans: return "yes"
        if "no" in ans: return "no"
        return "no"
    except Exception as e:
        logger.error(f"VLM API failed with error: {e}")
        return "no"

def _extract_object_from_question(question: str) -> str:
    """Extracts the object being asked about from a POPE-style question.
    e.g. 'Is there a snowboard in the image?' -> 'snowboard'
    """
    import re
    # Match "Is there a/an <object> in the image?"
    match = re.search(r"Is there an? (.+?) in the image", question, re.IGNORECASE)
    if match:
        return match.group(1).strip().lower()
    return ""

def evaluate(qa_pairs, img_dir, yolo_detector=None):
    """Runs evaluation and calculates metrics."""
    y_true = []
    y_pred = []
    
    logger.info(f"Running evaluation on {len(qa_pairs)} questions...") 
    for idx, qa in enumerate(qa_pairs):
        img_path = str(img_dir / qa["image"])
        
        if not os.path.exists(img_path):
            logger.error(f"Image not found: {img_path}")
            logger.error(f"Did you extract the val2014.zip file correctly? Make sure --coco_dir points directly to the folder containing the .jpg files (e.g., C:\\Downloads\\val2014\\val2014)")
            sys.exit(1)
            
        question = qa.get("text", qa.get("question", ""))
        label = qa["label"].lower()
        
        yolo_context = ""
        if yolo_detector:
            frame = cv2.imread(img_path)
            target_object = _extract_object_from_question(question)
            if target_object:
                yolo_context = yolo_detector.detect_object(frame, target_object)
                
        pred = ask_vlm(img_path, question, yolo_context)
        
        # Convert to binary for sklearn metrics
        y_true.append(1 if label == "yes" else 0)
        y_pred.append(1 if pred == "yes" else 0)
        
        print(f"[{idx+1}/{len(qa_pairs)}] Q: {question} | True: {label} | Pred: {pred} | Context: {yolo_context}")
        
    acc = accuracy_score(y_true, y_pred) * 100
    prec = precision_score(y_true, y_pred, zero_division=0) * 100
    rec = recall_score(y_true, y_pred, zero_division=0) * 100
    f1 = f1_score(y_true, y_pred, zero_division=0) * 100
    
    return acc, prec, rec, f1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["mini", "full"], default="mini")
    parser.add_argument("--coco_dir", type=str, default="", help="Path to COCO val2014 images")
    parser.add_argument("--repope_dir", type=str, default="", help="Path to RePOPE annotations directory")
    parser.add_argument("--max_per_split", type=int, default=0,
                        help="Limit questions per split (0 = use all). E.g. --max_per_split 200 for a ~1hr run.")
    args = parser.parse_args()
    
    qa_pairs = []
    if args.mode == "full":
        if not args.coco_dir or not args.repope_dir:
            logger.error("--coco_dir and --repope_dir are required for full mode.")
            return
        img_dir = Path(args.coco_dir)
        
        # Load all three RePOPE splits
        splits_mapping = {
            "adversarial": "coco_repope_adversarial.json",
            "popular": "coco_repope_popular.json",
            "random": "coco_repope_random.json"
        }
        for split_name, filename in splits_mapping.items():
            filepath = Path(args.repope_dir) / filename
            if filepath.exists():
                with open(filepath, "r") as f:
                    # Depending on format (list of dicts or JSONL)
                    try:
                        data = json.load(f)
                        for item in data:
                            item["split"] = split_name
                            qa_pairs.append(item)
                    except json.JSONDecodeError:
                        # Fallback for JSONL
                        f.seek(0)
                        for line in f:
                            item = json.loads(line)
                            item["split"] = split_name
                            qa_pairs.append(item)
    else:
        # MINI MODE
        data_dir = Path("tests/data")
        img_dir = download_mini_dataset(data_dir)
        qa_pairs = MINI_QA
        
    # Group by split for Table 1 reproduction
    splits = ["adversarial", "random", "popular"]
    results_table = []
    
    logger.info("=== Loading YOLO-World ===")
    # Initialize with all 80 COCO classes ONCE so it can detect any queried object.
    # Per-question, the evaluate() function filters to only the relevant detection.
    yolo = BenchmarkYOLO("models/yolov8s-worldv2.pt", COCO_80_CLASSES)
    
    for split in splits:
        split_qa = [qa for qa in qa_pairs if qa["split"] == split]
        if not split_qa:
            continue
        
        # Optionally cap the number of questions for a faster run
        if args.max_per_split and args.max_per_split > 0:
            split_qa = split_qa[:args.max_per_split]
            logger.info(f"[Capped to {args.max_per_split} questions per split]")
            
        logger.info(f"\n--- Evaluating Split: {split.upper()} ---")
        
        logger.info("1. Baseline (Qwen-VL Only)")
        b_acc, b_prec, b_rec, b_f1 = evaluate(split_qa, img_dir, yolo_detector=None)
        
        logger.info("2. Proposed (Qwen-VL + YOLO)")
        p_acc, p_prec, p_rec, p_f1 = evaluate(split_qa, img_dir, yolo_detector=yolo)
        
        results_table.append([
            split.capitalize(), "Qwen-VL", 
            f"{b_acc:.2f}", f"{b_prec:.2f}", f"{b_rec:.2f}", f"{b_f1:.2f}"
        ])
        results_table.append([
            "", "Qwen-VL + YOLO", 
            f"{p_acc:.2f}", f"{p_prec:.2f}", f"{p_rec:.2f}", f"{p_f1:.2f}"
        ])
        
    # Print Table 1
    print("\n\n" + "="*80)
    print("Table 1. Performance comparison on the RePOPE dataset.")
    print("="*80)
    headers = ["Setting", "Model", "Accuracy (%)", "Precision (%)", "Recall (%)", "F1-Score (%)"]
    print(tabulate(results_table, headers=headers, tablefmt="github"))
    print("="*80)

if __name__ == "__main__":
    main()
