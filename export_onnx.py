import argparse
from ultralytics import YOLO
from loguru import logger
import os

def export_yolo_to_onnx(model_path: str = "yolov8s-world.pt"):
    """
    Exports the YOLO-World PyTorch model to ONNX for faster inference.
    """
    logger.info(f"Loading YOLO model from {model_path} for export...")
    if not os.path.exists(model_path):
        logger.error(f"Model file not found: {model_path}. Run the system once to download it.")
        return

    try:
        model = YOLO(model_path)
        logger.info("Exporting to ONNX format...")
        # export to onnx
        success = model.export(format="onnx", half=True)  # half precision for speed
        logger.info(f"Export successful. ONNX model saved to: {success}")
        logger.info("Update config or yolo_detector.py to point to the new .onnx file to use it.")
    except Exception as e:
        logger.error(f"Failed to export model: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export YOLO model to ONNX for performance.")
    parser.add_argument("--model", type=str, default="yolov8s-world.pt", help="Path to the PyTorch model (.pt)")
    args = parser.parse_args()

    export_yolo_to_onnx(args.model)
