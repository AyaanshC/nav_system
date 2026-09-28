import base64
import cv2
import requests
import json

MODEL_NAME = "qwen2.5vl:3b"
img_path = "C:\\Users\\cayaa\\Downloads\\val2014\\val2014\\COCO_val2014_000000310196.jpg"

img = cv2.imread(img_path)
_, buf = cv2.imencode(".jpg", img)
b64_img = base64.b64encode(buf.tobytes()).decode("utf-8")

payload = {
    "model": MODEL_NAME,
    "messages": [
        {
            "role": "user",
            "content": "Is there a snowboard in the image? Answer strictly yes or no.",
            "images": [b64_img]
        }
    ],
    "stream": False,
    "options": {
        "temperature": 0.1,
        "repeat_penalty": 1.1
    }
}

resp = requests.post("http://localhost:11434/api/chat", json=payload)
print(resp.json())
