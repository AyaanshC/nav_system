"""
Isolates the exact problem - test what Qwen returns for a real image question.
"""
import base64
import cv2
from openai import OpenAI

client = OpenAI(api_key="ollama", base_url="http://localhost:11434/v1")
img_path = "C:\\Users\\cayaa\\Downloads\\val2014\\val2014\\COCO_val2014_000000310196.jpg"

# Read + resize same as evaluate_repope.py
img = cv2.imread(img_path)
h, w = img.shape[:2]
new_h = int(h * 320 / w)
img = cv2.resize(img, (320, new_h))
_, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])
b64_img = base64.b64encode(buf.tobytes()).decode("utf-8")

print(f"Image size after resize: {img.shape}")
print(f"Base64 length: {len(b64_img)}")

instruction = "You are a visual assistant. Briefly describe if the object is present, then end your answer with strictly [Yes] or [No]."
question = "Is there a snowboard in the image?"
user_text = f"{instruction}\n\nQuestion: {question}"

content = [
    {"type": "text", "text": user_text},
    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64_img}", "detail": "low"}}
]

# Test 3b first
print("\n--- Testing qwen2.5vl:3b ---")
try:
    resp = client.chat.completions.create(
        model="qwen2.5vl:3b",
        max_tokens=50,
        temperature=0.7,
        messages=[{"role": "user", "content": content}]
    )
    print(f"3b Raw: {repr(resp.choices[0].message.content)}")
except Exception as e:
    print(f"3b Error: {e}")

# Test 7b
print("\n--- Testing qwen2.5vl:7b ---")
try:
    resp = client.chat.completions.create(
        model="qwen2.5vl:7b",
        max_tokens=50,
        temperature=0.7,
        messages=[{"role": "user", "content": content}]
    )
    print(f"7b Raw: {repr(resp.choices[0].message.content)}")
except Exception as e:
    print(f"7b Error: {e}")
