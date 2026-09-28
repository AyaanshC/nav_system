import base64
import cv2
from openai import OpenAI

client = OpenAI(api_key="ollama", base_url="http://localhost:11434/v1")

# Use resized image exactly like evaluate_repope.py
img_path = "C:\\Users\\cayaa\\Downloads\\val2014\\val2014\\COCO_val2014_000000310196.jpg"
img = cv2.imread(img_path)
h, w = img.shape[:2]
new_h = int(h * 320 / w)
img = cv2.resize(img, (320, new_h))
_, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])
b64_img = base64.b64encode(buf.tobytes()).decode("utf-8")

content = [
    {"type": "text", "text": "What objects do you see in this image? Describe them."},
    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64_img}", "detail": "low"}}
]

resp = client.chat.completions.create(
    model="qwen2.5vl:3b",
    messages=[{"role": "user", "content": content}]
)
print("Description:", resp.choices[0].message.content)

content2 = [
    {"type": "text", "text": "Is there a person in the image? Answer strictly 'yes' or 'no'."},
    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64_img}", "detail": "low"}}
]
resp2 = client.chat.completions.create(
    model="qwen2.5vl:3b",
    temperature=0.0,
    messages=[
        {"role": "system", "content": "You are a visual assistant. Answer strictly 'yes' or 'no'."},
        {"role": "user", "content": content2}
    ]
)
print("Yes/No Answer:", resp2.choices[0].message.content)
