import base64
import cv2
from openai import OpenAI

client = OpenAI(api_key="ollama", base_url="http://localhost:11434/v1")
MODEL_NAME = "qwen2.5vl:3b"

img_path = "C:\\Users\\cayaa\\Downloads\\val2014\\val2014\\COCO_val2014_000000310196.jpg"

img = cv2.imread(img_path)
if img is None:
    print("Failed to read image")
    exit(1)

_, buf = cv2.imencode(".jpg", img)
b64_img = base64.b64encode(buf.tobytes()).decode("utf-8")

content = [
    {"type": "text", "text": "Is there a snowboard in the image?"},
    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64_img}"}}
]

response = client.chat.completions.create(
    model=MODEL_NAME,
    max_tokens=20,
    temperature=0.0,
    messages=[
        {"role": "system", "content": "You are a visual assistant. Answer strictly 'yes' or 'no'."},
        {"role": "user", "content": content}
    ]
)

ans = response.choices[0].message.content
print(f"Raw response: {repr(ans)}")
