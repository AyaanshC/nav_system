import base64
import cv2
from openai import OpenAI

client = OpenAI(api_key="ollama", base_url="http://localhost:11434/v1")

img_path = "C:\\Users\\cayaa\\Downloads\\val2014\\val2014\\COCO_val2014_000000310196.jpg"
img = cv2.imread(img_path)
h, w = img.shape[:2]
new_h = int(h * 320 / w)
img = cv2.resize(img, (320, new_h))
_, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])
b64_img = base64.b64encode(buf.tobytes()).decode("utf-8")

content = [
    {"type": "text", "text": "Please answer the following question with 'Yes' or 'No'.\n\nQuestion: Is there a snowboard in the image?"},
    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64_img}"}}
]

try:
    resp = client.chat.completions.create(
        model="qwen2.5vl:3b",
        temperature=0.1,
        max_tokens=20,
        messages=[{"role": "user", "content": content}]
    )
    print("Answer:", resp.choices[0].message.content)
except Exception as e:
    print("Error:", e)
