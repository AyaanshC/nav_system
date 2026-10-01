import base64
from openai import OpenAI

client = OpenAI(api_key="ollama", base_url="http://localhost:11434/v1")

img_path = "C:\\Users\\cayaa\\Downloads\\val2014\\val2014\\COCO_val2014_000000310196.jpg"

with open(img_path, "rb") as f:
    b64_img = base64.b64encode(f.read()).decode("utf-8")

content = [
    {"type": "text", "text": "Is there a snowboard in the image? Answer strictly yes or no."},
    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64_img}"}}
]

try:
    resp = client.chat.completions.create(
        model="qwen2.5vl:3b",
        temperature=0.0,
        max_tokens=20,
        messages=[{"role": "user", "content": content}]
    )
    print("Answer:", resp.choices[0].message.content)
except Exception as e:
    print("Error:", e)
