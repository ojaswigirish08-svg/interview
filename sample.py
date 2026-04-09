import requests
import base64
import os
from dotenv import load_dotenv
load_dotenv()

API_KEY = os.getenv("MISTRAL_API_KEY")

# Convert audio → base64
with open("output.mp3", "rb") as f:
    ref_audio = base64.b64encode(f.read()).decode()

url = "https://api.mistral.ai/v1/audio/speech"

headers = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json"
}

data = {
    "model": "voxtral-mini-tts-2603",
    "input": """Hello Surya. Let's begin your interview.


Can you explain the difference between blocking and non-blocking assignments in Verilog?""",
    "ref_audio": ref_audio,
    "response_format": "mp3"
}

response = requests.post(url, headers=headers, json=data)

print("Status:", response.status_code)

if response.status_code == 200:
    result = response.json()   # ✅ parse JSON

    audio_base64 = result["audio_data"]   # ✅ extract
    audio_bytes = base64.b64decode(audio_base64)  # ✅ decode

    with open("sample.mp3", "wb") as f:
        f.write(audio_bytes)

    print("✅ sample.mp3 saved (working audio)")
else:
    print("❌ Error:", response.text)