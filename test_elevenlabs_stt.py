import os
from dotenv import load_dotenv
from io import BytesIO
import requests
from elevenlabs.client import ElevenLabs

load_dotenv()

elevenlabs = ElevenLabs(
    api_key=os.getenv("ELEVENLABS_API_KEY"),
)

# Option 1: Transcribe from a URL (sample audio)
audio_url = "https://storage.googleapis.com/eleven-public-cdn/audio/marketing/nicole.mp3"
response = requests.get(audio_url)
audio_data = BytesIO(response.content)

# Option 2: Transcribe from a local file (uncomment to use)
# with open("your_audio.mp3", "rb") as f:
#     audio_data = f

transcription = elevenlabs.speech_to_text.convert(
    file=audio_data,
    model_id="scribe_v2",
    tag_audio_events=True,
    language_code="eng",
    diarize=True,
)

# Print full result
print("=== Full Transcription ===")
print(transcription.text)

# Print word-level details if available
if transcription.words:
    print("\n=== Word-level Details ===")
    for word in transcription.words[:20]:  # first 20 words
        print(f"  {word.text} ({word.start:.2f}s - {word.end:.2f}s)")
