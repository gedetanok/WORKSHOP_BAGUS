from openai import OpenAI
import os
from dotenv import load_dotenv
from pathlib import Path

load_dotenv()

client = OpenAI(
    api_key=os.getenv('OPENAI_API_KEY') # sesuaikan
)
text = "Halo, saya Bagusde!"

audio_file = Path('audio_output_1.mp3')

with client.audio.speech.with_streaming_response.create(
    model='openai/gpt-4o-mini-tts',
    voice='alloy',
    input=text
) as response:
    response.stream_to_file(audio_file)

