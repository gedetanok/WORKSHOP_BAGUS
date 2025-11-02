from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(
    api_key=os.getenv('OPENAI_API_KEY') # sesuaikan
)

audio_path = 'audio_dummy.mp3'

# client.audio.transcriptions.create()
with open(audio_path, 'rb') as audio_file:
    result = client.audio.transcriptions.create(
        model='whisper-1',
        file=audio_file,
        response_format='text'
    )

print(f'Hasil transkripsi: \n{result}')