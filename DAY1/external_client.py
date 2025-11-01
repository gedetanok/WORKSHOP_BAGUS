from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()

client = OpenAI(
    base_url='https://openrouter.ai/api/v1', # ganti kalau mau ganti
    api_key=os.getenv('OPENROUTER_API_KEY') # sesuaikan
)

# client.chat.completions.create()
response = client.chat.completions.create(
    model='x-ai/grok-4-fast', # ganti kalau mau ganti
    messages=[
        {'role':'system', 'content':'You are a helpful assistant.'},
        {'role':'user', 'content':'Jelaskan kepada saya tentang quantum computing.'},
    ]
)

print(response.choices[0].message.content)