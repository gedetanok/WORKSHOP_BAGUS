from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()

client = OpenAI(
    api_key=os.getenv('OPENAI_API_KEY')
)

try:
    models = client.models.list()
    print('Sukses!')
    print(f'All models: {models.data}')
except Exception as e:
    print(f'{e}')