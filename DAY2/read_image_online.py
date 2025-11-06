from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(
    api_key=os.getenv('OPENAI_API_KEY')
)

image_url = 'https://imageio.forbes.com/specials-images/imageserve/664b9c049314ec4607e8ee46/Porsche-Sonderwunsch-Haus--The-Taycan-4S-Cross-Turismo-For-Jennie-Ruby-Jane---/0x0.jpg?format=jpg&crop=2432,1824,x0,y291,safe&width=960'

response = client.chat.completions.create(
    model='gpt-4o',
    messages=[
        {'role':'user', 'content':[
            {'type':'text', 'text':'Can you please describe this image?'},
            {'type':'image_url', 'image_url':{'url':image_url}},
        ]}
    ]
)

print(response.choices[0].message.content)
