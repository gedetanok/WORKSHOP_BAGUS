from openai import OpenAI
import os
from dotenv import load_dotenv
import base64

load_dotenv()

client = OpenAI(
    api_key=os.getenv('OPENAI_API_KEY')
)

# image_path = asset/kucing-ayam.jpg
def decode_image(image_path):
    with open(image_path, 'rb') as file:
        return base64.b64encode(file.read()).decode()

image_url = 'asset/kucing-ayam.jpg' 


# response = client.chat.completions.create(
#     model='gpt-4o',
#     messages=[
#         {'role':'user', 'content':[
#             {'type':'text', 'text':'Can you please describe this image?'},
#             {'type':'image_url', 'image_url':{'url':image_url}}
#         ]}
#     ]
# )

# print(response.choices[0].message.content)


print(decode_image(image_url))