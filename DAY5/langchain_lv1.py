from dotenv import load_dotenv
load_dotenv()

from langchain_openai import ChatOpenAI 
# perlu system prompt
from langchain_core.prompts import ChatPromptTemplate

# messages = []

# chain = input -> llm -> output
# chain = history message (chatrpmoop) -> llm (chatopenai) -> output

# prompt
SYSTEM_PROMPT = 'you are a helpful assistant.'

prompt = ChatPromptTemplate([
    ('system', SYSTEM_PROMPT), # {role: fjkasbf, content: blablaba}
    ('human', '{input}')
])

llm = ChatOpenAI(
    model='gpt-4o-mini',
)

chain = prompt | llm # prompt -> llm

# Chat looping
while True:
    user_text = input('\nYou: ').strip()
    
    ai_message = chain.invoke({'input':user_text})
    print(f'AI: {ai_message.content}')




