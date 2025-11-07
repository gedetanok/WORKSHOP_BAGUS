from dotenv import load_dotenv
load_dotenv()

from langchain_openai import ChatOpenAI 
# perlu system prompt
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

# messages = []

# chain = input -> llm -> output
# chain = history message (chatrpmoop) -> llm (chatopenai) -> output

# prompt
SYSTEM_PROMPT = 'you are a helpful assistant.'

prompt = ChatPromptTemplate.from_messages([
    ('system', SYSTEM_PROMPT), # {role: fjkasbf, content: blablaba}
    MessagesPlaceholder("chat_history"),
    ('human', '{input}')
])

llm = ChatOpenAI(
    model='gpt-4o-mini',
)

chain = prompt | llm # prompt -> llm

# store per session
session_store = {}
def get_history(session_id):
    if session_id not in session_store:
        session_store[session_id] = InMemoryChatMessageHistory()
    return session_store[session_id]


# chain
agent = RunnableWithMessageHistory(
    chain, 
    get_history,
    input_messages_key="input",
    history_messages_key="chat_history"
)

# Chat looping
session_id = 'demo-level-2'
while True:
    user_text = input('\nYou: ').strip()
    
    ai_message = agent.invoke({
        'input':user_text},
        config={'configurable':{'session_id':session_id}})
    
    print(f'AI: {ai_message.content}')




