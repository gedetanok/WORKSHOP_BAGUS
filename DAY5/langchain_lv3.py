from dotenv import load_dotenv
load_dotenv()

from langchain_openai import ChatOpenAI 
# perlu system prompt
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.tools import tool
from langchain.agents import create_tool_calling_agent, AgentExecutor

@tool
def multiply(a: float, b: float) -> float:
    """Kalikan dua angka dan kembalikan nilainya"""
    return a * b

@tool
def to_upper(txt: str) -> str:
    """Ubah semua teks menjadi huruf kapital"""
    return txt.upper()


TOOLS = [multiply, to_upper]
# messages = []

# chain = input -> llm -> output
# chain = history message (chatrpmoop) -> llm (chatopenai) -> output

# prompt
SYSTEM_PROMPT = 'You are a helpful assistant. Use tools when they help solve the users request.' \
'prefer "Multiply" for arithmetic like "x times y". Keep answers short and correct'

prompt = ChatPromptTemplate.from_messages([
    ('system', SYSTEM_PROMPT), # {role: fjkasbf, content: blablaba}
    MessagesPlaceholder("chat_history"),
    ('human', '{input}'),
    MessagesPlaceholder('agent_scratchpad')
])

llm = ChatOpenAI(
    model='gpt-4o-mini',
)

# perlu berpikir: prompt -> LLM -> Analize -> Execute
agent_runnable = create_tool_calling_agent(llm, TOOLS, prompt)
executor = AgentExecutor(agent=agent_runnable, tools=TOOLS)

# store per session
session_store = {}
def get_history(session_id):
    if session_id not in session_store:
        session_store[session_id] = InMemoryChatMessageHistory()
    return session_store[session_id]

# chain
agent_with_memory = RunnableWithMessageHistory(
    executor, 
    get_history,
    input_messages_key="input",
    history_messages_key="chat_history"
)

# Chat looping
session_id = 'demo-level-2'
while True:
    user_text = input('\nYou: ').strip()
    
    ai_message = agent_with_memory.invoke({
        'input':user_text},
        config={'configurable':{'session_id':session_id}})
    
    print(f"AI: {ai_message['output']}")




