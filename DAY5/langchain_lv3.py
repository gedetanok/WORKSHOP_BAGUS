from dotenv import load_dotenv
load_dotenv()

from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate
from langchain.agents import create_agent
from langgraph.checkpoint.memory import MemorySaver

# 1) Definisikan tools
@tool
def multiply(a: float, b: float) -> float:
    """Kalikan dua angka dan kembalikan hasilnya."""
    return a * b

@tool
def to_upper(text: str) -> str:
    """Ubah teks menjadi huruf kapital."""
    return text.upper()

TOOLS = [multiply, to_upper]

# 2) System prompt untuk agent
SYSTEM_PROMPT = """You are a helpful assistant.
Use tools when they help solve the user's request.
Prefer `multiply` for arithmetic like "x times y".
Keep answers short and correct.
"""

# 3) Buat LLM & Agent dengan memory
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
memory = MemorySaver()
agent_executor = create_agent(
    llm,
    TOOLS,
    checkpointer=memory,
    system_prompt=SYSTEM_PROMPT
)

# 4) Loop percakapan (agent + tools + memory)
if __name__ == "__main__":
    thread_id = "demo-user-2"
    config = {"configurable": {"thread_id": thread_id}}

    while True:
        user_text = input("\nYou: ").strip()
        if user_text.lower() in {"exit", "quit"}:
            print("AI: Bye!"); break

        # Invoke agent
        result = agent_executor.invoke(
            {"messages": [("user", user_text)]},
            config=config
        )

        # Tampilkan response terakhir dari AI
        if result["messages"]:
            last_message = result["messages"][-1]
            if hasattr(last_message, 'content'):
                print(f"AI: {last_message.content}")
