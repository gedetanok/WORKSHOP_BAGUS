# agent untuk menulis blog
# State -> menyimpan semua data (user_input, output: research
#.                              output: writer, Ouput: editor, final_article) -> AI Response = State['output_research']
# input -> research -> writer -> editor - output ke user (final artikel)
from dotenv import load_dotenv
load_dotenv()

from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_core.messages import HumanMessage, SystemMessage, BaseMessage
from typing import TypedDict, Annotated, List
from langgraph.checkpoint.memory import MemorySaver


class AgenState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    topic: str
    research_result: str
    draft_article: str # tempat hasil writer agent
    final_article: str # tempat hasil editor agent


llm = ChatOpenAI(model='gpt-4o-mini', temperature=0.7) # 0 - 1 -> 4932 x 3848 berapa? 

def research_agent(State: AgenState) -> AgenState:
    """Melakukan riset tentang topik"""

    print('Resarch Agent sedang melakukan tugasnya...')

    prompt = f"""Kamu adalah research assistant.
    Riset topik berikut dan berikan 3 - 5 poin penting: {State['messages']}

    Format output: 
    - Poin 1
    - Poin 2
    - Poin 3
    - dst...
    """

    response = llm.invoke([SystemMessage(prompt)])
    research = response.content

    print(f'HASIL: {research[:100]}...')

    return {
        'messages': [SystemMessage(content=f'Research: {research}')],
        'research_result': research
    }

def writer_agent(State: AgenState) -> AgenState:
    """Menulis draft artikel berdasarkan hasil riset"""

    print('Writer Agent sedang melakukan tugasnya...')

    prompt = f"""Kamu adalah content writer.
    Topik: {State['topic']}

    Berdasrakan reserch berikut: 
    {State['research_result']}

    Tulis artikel blog (300 - 400 kata) dengan struktur: 
    - Judul menarik
    - Intro
    - Body (3 Paragraf)
    - Kesimpulan
    """

    response = llm.invoke([SystemMessage(prompt)])
    draft = response.content

    print(f'HASIL: {draft[:100]}...')

    return {
        'messages': [SystemMessage(content=f'Research: {draft}')],
        'draft_article': draft
    }

def editor_agent(State: AgenState) -> AgenState:
    """Mengedit dan memperbaiki artikel"""

    print('Editor Agent sedang melakukan tugasnya...')

    prompt = f"""Kamu adalah seorang artikel editor profesional.

    Draft artikel: 
    {State['draft_article']}

    Tugasmu: 
    1. Perbaiki grammar dan typo
    2. Improve flow dan readability
    3. Pastikan struktur jelas
    4. Output final artikel yang sudah dipoles
    """

    response = llm.invoke([SystemMessage(prompt)])
    final = response.content

    print(f'HASIL: {final[:100]}...')

    return {
        'messages': [SystemMessage(content=f'Editor: {final}')],
        'final_article': final
    }


def create_sequential_flow():
    workflow = StateGraph(AgenState)

    # bangun flownya
    workflow.add_node('researcher', research_agent)
    workflow.add_node('writer', writer_agent)
    workflow.add_node('editor', editor_agent)

    workflow.add_edge(START, 'researcher')
    workflow.add_edge('researcher', 'writer')
    workflow.add_edge('writer', 'editor')
    workflow.add_edge('editor', END)

    checkpointer = MemorySaver()
    return workflow.compile(checkpointer=checkpointer)

if __name__ == "__main__":
    print('SEQUENTIAL FLOW: Blog article generator')
    app = create_sequential_flow()
    THREAD_ID = "jennie-cantik"

    user_input = input('\nYou: ').strip()
    #
    initial_state = {
        'messages': [HumanMessage(user_input)],
        'topic': user_input,
        "research_result": "",
        "draft_article": '', # tempat hasil writer agent
        "final_article": '' # tempat hasil editor agent
    }

    app.invoke(initial_state, config={'configurable':{'thread_id':THREAD_ID}})

    final_state = app.get_state(config={'configurable':{'thread_id':THREAD_ID}})

    print('ARTIKEL FINAL: ')
    print(f'{final_state.values.get('final_article')}')

    while True:
        user_input = input('\nYou: ').strip()

        delta = {
            'messages': [HumanMessage(content=user_input)]
        }

        app.invoke(delta, config={'configurable':{'thread_id':THREAD_ID}})

        current = app.get_state(config={'configurable':{'thread_id':THREAD_ID}})
        current_article = current.values.get('final_article')

        print('ARTIKEL TERBARU: ')
        print(current_article)