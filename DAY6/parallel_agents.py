# market research agent
from dotenv import load_dotenv
load_dotenv()

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, BaseMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from typing import TypedDict, Annotated, List

# 1. Define State
class MarketState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    product_name: str
    competitor_analysis: str
    customer_insights: str
    trend_analysis: str
    final_report: str

# 2. Initialize LLM
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)

# 3. Define Parallel Agent Nodes
def competitor_analyst(state: MarketState) -> MarketState:
    """Agent 1: Analisis kompetitor (berjalan parallel)"""
    print("\nCOMPETITOR ANALYST: Menganalisis kompetitor...")

    prompt = f"""Kamu adalah competitor analyst.
    Analisis 3 kompetitor utama untuk produk: {state['product_name']}

    Berikan:
    1. Nama kompetitor
    2. Kekuatan mereka
    3. Kelemahan mereka
    """

    response = llm.invoke([SystemMessage(content=prompt)])
    analysis = response.content

    return {
        "messages": [HumanMessage(content=f"Competitor: {analysis}")],
        "competitor_analysis": analysis
    }

def customer_insight_agent(state: MarketState) -> MarketState:
    """Agent 2: Analisis customer (berjalan parallel)"""
    print("\nCUSTOMER INSIGHT AGENT: Menganalisis customer needs...")

    prompt = f"""Kamu adalah customer research analyst.
    Analisis kebutuhan customer untuk produk: {state['product_name']}

    Berikan:
    1. Target customer segment
    2. Pain points mereka
    3. What they value most
    """

    response = llm.invoke([SystemMessage(content=prompt)])
    insights = response.content
    

    return {
        "messages": [HumanMessage(content=f"Customer: {insights}")],
        "customer_insights": insights
    }

def trend_analyst(state: MarketState) -> MarketState:
    """Agent 3: Analisis tren pasar (berjalan parallel)"""
    print("\nTREND ANALYST: Menganalisis market trends...")

    prompt = f"""Kamu adalah market trend analyst.
    Analisis tren pasar terkait: {state['product_name']}

    Berikan:
    1. Current market trends
    2. Emerging opportunities
    3. Potential threats
    """

    response = llm.invoke([SystemMessage(content=prompt)])
    trends = response.content

    return {
        "messages": [HumanMessage(content=f"Trends: {trends}")],
        "trend_analysis": trends
    }

def report_synthesizer(state: MarketState) -> MarketState:
    """Agent 4: Menggabungkan semua hasil parallel agents"""
    print("\nREPORT SYNTHESIZER: Menggabungkan semua hasil...")

    prompt = f"""Kamu adalah strategic analyst.

    Gabungkan 3 analisis berikut menjadi 1 executive summary:

    COMPETITOR ANALYSIS:
    {state['competitor_analysis']}

    CUSTOMER INSIGHTS:
    {state['customer_insights']}

    MARKET TRENDS:
    {state['trend_analysis']}

    Buat executive summary (200 kata) dengan:
    1. Key Findings
    2. Strategic Recommendations
    3. Next Steps
    """

    response = llm.invoke([SystemMessage(content=prompt)])
    report = response.content

    return {
        "messages": [HumanMessage(content=f"Report: {report}")],
        "final_report": report
    }

# 4. Build Graph - Parallel Flow
def create_parallel_workflow():
    workflow = StateGraph(MarketState)

    # Add nodes
    workflow.add_node("competitor_analyst", competitor_analyst)
    workflow.add_node("customer_insight", customer_insight_agent)
    workflow.add_node("trend_analyst", trend_analyst)
    workflow.add_node("synthesizer", report_synthesizer)

    # PARALLEL FLOW:
    # START -> 3 agents berjalan parallel
    workflow.add_edge(START, "competitor_analyst")
    workflow.add_edge(START, "customer_insight")
    workflow.add_edge(START, "trend_analyst")

    # Semua converge ke synthesizer
    workflow.add_edge("competitor_analyst", "synthesizer")
    workflow.add_edge("customer_insight", "synthesizer")
    workflow.add_edge("trend_analyst", "synthesizer")

    # Synthesizer -> END
    workflow.add_edge("synthesizer", END)

    return workflow.compile()

# 5. Run the workflow
if __name__ == "__main__":
    print("PARALLEL FLOW: Market Research System")


    # Create workflow
    app = create_parallel_workflow()

    # Initial state
    initial_state = {
        "messages": [],
        "product_name": "AI-powered productivity app untuk remote workers",
        "competitor_analysis": "",
        "customer_insights": "",
        "trend_analysis": "",
        "final_report": ""
    }

    # Execute workflow
    final_state = app.invoke(initial_state)

    # Show final result
    print("=� EXECUTIVE SUMMARY:")

    print(final_state["final_report"])
