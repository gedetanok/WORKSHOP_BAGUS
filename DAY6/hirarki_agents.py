# Customer support system

from dotenv import load_dotenv
load_dotenv()

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, BaseMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from typing import TypedDict, Annotated, Literal, List

# 1. Define State
class SupportState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    customer_query: str
    query_type: str  # "faq", "technical", "billing"
    worker_response: str
    final_response: str

# 2. Initialize LLM
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)

# 3. Define Manager Agent (Router)

def manager_agent(state: SupportState) -> SupportState:
    """Manager: Klasifikasi query dan route ke worker yang tepat"""
    print("\nMANAGER AGENT: Menganalisis customer query...")

    query = state["customer_query"]

    prompt = f"""Kamu adalah customer support manager.
    Customer query: "{query}"

    Klasifikasikan query ke salah satu kategori:
    - "faq": Pertanyaan umum (pricing, features, account)
    - "technical": Masalah teknis (error, bug, performance)
    - "billing": Masalah pembayaran (invoice, refund, subscription)

    Jawab HANYA dengan satu kata: faq, technical, atau billing
    """

    response = llm.invoke([SystemMessage(content=prompt)])
    query_type = response.content.strip().lower()

    # Validate
    if query_type not in ["faq", "technical", "billing"]:
        query_type = "faq"  # default

    return {
        "messages": [HumanMessage(content=f"Manager: Routing to {query_type}")],
        "query_type": query_type
    }

# 4. Define Worker Agents

def faq_worker(state: SupportState) -> SupportState:
    """FAQ Worker: Handle pertanyaan umum"""
    print("\nFAQ WORKER: Menjawab pertanyaan umum...")

    query = state["customer_query"]

    prompt = f"""Kamu adalah FAQ support agent.
    Customer bertanya: "{query}"

    Jawab dengan ramah dan informatif (2-3 kalimat).
    Referensikan documentation atau help page jika perlu.
    """

    response = llm.invoke([SystemMessage(content=prompt)])
    answer = response.content


    return {
        "messages": [HumanMessage(content=f"FAQ: {answer}")],
        "worker_response": answer
    }

def technical_worker(state: SupportState) -> SupportState:
    """Technical Worker: Handle masalah teknis"""
    print("\nTECHNICAL WORKER: Mengatasi masalah teknis...")

    query = state["customer_query"]

    prompt = f"""Kamu adalah technical support engineer.
    Customer melaporkan: "{query}"

    Berikan:
    1. Diagnosis masalah
    2. Troubleshooting steps (3-4 langkah)
    3. Kapan harus escalate ke senior engineer
    """

    response = llm.invoke([SystemMessage(content=prompt)])
    answer = response.content


    return {
        "messages": [HumanMessage(content=f"Technical: {answer}")],
        "worker_response": answer
    }

def billing_worker(state: SupportState) -> SupportState:
    """Billing Worker: Handle masalah pembayaran"""
    print("\nBILLING WORKER: Mengatasi masalah billing...")

    query = state["customer_query"]

    prompt = f"""Kamu adalah billing support specialist.
    Customer menanyakan: "{query}"

    Berikan:
    1. Penjelasan tentang billing issue
    2. Langkah-langkah resolusi
    3. Timeline penyelesaian
    """

    response = llm.invoke([SystemMessage(content=prompt)])
    answer = response.content

    return {
        "messages": [HumanMessage(content=f"Billing: {answer}")],
        "worker_response": answer
    }

def quality_checker(state: SupportState) -> SupportState:
    """Final QC: Manager review response sebelum dikirim ke customer"""
    print("\nQUALITY CHECKER: Manager review response...")

    worker_answer = state["worker_response"]

    prompt = f"""Kamu adalah QA manager.
    Review response berikut dan polish jika perlu:

    {worker_answer}

    Pastikan:
    - Professional dan empathetic
    - Clear dan actionable
    - Grammar correct

    Output response final yang siap dikirim ke customer.
    """

    response = llm.invoke([SystemMessage(content=prompt)])
    final = response.content

    return {
        "messages": [HumanMessage(content=f"Final: {final}")],
        "final_response": final
    }

# 5. Router Function (untuk conditional edges)

def route_to_worker(state: SupportState) -> Literal["faq_worker", "technical_worker", "billing_worker"]:
    """Routing logic berdasarkan query_type"""
    query_type = state["query_type"]

    if query_type == "faq":
        return "faq_worker"
    elif query_type == "technical":
        return "technical_worker"
    else:  # billing
        return "billing_worker"

# 6. Build Graph - Hierarchical Flow

def create_hierarchical_workflow():
    workflow = StateGraph(SupportState)

    # Add nodes
    workflow.add_node("manager", manager_agent)
    workflow.add_node("faq_worker", faq_worker)
    workflow.add_node("technical_worker", technical_worker)
    workflow.add_node("billing_worker", billing_worker)
    workflow.add_node("quality_checker", quality_checker)

    # Define edges
    workflow.add_edge(START, "manager")

    # CONDITIONAL ROUTING: Manager routes ke worker yang sesuai
    workflow.add_conditional_edges(
        "manager",
        route_to_worker,
        {
            "faq_worker": "faq_worker",
            "technical_worker": "technical_worker",
            "billing_worker": "billing_worker"
        }
    )

    # Semua workers -> quality checker
    workflow.add_edge("faq_worker", "quality_checker")
    workflow.add_edge("technical_worker", "quality_checker")
    workflow.add_edge("billing_worker", "quality_checker")

    # Quality checker -> END
    workflow.add_edge("quality_checker", END)

    return workflow.compile()

# 7. Run the workflow
if __name__ == "__main__":
    print("HIERARCHICAL FLOW: Customer Support System (Manager-Worker)")

    # Create workflow
    app = create_hierarchical_workflow()

    # Test with different query types
    test_queries = [
        "Berapa harga paket premium?",
        "Aplikasi saya crash setiap kali login",
        "Saya belum menerima invoice untuk bulan ini"
    ]

    for idx, query in enumerate(test_queries, 1):
        print(f"\n{'='*70}")
        print(f"TEST CASE #{idx}")
        print(f"{'='*70}")

        initial_state = {
            "messages": [],
            "customer_query": query,
            "query_type": "",
            "worker_response": "",
            "final_response": ""
        }

        final_state = app.invoke(initial_state)
        print(f"FINAL RESPONSE TO CUSTOMER:")
        print(final_state["final_response"])
