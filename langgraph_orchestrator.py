# langgraph_orchestrator.py

from typing import TypedDict, Literal
from langgraph.graph import StateGraph, END
from weaviate_store import weaviate_search
from langchain_helper import get_agent_answer

# Define the state that flows through the graph
class AgentState(TypedDict):
    question: str
    enhanced_question: str
    agent: str
    context: str
    final_answer: str
    status: str
    results: list

# ── Router node ──────────────────────────────────────────────
def router_node(state: AgentState) -> AgentState:
    """
    Decides which agent should handle the question.
    Uses keyword matching — same logic as before but now
    runs as a LangGraph node.
    """
    question = state["enhanced_question"].lower()

    biology_keywords = [
        "dna", "cell", "cells", "protein", "photosynthesis",
        "gene", "genes", "crispr", "mutation", "mutations",
        "rna", "mitochondria", "enzyme", "enzymes", "ribosome",
        "mitosis", "meiosis", "genome", "chromosome", "helix"
    ]
    disease_keywords = [
        "diabetes", "hypertension", "fever", "disease", "blood sugar",
        "corona", "coronavirus", "covid", "covid-19", "virus", "infection",
        "cancer", "tumor", "tumour", "chemotherapy", "metastasis",
        "asthma", "alzheimer"
    ]
    medicine_keywords = [
        "insulin", "aspirin", "antibiotic", "antibiotics",
        "vaccine", "vaccines", "medicine", "paracetamol", "ibuprofen"
    ]
    hospital_keywords = [
        "doctor", "doctors", "nurse", "nurses",
        "hospital", "hospitals", "clinic",
        "pharmacist", "pharmacists", "icu", "surgeon", "radiologist"
    ]
    nutrition_keywords = [
        "nutrition", "carbohydrates", "vitamins", "minerals",
        "fat", "diet", "fiber", "calories", "omega"
    ]

    if any(w in question for w in biology_keywords):
        agent = "biology_agent"
    elif any(w in question for w in disease_keywords):
        agent = "disease_agent"
    elif any(w in question for w in medicine_keywords):
        agent = "medicine_agent"
    elif any(w in question for w in hospital_keywords):
        agent = "hospital_agent"
    elif any(w in question for w in nutrition_keywords):
        agent = "nutrition_agent"
    else:
        agent = "general_agent"

    return {**state, "agent": agent}

# ── Retrieval node ────────────────────────────────────────────
def retrieval_node(state: AgentState) -> AgentState:
    """
    Searches Chroma vector DB for relevant context
    based on the assigned agent.
    """
    agent = state["agent"]
    question = state["enhanced_question"]

    if agent == "general_agent":
        results = weaviate_search(question, top_k=3)
    else:
        results = weaviate_search(question, agent_name=agent, top_k=3)

    context = "\n".join([r["answer"] for r in results])

    return {**state, "context": context, "results": results}

# ── Agent nodes ───────────────────────────────────────────────
def biology_node(state: AgentState) -> AgentState:
    answer = get_agent_answer(
        state["question"],
        state["context"],
        domain="biology and genetics"
    )
    return {**state, "final_answer": answer, "status": "success"}

def disease_node(state: AgentState) -> AgentState:
    answer = get_agent_answer(
        state["question"],
        state["context"],
        domain="diseases and medical conditions"
    )
    return {**state, "final_answer": answer, "status": "success"}

def medicine_node(state: AgentState) -> AgentState:
    answer = get_agent_answer(
        state["question"],
        state["context"],
        domain="medicines and treatments"
    )
    return {**state, "final_answer": answer, "status": "success"}

def hospital_node(state: AgentState) -> AgentState:
    answer = get_agent_answer(
        state["question"],
        state["context"],
        domain="hospital and healthcare"
    )
    return {**state, "final_answer": answer, "status": "success"}

def nutrition_node(state: AgentState) -> AgentState:
    answer = get_agent_answer(
        state["question"],
        state["context"],
        domain="nutrition and diet"
    )
    return {**state, "final_answer": answer, "status": "success"}

def general_node(state: AgentState) -> AgentState:
    answer = get_agent_answer(
        state["question"],
        state["context"],
        domain="life sciences"
    )
    return {**state, "final_answer": answer, "status": "success"}

# ── Routing function ──────────────────────────────────────────
def route_to_agent(state: AgentState) -> Literal[
    "biology_node", "disease_node", "medicine_node",
    "hospital_node", "nutrition_node", "general_node"
]:
    """
    Tells LangGraph which node to go to after routing.
    """
    mapping = {
        "biology_agent":   "biology_node",
        "disease_agent":   "disease_node",
        "medicine_agent":  "medicine_node",
        "hospital_agent":  "hospital_node",
        "nutrition_agent": "nutrition_node",
        "general_agent":   "general_node"
    }
    return mapping.get(state["agent"], "general_node")

# ── Build the graph ───────────────────────────────────────────
def build_graph():
    graph = StateGraph(AgentState)

    # Add all nodes
    graph.add_node("router", router_node)
    graph.add_node("retrieval", retrieval_node)
    graph.add_node("biology_node", biology_node)
    graph.add_node("disease_node", disease_node)
    graph.add_node("medicine_node", medicine_node)
    graph.add_node("hospital_node", hospital_node)
    graph.add_node("nutrition_node", nutrition_node)
    graph.add_node("general_node", general_node)

    # Define the flow
    graph.set_entry_point("router")
    graph.add_edge("router", "retrieval")

    # After retrieval, route to correct agent node
    graph.add_conditional_edges(
        "retrieval",
        route_to_agent
    )

    # All agent nodes end the graph
    graph.add_edge("biology_node", END)
    graph.add_edge("disease_node", END)
    graph.add_edge("medicine_node", END)
    graph.add_edge("hospital_node", END)
    graph.add_edge("nutrition_node", END)
    graph.add_edge("general_node", END)

    return graph.compile()

# Compile once at startup
chatbot_graph = build_graph()

def run_graph(question: str, enhanced_question: str) -> dict:
    """
    Main entry point — runs the full LangGraph pipeline.
    Returns the same format as the old route_question().
    """
    initial_state = AgentState(
        question=question,
        enhanced_question=enhanced_question,
        agent="",
        context="",
        final_answer="",
        status="",
        results=[]
    )

    result = chatbot_graph.invoke(initial_state)

    return {
        "agent": result["agent"],
        "question": question,
        "final_answer": result["final_answer"],
        "results": result["results"],
        "status": result["status"]
    }