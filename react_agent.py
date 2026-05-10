# react_agent.py

from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, ToolMessage
from vector_store import chroma_search
from dotenv import load_dotenv
import os
import json

load_dotenv()

llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.3,
    api_key=os.getenv("OPENAI_API_KEY")
)

# ── Define tools ──────────────────────────────────────────────

@tool
def search_biology(query: str) -> str:
    """Search biology knowledge base for DNA, cells, genes, proteins, CRISPR."""
    results = chroma_search(query, agent_name="biology_agent", top_k=3)
    return "\n".join([r["answer"] for r in results]) if results else "No results found."

@tool
def search_disease(query: str) -> str:
    """Search disease knowledge base for diabetes, cancer, COVID-19, infections."""
    results = chroma_search(query, agent_name="disease_agent", top_k=3)
    return "\n".join([r["answer"] for r in results]) if results else "No results found."

@tool
def search_medicine(query: str) -> str:
    """Search medicine knowledge base for insulin, antibiotics, vaccines."""
    results = chroma_search(query, agent_name="medicine_agent", top_k=3)
    return "\n".join([r["answer"] for r in results]) if results else "No results found."

@tool
def search_hospital(query: str) -> str:
    """Search hospital knowledge base for doctors, nurses, clinics."""
    results = chroma_search(query, agent_name="hospital_agent", top_k=3)
    return "\n".join([r["answer"] for r in results]) if results else "No results found."

@tool
def search_nutrition(query: str) -> str:
    """Search nutrition knowledge base for vitamins, diet, minerals."""
    results = chroma_search(query, agent_name="nutrition_agent", top_k=3)
    return "\n".join([r["answer"] for r in results]) if results else "No results found."

tools = [
    search_biology,
    search_disease,
    search_medicine,
    search_hospital,
    search_nutrition
]

# Bind tools to LLM
llm_with_tools = llm.bind_tools(tools)

# Tool name to function map
tool_map = {t.name: t for t in tools}

def run_react_agent(question: str) -> dict:
    """
    Manual ReAct loop using tool calling.
    Shows reasoning steps clearly.
    """
    messages = [
        HumanMessage(content=f"""You are a life sciences AI assistant.
Use the available tools to search for relevant information, 
then provide a detailed answer.

Question: {question}
""")
    ]

    reasoning_steps = []
    final_answer = ""
    max_iterations = 5

    for i in range(max_iterations):
        response = llm_with_tools.invoke(messages)
        messages.append(response)

        # If no tool calls — we have final answer
        if not response.tool_calls:
            final_answer = response.content
            break

        # Process each tool call
        for tool_call in response.tool_calls:
            tool_name = tool_call["name"]
            tool_input = tool_call["args"]

            # Record reasoning step
            reasoning_steps.append({
                "thought": f"I need to search for information about this topic",
                "action": tool_name,
                "input": str(tool_input.get("query", "")),
                "observation": ""
            })

            # Run the tool
            if tool_name in tool_map:
                observation = tool_map[tool_name].invoke(tool_input)
            else:
                observation = "Tool not found."

            # Update observation in reasoning steps
            reasoning_steps[-1]["observation"] = str(observation)[:300]

            # Add tool result to messages
            messages.append(
                ToolMessage(
                    content=str(observation),
                    tool_call_id=tool_call["id"]
                )
            )

    # If no final answer yet, get one
    if not final_answer:
        final_response = llm_with_tools.invoke(messages)
        final_answer = final_response.content or "No answer found."

    return {
        "final_answer": final_answer,
        "reasoning_steps": reasoning_steps,
        "status": "react_success"
    }