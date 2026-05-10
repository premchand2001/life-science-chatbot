# react_agent.py
# Now uses proper MCP tool schemas

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, ToolMessage
from mcp_tools import get_mcp_tools, get_mcp_schemas
from dotenv import load_dotenv
import os

load_dotenv()

llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.3,
    api_key=os.getenv("OPENAI_API_KEY")
)

# Load MCP tools
tools = get_mcp_tools()

# Bind MCP tools to LLM
llm_with_tools = llm.bind_tools(tools)

# Tool name to function map
tool_map = {t.name: t for t in tools}

def run_react_agent(question: str) -> dict:
    """
    Runs ReAct agent using MCP tools.
    Shows reasoning steps and which MCP tool was used.
    """
    messages = [
        HumanMessage(content=f"""You are a life sciences AI assistant.
You have access to MCP tools that can search knowledge bases
and real external APIs like OpenFDA and ClinicalTrials.gov.

Use the most relevant tool(s) to answer this question accurately.

Question: {question}
""")
    ]

    reasoning_steps = []
    final_answer = ""
    max_iterations = 5

    for i in range(max_iterations):
        response = llm_with_tools.invoke(messages)
        messages.append(response)

        # No tool calls means we have final answer
        if not response.tool_calls:
            final_answer = response.content
            break

        # Process each tool call
        for tool_call in response.tool_calls:
            tool_name = tool_call["name"]
            tool_input = tool_call["args"]

            # Get MCP schema for this tool
            from mcp_tools import MCP_TOOL_SCHEMAS
            schema = MCP_TOOL_SCHEMAS.get(tool_name, {})

            # Record reasoning step with MCP info
            reasoning_steps.append({
                "thought": f"Using MCP tool to find relevant information",
                "action": tool_name,
                "input": str(tool_input),
                "observation": "",
                "mcp_schema": schema.get("description", "")
            })

            # Run the MCP tool
            if tool_name in tool_map:
                observation = tool_map[tool_name].invoke(tool_input)
            else:
                observation = "MCP tool not found."

            # Update observation
            reasoning_steps[-1]["observation"] = str(observation)[:400]

            # Add tool result to messages
            messages.append(
                ToolMessage(
                    content=str(observation),
                    tool_call_id=tool_call["id"]
                )
            )

    # Get final answer if not yet received
    if not final_answer:
        final_response = llm_with_tools.invoke(messages)
        final_answer = final_response.content or "No answer found."

    return {
        "final_answer": final_answer,
        "reasoning_steps": reasoning_steps,
        "status": "react_success"
    }

def get_available_mcp_tools() -> list:
    """Returns list of available MCP tool names and descriptions."""
    schemas = get_mcp_schemas()
    return [
        {
            "name": name,
            "description": schema["description"]
        }
        for name, schema in schemas.items()
    ]