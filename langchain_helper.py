# langchain_helper.py

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
import os

load_dotenv()

# Initialize the LangChain LLM
llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.3,
    api_key=os.getenv("OPENAI_API_KEY")
)

# Output parser — converts LLM output to plain string
parser = StrOutputParser()

# Main prompt template for life sciences QA
qa_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are a helpful, accurate life sciences AI assistant.
You specialize in biology, medicine, disease, nutrition, and hospital topics.
Give detailed, beginner-friendly answers.
When answering follow-up questions, always use the provided context
to give a complete and connected answer.

Retrieved knowledge:
{context}
"""
    ),
    ("human", "{question}")
])

# Agent-specific prompt template
agent_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are a specialized life sciences AI assistant focusing on {domain}.
Use the following retrieved knowledge to answer accurately:

{context}

If the retrieved knowledge does not fully answer the question,
use your own knowledge to fill in the gaps.
Always be clear, detailed and beginner-friendly.
"""
    ),
    ("human", "{question}")
])

# Build the chains
qa_chain = qa_prompt | llm | parser
agent_chain = agent_prompt | llm | parser

def get_langchain_answer(question: str, context: str = "") -> str:
    """
    Uses LangChain QA chain to answer a question with context.
    Replaces the old get_ai_answer() function.
    """
    return qa_chain.invoke({
        "question": question,
        "context": context if context else "No specific context retrieved."
    })

def get_agent_answer(question: str, context: str = "", domain: str = "life sciences") -> str:
    """
    Uses LangChain agent chain with domain specialization.
    """
    return agent_chain.invoke({
        "question": question,
        "context": context if context else "No specific context retrieved.",
        "domain": domain
    })

def get_langchain_answer_stream(question: str, context: str = ""):
    """
    Streaming version — yields tokens one by one.
    Used by the /stream endpoint.
    """
    for chunk in qa_chain.stream({
        "question": question,
        "context": context if context else "No specific context retrieved."
    }):
        yield chunk