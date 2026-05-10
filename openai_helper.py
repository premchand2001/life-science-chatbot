# openai_helper.py

from langchain_helper import get_langchain_answer, get_langchain_answer_stream

def get_ai_answer(question: str, context: str = "") -> str:
    """
    Now powered by LangChain instead of raw OpenAI calls.
    """
    return get_langchain_answer(question, context)

def get_ai_answer_stream(question: str, context: str = ""):
    """
    Streaming version — now powered by LangChain.
    """
    for token in get_langchain_answer_stream(question, context):
        yield token