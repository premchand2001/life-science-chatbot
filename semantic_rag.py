# semantic_rag.py

from langchain_openai import OpenAIEmbeddings
from knowledge_base import documents
import os
from dotenv import load_dotenv

load_dotenv()

embeddings_model = OpenAIEmbeddings(
    api_key=os.getenv("OPENAI_API_KEY")
)

def semantic_search(question, agent_name=None, top_k=3):
    query_embedding = embeddings_model.embed_query(question)

    all_results = []
    search_agents = [agent_name] if agent_name else list(documents.keys())

    for agent in search_agents:
        if agent not in documents:
            continue
        texts = documents[agent]
        for text in texts:
            text_embedding = embeddings_model.embed_query(text)
            # Cosine similarity
            dot = sum(a*b for a, b in zip(query_embedding, text_embedding))
            mag1 = sum(a**2 for a in query_embedding) ** 0.5
            mag2 = sum(b**2 for b in text_embedding) ** 0.5
            score = dot / (mag1 * mag2) if mag1 and mag2 else 0
            all_results.append({
                "answer": text,
                "score": round(score, 4),
                "source": agent
            })

    all_results.sort(key=lambda x: x["score"], reverse=True)
    return all_results[:top_k]