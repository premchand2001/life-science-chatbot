# semantic_rag.py

from sentence_transformers import SentenceTransformer, util
from knowledge_base import documents

model = SentenceTransformer("all-MiniLM-L6-v2")

# Pre-encode all documents at startup
document_embeddings = {}
for agent, texts in documents.items():
    document_embeddings[agent] = model.encode(texts, convert_to_tensor=True)

def semantic_search(question, agent_name=None, top_k=3):
    query_embedding = model.encode(question, convert_to_tensor=True)

    if agent_name and agent_name in documents:
        texts = documents[agent_name]
        embeddings = document_embeddings[agent_name]
        scores = util.cos_sim(query_embedding, embeddings)[0]
        ranked = sorted(
            zip(texts, scores),
            key=lambda x: float(x[1]),
            reverse=True
        )[:top_k]
        return [
            {"answer": text, "score": round(float(score), 4), "source": agent_name}
            for text, score in ranked
        ]

    # Search across all agents if no specific agent given
    all_results = []
    for agent, texts in documents.items():
        embeddings = document_embeddings[agent]
        scores = util.cos_sim(query_embedding, embeddings)[0]
        ranked = sorted(
            zip(texts, scores),
            key=lambda x: float(x[1]),
            reverse=True
        )[:top_k]
        for text, score in ranked:
            all_results.append({
                "answer": text,
                "score": round(float(score), 4),
                "source": agent
            })

    all_results.sort(key=lambda x: x["score"], reverse=True)
    return all_results[:top_k]