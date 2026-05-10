# vector_store.py

from chromadb import PersistentClient
from sentence_transformers import SentenceTransformer
from knowledge_base import documents

# Embedding model — same one already used in semantic_rag.py
model = SentenceTransformer("all-MiniLM-L6-v2")

# Chroma saves data to this folder permanently on your disk
client = PersistentClient(path="./chroma_db")

def get_or_create_collection(agent_name: str):
    return client.get_or_create_collection(name=agent_name)

def build_vector_store():
    """
    Loads all documents from knowledge_base.py into Chroma.
    Safe to run multiple times — skips docs already stored.
    """
    print("Building Chroma vector store...")
    for agent_name, texts in documents.items():
        collection = get_or_create_collection(agent_name)
        existing = collection.get()
        existing_ids = set(existing["ids"])

        new_texts = []
        new_embeddings = []
        new_ids = []
        new_metadatas = []

        for i, text in enumerate(texts):
            doc_id = f"{agent_name}_{i}"
            if doc_id not in existing_ids:
                embedding = model.encode(text).tolist()
                new_texts.append(text)
                new_embeddings.append(embedding)
                new_ids.append(doc_id)
                new_metadatas.append({"source": agent_name})

        if new_texts:
            collection.add(
                documents=new_texts,
                embeddings=new_embeddings,
                ids=new_ids,
                metadatas=new_metadatas
            )
            print(f"  ✅ {agent_name}: added {len(new_texts)} documents")
        else:
            print(f"  ⏭️  {agent_name}: already up to date")

    print("Vector store ready!")

def chroma_search(question: str, agent_name: str = None, top_k: int = 3):
    """
    Search Chroma for most relevant documents.
    If agent_name given — searches that agent only.
    Otherwise searches all agents.
    """
    query_embedding = model.encode(question).tolist()

    if agent_name:
        collection = get_or_create_collection(agent_name)
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=min(top_k, collection.count())
        )
        docs = results["documents"][0]
        distances = results["distances"][0]
        return [
            {
                "answer": doc,
                "score": round(1 - dist, 4),
                "source": agent_name
            }
            for doc, dist in zip(docs, distances)
        ]

    # Search all agents and merge results
    all_results = []
    for name in documents.keys():
        collection = get_or_create_collection(name)
        if collection.count() == 0:
            continue
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=min(top_k, collection.count())
        )
        for doc, dist in zip(results["documents"][0], results["distances"][0]):
            all_results.append({
                "answer": doc,
                "score": round(1 - dist, 4),
                "source": name
            })

    all_results.sort(key=lambda x: x["score"], reverse=True)
    return all_results[:top_k]