# vector_store.py

from chromadb import PersistentClient
from langchain_openai import OpenAIEmbeddings
from knowledge_base import documents
import os
from dotenv import load_dotenv

load_dotenv()

client = PersistentClient(path="./chroma_db")

# OpenAI embeddings — runs on OpenAI servers, uses almost zero local memory
embeddings_model = OpenAIEmbeddings(
    api_key=os.getenv("OPENAI_API_KEY")
)

def get_or_create_collection(agent_name: str):
    return client.get_or_create_collection(name=agent_name)

def build_vector_store():
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
                embedding = embeddings_model.embed_query(text)
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
    query_embedding = embeddings_model.embed_query(question)

    if agent_name:
        collection = get_or_create_collection(agent_name)
        count = collection.count()
        if count == 0:
            return []
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=min(top_k, count)
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