# weaviate_store.py
# Weaviate vector database integration

import weaviate
from weaviate.classes.init import Auth
from weaviate.classes.config import Property, DataType
from knowledge_base import documents
import os
from dotenv import load_dotenv

load_dotenv()

# Smart switcher — HuggingFace locally, OpenAI on Render cloud
if os.getenv("RENDER"):
    from langchain_openai import OpenAIEmbeddings
    hf_embeddings = OpenAIEmbeddings(
        api_key=os.getenv("OPENAI_API_KEY")
    )
    print("Using OpenAI embeddings (cloud deployment)")
else:
    from langchain_huggingface import HuggingFaceEmbeddings
    hf_embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2"
    )
    print("Using HuggingFace embeddings (local development)")

def get_weaviate_client():
    """Connect to Weaviate cloud cluster."""
    client = weaviate.connect_to_weaviate_cloud(
        cluster_url=os.getenv("WEAVIATE_URL"),
        auth_credentials=Auth.api_key(os.getenv("WEAVIATE_API_KEY"))
    )
    return client

def build_weaviate_store():
    """
    Loads all documents from knowledge_base.py into Weaviate.
    Uses HuggingFace locally and OpenAI on cloud.
    """
    print("Building Weaviate vector store...")
    client = get_weaviate_client()

    try:
        for agent_name, texts in documents.items():
            # Collection name — capitalize properly
            collection_name = agent_name.replace("_", " ").title().replace(" ", "")

            # Create collection if it doesn't exist
            if not client.collections.exists(collection_name):
                client.collections.create(
                    name=collection_name,
                    properties=[
                        Property(
                            name="content",
                            data_type=DataType.TEXT
                        ),
                        Property(
                            name="source",
                            data_type=DataType.TEXT
                        ),
                        Property(
                            name="agent",
                            data_type=DataType.TEXT
                        )
                    ]
                )
                print(f"  Created collection: {collection_name}")

            collection = client.collections.get(collection_name)

            # Check existing count
            existing_count = len(
                collection.query.fetch_objects(limit=1000).objects
            )

            if existing_count >= len(texts):
                print(f"  ⏭️  {agent_name}: already up to date")
                continue

            # Add documents with embeddings
            with collection.batch.dynamic() as batch:
                for i, text in enumerate(texts):
                    embedding = hf_embeddings.embed_query(text)
                    batch.add_object(
                        properties={
                            "content": text,
                            "source": agent_name,
                            "agent": agent_name
                        },
                        vector=embedding
                    )

            print(f"  ✅ {agent_name}: added {len(texts)} documents")

    finally:
        client.close()

    print("Weaviate vector store ready!")

def weaviate_search(question: str, agent_name: str = None, top_k: int = 3) -> list:
    """
    Search Weaviate for most relevant documents.
    Uses HuggingFace locally and OpenAI on cloud.
    """
    client = get_weaviate_client()

    try:
        query_embedding = hf_embeddings.embed_query(question)

        if agent_name:
            collection_name = agent_name.replace("_", " ").title().replace(" ", "")

            if not client.collections.exists(collection_name):
                return []

            collection = client.collections.get(collection_name)
            results = collection.query.near_vector(
                near_vector=query_embedding,
                limit=top_k,
                return_metadata=["distance"]
            )

            return [
                {
                    "answer": obj.properties.get("content", ""),
                    "score": round(1 - obj.metadata.distance, 4),
                    "source": agent_name
                }
                for obj in results.objects
            ]

        # Search all agents
        all_results = []
        for agent in documents.keys():
            collection_name = agent.replace("_", " ").title().replace(" ", "")

            if not client.collections.exists(collection_name):
                continue

            collection = client.collections.get(collection_name)
            results = collection.query.near_vector(
                near_vector=query_embedding,
                limit=top_k,
                return_metadata=["distance"]
            )

            for obj in results.objects:
                all_results.append({
                    "answer": obj.properties.get("content", ""),
                    "score": round(1 - obj.metadata.distance, 4),
                    "source": agent
                })

        all_results.sort(key=lambda x: x["score"], reverse=True)
        return all_results[:top_k]

    finally:
        client.close()