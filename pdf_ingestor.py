# pdf_ingestor.py

import os
from pypdf import PdfReader
from chromadb import PersistentClient
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

embeddings_model = OpenAIEmbeddings(
    api_key=os.getenv("OPENAI_API_KEY")
)
client = PersistentClient(path="./chroma_db")

def extract_text_from_pdf(pdf_path: str) -> list[str]:
    """
    Reads a PDF file and splits it into chunks of ~500 characters.
    Returns a list of text chunks.
    """
    reader = PdfReader(pdf_path)
    full_text = ""

    for page in reader.pages:
        text = page.extract_text()
        if text:
            full_text += text + " "

    # Split into chunks of 500 characters with 50 char overlap
    chunks = []
    chunk_size = 500
    overlap = 50
    start = 0

    while start < len(full_text):
        end = start + chunk_size
        chunk = full_text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start = end - overlap

    print(f"  Extracted {len(chunks)} chunks from {os.path.basename(pdf_path)}")
    return chunks

def ingest_pdf(pdf_path: str, agent_name: str):
    """
    Ingests a PDF into the Chroma vector store under a specific agent.
    """
    print(f"\nIngesting: {os.path.basename(pdf_path)} → {agent_name}")

    chunks = extract_text_from_pdf(pdf_path)

    if not chunks:
        print("  No text found in PDF. It may be a scanned image PDF.")
        return

    collection = client.get_or_create_collection(name=agent_name)
    existing = collection.get()
    existing_ids = set(existing["ids"])

    new_texts = []
    new_embeddings = []
    new_ids = []
    new_metadatas = []

    filename = os.path.basename(pdf_path)

    for i, chunk in enumerate(chunks):
        doc_id = f"{filename}_chunk_{i}"

        if doc_id in existing_ids:
            continue

        # Now uses OpenAI embeddings instead of sentence-transformers
        embedding = embeddings_model.embed_query(chunk)
        new_texts.append(chunk)
        new_embeddings.append(embedding)
        new_ids.append(doc_id)
        new_metadatas.append({
            "source": filename,
            "agent": agent_name,
            "chunk": i
        })

    if new_texts:
        collection.add(
            documents=new_texts,
            embeddings=new_embeddings,
            ids=new_ids,
            metadatas=new_metadatas
        )
        print(f"  ✅ Added {len(new_texts)} new chunks to {agent_name}")
    else:
        print(f"  ⏭️  Already ingested — no new chunks added")

def list_ingested_pdfs():
    """
    Shows all PDFs currently stored in the vector DB.
    """
    print("\nCurrently ingested PDFs:")
    collections = client.list_collections()
    found_any = False

    for col_name in collections:
        collection = client.get_collection(col_name)
        results = collection.get(include=["metadatas"])
        sources = set()
        for meta in results["metadatas"]:
            if meta and "source" in meta:
                src = meta["source"]
                if src.endswith(".pdf"):
                    sources.add(src)
        if sources:
            found_any = True
            for src in sources:
                print(f"  📄 {src} → {col_name}")

    if not found_any:
        print("  No PDFs ingested yet.")