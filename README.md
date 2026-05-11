# Multi-Agent Life Science Chatbot

A production-grade GenAI solution built with FastAPI, LangChain, LangGraph, Weaviate, and HuggingFace models.

## Live Demo
https://life-science-chatbot.onrender.com

## Architecture

- **FastAPI** — Backend REST API
- **LangGraph** — Multi-agent orchestration graph
- **LangChain** — LLM chain management
- **Weaviate** — Cloud vector database
- **HuggingFace** — Embedding model (all-MiniLM-L6-v2)
- **OpenAI GPT-4o-mini** — Language model
- **MCP** — Model Context Protocol tool schemas
- **SQLite** — Conversation memory database

## Agents

| Agent | Domain |
|---|---|
| biology_agent | DNA, cells, genes, CRISPR |
| disease_agent | Diabetes, cancer, COVID-19 |
| medicine_agent | Insulin, antibiotics, vaccines |
| hospital_agent | Doctors, nurses, hospitals |
| nutrition_agent | Vitamins, diet, minerals |

## Features

- Multi-agent routing via LangGraph
- Semantic search with Weaviate vector DB
- RAG (Retrieval Augmented Generation)
- MCP schemas with OpenFDA and ClinicalTrials.gov
- ReAct reasoning with visible thinking steps
- PDF document ingestion
- Live streaming responses
- Database-backed conversation memory
- Admin dashboard
- Cloud deployed on Render

## Setup

### 1. Clone the repository
git clone https://github.com/premchand2001/life-science-chatbot.git
cd life-science-chatbot

### 2. Create virtual environment
python -m venv .venv
.venv\Scripts\activate

### 3. Install dependencies
pip install -r requirements.txt

### 4. Set environment variables
Create a `.env` file:
OPENAI_API_KEY=your-openai-key
WEAVIATE_URL=your-weaviate-url
WEAVIATE_API_KEY=your-weaviate-key

### 5. Build vector store
python -c "from weaviate_store import build_weaviate_store; build_weaviate_store()"

### 6. Run the app
uvicorn main:app --reload

## API Endpoints

| Endpoint | Method | Description |
|---|---|---|
| /ui | GET | Main chat interface |
| /ask | POST | Ask a question |
| /stream | POST | Streaming response |
| /react | POST | ReAct reasoning |
| /upload-pdf | POST | Upload PDF to knowledge base |
| /mcp/schemas | GET | View MCP tool schemas |
| /mcp/fda/{drug} | GET | FDA drug lookup |
| /mcp/trials/{condition} | GET | Clinical trials search |
| /admin | GET | Admin dashboard |
| /session-history | GET | Current session history |

## Design Decisions

- **Weaviate over Pinecone** — Better free tier, easier integration
- **Chroma for local dev** — Zero setup for development
- **Smart embedding switcher** — HuggingFace locally, OpenAI on cloud
- **SQLite for memory** — Zero setup persistent storage
- **LangGraph over manual routing** — Proper graph-based orchestration

## Team
Built by Premchand K during BTransforms GenAI  
