# main.py

from fastapi import FastAPI, Request, Form, UploadFile, File
from fastapi.responses import RedirectResponse, StreamingResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from agents import route_question
from langgraph_orchestrator import run_graph
from openai_helper import get_ai_answer, get_ai_answer_stream
from pdf_ingestor import ingest_pdf, list_ingested_pdfs
from database import init_db, save_message, get_session_messages, clear_session
from react_agent import run_react_agent
import os
import shutil
import uuid

app = FastAPI()
templates = Jinja2Templates(directory="templates")

# Initialize database on startup
init_db()

# Auto-rebuild Chroma vector store on startup
from vector_store import build_vector_store
build_vector_store()

# In-memory for current session
conversation_memory = []
chat_history = []

# Current session ID — changes when user clears chat
current_session_id = str(uuid.uuid4())

class QuestionRequest(BaseModel):
    question: str

@app.get("/")
def home():
    return RedirectResponse(url="/ui")

@app.get("/ui")
def ui_page(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={"result": None, "history": chat_history}
    )

@app.post("/ask")
def ask_question(request: Request, question: str = Form(...)):

    # Detect follow-up questions
    follow_up_words = ["that", "it", "this", "more", "explain",
                       "elaborate", "continue", "detail", "tell me more",
                       "what about", "how about", "why", "how"]
    is_follow_up = any(word in question.lower().split() for word in follow_up_words)

    enhanced_question = question
    if is_follow_up and len(conversation_memory) >= 2:
        # Grab last user question + last bot answer for full context
        last_user = next(
            (m["message"] for m in reversed(conversation_memory) if m["role"] == "user"), ""
        )
        last_bot = next(
            (m["message"] for m in reversed(conversation_memory) if m["role"] == "assistant"), ""
        )
        enhanced_question = f"{last_user} {last_bot} {question}"

    # Store user message
    conversation_memory.append({"role": "user", "message": question})

    # Route through LangGraph orchestration
    result = run_graph(question, enhanced_question)
    result["question"] = question

    # OpenAI fallback if agent found nothing useful
    if (
        result.get("status") == "no_match"
        or not result.get("final_answer")
        or result.get("final_answer") == "No strong supporting result found in the knowledge base."
    ):
        try:
            context = "\n".join([r["answer"] for r in result.get("results", [])])
            ai_answer = get_ai_answer(enhanced_question, context)
            result["agent"] = "openai_fallback"
            result["final_answer"] = ai_answer
            result["status"] = "ai_success"
            result["results"] = []
        except Exception as e:
            result["agent"] = "openai_fallback"
            result["final_answer"] = f"OpenAI error: {str(e)}"
            result["status"] = "ai_error"
            result["results"] = []

    # Store bot answer in memory AND database
    conversation_memory.append({"role": "assistant", "message": result["final_answer"]})
    save_message(
        current_session_id,
        "assistant",
        result["final_answer"],
        agent=result.get("agent"),
        status=result.get("status")
    )

    # Keep memory capped at last 10 messages
    if len(conversation_memory) > 10:
        conversation_memory.pop(0)

    # Add to chat history for display
    chat_history.append({
        "question": question,
        "final_answer": result.get("final_answer", "No answer found."),
        "agent": result.get("agent", "unknown"),
        "status": result.get("status", "unknown")
    })

    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={"result": result, "history": chat_history}
    )

@app.post("/clear")
def clear_history(request: Request):
    global current_session_id
    chat_history.clear()
    conversation_memory.clear()
    # Start a fresh session ID when chat is cleared
    current_session_id = str(uuid.uuid4())
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={"result": None, "history": chat_history}
    )

@app.post("/chat")
def chat(request_body: QuestionRequest):
    result = route_question(request_body.question)
    return result

# PDF upload endpoint
@app.post("/upload-pdf")
async def upload_pdf(
    request: Request,
    file: UploadFile = File(...),
    agent_name: str = Form(...)
):
    # Save uploaded PDF temporarily
    upload_dir = "uploaded_pdfs"
    os.makedirs(upload_dir, exist_ok=True)
    file_path = os.path.join(upload_dir, file.filename)

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Ingest into Chroma
    try:
        ingest_pdf(file_path, agent_name)
        message = f"✅ Successfully ingested {file.filename} into {agent_name}"
    except Exception as e:
        message = f"❌ Error ingesting PDF: {str(e)}"

    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={
            "result": None,
            "history": chat_history,
            "upload_message": message
        }
    )

# See all ingested PDFs
@app.get("/ingested-pdfs")
def ingested_pdfs():
    collections = []
    from chromadb import PersistentClient
    client = PersistentClient(path="./chroma_db")
    all_cols = client.list_collections()
    for col_name in all_cols:
        collection = client.get_collection(col_name)
        results = collection.get(include=["metadatas"])
        sources = set()
        for meta in results["metadatas"]:
            if meta and "source" in meta:
                if meta["source"].endswith(".pdf"):
                    sources.add(meta["source"])
        if sources:
            collections.append({
                "agent": col_name,
                "pdfs": list(sources)
            })
    return {"ingested": collections}

# Streaming endpoint
@app.post("/stream")
async def stream_answer(request: Request, question: str = Form(...)):
    """
    Streams the AI answer token by token.
    Called by the frontend JavaScript fetch() for live typing effect.
    """
    # Build context from memory
    follow_up_words = ["that", "it", "this", "more", "explain",
                       "elaborate", "continue", "detail", "tell me more",
                       "what about", "how about", "why", "how"]
    is_follow_up = any(word in question.lower().split() for word in follow_up_words)

    enhanced_question = question
    if is_follow_up and len(conversation_memory) >= 2:
        # Only grab the LAST exchange — not older ones
        last_user = ""
        last_bot = ""
        for m in reversed(conversation_memory):
            if m["role"] == "assistant" and not last_bot:
                last_bot = m["message"][:300]  # limit to 300 chars
            if m["role"] == "user" and not last_user:
                last_user = m["message"]
            if last_user and last_bot:
                break
        enhanced_question = f"Regarding '{last_user}': {question}"

    # Get context from vector DB
    result = route_question(enhanced_question)
    context = "\n".join([r["answer"] for r in result.get("results", [])])

    # Store user message in memory AND database
    conversation_memory.append({"role": "user", "message": question})
    save_message(current_session_id, "user", question)

    # Collect full answer for memory storage
    full_answer = []

    def generate():
        for token in get_ai_answer_stream(enhanced_question, context):
            full_answer.append(token)
            yield token
        # Store complete answer in memory after streaming
        conversation_memory.append({
            "role": "assistant",
            "message": "".join(full_answer)
        })
        # Keep memory capped
        if len(conversation_memory) > 10:
            conversation_memory.pop(0)

    return StreamingResponse(
        generate(),
        media_type="text/plain"
    )

# Get full history for current session from database
@app.get("/session-history")
def session_history():
    messages = get_session_messages(current_session_id)
    return {
        "session_id": current_session_id,
        "messages": messages
    }

# Get all past sessions — useful for admin/boss demo
@app.get("/all-sessions")
def all_sessions():
    from database import get_all_sessions
    sessions = get_all_sessions()
    return {"sessions": sessions}

# ReAct reasoning endpoint
@app.post("/react")
async def react_answer(request: Request, question: str = Form(...)):
    """
    Runs the ReAct agent which shows its reasoning steps.
    """
    # Build context from memory
    follow_up_words = ["that", "it", "this", "more", "explain",
                       "elaborate", "continue", "detail"]
    is_follow_up = any(word in question.lower().split() for word in follow_up_words)

    enhanced_question = question
    if is_follow_up and len(conversation_memory) >= 2:
        last_user = ""
        for m in reversed(conversation_memory):
            if m["role"] == "user":
                last_user = m["message"]
                break
        enhanced_question = f"Regarding '{last_user}': {question}"

    # Run ReAct agent
    result = run_react_agent(enhanced_question)

    # Store in memory and database
    conversation_memory.append({"role": "user", "message": question})
    conversation_memory.append({
        "role": "assistant",
        "message": result["final_answer"]
    })
    save_message(current_session_id, "user", question)
    save_message(
        current_session_id,
        "assistant",
        result["final_answer"],
        agent="react_agent",
        status=result["status"]
    )

    # Add to chat history
    chat_history.append({
        "question": question,
        "final_answer": result["final_answer"],
        "agent": "react_agent",
        "status": result["status"]
    })

    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={
            "result": {
                "question": question,
                "final_answer": result["final_answer"],
                "agent": "react_agent",
                "status": result["status"],
                "results": [],
                "reasoning_steps": result["reasoning_steps"]
            },
            "history": chat_history
        }
    )

# Admin dashboard endpoint
@app.get("/admin")
def admin_dashboard(request: Request):
    from database import get_all_sessions, get_connection

    # Get all sessions
    sessions = get_all_sessions()

    # Get all messages (last 50)
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT role, message, agent, status, timestamp
        FROM messages
        ORDER BY id DESC
        LIMIT 50
    """)
    rows = cursor.fetchall()
    conn.close()
    messages = [dict(row) for row in rows]

    # Count agents used
    agent_counts = {}
    for msg in messages:
        if msg["role"] == "assistant" and msg["agent"]:
            agent = msg["agent"]
            agent_counts[agent] = agent_counts.get(agent, 0) + 1

    # Count total questions
    total_questions = sum(
        1 for msg in messages if msg["role"] == "user"
    )

    # Count ingested PDFs
    try:
        from chromadb import PersistentClient
        client = PersistentClient(path="./chroma_db")
        all_cols = client.list_collections()
        total_pdfs = 0
        for col_name in all_cols:
            collection = client.get_collection(col_name)
            results = collection.get(include=["metadatas"])
            for meta in results["metadatas"]:
                if meta and meta.get("source", "").endswith(".pdf"):
                    total_pdfs += 1
    except:
        total_pdfs = 0

    stats = {
        "total_sessions": len(sessions),
        "total_messages": len(messages),
        "total_questions": total_questions,
        "total_pdfs": total_pdfs,
        "agent_counts": agent_counts
    }

    return templates.TemplateResponse(
        request=request,
        name="admin.html",
        context={
            "stats": stats,
            "messages": messages,
            "sessions": sessions
        }
    )