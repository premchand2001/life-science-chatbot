from fastapi import FastAPI, Request, Form
from fastapi.responses import RedirectResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from agents import route_question

app = FastAPI()

templates = Jinja2Templates(directory="templates")

class QuestionRequest(BaseModel):
    question: str

chat_history = []

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
    result = route_question(question)

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
    chat_history.clear()
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={"result": None, "history": chat_history}
    )

@app.post("/chat")
def chat(request_body: QuestionRequest):
    result = route_question(request_body.question)
    return result