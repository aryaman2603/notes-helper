from fastapi import FastAPI
from pydantic import BaseModel
from rag.retriever import retrieve
from agent.llm import generate_answer
from agent.memory import get_memory, append_memory
app = FastAPI(title="Notes Helper API")

class QueryRequest(BaseModel):
    query: str
    subject: str
    session_id: str

class Source(BaseModel):
    source: str
    page: int | None
    chunk_id: int

class AnswerResponse(BaseModel):
    answer: str
    sources: list[Source]

@app.post("/ask_llm", response_model=AnswerResponse)
def ask_with_llm(req: QueryRequest):
    context, sources = retrieve(req.query, subject=req.subject)

    memory = get_memory(req.session_id, req.subject)
    answer = generate_answer(req.query, context, memory)

    append_memory(req.session_id, req.subject, "user", req.query)
    append_memory(req.session_id, req.subject, "assistant", answer)

    return AnswerResponse(
        answer=answer,
        sources=sources
    )


