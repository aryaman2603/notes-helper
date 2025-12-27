from fastapi import FastAPI
from pydantic import BaseModel
from rag.retriever import retrieve
from agent.llm import generate_answer

app = FastAPI(title="Notes Helper API")

class QueryRequest(BaseModel):
    query: str

class AnswerResponse(BaseModel):
    answer: str
    context: str

@app.post("/ask_llm", response_model=AnswerResponse)
def ask_with_llm(req: QueryRequest):
    context = retrieve(req.query)
    answer = generate_answer(req.query, context)

    return AnswerResponse(
        answer=answer,
        context=context
    )


