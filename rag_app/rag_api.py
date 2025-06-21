from fastapi import FastAPI
from pydantic import BaseModel
from rag import ask_question

app = FastAPI()

class Query(BaseModel):
    question: str

@app.post("/ask")
async def ask(query: Query):
    answer = ask_question(query.question)
    return {"answer": answer}