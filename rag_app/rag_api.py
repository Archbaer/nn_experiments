from fastapi import FastAPI
from pydantic import BaseModel
from rag import ask_question
import uvicorn

app = FastAPI()

class Query(BaseModel):
    question: str

@app.post("/ask")
async def ask(query: Query):
    answer = ask_question(query.question)
    return {"answer": answer}

if __name__ == "__main__":
    # host: '0.0.0.0' for docker
    # host: '127.0.0.1' for local
    uvicorn.run("rag_api:app", host="0.0.0.0", port=3584, reload=True)