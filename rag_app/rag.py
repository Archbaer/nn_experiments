from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_postgres.vectorstores import PGVector
from langchain_ollama import ChatOllama
from langchain.chains import RetrievalQA
import sys

# Embeddings model
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Vector store setup
vectorstore = PGVector(
    embeddings=embeddings,
    collection_name='knowledge_base', 
    connection='postgresql+psycopg://pguser:pgpass@localhost:5432/pgdb'
                       )

# Chat model setup
llm = ChatOllama(model="deepseek-r1:7b", base_url="http://localhost:11434")

# Retriever setup
# k=25 means we retrieve the top 25 relevant chunks
retriever = vectorstore.as_retriever(search_kwargs={"k": 25})

# RAG chain setup
rag_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

# Query function
def ask_question(query: str) -> str:
    return rag_chain.invoke(query)

if __name__ == "__main__":
    # Example query
    if len(sys.argv) > 1:
        question = sys.argv[1]
        answer = ask_question(question)
        print(f"Question: {question}\nAnswer: {answer}")
    else:
        print("Please provide a question as a command line argument.")
        sys.exit(1)