from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.runnables import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
from langchain_postgres.vectorstores import PGVector
from langchain_openai import OpenAIEmbeddings
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
import os
import sys

load_dotenv(override=True)
os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY', 'your-key-if-not-using-env')


# Embeddings model
embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-large-en-v1.5")

# Vector store setup
vectorstore = PGVector(
    embeddings=embeddings,
    collection_name='knowledge_base', 
    connection='postgresql+psycopg://pguser:pgpass@localhost:5432/pgdb'
                       )

# Chat model setup
# When deploying, or you don't have ollama. 
# Either install ollama for an open source LLM or use open ai's gpt-4o-mini
llm = ChatOllama(model="llama3.2", base_url="http://localhost:11434")

# Retriever setup
# k=10 means we retrieve the top 10 relevant chunks
retriever = vectorstore.as_retriever(search_kwargs={"k": 10})

# Creating a custom prompt template
PROMPT_TEMPLATE = """
You are an assistant for question-answering tasks.
Use the following pieces of retrieved context to answer the question.
If you don't know the answer, say that you don't know. Don't Make Up Anything.

{context}

---

Answer the question based on the above context: {question}
"""
prompt = PromptTemplate(
    input_variables=["context", "question"],
    template=PROMPT_TEMPLATE,
)

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# RAG chain setup using LCEL
rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

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