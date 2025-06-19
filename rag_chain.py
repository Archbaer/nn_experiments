
from langchain_community.document_loaders import PyPDFLoader # Or other loaders
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.chat_models import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

# Load Documents
loader = PyPDFLoader("knowledge-base/AI Engineering.pdf") # Replace with your document path
documents = loader.load()

# Split Documents
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = text_splitter.split_documents(documents)

# Generate Embeddings (using a free Hugging Face model)
embedding_model_name = "sentence-transformers/all-MiniLM-L6-v2" # Or "BAAI/bge-small-en-v1.5" etc.
embeddings = HuggingFaceEmbeddings(model_name=embedding_model_name)

# Create Vector Store (ChromaDB example)
vector_store = Chroma.from_documents(chunks, embeddings, persist_directory="./chroma_db")
retriever = vector_store.as_retriever()

# Load DeepSeek-R1 LLM via Ollama (ollama run deepseek-r1:7b)
llm = ChatOllama(model="deepseek-r1") # Ensure this model is pulled in Ollama

# 6. Define Prompt Template
prompt = ChatPromptTemplate.from_messages([
    ("system", "Answer the user's question based on the provided context:\n\n{context}"),
    ("user", "{input}")
])

# 7. Create RAG Chain
document_chain = create_stuff_documents_chain(llm, prompt)
retrieval_chain = create_retrieval_chain(retriever, document_chain)

# 8. Invoke the RAG Chain
question = "What is the main topic of the document?"
response = retrieval_chain.invoke({"input": question})

print(response["answer"])
