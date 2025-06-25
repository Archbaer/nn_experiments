from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_postgres.vectorstores import PGVector
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
import os

# To load your open ai api key, if you have one
load_dotenv(override=True)
os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY', 'your-key-if-not-using-env')

def populate_rag():
    """
    Populates the RAG database with documents from the Bookshelf directory.
    """ 
    # Loads Pdfs from this specified file
    loader = DirectoryLoader(
        path="./Bookshelf",
        glob="**/*.pdf",
        loader_cls=PyPDFLoader
    )
    documents = loader.load()
    print(f"Loaded {len(documents)} documents.")

    # Splitting the documents into smaller chunks 
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=200)
    chunks = text_splitter.split_documents(documents)

    # Embedding function
    embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-large-en-v1.5")
                            
    print(f"Split into {len(chunks)} chunks.")
    print("Starting to populate the RAG database...")
    
    # Populating the database
    # Use localhost when running local and service name when running on docker or k8s
    try:
        vectorstore = PGVector.from_documents(
            documents=chunks,
            collection_name='knowledge_base',
            connection="postgresql+psycopg://pguser:pgpass@localhost:5432/pgdb",
            embedding=embeddings
        )
        print("✅ RAG database populated successfully!")
    except Exception as e:
        print(f"Error connecting to the database: {e}")
        import traceback
        traceback.print_exc()
        raise
    return vectorstore

if __name__ == "__main__":
    populate_rag()