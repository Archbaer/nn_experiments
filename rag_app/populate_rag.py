from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_postgres.vectorstores import PGVector
from langchain_huggingface import HuggingFaceEmbeddings

def populate_rag():
    """
    Populates the RAG database with documents from the Bookshelf directory.
    """ 
    loader = DirectoryLoader(
        path="./Bookshelf",
        glob="**/*.pdf",
        loader_cls=PyPDFLoader
    )
    documents = loader.load()
    print(f"Loaded {len(documents)} documents.")

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(documents)


    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
                            
    print(f"Split into {len(chunks)} chunks.")
    print("Starting to populate the RAG database...")
    
    # Populating the database
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