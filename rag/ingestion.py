import os
import sys
from dotenv import load_dotenv
try:
    from langchain.text_splitter import RecursiveCharacterTextSplitter
except ImportError:
    try:
        from langchain_text_splitters import RecursiveCharacterTextSplitter
    except ImportError:
        from langchain.text_splitter import RecursiveCharacterTextSplitter

try:
    from langchain.vectorstores import FAISS
except ImportError:
    from langchain_community.vectorstores import FAISS

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import Config

load_dotenv()

def get_embeddings():
    """Returns OpenAIEmbeddings if API key is present, otherwise falls back to local HuggingFaceEmbeddings."""
    openai_key = os.getenv("OPENAI_API_KEY")
    if openai_key:
        print("[*] OpenAI API Key found. Using OpenAIEmbeddings.")
        try:
            from langchain.embeddings import OpenAIEmbeddings
            return OpenAIEmbeddings()
        except ImportError:
            try:
                from langchain_community.embeddings import OpenAIEmbeddings
                return OpenAIEmbeddings()
            except ImportError:
                from langchain_openai import OpenAIEmbeddings
                return OpenAIEmbeddings()
    else:
        print("[!] No OpenAI API Key found. Falling back to local HuggingFaceEmbeddings ('all-MiniLM-L6-v2').")
        try:
            from langchain.embeddings import HuggingFaceEmbeddings
            return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        except ImportError:
            from langchain_community.embeddings import HuggingFaceEmbeddings
            return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

def ingest_manual():
    print("=" * 60)
    print("STARTING MANUAL INGESTION TO VECTOR STORE")
    print("=" * 60)
    
    manual_path = Config.MAINTENANCE_MANUAL_PATH
    if not os.path.exists(manual_path):
        raise FileNotFoundError(f"Maintenance manual not found at {manual_path}")
        
    print(f"[*] Reading manual file: {manual_path}")
    with open(manual_path, "r", encoding="utf-8") as f:
        text = f.read()
        
    # Split text into logical chunks
    print("[*] Splitting document into chunks...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=600,
        chunk_overlap=100,
        separators=["\n\n", "\n", " ", ""]
    )
    # Wrap in LangChain Document structure
    try:
        from langchain.docstore.document import Document
    except ImportError:
        try:
            from langchain.schema import Document
        except ImportError:
            from langchain_core.documents import Document
    chunks = splitter.create_documents([text])
    print(f"[+] Document split into {len(chunks)} chunks.")
    
    # Initialize embeddings
    embeddings = get_embeddings()
    
    # Create and save FAISS vector store
    print("[*] Generating embeddings and building FAISS index...")
    vectorstore = FAISS.from_documents(chunks, embeddings)
    
    index_path = Config.FAISS_INDEX_DIR
    print(f"[*] Saving FAISS index to: {index_path}")
    vectorstore.save_local(index_path)
    
    print("=" * 60)
    print("INGESTION PIPELINE COMPLETE")
    print("=" * 60)

if __name__ == "__main__":
    ingest_manual()