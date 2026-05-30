import os
import sys
from dotenv import load_dotenv
try:
    from langchain.vectorstores import FAISS
except ImportError:
    from langchain_community.vectorstores import FAISS

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import Config
from rag.ingestion import get_embeddings

load_dotenv()

class MaintenanceCopilot:
    def __init__(self):
        self.index_path = Config.FAISS_INDEX_DIR
        if not os.path.exists(self.index_path) or not os.path.exists(os.path.join(self.index_path, "index.faiss")):
            # If the index doesn't exist, we will try to build it first
            print("[!] FAISS index not found. Initiating auto-ingestion...")
            from rag.ingestion import ingest_manual
            ingest_manual()
            
        self.embeddings = get_embeddings()
        self.vectorstore = FAISS.load_local(self.index_path, self.embeddings)
        
    def query(self, user_query: str) -> dict:
        """Retrieves matching documents and generates a response based on the query."""
        print(f"[*] Querying Maintenance Copilot for: '{user_query}'")
        
        # 1. Similarity search
        retrieved_docs = self.vectorstore.similarity_search(user_query, k=3)
        context_list = [doc.page_content for doc in retrieved_docs]
        context_text = "\n\n---\n\n".join(context_list)
        
        # 2. Check for OpenAI key to decide on LLM generation vs. local compilation
        openai_key = os.getenv("OPENAI_API_KEY")
        if openai_key:
            try:
                print("[*] Generating response using OpenAI LLM...")
                try:
                    from langchain.chat_models import ChatOpenAI
                except ImportError:
                    try:
                        from langchain_community.chat_models import ChatOpenAI
                    except ImportError:
                        from langchain_openai import ChatOpenAI

                try:
                    from langchain.chains import RetrievalQA
                except ImportError:
                    try:
                        from langchain.chains import RetrievalQA
                    except ImportError:
                        from langchain.chains.retrieval import RetrievalQA
                
                llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.2)
                qa_chain = RetrievalQA.from_chain_type(
                    llm=llm,
                    chain_type="stuff",
                    retriever=self.vectorstore.as_retriever(search_kwargs={"k": 3})
                )
                response = qa_chain.run(user_query)
                
                return {
                    "query": user_query,
                    "answer": response,
                    "mode": "LLM-enhanced",
                    "sources": context_list
                }
            except Exception as e:
                print(f"[!] Error calling OpenAI API: {e}. Falling back to local offline search.")
                
        # Fallback offline response compilation
        print("[*] Compiling offline response from retrieved handbook context...")
        
        # Format a professional response based on retrieved fragments
        fallback_answer = (
            f"### 📋 Manual Search Results (Offline Mode)\n"
            f"Here are the matching troubleshooting procedures found in the F-100 Maintenance Manual:\n\n"
        )
        for i, doc in enumerate(retrieved_docs, 1):
            cleaned_content = doc.page_content.strip()
            # Try to grab header if present, else show index
            fallback_answer += f"**Reference Block {i}:**\n{cleaned_content}\n\n"
            
        fallback_answer += (
            "\n> ⚠️ *Note: Running in Offline Mode. Provide an `OPENAI_API_KEY` in the environment variables "
            "for AI-synthesized responses.*"
        )
        
        return {
            "query": user_query,
            "answer": fallback_answer,
            "mode": "Offline-retrieval",
            "sources": context_list
        }

if __name__ == "__main__":
    print("Testing Maintenance Copilot...")
    try:
        copilot = MaintenanceCopilot()
        res = copilot.query("How to troubleshoot Sensor 11?")
        print("\nCOPILOT RESPONSE:")
        print(res["answer"])
    except Exception as e:
        print("Error testing Copilot:", e)
