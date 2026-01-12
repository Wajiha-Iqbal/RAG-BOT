import os
import shutil
import stat
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_openai import  OpenAIEmbeddings
from dotenv import load_dotenv

class VectorStoreManager:
    def __init__(self, vector_store_path="vector_store"):
        load_dotenv()
        self.vector_store_path = vector_store_path
        self.embeddings = GoogleGenerativeAIEmbeddings(
            model="models/text-embedding-004",
            google_api_key=os.getenv("GOOGLE_API_KEY")
        )

    # def __init__(self, vector_store_path="vector_store", model_name="openai/text-embedding-3-small"):
    #     load_dotenv()
    #     self.vector_store_path = vector_store_path
        
    #     # Configure OpenRouter via the OpenAIEmbeddings class
    #     self.embeddings = OpenAIEmbeddings(
    #         model=model_name,
    #         openai_api_key=os.getenv("OPENROUTER_API_KEY"),
    #         openai_api_base="https://openrouter.ai/api/v1",
    #         # Optional: identify your app for OpenRouter's rankings
    #         default_headers={
    #             "HTTP-Referer": "https://your-site-url.com", # Your site URL
    #             "X-Title": "My App Name",                   # Your app name
    #         }
    #     )
    
    def remove_readonly(self, func, path, _):
        """Force delete even if files are read-only (Windows-safe)."""
        os.chmod(path, stat.S_IWRITE)
        func(path)
    
    def clear_vector_store(self):
        if os.path.exists(self.vector_store_path):
            print("[0] Clearing existing vector store...")
            shutil.rmtree(self.vector_store_path, onerror=self.remove_readonly)
            print("✅ Old vector store removed.")
    
    def create_vector_store(self, documents):
        print("[3] Generating and storing embeddings in Chroma...")
        if not documents:
            raise ValueError("No documents provided to create vector store")
        
        vectorstore = Chroma.from_documents(
            documents=documents,
            embedding=self.embeddings,
            persist_directory=self.vector_store_path
        )
        print(f"✅ Vector store saved to '{self.vector_store_path}' directory.")
        return vectorstore
    
    def load_vector_store(self):
        if not os.path.exists(self.vector_store_path):
            raise FileNotFoundError(f"Vector store not found at {self.vector_store_path}")
        
        return Chroma(
            persist_directory=self.vector_store_path,
            embedding_function=self.embeddings
        )
    
    def vector_store_exists(self):
        return os.path.exists(self.vector_store_path)