import os
# DISABLE ALL TELEMETRY
os.environ['ANONYMIZED_TELEMETRY'] = 'False'
os.environ['CHROMA_TELEMETRY'] = 'False'
os.environ['DISABLE_TELEMETRY'] = 'True'

from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize chatbot as None first
chatbot = None

try:
    # Load environment variables
    load_dotenv()
    
    # Get API key
    API_KEY = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    
    if not API_KEY:
        logger.error("❌ No API key found!")
        
        class FallbackChatbot:
            def ask(self, message):
                return "AI service is currently unavailable. Please add GEMINI_API_KEY or GOOGLE_API_KEY to your .env file."
        chatbot = FallbackChatbot()
    else:
        logger.info(f"✅ API key loaded (first 5 chars: {API_KEY[:5]}...)")
        
        # Set the API key for Google Generative AI
        os.environ["GOOGLE_API_KEY"] = API_KEY
        
        # Import your modules
        from pdf_processor import PDFProcessor
        from vector_store_manager import VectorStoreManager
        from memory_manager import MemoryManager
        from chatbot import LAMAChatbot
        
        # Initialize vector store
        PDF_PATH = "Lama.pdf"
        vector_manager = VectorStoreManager()
        if not vector_manager.vector_store_exists():
            logger.info("Creating vector store...")
            processor = PDFProcessor()
            chunks = processor.process_pdf(PDF_PATH)
            vector_manager.create_vector_store(chunks)
        
        vector_store = vector_manager.load_vector_store()
        memory_manager = MemoryManager()
        chatbot = LAMAChatbot(vector_store, memory_manager)
        
        # ✅ CRITICAL: Create and set the agent executor
        # Import LangChain agent modules
        from langchain.agents import AgentExecutor, create_openai_functions_agent
        
        # Create the agent
        agent = create_openai_functions_agent(
            llm=chatbot.llm,
            tools=[chatbot.retriever_tool],
            prompt=chatbot.prompt
        )
        
        # Create the executor
        executor = AgentExecutor(
            agent=agent,
            tools=[chatbot.retriever_tool],
            memory=memory_manager.get_memory(),
            verbose=True,
            handle_parsing_errors=True
        )
        
        # Set the executor in the chatbot
        chatbot.set_agent_executor(executor)
        
        logger.info("✅ Backend initialized successfully!")
        logger.info(f"🤖 Agent executor ready: {chatbot.agent_executor is not None}")
        
except Exception as e:
    logger.error(f"❌ Failed to initialize backend: {str(e)}")
    import traceback
    traceback.print_exc()
    
    class DummyChatbot:
        def ask(self, message):
            return f"I'm sorry, but the chatbot initialization failed: {str(e)}. Please check the backend logs."
    
    chatbot = DummyChatbot()

@app.post("/chat")
async def chat(request: Request):
    try:
        data = await request.json()
        message = data.get("message", "")
        
        if not message:
            raise HTTPException(status_code=400, detail="Empty message")
        
        if chatbot is None:
            raise HTTPException(status_code=500, detail="Chatbot not initialized")
        
        logger.info(f"📨 Received: {message[:50]}...")
        
        # Check if agent executor is ready
        if hasattr(chatbot, 'agent_executor') and chatbot.agent_executor is None:
            logger.error("❌ Agent executor is None!")
            return {"response": "Agent executor is not initialized. Please check backend logs."}
        
        response = chatbot.ask(message)
        
        logger.info(f"📤 Response: {response[:50]}...")
        
        return {"response": response}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error in /chat endpoint: {str(e)}")
        import traceback
        traceback.print_exc()
        return {"error": str(e), "response": "I encountered an error processing your request."}

@app.get("/health")
async def health_check():
    import datetime
    agent_ready = False
    if chatbot is not None and hasattr(chatbot, 'agent_executor'):
        agent_ready = chatbot.agent_executor is not None
    
    return {
        "status": "healthy" if chatbot else "degraded",
        "chatbot_ready": chatbot is not None and hasattr(chatbot, 'ask'),
        "agent_executor_ready": agent_ready,
        "timestamp": datetime.datetime.now().isoformat()
    }

@app.get("/")
async def root():
    return {
        "message": "LAMA Retail AI Backend API",
        "endpoints": {
            "chat": "POST /chat",
            "health": "GET /health"
        },
        "status": "running"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)