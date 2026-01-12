import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.tools.retriever import create_retriever_tool
from langchain.schema import SystemMessage
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder

class LAMAChatbot:
    def __init__(self, vector_store, memory_manager):
        load_dotenv()

        # Initialize Gemini model
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            google_api_key=os.getenv("GOOGLE_API_KEY"),
            temperature=0.1,
            convert_system_message_to_human=True
        )

        # Retriever tool
        retriever = vector_store.as_retriever(search_kwargs={"k": 3})
        self.retriever_tool = create_retriever_tool(
            retriever,
            "lama_knowledge_search",
            """You are LAMA's customer support AI. Use the search tool to find accurate answers in the provided knowledge base. You can now also authoritatively answer questions about:

Company Details (e.g., HR email, social media, head office location)
Website & Account Management (e.g., creating an account, color accuracy, out-of-stock items)
Payment & Security (e.g., payment methods, SSL security, guest checkout)
Order Issues (e.g., canceled paid orders, missing confirmation emails, items disappearing from cart)
Promotions & Support (e.g., discount codes, customer support timings)
Services (e.g., no repair service, no gift cards)
Policies (e.g., detailed Return & Exchange, Privacy Policy)

Crucial Policy Note: If you find conflicting exchange periods (7 days in FAQ vs. 15 days in Policy), default to 7 days and note the discrepancy. Never make up information."""
        )
        
        # Define prompt template
        self.prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content="You are LAMA's customer support AI. Use the available tools to provide accurate information from the knowledge base."),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ])
        
        self.memory_manager = memory_manager
        self.agent_executor = None
    
    def set_agent_executor(self, executor):
        self.agent_executor = executor

    def ask(self, question):
        try:
            # Always use the agent for fresh responses
            if self.agent_executor is None:
                return "Agent executor not initialized."

            response = self.agent_executor.invoke({"input": question})

            # Extract text from response
            if isinstance(response, dict) and "output" in response:
                answer = response["output"]
            else:
                answer = str(response)

            # Store in conversation memory
            self.memory_manager.add_interaction(question, answer)

            return answer

        except Exception as e:
            return f"I apologize, but I encountered an error: {str(e)}"