from langchain.memory import ConversationBufferWindowMemory

class MemoryManager:
    """Simplified memory manager using only ConversationBufferWindowMemory"""
    def __init__(self, k=10):
        self.memory = ConversationBufferWindowMemory(
            memory_key="chat_history",
            return_messages=True,
            k=k
        )
    
    def add_interaction(self, question, answer):
        """Store human question and AI answer"""
        self.memory.chat_memory.add_user_message(question)
        self.memory.chat_memory.add_ai_message(answer)
    
    def get_memory(self):
        """Get LangChain memory object for agents"""
        return self.memory
    
    def clear_memory(self):
        """Clear all memory"""
        self.memory.clear()
    
    def get_history(self):
        """Get recent conversation history as text"""
        return self.memory.load_memory_variables({})