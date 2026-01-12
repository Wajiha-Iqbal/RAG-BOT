# File: main.py
import os
import re
from dotenv import load_dotenv
from pdf_processor import PDFProcessor
from vector_store_manager import VectorStoreManager
from memory_manager import MemoryManager
# from llm import LAMAChatbot
from chatbot import LAMAChatbot
from langchain.agents import AgentExecutor, create_openai_functions_agent


# ----------------- Setup Knowledge Base -----------------
def setup_knowledge_base(pdf_path="Lama1.pdf"):
    print("📚 Setting up knowledge base...")
    processor = PDFProcessor()
    chunks = processor.process_pdf(pdf_path)

    if not chunks:
        print("❌ Failed to process PDF.")
        return None

    vector_manager = VectorStoreManager()
    vector_manager.clear_vector_store()
    return vector_manager.create_vector_store(chunks)


def clean_markdown(text):
    if not text:
        return ""
    text = re.sub(r'\*\*|\*|__|_', '', text)
    text = re.sub(r'```.*?```', '', text, flags=re.DOTALL)
    text = re.sub(r'`[^`]*`', '', text)
    text = re.sub(r'^#+\s+', '', text, flags=re.MULTILINE)
    text = re.sub(r'^[*-]\s+', '', text, flags=re.MULTILINE)
    text = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


# ----------------- Main -----------------
def main():
    load_dotenv()
    print("🤖 Initializing LAMA Customer Support AI...")

    PDF_PATH = "Lama1.pdf"
    if not os.path.exists(PDF_PATH):
        print(f"❌ PDF '{PDF_PATH}' not found.")
        return

    vector_manager = VectorStoreManager()
    if not vector_manager.vector_store_exists():
        vector_store = setup_knowledge_base(PDF_PATH)
        if vector_store is None:
            return
    else:
        vector_store = vector_manager.load_vector_store()

    memory_manager = MemoryManager()
    chatbot = LAMAChatbot(vector_store, memory_manager)

    agent = create_openai_functions_agent(
        chatbot.llm,
        [chatbot.retriever_tool],
        prompt=chatbot.prompt
    )

    executor = AgentExecutor(
        agent=agent,
        tools=[chatbot.retriever_tool],
        memory=memory_manager.get_memory(),
        verbose=True
    )
    chatbot.set_agent_executor(executor)

    print("\n" + "=" * 65)
    print("🤖 LAMA Customer Support AI")
    print("💬 Start typing your question")
    print("❌ Type 'exit' or 'quit' to stop")
    print("=" * 65)

    # ----------------- Chat Loop -----------------
    while True:
        try:
            question = input("\nYou: ").strip()

            if not question:
                continue

            if question.lower() in ["exit", "quit", "bye", "goodbye"]:
                print("👋 Goodbye!")
                break

            print("🔍 Searching knowledge base...")
            response = chatbot.ask(question)

            cleaned_response = clean_markdown(response)
            print(f"\n🤖 LAMA Support:\n{cleaned_response}")

        except KeyboardInterrupt:
            print("\n👋 Interrupted. Exiting.")
            break
        except Exception as e:
            print(f"❌ Error: {e}")


if __name__ == "__main__":
    main()







# # File: main.py
# import os
# import re
# from dotenv import load_dotenv
# from pdf_processor import PDFProcessor
# from vector_store_manager import VectorStoreManager
# from memory_manager import MemoryManager
# from chatbot import LAMAChatbot
# from langchain.agents import AgentExecutor, create_openai_functions_agent

# # Whisper STT + TTS
# # from whisper import LiveWhisperSTT

# # ----------------- Setup Knowledge Base -----------------
# def setup_knowledge_base(pdf_path="Lama.pdf"):
#     print("📚 Setting up knowledge base...")
#     processor = PDFProcessor()
#     chunks = processor.process_pdf(pdf_path)

#     if not chunks:
#         print("❌ Failed to process PDF.")
#         return None

#     vector_manager = VectorStoreManager()
#     vector_manager.clear_vector_store()
#     return vector_manager.create_vector_store(chunks)

# def clean_markdown(text):
#     """Remove markdown formatting from text for clean TTS output"""
#     if not text:
#         return ""
    
#     # Remove bold/italic markers
#     text = re.sub(r'\*\*|\*|__|_', '', text)
#     # Remove code blocks
#     text = re.sub(r'```.*?```', '', text, flags=re.DOTALL)
#     # Remove inline code
#     text = re.sub(r'`[^`]*`', '', text)
#     # Remove headers
#     text = re.sub(r'^#+\s+', '', text, flags=re.MULTILINE)
#     # Remove lists
#     text = re.sub(r'^[*-]\s+', '', text, flags=re.MULTILINE)
#     # Remove links
#     text = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', text)
#     # Remove extra whitespace
#     text = re.sub(r'\s+', ' ', text)
#     # Clean up multiple spaces
#     text = ' '.join(text.split())
#     return text.strip()

# # ----------------- Main -----------------
# def main():
#     load_dotenv()
#     print("🤖 Initializing LAMA Customer Support AI...")

#     PDF_PATH = "Lama.pdf"
#     if not os.path.exists(PDF_PATH):
#         print(f"❌ PDF '{PDF_PATH}' not found.")
#         return

#     vector_manager = VectorStoreManager()
#     if not vector_manager.vector_store_exists():
#         vector_store = setup_knowledge_base(PDF_PATH)
#         if vector_store is None:
#             return
#     else:
#         vector_store = vector_manager.load_vector_store()

#     memory_manager = MemoryManager()
#     chatbot = LAMAChatbot(vector_store, memory_manager)

#     agent = create_openai_functions_agent(
#         chatbot.llm,
#         [chatbot.retriever_tool],
#         prompt=chatbot.prompt
#     )

#     executor = AgentExecutor(
#         agent=agent,
#         tools=[chatbot.retriever_tool],
#         memory=memory_manager.get_memory(),
#         verbose=True
#     )
#     chatbot.set_agent_executor(executor)

#     # STT + TTS
#     # voice = LiveWhisperSTT()
#     last_response = ""

#     print("\n" + "=" * 65)
#     print("🤖 LAMA Customer Support AI")
#     print("🎤 Press [V] → Speak")
#     print("⌨️  Press [T] → Type")
#     print("🔊 Press [S] → Speak last response")
#     print("❌ Press [Q] → Quit")
#     print("=" * 65)

#     # ----------------- Chat Loop -----------------
#     while True:
#         try:
#             choice = input("\nChoose (V / T / S / Q): ").strip().lower()

#             # Quit
#             if choice == "q":
#                 print("👋 Goodbye!")
#                 break

#             # Voice Input
#             elif choice == "v":
#                 print("\n🎤 Voice input started...")
#                 print("💡 Speak now (Press ENTER to stop recording)")
#                 try:
#                     voice.start_recording()
#                     question = voice.partial_text.strip()
#                     if question:
#                         print(f"\n🎤 You said: {question}")
#                     else:
#                         print("⚠️ No speech detected.")
#                         continue
#                 except Exception as e:
#                     print(f"❌ Voice input error: {e}")
#                     continue

#             # Text Input
#             elif choice == "t":
#                 question = input("\n💬 Type your question: ").strip()

#             # Speak last response
#             elif choice == "s":
#                 if last_response:
#                     cleaned_response = clean_markdown(last_response)
#                     print(f"\n🔊 Speaking: {cleaned_response[:100]}...")
#                     try:
#                         voice.speak_chatbot_text(cleaned_response)
#                     except Exception as e:
#                         print(f"❌ TTS Error: {e}")
#                 else:
#                     print("⚠️ No response to speak yet.")
#                 continue

#             else:
#                 print("⚠️ Invalid option.")
#                 continue

#             if not question:
#                 print("⚠️ Empty input.")
#                 continue

#             if question.lower() in ["quit", "exit", "bye", "goodbye"]:
#                 print("👋 Goodbye!")
#                 break

#             print("🔍 Searching knowledge base...")
#             response = chatbot.ask(question)
#             last_response = response

#             # Clean response for display and TTS
#             cleaned_response = clean_markdown(response)
            
#             print(f"\n🤖 LAMA Support:\n{response}")
            
#             # Automatically speak the response after voice input
#             if choice == "v":
#                 print(f"\n🔊 Speaking response...")
#                 try:
#                     voice.speak_chatbot_text(cleaned_response)
#                 except Exception as e:
#                     print(f"❌ TTS Error: {e}")

#         except KeyboardInterrupt:
#             print("\n👋 Interrupted. Exiting.")
#             break
#         except Exception as e:
#             print(f"❌ Error: {e}")

# if __name__ == "__main__":
#     main()