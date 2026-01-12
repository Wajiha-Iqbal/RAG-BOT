import fitz  # PyMuPDF
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

class PDFProcessor:
    def __init__(self, chunk_size=500, chunk_overlap=100):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
        )
    
    def extract_text_from_pdf(self, pdf_path):
        print("[1] Loading knowledge base from local file...")
        documents = []
        
        try:
            with fitz.open(pdf_path) as pdf:
                for page_num in range(len(pdf)):
                    page = pdf[page_num]
                    text = page.get_text()
                    
                    if text.strip():
                        documents.append(Document(
                            page_content=text,
                            metadata={
                                "source": pdf_path,
                                "page": page_num + 1
                            }
                        ))
            
            print(f"✅ Loaded {len(documents)} pages from PDF.")
            return documents
        except Exception as e:
            print(f"❌ Error loading PDF: {e}")
            return []
    
    def split_documents(self, documents):
        print("[2] Splitting text into chunks...")
        if not documents:
            print("❌ No documents to split.")
            return []
        
        chunks = self.text_splitter.split_documents(documents)
        print(f"✅ Created {len(chunks)} chunks.")
        return chunks

    def process_pdf(self, pdf_path):
        """Convenience method to extract and split in one call"""
        documents = self.extract_text_from_pdf(pdf_path)
        return self.split_documents(documents)