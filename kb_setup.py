import chromadb
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings

def initialize_knowledge_base():
    print("Initializing ChromaDB Vector Store...")
    
    # Initialize persistent ChromaDB client
    chroma_client = chromadb.PersistentClient(path="./chroma_db")
    
    # Use local, fast, free embeddings
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    # Create or get collection
    collection = chroma_client.get_or_create_collection(name="support_kb")
    
    # Dummy e-commerce Knowledge Base Data
    faq_data = [
        {"id": "doc1", "text": "Returns Policy: You can return items within 30 days of receipt. Items must be unworn and in original packaging. To start a return, provide your Order ID."},
        {"id": "doc2", "text": "Shipping Delays: Standard shipping takes 3-5 business days. During peak holiday seasons, allow up to 7 business days."},
        {"id": "doc3", "text": "Damaged Items: If your product arrives damaged, please upload a photo within 48 hours to receive an immediate replacement."},
        {"id": "doc4", "text": "Refund Process: Refunds are processed back to the original payment method within 5-7 business days after we receive the returned item."}
    ]
    
    # Insert vectors
    for doc in faq_data:
        vector = embeddings.embed_query(doc["text"])
        collection.upsert(
            documents=[doc["text"]],
            embeddings=[vector],
            ids=[doc["id"]]
        )
        
    print("Knowledge base successfully populated!")

if __name__ == "__main__":
    initialize_knowledge_base()