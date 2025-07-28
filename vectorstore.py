from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from chromadb.config import Settings 
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langsmith import traceable
import pdfplumber

# Initialize embedding model and vector DB path
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
VECTOR_DIR = "db"

@traceable(name="Index Text")


def index_text(text: str, metadata: dict):
    # print(f"[IndexText] Text length: {len(text)} characters")

    if not text.strip():
        print("[DEBUG] Skipping empty content.")
        return

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    chunks = splitter.create_documents([text], metadatas=[metadata])

    if not chunks:
        print("[DEBUG] No chunks created from text.")
        return

    # print(f"[DEBUG] Indexing {len(chunks)} chunks...")

    vectordb = Chroma(
        persist_directory=VECTOR_DIR,
        embedding_function=embedding_model,
        collection_name="default",
        client_settings=Settings(persist_directory=VECTOR_DIR)
    )

    try:
        print(f"initiated adding document to db")
        vectordb.add_documents(chunks)
        print(f'added documents to db')
    except ValueError as e:
        print(f"[ERROR] Error adding documents to Chroma: {e}")


def extract_text_from_pdf(file_path: str) -> str:
    try:
        with pdfplumber.open(file_path) as pdf:
            return "\n".join(page.extract_text() or "" for page in pdf.pages)
    except Exception as e:
        print(f"[ERROR] Failed to extract text from PDF: {e}")
        return ""

@traceable(name="Retrieve Context")
def retrieve_context(query: str, k: int = 3):
    print(f"[DEBUG] Retrieving context for query: {query}")

    vectordb = Chroma(
        persist_directory=VECTOR_DIR,
        embedding_function=embedding_model,
        collection_name="default",
        client_settings=Settings(persist_directory=VECTOR_DIR)
    )
    # print("[DEBUG] Collection document count:", vectordb._collection.count())

    docs = vectordb.similarity_search(query, k=10)
    # for i, doc in enumerate(docs):
        # print(f"[MATCH {i+1}] {doc.metadata.get('source')}:\n{doc.page_content[:200]}\n")

    try:
        retriever = vectordb.as_retriever(search_kwargs={"k": k})
        results = retriever.invoke(query)
        # print(f"[DEBUG] Retrieved {len(results)} documents.")
        # for i, doc in enumerate(results):
        #     print(f"[DEBUG] Result {i+1}: {doc.page_content[:300]}...")
        return results
    except Exception as e:
        print(f"[ERROR] Retrieval error: {e}")
        return []

# === Knowledge Base Management Functions ===

def get_knowledge_base_stats():
    """Get statistics about the knowledge base"""
    try:
        vectordb = Chroma(
            persist_directory=VECTOR_DIR,
            embedding_function=embedding_model,
            collection_name="default",
            client_settings=Settings(persist_directory=VECTOR_DIR)
        )
        
        # Get total document count
        total_docs = vectordb._collection.count()
        
        # Get unique sources
        sources = {}
        if total_docs > 0:
            # Get all documents to analyze sources
            all_docs = vectordb.get()
            if all_docs and 'metadatas' in all_docs:
                for metadata in all_docs['metadatas']:
                    if metadata and 'source' in metadata:
                        source = metadata['source']
                        sources[source] = sources.get(source, 0) + 1
        
        return {
            "total_documents": total_docs,
            "sources": sources,
            "vector_db_path": VECTOR_DIR
        }
    except Exception as e:
        print(f"[ERROR] Failed to get knowledge base stats: {e}")
        return {
            "total_documents": 0,
            "sources": {},
            "vector_db_path": VECTOR_DIR,
            "error": str(e)
        }

def get_indexed_documents():
    """Get list of all indexed documents with their metadata"""
    try:
        vectordb = Chroma(
            persist_directory=VECTOR_DIR,
            embedding_function=embedding_model,
            collection_name="default",
            client_settings=Settings(persist_directory=VECTOR_DIR)
        )
        
        all_docs = vectordb.get()
        documents = []
        
        if all_docs and 'metadatas' in all_docs and 'documents' in all_docs:
            for i, metadata in enumerate(all_docs['metadatas']):
                if metadata:
                    doc_info = {
                        "id": all_docs.get('ids', [])[i] if 'ids' in all_docs else f"doc_{i}",
                        "source": metadata.get('source', 'unknown'),
                        "content_preview": all_docs['documents'][i][:200] + "..." if len(all_docs['documents'][i]) > 200 else all_docs['documents'][i],
                        "metadata": metadata
                    }
                    documents.append(doc_info)
        
        return documents
    except Exception as e:
        print(f"[ERROR] Failed to get indexed documents: {e}")
        return []

def clear_knowledge_base():
    """Clear all documents from the knowledge base"""
    try:
        vectordb = Chroma(
            persist_directory=VECTOR_DIR,
            embedding_function=embedding_model,
            collection_name="default",
            client_settings=Settings(persist_directory=VECTOR_DIR)
        )
        
        # Delete all documents
        vectordb._collection.delete(where={})
        print("[INFO] Knowledge base cleared successfully")
        
    except Exception as e:
        print(f"[ERROR] Failed to clear knowledge base: {e}")
        raise e
