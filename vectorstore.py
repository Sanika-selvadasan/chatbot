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
    print(f"[IndexText] Text length: {len(text)} characters")

    if not text.strip():
        print("[DEBUG] Skipping empty content.")
        return

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    chunks = splitter.create_documents([text], metadatas=[metadata])

    if not chunks:
        print("[DEBUG] No chunks created from text.")
        return

    print(f"[DEBUG] Indexing {len(chunks)} chunks...")

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
    vectordb = Chroma(persist_directory=VECTOR_DIR, embedding_function=embedding_model)

    try:
        retriever = vectordb.as_retriever(search_kwargs={"k": k})
        results = retriever.invoke(query)
        print(f"[DEBUG] Retrieved {len(results)} documents.")
        for i, doc in enumerate(results):
            print(f"[DEBUG] Result {i+1}: {doc.page_content[:300]}...")
        return results
    except Exception as e:
        print(f"[ERROR] Retrieval error: {e}")
        return []
