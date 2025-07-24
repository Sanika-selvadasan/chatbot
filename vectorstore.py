from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
import pdfplumber
from langchain_core.documents import Document
from langsmith import traceable

embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
VECTOR_DIR = "db"



@traceable(name="Index Text")
def index_text(text: str, metadata: dict):
    if not text.strip():
        print(" Skipping empty content")
        return

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    chunks = splitter.create_documents([text], metadatas=[metadata])

    if not chunks:
        print("No chunks created from text")
        return

    print(f"Indexing {len(chunks)} chunks")

    vectordb = Chroma(persist_directory=VECTOR_DIR, embedding_function=embedding_model)

    try:
        vectordb.add_documents(chunks)
        vectordb.persist()
    except ValueError as e:
        print(f"Error adding documents to Chroma: {e}")


def extract_text_from_pdf(file_path: str) -> str:
    try:
        with pdfplumber.open(file_path) as pdf:
            return "\n".join(page.extract_text() or "" for page in pdf.pages)
    except Exception as e:
        print(f"Failed to extract text from PDF: {e}")
        return ""
    


@traceable(name="Retrieve Context")
def retrieve_context(query: str, k: int = 3):
    print(f" Searching for: {query}")
 # prints # of vectors stored

    vectordb = Chroma(persist_directory=VECTOR_DIR, embedding_function=embedding_model)
    print("Checking DB contents...")
    print(vectordb._collection.count()) 
    retriever = vectordb.as_retriever(search_kwargs={"k": k})

    try:
        results = retriever.invoke(query)
        print(f"Retrieved {len(results)} documents")
        for i, doc in enumerate(results):
            print(f"Result {i+1}: {doc.page_content[:300]}")
        return results
    except Exception as e:
        print("Retrieval error:", e)
        return []
