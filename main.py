from fastapi import FastAPI, Request, UploadFile, File
from fastapi.responses import JSONResponse
from fastapi import UploadFile, File, APIRouter
import os
from langsmith.run_helpers import traceable

from fastapi import Request
from pydantic import BaseModel
from typing import Optional, List
import os
from db import init_db, save_to_db
from groq_client import chat_with_groq
from utils import user_wants_human_agent, notify_human_agent, generate_brief_summary
from scraper import scrape_website, compute_hash
from vectorstore import index_text, extract_text_from_pdf, retrieve_context
from dotenv import load_dotenv
load_dotenv()

# === Setup ===
app = FastAPI()
init_db()
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

last_scraped_hash = None  # cache website state
chat_sessions = {}  # in-memory session cache


# === Models ===
class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    session_id: str
    message: str
    email: Optional[str] = None
    phone: Optional[str] = None


# === Core Chat Endpoint ===


@traceable(name="Chat Endpoint")
@app.post("/chat")
async def chat(req: ChatRequest):
    session_id = req.session_id
    user_message = req.message.strip()

    # Create new session if needed
    if session_id not in chat_sessions:
        chat_sessions[session_id] = {
            "history": [],
            "email": req.email,
            "phone": req.phone,
            "awaiting_email": False,
            "awaiting_phone": False,
            "escalated": False
        }

    session = chat_sessions[session_id]
    session["history"].append({"role": "user", "content": user_message})

    # === Escalation flow ===
    if session["awaiting_email"]:
        session["email"] = user_message
        session["awaiting_email"] = False
        session["awaiting_phone"] = True
        bot_reply = "✅ Got your email. Could you please provide your phone number too?"

    elif session["awaiting_phone"]:
        session["phone"] = user_message
        session["awaiting_phone"] = False
        notify_human_agent("User requested escalation", session["history"])
        session["escalated"] = True
        bot_reply = "📩 Thanks! I’ve escalated this to a human agent. You’ll be contacted shortly."

    elif user_wants_human_agent(user_message) and not session["escalated"]:
        if not session["email"]:
            session["awaiting_email"] = True
            bot_reply = "🔔 I can connect you to a human agent. First, please provide your email."
        elif not session["phone"]:
            session["awaiting_phone"] = True
            bot_reply = "🔔 Got your email. Could you now provide your phone number?"
        else:
            notify_human_agent(user_message, session["history"])
            session["escalated"] = True
            bot_reply = "📩 I’ve escalated this to a human agent. You’ll be contacted shortly."

    else:
        # === Retrieval-Augmented Generation ===
        retrieved_docs = retrieve_context(user_message, k=4)
        context_text = "\n".join([doc.page_content for doc in retrieved_docs])

        # === Inject system prompt with context
        system_prompt = (
            "You are a helpful customer service chatbot for HotelsByDay. "
            "Use the following context to answer accurately:\n\n"
            f"{context_text}\n\n"
            "Be polite and informative. Do not redirect users to the website."
        )

        messages = [{"role": "system", "content": system_prompt}] + session["history"]
        response = chat_with_groq(messages)
        bot_reply = response.content


    session["history"].append({"role": "assistant", "content": bot_reply})

    # Save summary once we have both email and phone
    if session["email"] and session["phone"]:
        summary = generate_brief_summary(session["history"])
        save_to_db(summary, session["email"], session["phone"])

    return JSONResponse({"reply": bot_reply})


# === Endpoint: Update vectorstore from website ===
@app.post("/update-website")
async def update_vector_store():
    global last_scraped_hash
    url = "https://www.hotelsbyday.com/en/hotels/united-states"
    content = scrape_website(url)
    if content.startswith("Error"):
        return {"status": "error", "message": content}

    current_hash = compute_hash(content)
    if current_hash != last_scraped_hash:
        index_text(content, metadata={"source": "website"})
        last_scraped_hash = current_hash
        return {"status": "updated", "message": "Website changed and content indexed."}
    else:
        return {"status": "unchanged", "message": "No changes detected on website."}


# === Endpoint: Upload and index admin PDF ===


@app.post("/upload-pdf")
async def upload_pdf(file: UploadFile = File(...)):
    if not file.filename.endswith(".pdf"):
        return {"status": "error", "message": "Only PDF files are supported."}

    file_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(file_path, "wb") as f:
        f.write(await file.read())

    content = extract_text_from_pdf(file_path)
    if not content.strip():
        return {"status": "error", "message": "Failed to extract text from PDF."}

    index_text(content, metadata={"source": file.filename})
    return {"status": "success", "message": f"{file.filename} uploaded and indexed."}

