from fastapi import FastAPI, Request, UploadFile, File
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional
import os
from db import init_db, save_to_db
from groq_client import chat_with_groq
from utils import user_wants_human_agent, generate_brief_summary
from scraper import scrape_website, compute_hash
from vectorstore import index_text, extract_text_from_pdf, retrieve_context
from dotenv import load_dotenv
from langsmith.run_helpers import traceable

load_dotenv()

app = FastAPI()
init_db()
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

last_scraped_hash = None
chat_sessions = {}


class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    session_id: str
    message: str
    email: Optional[str] = None
    phone: Optional[str] = None

LOW_CONFIDENCE_THRESHOLD = 0.4  # confidence threshold for escalation


@traceable(name="Chat Endpoint")


# === Core Chat Endpoint ===
@app.post("/chat")
async def chat(req: ChatRequest):
    session_id = req.session_id
    user_message = req.message.strip()
    print(f"[User:{session_id}] {user_message}")

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

    if session["awaiting_email"]:
        session["email"] = user_message
        session["awaiting_email"] = False
        session["awaiting_phone"] = True
        bot_reply = "✅ Got your email. Could you please provide your phone number too?"

    elif session["awaiting_phone"]:
        session["phone"] = user_message
        session["awaiting_phone"] = False
        escalate_to_human(session)
        bot_reply = "📩 Thanks! I’ve escalated this to a human agent. You’ll be contacted shortly."

    elif user_wants_human_agent(user_message) and not session["escalated"]:
        if not session["email"]:
            session["awaiting_email"] = True
            bot_reply = "🔔 I can connect you to a human agent. First, please provide your email."
        elif not session["phone"]:
            session["awaiting_phone"] = True
            bot_reply = "🔔 Got your email. Could you now provide your phone number?"
        else:
            escalate_to_human(session)
            bot_reply = "📩 I’ve escalated this to a human agent. You’ll be contacted shortly."

    else:
        retrieved_docs = retrieve_context(user_message, k=4)
        context_text = "\n".join([doc.page_content for doc in retrieved_docs])

        system_prompt = (
            "You are a helpful assistant. Use the following context to answer:\n\n"
            f"{context_text}\n\n"
            "If unsure, admit uncertainty. Be polite and concise."
        )

        messages = [{"role": "system", "content": system_prompt}] + session["history"]
        response = chat_with_groq(messages)
        bot_reply = response.content.strip()

        # Low confidence check (optional scoring logic)
        if "I'm not sure" in bot_reply.lower() or "I don’t have that information" in bot_reply.lower():
            print("[Confidence] Low response confidence. Escalating...")
            escalate_to_human(session)
            bot_reply = "⚠️ I’m escalating this to a human agent for better assistance."

    session["history"].append({"role": "assistant", "content": bot_reply})

    # Save summary if both email & phone exist
    if session["email"] and session["phone"]:
        summary = generate_brief_summary(session["history"])
        save_to_db(summary, session["email"], session["phone"])

    return JSONResponse({"reply": bot_reply})

# Escalation Handler
def escalate_to_human(session):
    session["escalated"] = True
    summary = generate_brief_summary(session["history"])
    print("\n=== URGENT HUMAN ALERT ===")
    print("Chat Summary:\n", summary)
    print("==========================\n")

@app.post("/update-website")
async def update_vector_store():
    print(f'inside update website')
    global last_scraped_hash
    url = "https://www.hotelsbyday.com/en/hotels/united-states"
    print("About to scrape website")
    content = scrape_website(url)
    print("scraping fininshed")
    if content.startswith("Error"):
        print(f"[ERROR] {content}")
        return {"status": "error", "message": content}

    current_hash = compute_hash(content)
    if current_hash != last_scraped_hash:
        print("[INFO] Website updated. Indexing...")
        index_text(content, metadata={"source": "website"})
        last_scraped_hash = current_hash
        return {"status": "updated", "message": "Website changed and content indexed."}
    else:
        print("[INFO] No website change detected.")
        return {"status": "unchanged", "message": "No changes detected on website."}


@app.post("/upload-pdf")
async def upload_pdf(file: UploadFile = File(...)):
    if not file.filename.endswith(".pdf"):
        return {"status": "error", "message": "Only PDF files are supported."}

    file_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(file_path, "wb") as f:
        f.write(await file.read())

    content = extract_text_from_pdf(file_path)
    print(f'Content pdf{content}')
    if not content.strip():
        print("[ERROR] Failed to extract text from PDF.")
        return {"status": "error", "message": "Failed to extract text from PDF."}

    print(f"[INFO] Indexing PDF: {file.filename}")
    index_text(content, metadata={"source": file.filename})
    return {"status": "success", "message": f"{file.filename} uploaded and indexed."}
