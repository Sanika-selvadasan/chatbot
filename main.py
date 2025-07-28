from fastapi import FastAPI, Request, UploadFile, File, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Dict
import os
from db import init_db, save_to_db
from groq_client import chat_with_groq, get_confidence_score
from utils import user_wants_human_agent, generate_brief_summary
from scraper import scrape_website, compute_hash
from vectorstore import index_text, extract_text_from_pdf, retrieve_context
from dotenv import load_dotenv
from langsmith.run_helpers import traceable
import uuid
from datetime import datetime
import json
import asyncio

load_dotenv()

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],  # Frontend URLs
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods
    allow_headers=["*"],  # Allow all headers
)
init_db()
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

last_scraped_hash = None
chat_sessions = {}
human_agent_sessions = {}  # Track human agent sessions

# WebSocket connection management
class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}
        self.session_connections: Dict[str, str] = {}  # session_id -> connection_id
        self.agent_connections: Dict[str, str] = {}    # agent_id -> connection_id

    async def connect(self, websocket: WebSocket, connection_id: str):
        await websocket.accept()
        self.active_connections[connection_id] = websocket

    def disconnect(self, connection_id: str):
        if connection_id in self.active_connections:
            del self.active_connections[connection_id]
        
        # Clean up session and agent mappings
        session_id = None
        agent_id = None
        for sid, cid in self.session_connections.items():
            if cid == connection_id:
                session_id = sid
                break
        for aid, cid in self.agent_connections.items():
            if cid == connection_id:
                agent_id = aid
                break
        
        if session_id:
            del self.session_connections[session_id]
        if agent_id:
            del self.agent_connections[agent_id]

    async def send_personal_message(self, message: str, connection_id: str):
        if connection_id in self.active_connections:
            await self.active_connections[connection_id].send_text(message)

    async def broadcast_to_session(self, message: str, session_id: str):
        if session_id in self.session_connections:
            connection_id = self.session_connections[session_id]
            await self.send_personal_message(message, connection_id)

    async def broadcast_to_agent(self, message: str, agent_id: str):
        if agent_id in self.agent_connections:
            connection_id = self.agent_connections[agent_id]
            await self.send_personal_message(message, connection_id)

manager = ConnectionManager()

# === Enhanced Models ===
class Message(BaseModel):
    role: str
    content: str
    confidence: Optional[float] = None
    timestamp: Optional[str] = None

class ChatRequest(BaseModel):
    session_id: str
    message: str
    email: Optional[str] = None
    phone: Optional[str] = None
    agent_id: Optional[str] = None  # For human agent messages

class HumanAgentRequest(BaseModel):
    session_id: str
    agent_id: str
    message: str

class SessionStatus(BaseModel):
    session_id: str
    is_escalated: bool
    agent_id: Optional[str] = None
    escalated_at: Optional[str] = None
    confidence_threshold: float = 0.4

LOW_CONFIDENCE_THRESHOLD = 0.4  # confidence threshold for escalation


# === Enhanced Chat Endpoint with Confidence Scoring ===
@app.post("/chat")
async def chat(req: ChatRequest):
    session_id = req.session_id
    user_message = req.message.strip()
    print(f"[User:{session_id}] {user_message}")

    # Check if session is escalated to human agent
    if session_id in chat_sessions and chat_sessions[session_id].get("escalated"):
        # Add user message to session history
        user_msg = {
            "role": "user", 
            "content": user_message,
            "timestamp": datetime.now().isoformat()
        }
        chat_sessions[session_id]["history"].append(user_msg)
        
        # Route message to admin agent
        agent_id = chat_sessions[session_id].get("agent_id")
        if agent_id and agent_id in human_agent_sessions:
            # Update agent session history
            human_agent_sessions[agent_id]["history"].append(user_msg)
            
            return JSONResponse({
                "reply": "Your message has been sent to the human agent. They will respond shortly.",
                "escalated": True,
                "agent_id": agent_id
            })
        else:
            return JSONResponse({
                "reply": "This conversation has been escalated to a human agent. Please wait for their response.",
                "escalated": True,
                "agent_id": agent_id
            })

    # Initialize session if needed
    if session_id not in chat_sessions:
        chat_sessions[session_id] = {
            "history": [],
            "escalated": False,
            "agent_id": None,
            "escalated_at": None,
            "confidence_scores": []
        }

    session = chat_sessions[session_id]
    
    # Add user message with timestamp
    user_msg = {
        "role": "user", 
        "content": user_message,
        "timestamp": datetime.now().isoformat()
    }
    session["history"].append(user_msg)

    # Get context for RAG
    retrieved_docs = retrieve_context(user_message, k=4)
    context_text = "\n".join([doc.page_content for doc in retrieved_docs])

    # Enhanced system prompt for confidence scoring
    system_prompt = f"""
    You are a professional customer service assistant for HotelsByDay, helping users book day-use hotel rooms in the United States.

    CRITICAL RULES:
    1. ONLY answer questions using information from the provided context.
    2. If the context is empty, insufficient, or doesn't contain relevant information, say "I don't have enough information to answer that question accurately. Let me connect you to a human agent who can help you."
    3. NEVER invent, speculate, or provide information not explicitly stated in the context.
    4. NEVER refer to external websites, contact information, or suggest users search online.
    5. If you're unsure about any detail, escalate to human agent.
    6. For pricing questions, only provide general information about pricing structure, not specific prices unless clearly stated in context.
    7. For booking requests, always escalate to human agent as you cannot make actual bookings.

    Your behavior:
    - Be concise, factual, and professional.
    - Only provide information that is clearly supported by the context.
    - If context is missing or unclear, escalate immediately.
    - Provide a confidence score (0.0-1.0) based on how well the context supports your answer.
    - For specific pricing, availability, or booking requests, escalate to human agent.

    Context provided:
    {context_text if context_text.strip() else "NO CONTEXT AVAILABLE"}

    IMPORTANT: End your response with a confidence score in this format: [CONFIDENCE: 0.85]
    If you cannot provide a helpful answer due to insufficient context, use confidence score 0.1
    """

    messages = [{"role": "system", "content": system_prompt}] + session["history"]
    response = chat_with_groq(messages)
    bot_reply = response.content.strip()

    # Extract confidence score from response
    confidence_score = get_confidence_score(bot_reply)
    bot_reply_clean = bot_reply.replace(f"[CONFIDENCE: {confidence_score}]", "").strip()
    
    # Store confidence score
    session["confidence_scores"].append(confidence_score)

    # Enhanced escalation logic
    should_escalate = (
        user_wants_human_agent(user_message) or
        confidence_score < LOW_CONFIDENCE_THRESHOLD or
        "I'm not sure" in bot_reply_clean.lower() or
        "I don't have that information" in bot_reply_clean.lower() or
        "I don't have enough information" in bot_reply_clean.lower() or
        "connect you to a human agent" in bot_reply_clean.lower() or
        "book" in user_message.lower() or
        "reservation" in user_message.lower() or
        "booking" in user_message.lower() or
        "price" in user_message.lower() or
        "cost" in user_message.lower() or
        "rate" in user_message.lower() or
        "availability" in user_message.lower() or
        "room" in user_message.lower() and ("available" in user_message.lower() or "book" in user_message.lower()) or
        "how much" in user_message.lower() or
        "what's the price" in user_message.lower() or
        "what is the cost" in user_message.lower() or
        "check availability" in user_message.lower() or
        "make a reservation" in user_message.lower() or
        not context_text.strip() or
        len(context_text.strip()) < 100 or
        context_text.strip() == "NO CONTEXT AVAILABLE"
    )

    if should_escalate and not session["escalated"]:
        agent_id = escalate_to_human(session_id, session)
        bot_reply_clean = f"Thank you for your question. I'm connecting you to a human agent who will assist you further. Your session ID is: {session_id}"
        
        return JSONResponse({
            "reply": bot_reply_clean,
            "escalated": True,
            "agent_id": agent_id,
            "confidence_score": confidence_score,
            "session_id": session_id
        })

    # Add bot response with confidence and timestamp
    bot_msg = {
        "role": "assistant", 
        "content": bot_reply_clean,
        "confidence": confidence_score,
        "timestamp": datetime.now().isoformat()
    }
    session["history"].append(bot_msg)

    return JSONResponse({
        "reply": bot_reply_clean,
        "escalated": False,
        "confidence_score": confidence_score,
        "session_id": session_id
    })

# === Debug Endpoint for Testing ===
@app.post("/debug/chat")
async def debug_chat(req: ChatRequest):
    """Debug endpoint to test chat functionality"""
    session_id = req.session_id
    user_message = req.message.strip()
    print(f"[DEBUG] User message: {user_message}")

    # Get context for RAG
    retrieved_docs = retrieve_context(user_message, k=4)
    context_text = "\n".join([doc.page_content for doc in retrieved_docs])
    
    print(f"[DEBUG] Retrieved {len(retrieved_docs)} documents")
    print(f"[DEBUG] Context length: {len(context_text)}")
    print(f"[DEBUG] Context preview: {context_text[:500]}...")

    # Enhanced system prompt for confidence scoring
    system_prompt = f"""
    You are a professional customer service assistant for HotelsByDay, helping users book day-use hotel rooms in the United States.

    CRITICAL RULES:
    1. ONLY answer questions using information from the provided context.
    2. If the context is empty, insufficient, or doesn't contain relevant information, say "I don't have enough information to answer that question accurately. Let me connect you to a human agent who can help you."
    3. NEVER invent, speculate, or provide information not explicitly stated in the context.
    4. NEVER refer to external websites, contact information, or suggest users search online.
    5. If you're unsure about any detail, escalate to human agent.

    Your behavior:
    - Be concise, factual, and professional.
    - Only provide information that is clearly supported by the context.
    - If context is missing or unclear, escalate immediately.
    - Provide a confidence score (0.0-1.0) based on how well the context supports your answer.

    Context provided:
    {context_text if context_text.strip() else "NO CONTEXT AVAILABLE"}

    IMPORTANT: End your response with a confidence score in this format: [CONFIDENCE: 0.85]
    If you cannot provide a helpful answer due to insufficient context, use confidence score 0.1
    """

    messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_message}]
    response = chat_with_groq(messages)
    bot_reply = response.content.strip()

    # Extract confidence score from response
    confidence_score = get_confidence_score(bot_reply)
    bot_reply_clean = bot_reply.replace(f"[CONFIDENCE: {confidence_score}]", "").strip()
    
    print(f"[DEBUG] Bot reply: {bot_reply_clean}")
    print(f"[DEBUG] Confidence score: {confidence_score}")

    return JSONResponse({
        "reply": bot_reply_clean,
        "confidence_score": confidence_score,
        "context_length": len(context_text),
        "documents_retrieved": len(retrieved_docs),
        "context_preview": context_text[:200] + "..." if len(context_text) > 200 else context_text
    })

# === Enhanced Escalation Handler ===
def escalate_to_human(session_id: str, session: dict):
    session["escalated"] = True
    session["escalated_at"] = datetime.now().isoformat()
    
    # Generate a unique agent ID for this session
    agent_id = f"agent_{uuid.uuid4().hex[:8]}"
    session["agent_id"] = agent_id
    
    # Store session in human agent sessions
    human_agent_sessions[agent_id] = {
        "session_id": session_id,  # Use the actual session_id parameter
        "history": session["history"].copy(),
        "escalated_at": session["escalated_at"],
        "status": "waiting"  # waiting, active, completed
    }
    
    summary = generate_brief_summary(session["history"])
    print(f"\n=== URGENT HUMAN ALERT ===")
    print(f"Session ID: {session_id}")
    print(f"Agent ID: {agent_id}")
    print(f"Chat Summary:\n{summary}")
    print("==========================\n")
    
    # Save to database with agent info and escalation status
    save_to_db(summary, agent_id=agent_id, session_id=session_id, escalated=True)
    
    return agent_id

def load_escalated_sessions_from_db():
    """Load escalated sessions from database on startup"""
    try:
        import sqlite3
        conn = sqlite3.connect('conversations.db')
        cursor = conn.cursor()
        
        # Get escalated sessions from database
        cursor.execute("""
            SELECT session_id, agent_id, escalated_at, summary 
            FROM conversations 
            WHERE escalated = 1 AND agent_id IS NOT NULL AND session_id IS NOT NULL
        """)
        
        escalated_sessions = cursor.fetchall()
        
        for session_data in escalated_sessions:
            session_id, agent_id, escalated_at, summary = session_data
            
            # Create session in chat_sessions if it doesn't exist
            if session_id not in chat_sessions:
                chat_sessions[session_id] = {
                    "history": [],
                    "escalated": True,
                    "agent_id": agent_id,
                    "escalated_at": escalated_at,
                    "confidence_scores": []
                }
            else:
                # Update existing session
                chat_sessions[session_id]["escalated"] = True
                chat_sessions[session_id]["agent_id"] = agent_id
                chat_sessions[session_id]["escalated_at"] = escalated_at
            
            # Recreate human agent session
            human_agent_sessions[agent_id] = {
                "session_id": session_id,
                "history": chat_sessions[session_id]["history"].copy(),
                "escalated_at": escalated_at,
                "status": "waiting"
            }
            
            print(f"[INFO] Restored escalated session: {session_id} -> {agent_id}")
        
        conn.close()
        print(f"[INFO] Loaded {len(escalated_sessions)} escalated sessions from database")
        
    except Exception as e:
        print(f"[ERROR] Failed to load escalated sessions from database: {e}")


@app.post("/update-website")
async def update_vector_store():
    print(f'inside update website')
    global last_scraped_hash
    url = "https://www.hotelsbyday.com"
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

# === Human Agent Chat Endpoint ===
@app.post("/agent-chat")
async def agent_chat(req: HumanAgentRequest):
    session_id = req.session_id
    agent_id = req.agent_id
    agent_message = req.message.strip()
    
    # Verify agent has access to this session
    if agent_id not in human_agent_sessions:
        return JSONResponse({
            "status": "error",
            "message": "Invalid agent ID or session not found"
        }, status_code=404)
    
    agent_session = human_agent_sessions[agent_id]
    if agent_session["session_id"] != session_id:
        return JSONResponse({
            "status": "error", 
            "message": "Agent not authorized for this session"
        }, status_code=403)
    
    # Add agent message to chat history
    if session_id in chat_sessions:
        chat_sessions[session_id]["history"].append({
            "role": "agent",
            "content": agent_message,
            "timestamp": datetime.now().isoformat(),
            "agent_id": agent_id
        })
        
        # Update agent session status
        agent_session["status"] = "active"
        agent_session["history"].append({
            "role": "agent",
            "content": agent_message,
            "timestamp": datetime.now().isoformat()
        })
    
    return JSONResponse({
        "status": "success",
        "message": "Agent message sent successfully"
    })

# === Take Session (Update Status) ===
@app.post("/agent/sessions/{agent_id}/take")
async def take_session(agent_id: str):
    """Mark a session as taken by an agent"""
    if agent_id not in human_agent_sessions:
        return JSONResponse({
            "status": "error",
            "message": "Agent session not found"
        }, status_code=404)
    
    # Update the session status to active
    human_agent_sessions[agent_id]["status"] = "active"
    
    return JSONResponse({
        "status": "success",
        "message": f"Session taken by agent {agent_id}",
        "agent_id": agent_id,
        "session_id": human_agent_sessions[agent_id]["session_id"]
    })

# === Get Session Status ===
@app.get("/session/{session_id}/status")
async def get_session_status(session_id: str):
    if session_id not in chat_sessions:
        return JSONResponse({
            "status": "error",
            "message": "Session not found"
        }, status_code=404)
    
    session = chat_sessions[session_id]
    return JSONResponse({
        "session_id": session_id,
        "is_escalated": session.get("escalated", False),
        "agent_id": session.get("agent_id"),
        "escalated_at": session.get("escalated_at"),
        "message_count": len(session.get("history", [])),
        "confidence_scores": session.get("confidence_scores", [])
    })

# === Get Session Chat History ===
@app.get("/session/{session_id}/history")
async def get_session_history(session_id: str):
    """Get the complete chat history for a session"""
    if session_id not in chat_sessions:
        return JSONResponse({
            "status": "error",
            "message": "Session not found"
        }, status_code=404)
    
    session = chat_sessions[session_id]
    return JSONResponse({
        "session_id": session_id,
        "history": session.get("history", []),
        "escalated": session.get("escalated", False),
        "agent_id": session.get("agent_id"),
        "escalated_at": session.get("escalated_at")
    })

# === Get Session Summary ===
@app.get("/session/{session_id}/summary")
async def get_session_summary(session_id: str):
    """Get a brief summary of the conversation for admin context"""
    if session_id not in chat_sessions:
        return JSONResponse({
            "status": "error",
            "message": "Session not found"
        }, status_code=404)
    
    session = chat_sessions[session_id]
    history = session.get("history", [])
    
    if not history:
        return JSONResponse({
            "status": "success",
            "summary": "No conversation history available.",
            "message_count": 0
        })
    
    # Generate summary using existing function
    summary = generate_brief_summary(history)
    
    return JSONResponse({
        "status": "success",
        "summary": summary,
        "message_count": len(history),
        "escalated": session.get("escalated", False),
        "agent_id": session.get("agent_id"),
        "escalated_at": session.get("escalated_at")
    })

# === Get Available Escalated Sessions ===
@app.get("/agent/sessions")
async def get_escalated_sessions():
    escalated_sessions = []
    for agent_id, session_data in human_agent_sessions.items():
        if session_data["status"] == "waiting":
            escalated_sessions.append({
                "agent_id": agent_id,
                "session_id": session_data["session_id"],
                "escalated_at": session_data["escalated_at"],
                "message_count": len(session_data["history"])
            })
    
    return JSONResponse({
        "escalated_sessions": escalated_sessions,
        "total_waiting": len(escalated_sessions)
    })

# === Admin Knowledge Base Management Endpoints ===

@app.get("/admin/knowledge-base/status")
async def get_knowledge_base_status():
    """Get current knowledge base status and statistics"""
    try:
        from vectorstore import get_knowledge_base_stats
        stats = get_knowledge_base_stats()
        return JSONResponse({
            "status": "success",
            "data": stats
        })
    except Exception as e:
        print(f"[ERROR] Failed to get knowledge base status: {e}")
        return JSONResponse({
            "status": "error",
            "message": f"Failed to get knowledge base status: {str(e)}"
        }, status_code=500)

@app.post("/admin/knowledge-base/force-update")
async def force_update_website():
    """Force re-scrape website regardless of hash changes"""
    try:
        global last_scraped_hash
        url = "https://www.hotelsbyday.com"
        print("[INFO] Force updating website content...")
        
        content = scrape_website(url)
        if content.startswith("Error"):
            print(f"[ERROR] {content}")
            return JSONResponse({
                "status": "error",
                "message": content
            }, status_code=500)

        print("[INFO] Force indexing website content...")
        index_text(content, metadata={"source": "website", "updated_at": datetime.now().isoformat()})
        
        # Update hash to current content
        current_hash = compute_hash(content)
        last_scraped_hash = current_hash
        
        return JSONResponse({
            "status": "success",
            "message": "Website content force updated and indexed successfully.",
            "hash": current_hash
        })
    except Exception as e:
        print(f"[ERROR] Force update failed: {e}")
        return JSONResponse({
            "status": "error",
            "message": f"Force update failed: {str(e)}"
        }, status_code=500)

@app.get("/admin/knowledge-base/documents")
async def get_indexed_documents():
    """Get list of indexed documents in knowledge base"""
    try:
        from vectorstore import get_indexed_documents
        documents = get_indexed_documents()
        return JSONResponse({
            "status": "success",
            "data": documents
        })
    except Exception as e:
        print(f"[ERROR] Failed to get indexed documents: {e}")
        return JSONResponse({
            "status": "error",
            "message": f"Failed to get indexed documents: {str(e)}"
        }, status_code=500)

@app.delete("/admin/knowledge-base/clear")
async def clear_knowledge_base():
    """Clear all indexed documents from knowledge base"""
    try:
        from vectorstore import clear_knowledge_base
        clear_knowledge_base()
        global last_scraped_hash
        last_scraped_hash = None
        
        return JSONResponse({
            "status": "success",
            "message": "Knowledge base cleared successfully."
        })
    except Exception as e:
        print(f"[ERROR] Failed to clear knowledge base: {e}")
        return JSONResponse({
            "status": "error",
            "message": f"Failed to clear knowledge base: {str(e)}"
        }, status_code=500)

# === Initialize Knowledge Base with Static Information ===
def initialize_knowledge_base():
    """Initialize knowledge base with static information about HotelsByDay"""
    static_info = """
    HotelsByDay - Day Use Hotel Bookings

    What is HotelsByDay?
    HotelsByDay is a service that allows you to book hotel rooms for day use, typically for 3-11 hours during the day. This is perfect for travelers who need a place to rest, work, or freshen up between flights or meetings.

    How it works:
    - Book a room for a specific time period (day use)
    - Time periods range from 3 to 11 hours
    - You cannot book "by the hour" - only pre-set time bands
    - Room type is typically "run of the house" (any available standard room)
    - Check-in and check-out times are fixed based on the time band you choose

    Common time bands:
    - Early morning: 8AM - 12PM
    - Afternoon: 12PM - 5PM
    - Evening: 5:30PM - 9PM
    - Late night: 9PM - 11:59PM

    Pricing and payment:
    - Prices vary by hotel and time band
    - You can pay now or pay later at the property
    - Free cancellation available
    - Up to 75% off compared to night stays
    - Loyalty program available (#MasterKey) with up to 5.5% back in coins

    Booking process:
    - Browse available hotels in your area
    - Select your preferred time band
    - Choose your payment method
    - Receive confirmation
    - Check in at the hotel during your time slot

    What's included:
    - Standard hotel room
    - Basic amenities
    - Access to hotel facilities during your stay
    - Professional service

    Limitations:
    - Specific room types cannot be guaranteed
    - Subject to hotel availability
    - Fixed check-in and check-out times
    - Cannot extend your stay beyond the booked time

    Customer service:
    - Available for booking assistance
    - Can help with special requests
    - Support for loyalty program members
    - Human agents available for complex inquiries
    """
    
    try:
        index_text(static_info, metadata={"source": "static_info", "type": "service_overview"})
        print("[INFO] Static knowledge base information initialized")
    except Exception as e:
        print(f"[ERROR] Failed to initialize static knowledge base: {e}")

# Initialize knowledge base on startup
initialize_knowledge_base()
load_escalated_sessions_from_db() # Load escalated sessions from DB on startup

# === WebSocket Endpoints ===

@app.websocket("/ws/session/{session_id}")
async def websocket_session(websocket: WebSocket, session_id: str):
    """WebSocket endpoint for customer sessions"""
    connection_id = f"session_{session_id}_{uuid.uuid4().hex[:8]}"
    
    try:
        await manager.connect(websocket, connection_id)
        manager.session_connections[session_id] = connection_id
        
        print(f"[WS] Customer connected to session {session_id}")
        
        # Send session status
        if session_id in chat_sessions:
            session = chat_sessions[session_id]
            status_message = {
                "type": "session_status",
                "escalated": session.get("escalated", False),
                "agent_id": session.get("agent_id"),
                "message_count": len(session.get("history", []))
            }
            await manager.send_personal_message(json.dumps(status_message), connection_id)
        
        while True:
            data = await websocket.receive_text()
            message_data = json.loads(data)
            
            if message_data["type"] == "user_message":
                # Handle user message through the regular chat endpoint
                user_message = message_data["message"]
                
                # Initialize session if needed
                if session_id not in chat_sessions:
                    chat_sessions[session_id] = {
                        "history": [],
                        "escalated": False,
                        "agent_id": None,
                        "escalated_at": None,
                        "confidence_scores": []
                    }

                session = chat_sessions[session_id]
                
                # Add user message to session history FIRST
                user_msg = {
                    "role": "user", 
                    "content": user_message,
                    "timestamp": datetime.now().isoformat()
                }
                session["history"].append(user_msg)
                
                # Check if already escalated
                if session.get("escalated"):
                    # Route user message to the admin agent
                    agent_id = session.get("agent_id")
                    if agent_id and agent_id in human_agent_sessions:
                        # Send user message to admin agent
                        agent_message = {
                            "type": "user_message",
                            "session_id": session_id,
                            "message": user_message,
                            "timestamp": datetime.now().isoformat()
                        }
                        await manager.broadcast_to_agent(json.dumps(agent_message), agent_id)
                        
                        # Send confirmation to user
                        response_message = {
                            "type": "bot_message",
                            "message": "Your message has been sent to the human agent. They will respond shortly.",
                            "escalated": True,
                            "agent_id": agent_id
                        }
                        await manager.send_personal_message(json.dumps(response_message), connection_id)
                    else:
                        # Fallback if agent not found
                        response_message = {
                            "type": "bot_message",
                            "message": "This conversation has been escalated to a human agent. Please wait for their response.",
                            "escalated": True,
                            "agent_id": session.get("agent_id")
                        }
                        await manager.send_personal_message(json.dumps(response_message), connection_id)
                    continue

                # Get context and generate response
                retrieved_docs = retrieve_context(user_message, k=4)
                context_text = "\n".join([doc.page_content for doc in retrieved_docs])
                
                system_prompt = f"""
                You are a professional customer service assistant for HotelsByDay, helping users book day-use hotel rooms in the United States.

                CRITICAL RULES:
                1. ONLY answer questions using information from the provided context.
                2. If the context is empty, insufficient, or doesn't contain relevant information, say "I don't have enough information to answer that question accurately. Let me connect you to a human agent who can help you."
                3. NEVER invent, speculate, or provide information not explicitly stated in the context.
                4. NEVER refer to external websites, contact information, or suggest users search online.
                5. If you're unsure about any detail, escalate to human agent.
                6. For pricing questions, only provide general information about pricing structure, not specific prices unless clearly stated in context.
                7. For booking requests, always escalate to human agent as you cannot make actual bookings.

                Your behavior:
                - Be concise, factual, and professional.
                - Only provide information that is clearly supported by the context.
                - If context is missing or unclear, escalate immediately.
                - Provide a confidence score (0.0-1.0) based on how well the context supports your answer.
                - For specific pricing, availability, or booking requests, escalate to human agent.

                Context provided:
                {context_text if context_text.strip() else "NO CONTEXT AVAILABLE"}

                IMPORTANT: End your response with a confidence score in this format: [CONFIDENCE: 0.85]
                If you cannot provide a helpful answer due to insufficient context, use confidence score 0.1
                """

                # Generate bot response
                messages = [{"role": "system", "content": system_prompt}] + session["history"]
                response = chat_with_groq(messages)
                bot_reply = response.content.strip()

                # Extract confidence score
                confidence_score = get_confidence_score(bot_reply)
                bot_reply_clean = bot_reply.replace(f"[CONFIDENCE: {confidence_score}]", "").strip()
                
                session["confidence_scores"].append(confidence_score)

                # Check for escalation
                should_escalate = (
                    user_wants_human_agent(user_message) or
                    confidence_score < LOW_CONFIDENCE_THRESHOLD or
                    "I'm not sure" in bot_reply_clean.lower() or
                    "I don't have that information" in bot_reply_clean.lower() or
                    "I don't have enough information" in bot_reply_clean.lower() or
                    "connect you to a human agent" in bot_reply_clean.lower() or
                    "book" in user_message.lower() or
                    "reservation" in user_message.lower() or
                    "booking" in user_message.lower() or
                    "price" in user_message.lower() or
                    "cost" in user_message.lower() or
                    "rate" in user_message.lower() or
                    "availability" in user_message.lower() or
                    "room" in user_message.lower() and ("available" in user_message.lower() or "book" in user_message.lower()) or
                    "how much" in user_message.lower() or
                    "what's the price" in user_message.lower() or
                    "what is the cost" in user_message.lower() or
                    "check availability" in user_message.lower() or
                    "make a reservation" in user_message.lower() or
                    not context_text.strip() or
                    len(context_text.strip()) < 100 or
                    context_text.strip() == "NO CONTEXT AVAILABLE"
                )

                if should_escalate and not session["escalated"]:
                    agent_id = escalate_to_human(session_id, session)
                    bot_reply_clean = f"Thank you for your question. I'm connecting you to a human agent who will assist you further. Your session ID is: {session_id}"
                    
                    response_message = {
                        "type": "bot_message",
                        "message": bot_reply_clean,
                        "escalated": True,
                        "agent_id": agent_id,
                        "confidence_score": confidence_score
                    }
                    await manager.send_personal_message(json.dumps(response_message), connection_id)
                    
                    # Notify agent about new escalated session
                    await manager.broadcast_to_agent(json.dumps({
                        "type": "new_escalated_session",
                        "session_id": session_id,
                        "agent_id": agent_id,
                        "message": f"New escalated session: {session_id}"
                    }), agent_id)
                else:
                    # Add bot response to session history
                    bot_msg = {
                        "role": "assistant", 
                        "content": bot_reply_clean,
                        "confidence": confidence_score,
                        "timestamp": datetime.now().isoformat()
                    }
                    session["history"].append(bot_msg)

                    response_message = {
                        "type": "bot_message",
                        "message": bot_reply_clean,
                        "escalated": False,
                        "confidence_score": confidence_score
                    }
                    await manager.send_personal_message(json.dumps(response_message), connection_id)
    
    except WebSocketDisconnect:
        manager.disconnect(connection_id)
        print(f"[WS] Customer disconnected from session {session_id}")

@app.websocket("/ws/agent/{agent_id}")
async def websocket_agent(websocket: WebSocket, agent_id: str):
    """WebSocket endpoint for admin agents"""
    connection_id = f"agent_{agent_id}_{uuid.uuid4().hex[:8]}"
    
    try:
        await manager.connect(websocket, connection_id)
        manager.agent_connections[agent_id] = connection_id
        
        print(f"[WS] Agent {agent_id} connected")
        
        # Send current escalated sessions
        escalated_sessions = []
        for aid, session_data in human_agent_sessions.items():
            if session_data["status"] == "waiting":
                escalated_sessions.append({
                    "agent_id": aid,
                    "session_id": session_data["session_id"],
                    "escalated_at": session_data["escalated_at"],
                    "message_count": len(session_data["history"])
                })
        
        status_message = {
            "type": "agent_status",
            "agent_id": agent_id,
            "escalated_sessions": escalated_sessions
        }
        await manager.send_personal_message(json.dumps(status_message), connection_id)
        
        while True:
            data = await websocket.receive_text()
            message_data = json.loads(data)
            
            if message_data["type"] == "agent_message":
                session_id = message_data["session_id"]
                agent_message = message_data["message"]
                
                # Verify agent has access to this session
                if agent_id not in human_agent_sessions:
                    error_message = {
                        "type": "error",
                        "message": "Invalid agent ID or session not found"
                    }
                    await manager.send_personal_message(json.dumps(error_message), connection_id)
                    continue
                
                agent_session = human_agent_sessions[agent_id]
                if agent_session["session_id"] != session_id:
                    error_message = {
                        "type": "error",
                        "message": "Agent not authorized for this session"
                    }
                    await manager.send_personal_message(json.dumps(error_message), connection_id)
                    continue
                
                # Add agent message to chat history
                if session_id in chat_sessions:
                    chat_sessions[session_id]["history"].append({
                        "role": "agent",
                        "content": agent_message,
                        "timestamp": datetime.now().isoformat(),
                        "agent_id": agent_id
                    })
                    
                    # Update agent session status
                    agent_session["status"] = "active"
                    agent_session["history"].append({
                        "role": "agent",
                        "content": agent_message,
                        "timestamp": datetime.now().isoformat()
                    })
                
                # Send message to customer
                customer_message = {
                    "type": "agent_message",
                    "message": agent_message,
                    "agent_id": agent_id,
                    "timestamp": datetime.now().isoformat()
                }
                await manager.broadcast_to_session(json.dumps(customer_message), session_id)
                
                # Confirm message sent to agent
                confirm_message = {
                    "type": "message_sent",
                    "session_id": session_id,
                    "message": "Message sent successfully"
                }
                await manager.send_personal_message(json.dumps(confirm_message), connection_id)
            
            elif message_data["type"] == "user_message":
                # Handle incoming user message (routed from customer WebSocket)
                session_id = message_data["session_id"]
                user_message = message_data["message"]
                
                # Verify agent has access to this session
                if agent_id not in human_agent_sessions:
                    error_message = {
                        "type": "error",
                        "message": "Invalid agent ID or session not found"
                    }
                    await manager.send_personal_message(json.dumps(error_message), connection_id)
                    continue
                
                agent_session = human_agent_sessions[agent_id]
                if agent_session["session_id"] != session_id:
                    error_message = {
                        "type": "error",
                        "message": "Agent not authorized for this session"
                    }
                    await manager.send_personal_message(json.dumps(error_message), connection_id)
                    continue
                
                # Send user message to admin
                user_message_for_admin = {
                    "type": "user_message",
                    "session_id": session_id,
                    "message": user_message,
                    "timestamp": message_data.get("timestamp", datetime.now().isoformat())
                }
                await manager.send_personal_message(json.dumps(user_message_for_admin), connection_id)
    
    except WebSocketDisconnect:
        manager.disconnect(connection_id)
        print(f"[WS] Agent {agent_id} disconnected")
