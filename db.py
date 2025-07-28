import sqlite3
from datetime import datetime

DB_PATH = "conversations.db"

def init_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS conversations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            summary TEXT,
            email TEXT,
            phone TEXT,
            agent_id TEXT,
            session_id TEXT,
            escalated INTEGER DEFAULT 0,
            escalated_at TEXT
        )
    """)
    conn.commit()
    conn.close()

def save_to_db(summary: str, email: str = None, phone: str = None, agent_id: str = None, session_id: str = None, escalated: bool = False):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Add new columns if they don't exist
    try:
        cursor.execute("ALTER TABLE conversations ADD COLUMN session_id TEXT")
    except sqlite3.OperationalError:
        pass  # Column already exists
    
    try:
        cursor.execute("ALTER TABLE conversations ADD COLUMN escalated INTEGER DEFAULT 0")
    except sqlite3.OperationalError:
        pass  # Column already exists
    
    try:
        cursor.execute("ALTER TABLE conversations ADD COLUMN escalated_at TEXT")
    except sqlite3.OperationalError:
        pass  # Column already exists
    
    escalated_at = datetime.now().isoformat() if escalated else None
    
    cursor.execute(
        "INSERT INTO conversations (timestamp, summary, email, phone, agent_id, session_id, escalated, escalated_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (datetime.now().isoformat(), summary, email, phone, agent_id, session_id, 1 if escalated else 0, escalated_at)
    )
    conn.commit()
    conn.close()
