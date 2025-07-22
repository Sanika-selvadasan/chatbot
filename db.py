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
            phone TEXT
        )
    """)
    conn.commit()
    conn.close()

def save_to_db(summary, email, phone):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO conversations (timestamp, summary, email, phone) VALUES (?, ?, ?, ?)",
        (datetime.now().isoformat(), summary, email, phone)
    )
    conn.commit()
    conn.close()
