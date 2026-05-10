# database.py

import sqlite3
import os
from datetime import datetime

DB_PATH = "./chatbot_memory.db"

def get_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    """
    Creates the database tables if they don't exist yet.
    Safe to call every time the app starts.
    """
    conn = get_connection()
    cursor = conn.cursor()

    # Table for conversation sessions
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS sessions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            created_at TEXT NOT NULL
        )
    """)

    # Table for individual messages
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            role TEXT NOT NULL,
            message TEXT NOT NULL,
            agent TEXT,
            status TEXT,
            timestamp TEXT NOT NULL
        )
    """)

    conn.commit()
    conn.close()
    print("✅ Database ready!")

def save_message(session_id: str, role: str, message: str,
                 agent: str = None, status: str = None):
    """
    Saves a single message to the database.
    role = 'user' or 'assistant'
    """
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO messages (session_id, role, message, agent, status, timestamp)
        VALUES (?, ?, ?, ?, ?, ?)
    """, (session_id, role, message, agent, status,
          datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
    conn.commit()
    conn.close()

def get_session_messages(session_id: str) -> list:
    """
    Returns all messages for a given session, oldest first.
    """
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT role, message, agent, status, timestamp
        FROM messages
        WHERE session_id = ?
        ORDER BY id ASC
    """, (session_id,))
    rows = cursor.fetchall()
    conn.close()
    return [dict(row) for row in rows]

def get_all_sessions() -> list:
    """
    Returns a summary of all sessions — useful for admin view.
    """
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT session_id,
               COUNT(*) as message_count,
               MIN(timestamp) as started_at,
               MAX(timestamp) as last_active
        FROM messages
        GROUP BY session_id
        ORDER BY last_active DESC
    """)
    rows = cursor.fetchall()
    conn.close()
    return [dict(row) for row in rows]

def clear_session(session_id: str):
    """
    Deletes all messages for a given session.
    """
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM messages WHERE session_id = ?", (session_id,))
    conn.commit()
    conn.close()