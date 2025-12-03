import sqlite3
import uuid
import json
from datetime import datetime
from typing import List, Dict, Optional, Tuple
import os

class ChatDatabase:
    def __init__(self, db_path: str = None):
        if db_path is None:
            if os.path.exists("/app"):  
                self.db_path = "/app/backend/chat_data.db"
            else:
                self.db_path = "./chat_data.db"
        else:
            self.db_path = db_path

        self.init_database()

    def init_database(self):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        conn.execute("PRAGMA foreign_keys = ON")

        conn.execute('''
            CREATE TABLE IF NOT EXISTS sessions (
                id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                model_used TEXT NOT NULL,
                message_count INTEGER DEFAULT 0
            )
        ''')

        conn.execute('''
            CREATE TABLE IF NOT EXISTS messages (
                id TEXT PRIMARY KEY,
                session_id TEXT NOT NULL,
                content TEXT NOT NULL,
                sender TEXT NOT NULL CHECK (sender IN ('user','assistant')),
                timestamp TEXT NOT NULL,
                metadata TEXT DEFAULT '{}',
                FOREIGN KEY (session_id) REFERENCES sessions(id) ON DELETE CASCADE
            )
        ''')

        conn.execute('CREATE INDEX IF NOT EXISTS idx_messages_session_id ON messages(session_id)')
        conn.execute('CREATE INDEX IF NOT EXISTS idx_messages_timestamp ON messages(timestamp)')
        conn.execute('CREATE INDEX IF NOT EXISTS idx_sessions_updated_at ON sessions(updated_at)')

        conn.execute('''
            CREATE TABLE IF NOT EXISTS session_documents (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                file_path TEXT NOT NULL,
                indexed INTEGER DEFAULT 0,
                FOREIGN KEY (session_id) REFERENCES sessions(id) ON DELETE CASCADE
            )
        ''')

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS indexes (
                id TEXT PRIMARY KEY,
                name TEXT UNIQUE,
                description TEXT,
                created_at TEXT,
                updated_at TEXT,
                vector_table_name TEXT,
                metadata TEXT
            )
        ''')

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS index_documents (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                index_id TEXT,
                original_filename TEXT,
                stored_path TEXT,
                FOREIGN KEY(index_id) REFERENCES indexes(id)
            )
        ''')

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS session_indexes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT,
                index_id TEXT,
                linked_at TEXT,
                FOREIGN KEY(session_id) REFERENCES sessions(id),
                FOREIGN KEY(index_id) REFERENCES indexes(id)
            )
        ''')

        conn.commit()
        conn.close()
        print("Database initialized successfully")

    def create_session(self, title: str, model: str) -> str:
        session_id = str(uuid.uuid4())
        now = datetime.now().isoformat()

        conn = sqlite3.connect(self.db_path)
        conn.execute('''
            INSERT INTO sessions (id, title, created_at, updated_at, model_used)
            VALUES (?, ?, ?, ?, ?)
        ''', (session_id, title, now, now, model))
        conn.commit()
        conn.close()

        return session_id

    def get_sessions(self, limit: int = 50) -> List[Dict]:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.execute('''
            SELECT id, title, created_at, updated_at, model_used, message_count
            FROM sessions
            ORDER BY updated_at DESC
            LIMIT ?
        ''', (limit,))
        sessions = [dict(row) for row in cursor.fetchall()]
        conn.close()
        return sessions

    def get_session(self, session_id: str) -> Optional[Dict]:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.execute('''
            SELECT id, title, created_at, updated_at, model_used, message_count
            FROM sessions
            WHERE id = ?
        ''', (session_id,))
        row = cursor.fetchone()
        conn.close()
        return dict(row) if row else None

    def add_message(self, session_id: str, content: str, sender: str, metadata: Dict = None) -> str:
        message_id = str(uuid.uuid4())
        now = datetime.now().isoformat()
        metadata_json = json.dumps(metadata or {})

        conn = sqlite3.connect(self.db_path)
        conn.execute('''
            INSERT INTO messages (id, session_id, content, sender, timestamp, metadata)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (message_id, session_id, content, sender, now, metadata_json))

        conn.execute('''
            UPDATE sessions
            SET updated_at = ?, 
                message_count = message_count + 1
            WHERE id = ?
        ''', (now, session_id))

        conn.commit()
        conn.close()

        return message_id

    def get_messages(self, session_id: str, limit: int = 100) -> List[Dict]:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.execute('''
            SELECT id, content, sender, timestamp, metadata
            FROM messages
            WHERE session_id = ?
            ORDER BY timestamp ASC
            LIMIT ?
        ''', (session_id, limit))

        messages = []
        for row in cursor.fetchall():
            message = dict(row)
            message["metadata"] = json.loads(message["metadata"])
            messages.append(message)

        conn.close()
        return messages

    def get_conversation_history(self, session_id: str) -> List[Dict]:
        messages = self.get_messages(session_id)
        return [{"role": msg["sender"], "content": msg["content"]} for msg in messages]

    # -------------------------------
    # ✅ THE MISSING FUNCTION (FIX)
    # -------------------------------
    def cleanup_empty_sessions(self) -> int:
        conn = sqlite3.connect(self.db_path)
        cursor = conn.execute('''
            SELECT s.id FROM sessions s
            LEFT JOIN messages m ON s.id = m.session_id
            WHERE m.id IS NULL
        ''')

        empty_sessions = [row[0] for row in cursor.fetchall()]
        deleted = 0

        for sid in empty_sessions:
            conn.execute('DELETE FROM sessions WHERE id = ?', (sid,))
            deleted += 1

        conn.commit()
        conn.close()
        return deleted

    # -------------------------------
    # REQUIRED BY backend/server.py
    # -------------------------------
    def get_stats(self) -> Dict:
        conn = sqlite3.connect(self.db_path)

        session_count = conn.execute('SELECT COUNT(*) FROM sessions').fetchone()[0]
        message_count = conn.execute('SELECT COUNT(*) FROM messages').fetchone()[0]

        row = conn.execute('''
            SELECT model_used, COUNT(*) 
            FROM sessions 
            GROUP BY model_used 
            ORDER BY COUNT(*) DESC
            LIMIT 1
        ''').fetchone()

        conn.close()

        return {
            "total_sessions": session_count,
            "total_messages": message_count,
            "most_used_model": row[0] if row else None
        }


def generate_session_title(first_message: str, max_length: int = 50) -> str:
    title = first_message.strip()
    prefixes = ["hey", "hi", "hello", "can you", "please", "i want", "i need"]

    tl = title.lower()
    for p in prefixes:
        if tl.startswith(p):
            title = title[len(p):].strip()
            break

    if title:
        title = title[0].upper() + title[1:]

    if len(title) > max_length:
        title = title[:max_length] + "..."

    if len(title) < 3:
        title = "New Chat"

    return title


db = ChatDatabase()

if __name__ == "__main__":
    print("🧪 Testing DB...")

    session_id = db.create_session("Test Chat", "qwen3:8b")
    db.add_message(session_id, "Hello!", "user")
    db.add_message(session_id, "Hi!", "assistant")

    print(db.get_messages(session_id))
    print(db.get_stats())

    print("✅ Done")
