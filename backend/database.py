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
                message_count INTEGER DEFAULT 0,
                user_id TEXT
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

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS user_preferences (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                key TEXT UNIQUE NOT NULL,
                value TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
        ''')

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id TEXT PRIMARY KEY,
                username TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                email TEXT,
                created_at TEXT NOT NULL,
                last_login TEXT
            )
        ''')

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS api_keys (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                api_key TEXT UNIQUE NOT NULL,
                created_at TEXT NOT NULL,
                expires_at TEXT,
                FOREIGN KEY(user_id) REFERENCES users(id)
            )
        ''')

        conn.commit()
        conn.close()
        print("✅ Database initialized successfully")

    def create_session(self, title: str, model: str, user_id: str = None) -> str:
        session_id = str(uuid.uuid4())
        now = datetime.now().isoformat()

        conn = sqlite3.connect(self.db_path)
        conn.execute('''
            INSERT INTO sessions (id, title, created_at, updated_at, model_used, user_id)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (session_id, title, now, now, model, user_id))
        conn.commit()
        conn.close()

        return session_id

    def get_sessions(self, limit: int = 50, user_id: str = None) -> List[Dict]:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        
        if user_id:
            cursor = conn.execute('''
                SELECT id, title, created_at, updated_at, model_used, message_count, user_id
                FROM sessions
                WHERE user_id = ?
                ORDER BY updated_at DESC
                LIMIT ?
            ''', (user_id, limit))
        else:
            cursor = conn.execute('''
                SELECT id, title, created_at, updated_at, model_used, message_count, user_id
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

    def delete_session(self, session_id: str) -> bool:
        """Delete a session and all its messages"""
        try:
            conn = sqlite3.connect(self.db_path)
            conn.execute('DELETE FROM messages WHERE session_id = ?', (session_id,))
            conn.execute('DELETE FROM sessions WHERE id = ?', (session_id,))
            conn.commit()
            conn.close()
            return True
        except Exception as e:
            print(f"[ERROR] Could not delete session: {e}")
            return False

    # ✅ FIXED: Match server.py call signature
    def add_message(self, session_id: str, sender: str, content: str, metadata: Dict = None) -> str:
        """
        Add message to session
        Args:
            session_id: Session ID
            sender: 'user' or 'assistant'
            content: Message text
            metadata: Optional metadata dict
        """
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
        """Get all messages for a session in chronological order"""
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
            try:
                message["metadata"] = json.loads(message["metadata"])
            except:
                message["metadata"] = {}
            
            # Return in format server.py expects
            messages.append({
                "role": message["sender"],
                "content": message["content"],
                "timestamp": message["timestamp"],
                "id": message["id"]
            })

        conn.close()
        return messages

    def get_conversation_history(self, session_id: str) -> List[Dict]:
        """Get conversation in OpenAI format"""
        messages = self.get_messages(session_id)
        return [{"role": msg["role"], "content": msg["content"]} for msg in messages]

    def get_documents_for_session(self, session_id: str) -> List[str]:
        """Get document paths for a session"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.execute('''
            SELECT file_path FROM session_documents WHERE session_id = ?
        ''', (session_id,))
        paths = [row[0] for row in cursor.fetchall()]
        conn.close()
        return paths


    def update_session_title(self, session_id: str, title: str) -> bool:
        """Update session title"""
        try:
            conn = sqlite3.connect(self.db_path)
            conn.execute("UPDATE sessions SET title=? WHERE id=?", (title, session_id))
            conn.commit()
            conn.close()
            return True
        except Exception as e:
            print(f"[ERROR] update_session_title: {e}")
            return False
    def cleanup_empty_sessions(self) -> int:
        """Remove sessions with no messages"""
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

    def get_stats(self) -> Dict:
        """Get database statistics"""
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

    # ==============================
    # USER PREFERENCES
    # ==============================
    def set_preference(self, key: str, value: str) -> bool:
        """Store user preference"""
        try:
            now = datetime.now().isoformat()
            conn = sqlite3.connect(self.db_path)
            conn.execute('''
                INSERT INTO user_preferences (key, value, updated_at)
                VALUES (?, ?, ?)
                ON CONFLICT(key) DO UPDATE SET value=?, updated_at=?
            ''', (key, value, now, value, now))
            conn.commit()
            conn.close()
            return True
        except Exception as e:
            print(f"[ERROR] set_preference: {e}")
            return False

    def get_preference(self, key: str, default=None):
        """Get user preference"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.execute('SELECT value FROM user_preferences WHERE key = ?', (key,))
            row = cursor.fetchone()
            conn.close()
            return row[0] if row else default
        except Exception as e:
            print(f"[ERROR] get_preference: {e}")
            return default

    def get_all_preferences(self) -> Dict:
        """Get all user preferences"""
        try:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.execute('SELECT key, value FROM user_preferences')
            prefs = {row['key']: row['value'] for row in cursor.fetchall()}
            conn.close()
            return prefs
        except Exception as e:
            print(f"[ERROR] get_all_preferences: {e}")
            return {}

    # ==============================
    # USER AUTHENTICATION
    # ==============================
    def create_user(self, username: str, password: str, email: str = None) -> str:
        """Create new user with hashed password"""
        import hashlib
        user_id = str(uuid.uuid4())
        password_hash = hashlib.sha256(password.encode()).hexdigest()
        now = datetime.now().isoformat()
        
        try:
            conn = sqlite3.connect(self.db_path)
            conn.execute('''
                INSERT INTO users (id, username, password_hash, email, created_at)
                VALUES (?, ?, ?, ?, ?)
            ''', (user_id, username, password_hash, email, now))
            conn.commit()
            conn.close()
            return user_id
        except Exception as e:
            print(f"[ERROR] create_user: {e}")
            return None

    def verify_user(self, username: str, password: str) -> Optional[str]:
        """Verify user credentials, return user_id if valid"""
        import hashlib
        password_hash = hashlib.sha256(password.encode()).hexdigest()
        
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.execute('''
                SELECT id FROM users 
                WHERE username = ? AND password_hash = ?
            ''', (username, password_hash))
            row = cursor.fetchone()
            conn.close()
            return row[0] if row else None
        except Exception as e:
            print(f"[ERROR] verify_user: {e}")
            return None

    def create_api_key(self, user_id: str) -> str:
        """Generate API key for user"""
        import secrets
        api_key = f"lgpt_{secrets.token_urlsafe(32)}"
        now = datetime.now().isoformat()
        
        try:
            conn = sqlite3.connect(self.db_path)
            conn.execute('''
                INSERT INTO api_keys (user_id, api_key, created_at)
                VALUES (?, ?, ?)
            ''', (user_id, api_key, now))
            conn.commit()
            conn.close()
            return api_key
        except Exception as e:
            print(f"[ERROR] create_api_key: {e}")
            return None

    def verify_api_key(self, api_key: str) -> Optional[str]:
        """Verify API key, return user_id if valid"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.execute('''
                SELECT user_id FROM api_keys WHERE api_key = ?
            ''', (api_key,))
            row = cursor.fetchone()
            conn.close()
            return row[0] if row else None
        except Exception as e:
            print(f"[ERROR] verify_api_key: {e}")
            return None


def generate_session_title(first_message: str = "", max_length: int = 50) -> str:
    """Generate a session title from first message"""
    if not first_message:
        return f"Chat {datetime.now().strftime('%H:%M')}"
    
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


# Create global instance
db = ChatDatabase()

if __name__ == "__main__":
    print("🧪 Testing DB...")

    session_id = db.create_session("Test Chat", "qwen2.5:7b-instruct-q4_K_M")
    db.add_message(session_id, "user", "Hello!")
    db.add_message(session_id, "assistant", "Hi!")

    print(db.get_messages(session_id))
    print(db.get_stats())

    print("✅ Done")
