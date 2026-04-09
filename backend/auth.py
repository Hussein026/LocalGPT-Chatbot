import jwt
import bcrypt
import uuid
import sqlite3
from datetime import datetime, timedelta

SECRET_KEY = "localgpt-alzheimer-secret-2024"
TOKEN_EXPIRY_DAYS = 7

def get_db_path():
    return "/root/LocalGPT-Chatbot/backend/chat_data.db"

def hash_password(password):
    return bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()

def verify_password(password, hashed):
    try:
        return bcrypt.checkpw(password.encode(), hashed.encode())
    except:
        return False

def create_token(user_id, email):
    payload = {"user_id": user_id, "email": email, "exp": datetime.utcnow() + timedelta(days=TOKEN_EXPIRY_DAYS)}
    return jwt.encode(payload, SECRET_KEY, algorithm="HS256")

def verify_token(token):
    try:
        return jwt.decode(token, SECRET_KEY, algorithms=["HS256"])
    except:
        return None

def register_user(email, password):
    if not email or not password:
        return {"error": "Email and password required"}
    if len(password) < 6:
        return {"error": "Password must be at least 6 characters"}
    if "@" not in email:
        return {"error": "Invalid email address"}
    try:
        conn = sqlite3.connect(get_db_path())
        existing = conn.execute("SELECT id FROM users WHERE email=? OR username=?", (email, email)).fetchone()
        if existing:
            conn.close()
            return {"error": "Email already registered"}
        user_id = str(uuid.uuid4())
        password_hash = hash_password(password)
        now = datetime.now().isoformat()
        conn.execute("INSERT INTO users (id, username, password_hash, email, created_at) VALUES (?,?,?,?,?)", (user_id, email, password_hash, email, now))
        conn.commit()
        conn.close()
        return {"token": create_token(user_id, email), "user_id": user_id, "email": email}
    except Exception as e:
        return {"error": str(e)}

def login_user(email, password):
    if not email or not password:
        return {"error": "Email and password required"}
    try:
        conn = sqlite3.connect(get_db_path())
        conn.row_factory = sqlite3.Row
        row = conn.execute("SELECT id, password_hash, email FROM users WHERE email=? OR username=?", (email, email)).fetchone()
        conn.close()
        if not row:
            return {"error": "Invalid email or password"}
        if not verify_password(password, row["password_hash"]):
            return {"error": "Invalid email or password"}
        conn = sqlite3.connect(get_db_path())
        conn.execute("UPDATE users SET last_login=? WHERE id=?", (datetime.now().isoformat(), row["id"]))
        conn.commit()
        conn.close()
        return {"token": create_token(row["id"], row["email"]), "user_id": row["id"], "email": row["email"]}
    except Exception as e:
        return {"error": str(e)}

def get_user_from_request(headers):
    auth = headers.get("Authorization", "")
    if not auth.startswith("Bearer "):
        return None
    payload = verify_token(auth[7:])
    if not payload:
        return None
    return payload.get("user_id")
