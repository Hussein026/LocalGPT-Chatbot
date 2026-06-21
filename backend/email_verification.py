import smtplib
import random
import string
import sqlite3
from datetime import datetime, timedelta
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

GMAIL_USER = "moukhaderh7@gmail.com"
GMAIL_APP_PASSWORD = "zvxfdpekrkfdwfen"
DB_PATH = "/root/LocalGPT-Chatbot/backend/chat_data.db"

def generate_code():
    return ''.join(random.choices(string.digits, k=6))

def send_verification_email(to_email, code, full_name):
    try:
        msg = MIMEMultipart()
        msg['From'] = GMAIL_USER
        msg['To'] = to_email
        msg['Subject'] = "Your LocalGPT Verification Code"
        body = f"""Dear Dr. {full_name},

Your verification code for the Alzheimer's Clinical Assistant is:

    {code}

This code expires in 15 minutes.

If you did not request this, please ignore this email.

Alzheimer's Clinical Assistant Team"""
        msg.attach(MIMEText(body, 'plain'))
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()
        server.login(GMAIL_USER, GMAIL_APP_PASSWORD)
        server.sendmail(GMAIL_USER, to_email, msg.as_string())
        server.quit()
        return True
    except Exception as e:
        print(f"Email error: {e}")
        return False

def save_verification_code(email, code):
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.execute("""CREATE TABLE IF NOT EXISTS verification_codes
            (email TEXT, code TEXT, expires_at TEXT, used INTEGER DEFAULT 0)""")
        conn.execute("DELETE FROM verification_codes WHERE email=?", (email,))
        expires = (datetime.now() + timedelta(minutes=15)).isoformat()
        conn.execute("INSERT INTO verification_codes (email, code, expires_at) VALUES (?,?,?)",
            (email, code, expires))
        conn.commit()
        conn.close()
        return True
    except Exception as e:
        print(f"DB error: {e}")
        return False

def verify_code(email, code):
    try:
        conn = sqlite3.connect(DB_PATH)
        row = conn.execute(
            "SELECT code, expires_at FROM verification_codes WHERE email=? AND used=0 ORDER BY rowid DESC LIMIT 1",
            (email,)).fetchone()
        conn.close()
        if not row:
            return False, "No verification code found"
        if datetime.now() > datetime.fromisoformat(row[1]):
            return False, "Code expired"
        if row[0] != code:
            return False, "Wrong code"
        conn = sqlite3.connect(DB_PATH)
        conn.execute("UPDATE verification_codes SET used=1 WHERE email=?", (email,))
        conn.execute("UPDATE users SET is_verified=1 WHERE email=?", (email,))
        conn.commit()
        conn.close()
        return True, "Verified!"
    except Exception as e:
        return False, str(e)
