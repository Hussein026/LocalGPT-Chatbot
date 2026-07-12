import sqlite3
import smtplib
import threading
import time
from datetime import datetime
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

GMAIL_USER = "moukhaderh7@gmail.com"
GMAIL_APP_PASSWORD = "zvxfdpekrkfdwfen"
DB_PATH = "/root/LocalGPT-Chatbot/backend/chat_data.db"

def send_reminder(patient_name, patient_email, medication_name, dosage, notes):
    try:
        msg = MIMEMultipart()
        msg["From"] = GMAIL_USER
        msg["To"] = patient_email
        msg["Subject"] = "Medication Reminder: " + medication_name
        body = "Dear " + patient_name + ",\n\nThis is a reminder to take your medication:\n\nMedication: " + medication_name + "\nDosage: " + dosage + "\n\nPlease take your medication as prescribed.\n\nAlzheimer Clinical Assistant"
        msg.attach(MIMEText(body, "plain"))
        server = smtplib.SMTP("smtp.gmail.com", 587)
        server.starttls()
        server.login(GMAIL_USER, GMAIL_APP_PASSWORD)
        server.sendmail(GMAIL_USER, patient_email, msg.as_string())
        server.quit()
        print("[REMINDER] Sent to " + patient_email)
        return True
    except Exception as e:
        print("[REMINDER ERROR] " + str(e))
        return False

def check_and_send_reminders():
    while True:
        try:
            current_time = datetime.now().strftime("%H:%M")
            conn = sqlite3.connect(DB_PATH)
            conn.row_factory = sqlite3.Row
            rows = conn.execute("SELECT * FROM medications WHERE active=1 AND reminder_time=?", (current_time,)).fetchall()
            conn.close()
            for row in rows:
                send_reminder(row["patient_name"], row["patient_email"], row["medication_name"], row["dosage"], row["notes"])
        except Exception as e:
            print("[REMINDER SCHEDULER ERROR] " + str(e))
        time.sleep(60)

def start_reminder_scheduler():
    t = threading.Thread(target=check_and_send_reminders, daemon=True)
    t.start()
    print("[REMINDER] Scheduler started")
