# LocalGPT — AI Clinical Assistant for Alzheimer's Care

> Built and deployed during a 360-hour internship at CRIBO Training Group, Bucharest (Jun–Aug 2026)
> 
> 🌐 **Live at: [alzheimer-gpt.com](https://alzheimer-gpt.com)**

---

## What is this?

LocalGPT is a **privacy-first AI clinical decision assistant** for medical doctors treating Alzheimer's disease. It runs on a private VPS server — no patient data ever leaves the server.

Doctors can ask clinical questions, search the latest research, track patients, and set medication reminders — all in one platform.

---

## Features

### 🧠 AI Knowledge Base (RAG System)
- Processes **25 peer-reviewed Alzheimer's research papers**
- **3,214 text chunks** indexed with custom keyword + synonym search
- Medical synonym expansion (e.g. "alzheimer" → "dementia", "ad", "neurodegenerative")
- Every AI answer includes a **source citation** from the knowledge base
- Never hallucinates — if unsure, says so

### 🔐 Doctor Authentication
- Doctor-only registration (full name, specialty, medical license number)
- **Email verification** — 6-digit code expires in 15 minutes
- JWT token authentication (7-day expiry, bcrypt password hashing)
- Unverified accounts cannot access the chat

### 💬 Clinical Chat Interface
- 3 pre-built clinical tool tabs: **Case Analysis**, **Literature Search**, **Treatment Options**
- Conversation memory (last 4 messages kept in context)
- **Voice input** via Web Speech API — doctors can speak queries
- Streaming responses — tokens appear in real time
- Mobile-first responsive design

### 👥 Patient Tracker
- Add and manage patients with diagnosis date and demographics
- Track cognitive scores per visit: **MMSE (0–30)** and **MoCA (0–30)**
- Full visit history with symptoms and clinical notes

### 💊 Medication Reminder System
- Set daily medication reminders per patient
- Background scheduler checks every 60 seconds
- Sends email reminders automatically via SMTP

### 📊 Admin Dashboard
- Protected dashboard at `/admin.html`
- Doctors can flag incorrect AI answers
- Admin reviews flagged answers and corrections
- Feedback loop improves system over time

### 🩺 Additional Tools
- **Symptom Checker** — structured symptom entry for AI-assisted staging
- **Caregiver Assistant** — translates clinical info into plain language for families

---

## Tech Stack

| Layer | Technology |
|---|---|
| Backend | Python (BaseHTTPRequestHandler), SQLite |
| AI Inference | Ollama (Qwen 2.5 3B), local VPS |
| RAG System | PyPDF2, custom keyword+synonym search, pickle serialization |
| Authentication | JWT (PyJWT), bcrypt, SMTP email verification |
| Frontend | Vanilla JavaScript, HTML/CSS, Web Speech API |
| Infrastructure | Ubuntu 24.04 VPS, Nginx reverse proxy, SSL (Let's Encrypt), systemd |
| Domain | alzheimer-gpt.com (Porkbun, $11.08/year) |

---

## System Architecture

```
Doctor → alzheimer-gpt.com (HTTPS)
       → Nginx reverse proxy
       → Python HTTP server (port 8000)
       → SQLite database (users, sessions, patients, feedback)
       → Ollama AI inference (local, port 11434)
       → RAG knowledge base (25 PDFs, 3,214 chunks)
```

---

## Development Timeline (7 Sprints)

| Sprint | Weeks | What was built |
|---|---|---|
| 1 | 1–2 | VPS setup, Ollama install, Cloudflare tunnel, HTTP server |
| 2 | 3–4 | RAG system, PDF ingestion, chat interface |
| 3 | 5–6 | JWT auth, email verification, doctor-only access |
| 4 | 7–8 | Clinical tabs, patient tracker, conversation memory |
| 5 | 9–10 | Email reminders, appointment manager, medication system |
| 6 | 11–12 | Production deployment, Nginx, SSL, performance optimization |
| 7 | 13–14 | Voice input, symptom checker, caregiver assistant, admin dashboard |

---

## Clinical Validation

The system was evaluated by a **clinical psychologist consultant** who raised 6 expert questions that drove measurable improvements:

- Added explicit source citations in every AI response
- Added "Research prototype" disclaimer banner
- Implemented voice input feature
- Built feedback system and admin dashboard
- Fixed inference timeout issues
- Optimized response time from **40 seconds → under 20 seconds**

---

## Infrastructure Cost

| Component | Cost |
|---|---|
| Contabo VPS Server | ~€6/month |
| Domain (alzheimer-gpt.com) | $11.08/year |
| SSL Certificate | Free (Let's Encrypt) |
| AI Model (Qwen 2.5 3B) | Free (Ollama open source) |
| All other software | Free / Open Source |

---

## Author

**Hussein Mokhadder**  
Computer Engineering Student — IoT & AI  
Politehnica University of Bucharest  
[github.com/Hussein026](https://github.com/Hussein026) | [linkedin.com/in/hussein-moukhader-468b39274](https://linkedin.com/in/hussein-moukhader-468b39274)

---

*Built during a 360-hour internship at CRIBO Training Group, Bucharest, 2026.*
