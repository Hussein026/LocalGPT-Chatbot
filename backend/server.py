#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import pathlib
import json
import http.server
import socketserver
import cgi
import os
import uuid
import sys
import traceback
from urllib.parse import urlparse
from concurrent.futures import ThreadPoolExecutor
import threading

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from database import db, generate_session_title
import simple_pdf_processor as pdf_module
from ollama_client import OllamaClient

# ------------------------------
# MODEL CONFIGURATION
# ------------------------------
DEFAULT_MODEL = "qwen2.5:0.5b-instruct-q4_K_M"  # Fast model
SMART_MODEL = "qwen2.5:3b-instruct-q4_K_M"  # Smart model for complex queries
VISION_MODEL = "llava:7b"  # Vision model for images
SYSTEM_PROMPT = "You are a concise, accurate assistant."

# Auto-naming configuration
AUTO_NAME_CHATS = True  # Enable automatic chat naming
RENAME_AFTER_FIRST_MESSAGE = True  # Rename chat after first user message

# Keywords that trigger smart model
COMPLEX_KEYWORDS = ["explain", "analyze", "compare", "detailed", "why", "how does", "complex", "technical"]

# Memory configuration
MAX_CONTEXT_MESSAGES = 20  # Keep last 20 messages in context
MAX_TOKENS_PER_MESSAGE = 500  # Trim long messages

# Tool configuration
ENABLE_TOOLS = True

# ------------------------------
# ONE GLOBAL OLLAMA CLIENT
# ------------------------------
ollama = OllamaClient(
    api_url="http://localhost:11434",
    default_model=DEFAULT_MODEL
)

# ==============================
# TCP Server with Threading
# ==============================
class ThreadedTCPServer(socketserver.ThreadingMixIn, socketserver.TCPServer):
    allow_reuse_address = True
    daemon_threads = True  # Allow graceful shutdown


# ==============================
# HTTP HANDLER
# ==============================
class ChatHandler(http.server.BaseHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        # DO NOT CREATE NEW CLIENTS HERE
        super().__init__(*args, **kwargs)

    # ----- CORS -----
    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, DELETE, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization")
        self.send_header("Access-Control-Allow-Credentials", "true")
        self.end_headers()

    # ----- GET -----
    def do_GET(self):
        path = urlparse(self.path).path

        if path == "/" or path == "":
            try:
                with open("public/index.html", "rb") as f:
                    self.send_response(200)
                    self.send_header("Content-Type", "text/html")
                    self.end_headers()
                    self.wfile.write(f.read())
                    return
            except FileNotFoundError:
                return self._send_json({"error": "Frontend not found"}, 404)

        if path.startswith("/static/") or path.endswith((".js", ".css")):
            file_path = f"public{path}"
            if os.path.exists(file_path):
                self.send_response(200)
                if file_path.endswith(".js"):
                    self.send_header("Content-Type", "application/javascript")
                elif file_path.endswith(".css"):
                    self.send_header("Content-Type", "text/css")
                else:
                    self.send_header("Content-Type", "application/octet-stream")
                self.end_headers()
                with open(file_path, "rb") as f:
                    self.wfile.write(f.read())
                return
            else:
                return self._send_json({"error": "File not found"}, 404)

        if path == "/auth/me":
            return self._handle_auth_me()
        if path == "/health":
            return self._health()
        if path == "/sessions":
            return self._get_sessions()
        if path.startswith("/sessions/") and path.endswith("/documents"):
            return self._get_session_documents(path.split("/")[-2])
        if path.startswith("/sessions/") and path.count("/") == 2:
            return self._get_session(path.split("/")[-1])
        if path == "/models":
            return self._get_models()
        if path == "/indexes":
            return self._get_indexes()
        if path.startswith("/indexes/") and path.count("/") == 2:
            return self._get_index(path.split("/")[-1])
        if path == "/preferences":
            return self._get_preferences()
        if path == "/available-models":
            return self._get_available_models()

        return self._not_found()

    # ----- POST -----
    def do_POST(self):
        path = urlparse(self.path).path

        if path == "/auth/register":
            return self._handle_register()
        if path == "/auth/login":
            return self._handle_login()
        if path == "/chat":
            return self._chat_legacy()
        if path == "/sessions":
            return self._create_session()
        if path.startswith("/sessions/") and path.endswith("/messages"):
            return self._session_chat(path.split("/")[-2])
        if path.startswith("/sessions/") and path.endswith("/upload"):
            return self._file_upload(path.split("/")[-2])
        if path.startswith("/sessions/") and path.endswith("/rename"):
            return self._rename_session(path.split("/")[-2])
        if path == "/indexes":
            return self._create_index()
        if path.startswith("/indexes/") and path.endswith("/upload"):
            return self._index_file_upload(path.split("/")[-2])
        if path == "/preferences":
            return self._set_preference()
        if path.startswith("/sessions/") and path.endswith("/regenerate"):
            return self._regenerate_response(path.split("/")[-2])
        if path.startswith("/sessions/") and path.endswith("/stop"):
            return self._stop_generation(path.split("/")[-2])

        return self._not_found()

    # ----- DELETE -----
    def do_DELETE(self):
        path = urlparse(self.path).path

        if path.startswith("/sessions/") and path.count("/") == 2:
            return self._delete_session(path.split("/")[-1])
        if path.startswith("/indexes/") and path.count("/") == 2:
            return self._delete_index(path.split("/")[-1])

        return self._not_found()

    # ==============================
    # HEALTH
    # ==============================
    def _handle_register(self):
        try:
            import auth as _auth
            length = int(self.headers.get("Content-Length", 0))
            data = __import__('json').loads(self.rfile.read(length).decode())
            result = _auth.register_user(data.get("email",""), data.get("password",""))
            return self._send_json(result, 400 if "error" in result else 200)
        except Exception as e:
            return self._send_json({"error": str(e)}, 500)

    def _handle_login(self):
        try:
            import auth as _auth
            length = int(self.headers.get("Content-Length", 0))
            data = __import__('json').loads(self.rfile.read(length).decode())
            result = _auth.login_user(data.get("email",""), data.get("password",""))
            return self._send_json(result, 401 if "error" in result else 200)
        except Exception as e:
            return self._send_json({"error": str(e)}, 500)

    def _handle_auth_me(self):
        try:
            import auth as _auth
            user_id = _auth.get_user_from_request(self.headers)
            if not user_id:
                return self._send_json({"error": "Unauthorized"}, 401)
            return self._send_json({"user_id": user_id, "authenticated": True})
        except Exception as e:
            return self._send_json({"error": str(e)}, 500)

    def _health(self):
        try:
            all_models = ollama.list_models()
            model_names = []
            if isinstance(all_models, dict):
                models_iter = all_models.get("models", []) or []
            else:
                models_iter = all_models or []
            for m in models_iter:
                if isinstance(m, dict):
                    name = m.get("name") or m.get("model") or m.get("id")
                    if name:
                        model_names.append(name)
                else:
                    model_names.append(str(m))
            return self._send_json({
                "status": "ok",
                "ollama_running": True,
                "available_models": model_names
            })
        except Exception as e:
            print(f"[ERROR] Health check failed: {e}")
            traceback.print_exc()
            return self._send_json({
                "status": "error",
                "ollama_running": False,
                "available_models": [],
                "error": str(e)
            }, status=500)

    # ==============================
    # GET MODELS
    # ==============================
    def _get_models(self):
        try:
            all_models = ollama.list_models()
            if isinstance(all_models, dict):
                models_iter = all_models.get("models", []) or []
            else:
                models_iter = all_models or []

            model_names = []
            for m in models_iter:
                if isinstance(m, dict):
                    name = m.get("name") or m.get("model") or m.get("id")
                    if name:
                        model_names.append(name)
                else:
                    model_names.append(str(m))

            embedding_keywords = ["embed", "embedding", "bge"]
            embedding_models = [m for m in model_names if any(k in m.lower() for k in embedding_keywords)]
            generation_models = [m for m in model_names if m not in embedding_models]

            generation_models.sort()
            embedding_models.sort()

            return self._send_json({
                "generation_models": generation_models,
                "embedding_models": embedding_models
            })
        except Exception as e:
            print(f"[ERROR] _get_models crashed: {e}")
            traceback.print_exc()
            return self._send_json({"error": f"Could not read models: {str(e)}"}, 500)

    # ==============================
    # INDEX / SESSION handlers
    # ==============================
    def _get_sessions(self):
        try:
            user_id = self.path.split("user_id=")[-1] if "user_id=" in self.path else None
            s = db.get_sessions(user_id=user_id)
            return self._send_json({"sessions": s, "total": len(s)})
        except Exception as e:
            print(f"[ERROR] _get_sessions: {e}")
            traceback.print_exc()
            return self._send_json({"error": f"Could not list sessions: {str(e)}"}, 500)

    def _get_session(self, sid):
        try:
            s = db.get_session(sid)
            if not s:
                return self._send_json({"error": "Session not found"}, 404)
            msgs = db.get_messages(sid)
            return self._send_json({"session": s, "messages": msgs})
        except Exception as e:
            print(f"[ERROR] _get_session: {e}")
            traceback.print_exc()
            return self._send_json({"error": f"Could not get session: {str(e)}"}, 500)

    def _get_session_documents(self, sid):
        try:
            s = db.get_session(sid)
            if not s:
                return self._send_json({"error": "Session not found"}, 404)
            paths = db.get_documents_for_session(sid)
            names = [os.path.basename(p) for p in paths]
            return self._send_json({"session": s, "files": names, "file_count": len(paths)})
        except Exception as e:
            print(f"[ERROR] _get_session_documents: {e}")
            traceback.print_exc()
            return self._send_json({"error": f"Could not list documents: {str(e)}"}, 500)

    def _get_indexes(self):
        try:
            idxs = db.list_indexes() if hasattr(db, "list_indexes") else []
            return self._send_json({"indexes": idxs})
        except Exception as e:
            print(f"[ERROR] _get_indexes: {e}")
            traceback.print_exc()
            return self._send_json({"error": f"Could not list indexes: {str(e)}"}, 500)

    def _get_index(self, idx):
        try:
            if hasattr(db, "get_index"):
                data = db.get_index(idx)
                if not data:
                    return self._send_json({"error": "Index not found"}, 404)
                return self._send_json({"index": data})
            return self._send_json({"error": "Indexing not available"}, 404)
        except Exception as e:
            print(f"[ERROR] _get_index: {e}")
            traceback.print_exc()
            return self._send_json({"error": f"Could not get index: {str(e)}"}, 500)

    # ==============================
    # CREATE SESSION
    # ==============================
    def _create_session(self):
        try:
            length = int(self.headers.get("Content-Length", 0))
            raw = self.rfile.read(length) if length > 0 else b"{}"
            data = json.loads(raw.decode("utf-8") or "{}")
            title = data.get("title") or generate_session_title()
            model = data.get("model") or DEFAULT_MODEL

            sid = None
            try:
                if hasattr(db, "create_session"):
                    sid = db.create_session(title, model, data.get("user_id"))
                elif hasattr(db, "add_session"):
                    sid = db.add_session({"title": title, "model": model})
                else:
                    sid = str(uuid.uuid4())
                    if hasattr(db, "save_session"):
                        db.save_session(sid, {"title": title, "model": model})
            except Exception:
                print("[WARN] db.create_session failed; returning generated id")
                traceback.print_exc()
                if not sid:
                    sid = str(uuid.uuid4())

            return self._send_json({"id": sid, "title": title, "model": model})
        except Exception as e:
            print(f"[ERROR] _create_session: {e}")
            traceback.print_exc()
            return self._send_json({"error": f"Could not create session: {str(e)}"}, 500)

    # ==============================
    # TOOL CALLING
    # ==============================
    def _use_calculator(self, expression: str) -> str:
        """Simple calculator tool"""
        try:
            # Safe eval with limited scope
            allowed_names = {"abs": abs, "round": round, "min": min, "max": max}
            result = eval(expression, {"__builtins__": {}}, allowed_names)
            return f"Calculation result: {result}"
        except Exception as e:
            return f"Calculator error: {str(e)}"

    def _detect_tool_use(self, message: str) -> tuple:
        """Detect if message needs tool"""
        msg_lower = message.lower()
        
        # Calculator detection
        calc_keywords = ["calculate", "compute", "what is", "solve"]
        has_math = any(op in message for op in ["+", "-", "*", "/", "^"])
        
        if has_math and any(kw in msg_lower for kw in calc_keywords):
            # Extract expression
            import re
            match = re.search(r'[\d\+\-\*/\(\)\.\s]+', message)
            if match:
                return ("calculator", match.group(0).strip())
        
        return (None, None)

    # ==============================
    # CONTEXT MANAGEMENT
    # ==============================
    def _trim_context(self, messages: list) -> list:
        """Trim messages to fit context window"""
        # Keep system prompt
        system_msg = [m for m in messages if m.get("role") == "system"]
        other_msgs = [m for m in messages if m.get("role") != "system"]
        
        # Keep last N messages
        if len(other_msgs) > MAX_CONTEXT_MESSAGES:
            other_msgs = other_msgs[-MAX_CONTEXT_MESSAGES:]
        
        # Trim long message content
        for msg in other_msgs:
            content = msg.get("content", "")
            if len(content) > MAX_TOKENS_PER_MESSAGE * 4:  # ~4 chars per token
                msg["content"] = content[:MAX_TOKENS_PER_MESSAGE * 4] + "..."
        
        return system_msg + other_msgs

    def _summarize_old_context(self, messages: list) -> str:
        """Summarize old messages if context too long"""
        if len(messages) <= MAX_CONTEXT_MESSAGES:
            return None
        
        old_messages = messages[1:-MAX_CONTEXT_MESSAGES]  # Skip system, keep recent
        if not old_messages:
            return None
        
        summary_parts = []
        for msg in old_messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")[:100]  # First 100 chars
            summary_parts.append(f"{role}: {content}")
        
        return "Previous conversation summary: " + " | ".join(summary_parts)

    # ==============================
    # MODEL ROUTING
    # ==============================
    def _select_model(self, user_message: str, requested_model: str = None) -> str:
        """Route to smart model for complex queries"""
        if requested_model:
            return requested_model
        
        msg_lower = user_message.lower()
        for keyword in COMPLEX_KEYWORDS:
            if keyword in msg_lower:
                return SMART_MODEL
        
        return DEFAULT_MODEL

    # ==============================
    # LEGACY CHAT (with system prompt)
    # ==============================
    def _chat_legacy(self):
        try:
            length = int(self.headers.get("Content-Length", 0))
            raw = self.rfile.read(length) if length > 0 else b"{}"
            data = json.loads(raw.decode("utf-8") or "{}")

            requested_model = data.get("model")
            user_msg = data.get("message", "").strip()

            if not user_msg:
                messages = data.get("messages") or []
                if messages:
                    last = messages[-1]
                    if isinstance(last, dict):
                        user_msg = str(last.get("content") or "").strip()
                    else:
                        user_msg = str(last).strip()

            if not user_msg:
                return self._send_json({"error": "Empty message"}, 400)

            # Route to appropriate model
            model = self._select_model(user_msg, requested_model)

            # Build messages with system prompt
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_msg}
            ]

            resp = ollama.chat(model, messages)
            reply = resp.get("response", "") if isinstance(resp, dict) else str(resp)
            return self._send_json({"response": reply, "model_used": model})
        except Exception as e:
            print(f"[ERROR] _chat_legacy: {e}")
            traceback.print_exc()
            return self._send_json({"error": f"Chat failed: {str(e)}"}, 500)

    # ==============================
    # SESSION CHAT (with full history)
    # ==============================
    def _session_chat(self, sid):
        try:
            length = int(self.headers.get("Content-Length", 0))
            raw = self.rfile.read(length) if length > 0 else b"{}"
            data = json.loads(raw.decode("utf-8") or "{}")

            user_message = data.get("message", "").strip()
            if not user_message:
                return self._send_json({"error": "Empty message"}, 400)

            requested_model = data.get("model")
            image_path = data.get("image_path")  # For vision models

            # CHECK FOR TOOL USE
            tool_result = None
            if ENABLE_TOOLS and not image_path:
                tool_name, tool_input = self._detect_tool_use(user_message)
                if tool_name == "calculator":
                    tool_result = self._use_calculator(tool_input)

            # GET MESSAGE HISTORY
            history = []
            try:
                if hasattr(db, "get_messages"):
                    history = db.get_messages(sid) or []
            except Exception as e:
                print(f"[WARN] Could not load history: {e}")
                history = []

            # AUTO-RENAME CHAT based on first user message
            if AUTO_NAME_CHATS and RENAME_AFTER_FIRST_MESSAGE and len(history) == 0:
                try:
                    # Generate title from first message
                    title = self._generate_chat_title(user_message)
                    if hasattr(db, "update_session_title"):
                        db.update_session_title(sid, title)
                        print(f"[INFO] Auto-renamed chat to: {title}")
                except Exception as e:
                    print(f"[WARN] Could not auto-rename chat: {e}")

            # Route to appropriate model (use vision model if image)
            if image_path:
                model = VISION_MODEL
            else:
                model = self._select_model(user_message, requested_model)

            # BUILD FULL MESSAGE LIST
            messages = []

            # 1. SYSTEM PROMPT (always first)
            messages.append({"role": "system", "content": SYSTEM_PROMPT})

            # 2. LOAD CONVERSATION HISTORY FROM DATABASE
            try:
                past_messages = db.get_conversation_history(sid)
                # Keep last 10 exchanges (20 messages) to avoid token overflow
                if len(past_messages) > 20:
                    past_messages = past_messages[-20:]
                messages.extend(past_messages)
                print(f"[MEMORY] Loaded {len(past_messages)} past messages for session {sid[:8]}")
            except Exception as e:
                print(f"[WARN] Could not load history: {e}")

            # 3. ADD TOOL RESULT IF AVAILABLE
            if tool_result:
                messages.append({"role": "system", "content": f"Tool result: {tool_result}"})

            # 4. ADD CURRENT USER MESSAGE
            messages.append({"role": "user", "content": user_message})

            # CALL OLLAMA (with image if present)
            if image_path:
                resp = ollama.chat_with_image(model, messages, image_path)
            else:
                messages = self._trim_context(messages)
                resp = ollama.chat(model, messages)
                
            reply = resp.get("response", "") if isinstance(resp, dict) else str(resp)

            if not reply:
                reply = "Error: Empty response from model"

            # SAVE TO DATABASE (persistent memory)
            try:
                if hasattr(db, "add_message"):
                    db.add_message(sid, "user", user_message)
                    db.add_message(sid, "assistant", reply)
            except Exception as e:
                print(f"[WARN] Could not persist messages to DB: {e}")
                traceback.print_exc()

            return self._send_json({
                "reply": reply,
                "model_used": model,
                "has_image": image_path is not None
            })
        except Exception as e:
            print(f"[ERROR] _session_chat: {e}")
            traceback.print_exc()
            return self._send_json({"error": f"Chat failed: {str(e)}"}, 500)

    # ==============================
    # GENERATE CHAT TITLE
    # ==============================
    def _generate_chat_title(self, first_message: str) -> str:
        """Generate a descriptive title from the first message"""
        # Clean message
        msg = first_message.strip()[:100]  # First 100 chars
        
        # Remove common prefixes
        prefixes = ["can you", "could you", "please", "help me", "i want to", "i need to"]
        msg_lower = msg.lower()
        for prefix in prefixes:
            if msg_lower.startswith(prefix):
                msg = msg[len(prefix):].strip()
                break
        
        # Capitalize first letter
        if msg:
            msg = msg[0].upper() + msg[1:]
        
        # Limit length
        if len(msg) > 50:
            msg = msg[:47] + "..."
        
        return msg or "New Chat"

    # ==============================
    # FILE UPLOAD (FIXED)
    # ==============================
    def _file_upload(self, sid):
        """Handle file uploads (PDF, images, etc.)"""
        try:
            content_type = self.headers.get('Content-Type', '')
            
            if 'multipart/form-data' not in content_type:
                return self._send_json({"error": "Use multipart/form-data"}, 400)
            
            # Parse boundary
            boundary = content_type.split("boundary=")[-1].strip()
            length = int(self.headers.get('Content-Length', 0))
            
            if length == 0:
                return self._send_json({"error": "No data received"}, 400)
            
            # Read raw data
            data = self.rfile.read(length)
            
            # Simple multipart parsing
            parts = data.split(f'--{boundary}'.encode())
            
            file_data = None
            filename = None
            
            for part in parts:
                if b'Content-Disposition' in part and b'filename=' in part:
                    # Extract filename
                    lines = part.split(b'\r\n')
                    for line in lines:
                        if b'filename=' in line:
                            # Extract filename from: Content-Disposition: form-data; name="file"; filename="test.pdf"
                            filename_part = line.split(b'filename="')[1].split(b'"')[0]
                            filename = filename_part.decode('utf-8', errors='ignore')
                            break
                    
                    # Extract file data (after double CRLF, before final CRLF)
                    if b'\r\n\r\n' in part:
                        file_data = part.split(b'\r\n\r\n', 1)[1]
                        # Remove trailing CRLF
                        if file_data.endswith(b'\r\n'):
                            file_data = file_data[:-2]
                        break
            
            if not file_data or not filename:
                print("[ERROR] No file found in multipart data")
                return self._send_json({"error": "No file found in upload"}, 400)
            
            # Save file
            upload_dir = "/tmp/localgpt_uploads"
            os.makedirs(upload_dir, exist_ok=True)
            
            safe_filename = f"{uuid.uuid4()}_{filename}"
            file_path = os.path.join(upload_dir, safe_filename)
            
            with open(file_path, 'wb') as f:
                f.write(file_data)
            
            print(f"[INFO] File saved: {file_path} ({len(file_data)} bytes)")
            
            # Determine file type
            filename_lower = filename.lower()
            is_image = filename_lower.endswith(('.png', '.jpg', '.jpeg', '.gif', '.webp', '.bmp'))
            is_pdf = filename_lower.endswith('.pdf')
            
            # Extract PDF text
            pdf_text = None
            text_length = 0
            
            if is_pdf:
                try:
                    import PyPDF2
                    print(f"[INFO] Extracting PDF text from: {file_path}")
                    
                    with open(file_path, 'rb') as f:
                        reader = PyPDF2.PdfReader(f)
                        pdf_text = ""
                        
                        total_pages = len(reader.pages)
                        print(f"[INFO] PDF has {total_pages} pages")
                        
                        for page_num, page in enumerate(reader.pages):
                            try:
                                page_text = page.extract_text()
                                if page_text:
                                    pdf_text += f"\n--- Page {page_num + 1} ---\n{page_text}\n"
                            except Exception as e:
                                print(f"[WARN] Failed to extract page {page_num + 1}: {e}")
                                continue
                        
                        pdf_text = pdf_text.strip()
                        text_length = len(pdf_text)
                        
                        if text_length == 0:
                            print("[WARN] PDF text extraction returned empty result")
                            pdf_text = None
                        else:
                            print(f"[SUCCESS] PDF extracted: {text_length} characters from {total_pages} pages")
                            
                except Exception as e:
                    print(f"[ERROR] PDF extraction failed: {e}")
                    import traceback
                    traceback.print_exc()
                    pdf_text = None
            
            response_data = {
                "success": True,
                "filename": filename,
                "path": file_path,
                "type": "image" if is_image else ("pdf" if is_pdf else "document"),
                "pdf_text": pdf_text,
                "text_length": text_length
            }
            
            print(f"[INFO] Upload response: {response_data['type']}, {response_data['text_length']} chars")
            
            return self._send_json(response_data)
                
        except Exception as e:
            print(f"[ERROR] _file_upload crashed: {e}")
            import traceback
            traceback.print_exc()
            return self._send_json({"error": f"Upload failed: {str(e)}"}, 500)

    # ==============================
    # Minimal upload / index stubs
    # ==============================
    def _index_file_upload(self, idx):
        return self._send_json({"error": "Index file upload not implemented"}, 501)

    def _create_index(self):
        return self._send_json({"error": "Create index not implemented"}, 501)

    def _rename_session(self, sid):
        return self._send_json({"error": "Rename session not implemented"}, 501)

    # ==============================
    # USER PREFERENCES
    # ==============================
    def _get_preferences(self):
        try:
            prefs = db.get_all_preferences()
            return self._send_json({"preferences": prefs})
        except Exception as e:
            print(f"[ERROR] _get_preferences: {e}")
            traceback.print_exc()
            return self._send_json({"error": str(e)}, 500)

    def _set_preference(self):
        try:
            length = int(self.headers.get("Content-Length", 0))
            raw = self.rfile.read(length) if length > 0 else b"{}"
            data = json.loads(raw.decode("utf-8") or "{}")
            
            key = data.get("key")
            value = data.get("value")
            
            if not key:
                return self._send_json({"error": "Missing key"}, 400)
            
            success = db.set_preference(key, str(value))
            return self._send_json({"success": success, "key": key, "value": value})
        except Exception as e:
            print(f"[ERROR] _set_preference: {e}")
            traceback.print_exc()
            return self._send_json({"error": str(e)}, 500)

    # ==============================
    # ADVANCED FEATURES
    # ==============================
    def _get_available_models(self):
        """Get list of all models for selection"""
        return self._get_models()

    def _regenerate_response(self, sid):
        """Regenerate last assistant response"""
        try:
            # Get last 2 messages
            messages = db.get_messages(sid)
            if len(messages) < 2:
                return self._send_json({"error": "No message to regenerate"}, 400)
            
            # Delete last assistant message
            last_msg = messages[-1]
            if last_msg.get("role") != "assistant":
                return self._send_json({"error": "Last message not from assistant"}, 400)
            
            # Get user message
            user_msg = messages[-2].get("content", "")
            
            # Regenerate by calling chat again
            return self._session_chat(sid)
        except Exception as e:
            print(f"[ERROR] _regenerate_response: {e}")
            traceback.print_exc()
            return self._send_json({"error": str(e)}, 500)

    def _stop_generation(self, sid):
        """Stop ongoing generation (placeholder)"""
        # Note: Ollama doesn't support stopping mid-generation easily
        # This would require streaming implementation
        return self._send_json({"message": "Stop not implemented (requires streaming)"}, 501)

    # ==============================
    # DELETE helpers
    # ==============================
    def _delete_session(self, sid):
        try:
            if hasattr(db, "delete_session"):
                db.delete_session(sid)
                return self._send_json({"deleted": True})
            return self._send_json({"error": "Delete not supported by DB"}, 404)
        except Exception as e:
            print(f"[ERROR] _delete_session: {e}")
            traceback.print_exc()
            return self._send_json({"error": f"Could not delete session: {str(e)}"}, 500)

    def _delete_index(self, idx):
        try:
            if hasattr(db, "delete_index"):
                db.delete_index(idx)
                return self._send_json({"deleted": True})
            return self._send_json({"error": "Delete index not supported by DB"}, 404)
        except Exception as e:
            print(f"[ERROR] _delete_index: {e}")
            traceback.print_exc()
            return self._send_json({"error": f"Could not delete index: {str(e)}"}, 500)

    # ==============================
    # Helper Methods
    # ==============================
    def _send_json(self, data, status=200):
        try:
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Methods", "GET, POST, DELETE, OPTIONS")
            self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization")
            self.send_header("Access-Control-Allow-Credentials", "true")
            self.end_headers()
            self.wfile.write(json.dumps(data, ensure_ascii=False).encode("utf-8"))
        except Exception:
            pass

    def _not_found(self):
        return self._send_json({"error": "Not found"}, 404)

    def log_message(self, format, *args):
        print(f"[{self.date_time_string()}] {format % args}")


# ==============================
# MAIN
# ==============================
def main():
    PORT = int(os.environ.get("PORT", 8000))

    # Disable PDF processor to save memory
    # try:
    #     pdf_module.initialize_simple_pdf_processor()
    # except Exception:
    #     pass

    host = "0.0.0.0"

    with ThreadedTCPServer((host, PORT), ChatHandler) as httpd:
        print(f"🚀 Backend running on http://{host}:{PORT}")
        print(f"📊 Threading enabled - supports multiple concurrent requests")
        httpd.serve_forever()


if __name__ == "__main__":
    print("🔥 Starting backend...")

    # Warm model on startup (keeps it in memory)
    try:
        print("🔥 Warming model...")
        ollama.generate(DEFAULT_MODEL, "hi")
        print("✅ Model warmed and ready")
    except Exception as e:
        print(f"⚠️ Model warm failed: {e}")

    main()

# ==============================
# AUTH ROUTES — appended
# ==============================
import auth as auth_module

_original_do_post = ChatHandler.do_POST
_original_do_get = ChatHandler.do_GET

def _new_do_post(self):
    path = urlparse(self.path).path
    if path == "/auth/register":
        return self._auth_register()
    if path == "/auth/login":
        return self._auth_login()
    return _original_do_post(self)

def _new_do_get(self):
    path = urlparse(self.path).path
    if path == "/auth/me":
        return self._auth_me()
    return _original_do_get(self)

def _auth_register(self):
    try:
        length = int(self.headers.get("Content-Length", 0))
        data = json.loads(self.rfile.read(length).decode())
        result = auth_module.register_user(data.get("email",""), data.get("password",""))
        status = 400 if "error" in result else 200
        return self._send_json(result, status)
    except Exception as e:
        return self._send_json({"error": str(e)}, 500)

def _auth_login(self):
    try:
        length = int(self.headers.get("Content-Length", 0))
        data = json.loads(self.rfile.read(length).decode())
        result = auth_module.login_user(data.get("email",""), data.get("password",""))
        status = 401 if "error" in result else 200
        return self._send_json(result, status)
    except Exception as e:
        return self._send_json({"error": str(e)}, 500)

def _auth_me(self):
    user_id = auth_module.get_user_from_request(self.headers)
    if not user_id:
        return self._send_json({"error": "Unauthorized"}, 401)
    return self._send_json({"user_id": user_id, "authenticated": True})

ChatHandler.do_POST = _new_do_post
ChatHandler.do_GET = _new_do_get
ChatHandler._auth_register = _auth_register
ChatHandler._auth_login = _auth_login
ChatHandler._auth_me = _auth_me
