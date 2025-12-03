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

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from database import db, generate_session_title
import simple_pdf_processor as pdf_module
from ollama_client import OllamaClient

DEFAULT_MODEL = "qwen2.5:0.5b-instruct-q4_K_M"

# ==============================
# TCP Server
# ==============================
class ReusableTCPServer(socketserver.TCPServer):
    allow_reuse_address = True


# ==============================
# HTTP HANDLER
# ==============================
class ChatHandler(http.server.BaseHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        use_ollama = os.environ.get("USE_OLLAMA", "true").lower() in ("1", "true", "yes")
        self.ollama = None
        if use_ollama:
            try:
                self.ollama = OllamaClient()
            except Exception as e:
                print(f"[WARN] Failed to init OllamaClient: {e}")
                traceback.print_exc()
                self.ollama = None
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

        return self._not_found()

    # ----- POST -----
    def do_POST(self):
        path = urlparse(self.path).path

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
    def _health(self):
        try:
            if self.ollama:
                try:
                    all_models = self.ollama.list_models()
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
                    print(f"[WARN] Ollama.list() failed: {e}")
                    traceback.print_exc()
                    return self._send_json({
                        "status": "error",
                        "ollama_running": False,
                        "available_models": [],
                        "error": str(e)
                    }, status=500)
            return self._send_json({
                "status": "ok",
                "ollama_running": False,
                "available_models": []
            })
        except Exception as e:
            print(f"[ERROR] Health handler crashed: {e}")
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
            if not self.ollama:
                return self._send_json({
                    "generation_models": [],
                    "embedding_models": [],
                    "warning": "Ollama is not configured"
                })
            all_models = self.ollama.list_models()
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
            s = db.get_sessions()
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
                    sid = db.create_session(title, model)
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
    # LEGACY CHAT (fully fixed)
    # ==============================
    def _chat_legacy(self):
        try:
            length = int(self.headers.get("Content-Length", 0))
            raw = self.rfile.read(length) if length > 0 else b"{}"
            data = json.loads(raw.decode("utf-8") or "{}")

            model = data.get("model") or DEFAULT_MODEL
            messages = data.get("messages") or []
            user_msg = ""

            # handle messages list
            if messages:
                last = messages[-1]
                if isinstance(last, dict):
                    user_msg = str(last.get("content") or "").strip()
                else:
                    user_msg = str(last).strip()

            # fallback to "message" key
            if not user_msg:
                user_msg = str(data.get("message") or "").strip()

            if not user_msg:
                user_msg = "[empty message]"

            if not self.ollama:
                return self._send_json({"error": "Ollama not available"}, 503)

            resp = self.ollama.generate(model, user_msg)
            reply = resp.get("response") if isinstance(resp, dict) else str(resp)
            return self._send_json({"response": reply})
        except Exception as e:
            print(f"[ERROR] _chat_legacy: {e}")
            traceback.print_exc()
            return self._send_json({"error": f"Chat failed: {str(e)}"}, 500)

    # ==============================
    # SESSION CHAT (fully fixed)
    # ==============================
    def _session_chat(self, sid):
        try:
            length = int(self.headers.get("Content-Length", 0))
            raw = self.rfile.read(length) if length > 0 else b"{}"
            data = json.loads(raw.decode("utf-8") or "{}")

            message = data.get("message") or data.get("messages") or ""
            if isinstance(message, list) and message:
                last = message[-1]
                if isinstance(last, dict):
                    message = str(last.get("content") or "").strip()
                else:
                    message = str(last).strip()
            elif isinstance(message, str):
                message = message.strip()
            else:
                message = str(message or "").strip()

            if not message:
                message = "[empty message]"

            model = data.get("model") or DEFAULT_MODEL

            if not self.ollama:
                return self._send_json({"error": "Ollama not available"}, 503)

            resp = self.ollama.generate(model, message)
            reply = resp.get("response") if isinstance(resp, dict) else str(resp)

            try:
                if hasattr(db, "add_message"):
                    db.add_message(sid, "user", message)
                    db.add_message(sid, "assistant", reply)
                elif hasattr(db, "append_message"):
                    db.append_message(sid, {"role": "user", "content": message})
                    db.append_message(sid, {"role": "assistant", "content": reply})
            except Exception:
                print("[WARN] Could not persist messages to DB")
                traceback.print_exc()

            return self._send_json({"reply": reply})
        except Exception as e:
            print(f"[ERROR] _session_chat: {e}")
            traceback.print_exc()
            return self._send_json({"error": f"Chat failed: {str(e)}"}, 500)

    # ==============================
    # Minimal upload / index stubs
    # ==============================
    def _file_upload(self, sid):
        return self._send_json({"error": "File upload not implemented"}, 501)

    def _index_file_upload(self, idx):
        return self._send_json({"error": "Index file upload not implemented"}, 501)

    def _create_index(self):
        return self._send_json({"error": "Create index not implemented"}, 501)

    def _rename_session(self, sid):
        return self._send_json({"error": "Rename session not implemented"}, 501)

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

    try:
        pdf_module.initialize_simple_pdf_processor()
    except Exception:
        pass

    host = "0.0.0.0"

    with ReusableTCPServer((host, PORT), ChatHandler) as httpd:
        print(f"🚀 FAST backend running on http://{host}:{PORT}")
        httpd.serve_forever()
if __name__ == "__main__":
    print("🔥 Starting backend...")

    # OPTIONAL preload (won't freeze server if it fails)
    try:
        from ollama_client import OllamaClient
        print("🔥 Preloading model once...")
        OllamaClient().generate("qwen2.5:0.5b-instruct-q4_K_M", "hello")
    except Exception as e:
        print("⚠️ Preload failed (safe to continue):", e)

    # start server
    main()
