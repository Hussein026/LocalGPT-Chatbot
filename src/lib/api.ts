// -------------------- BASE URLS FROM ENV --------------------
// -------------------- BASE URLS FROM ENV --------------------
// Use env variable first, fallback to window location if available, else localhost
export const API_BASE_URL =
  process.env.NEXT_PUBLIC_API_BASE_URL ||
  (typeof window !== "undefined"
    ? `${window.location.protocol}//${window.location.hostname}:8000`
    : "http://localhost:8000");

export const RAG_API_BASE_URL =
  process.env.NEXT_PUBLIC_RAG_API_BASE_URL ||
  (typeof window !== "undefined"
    ? `${window.location.protocol}//${window.location.hostname}:8001`
    : "http://127.0.0.1:8001");


// -------------------- URL BUILDER (IMPORTANT) --------------------
// This function correctly uses API_BASE_URL and ensures proper path formatting.
const buildUrl = (path: string) => {
  return `${API_BASE_URL}${path.startsWith("/") ? path : `/${path}`}`;
};

// -------------------- UUID --------------------
export const generateUUID = () => {
  if (typeof window !== "undefined" && window.crypto?.randomUUID) {
    return window.crypto.randomUUID();
  }
  return "xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx".replace(/[xy]/g, (c) => {
    const r = (Math.random() * 16) | 0;
    const v = c === "x" ? r : (r & 0x3) | 0x8;
    return v.toString(16);
  });
};

// -------------------- INTERFACES --------------------
export interface Step {
  key: string;
  label: string;
  status: "pending" | "active" | "done";
  details: any;
}

export interface ChatMessage {
  id: string;
  content: string | Array<Record<string, any>> | { steps: Step[] };
  sender: "user" | "assistant";
  timestamp: string;
  isLoading?: boolean;
  metadata?: Record<string, unknown>;
}

export interface ChatSession {
  id: string;
  title: string;
  created_at: string;
  updated_at: string;
  model_used: string;
  message_count: number;
}

export interface ChatRequest {
  message: string;
  model?: string;
  conversation_history?: Array<{ role: "user" | "assistant"; content: string }>;
}

export interface ChatResponse {
  response: string;
  model: string;
  message_count: number;
}

export interface HealthResponse {
  status: string;
  ollama_running: boolean;
  available_models: string[];
  database_stats?: {
    total_sessions: number;
    total_messages: number;
    most_used_model: string | null;
  };
}

export interface ModelsResponse {
  generation_models: string[];
  embedding_models: string[];
}

export interface SessionResponse {
  sessions: ChatSession[];
  total: number;
}

export interface SessionChatResponse {
  response: string;
  session: ChatSession;
  user_message_id?: string;
  ai_message_id?: string;
}

// -------------------- API CLASS --------------------
interface UploadResult {
  uploaded_files: { filename: string; stored_path: string }[];
}
class ChatAPI {
  convertDbMessage(m: any): any {
    throw new Error('Method not implemented.');
  }

  // ------ HEALTH ------
  async checkHealth(): Promise<HealthResponse> {
    try {
      const response = await fetch(buildUrl("/health")); // FIX: Use buildUrl
      if (!response.ok) {
        //throw new Error(`Health check failed: ${response.status}`);
      }
      return await response.json();
    } catch (error) {
      console.error("Health check failed:", error);
      return {
        status: "error",
        ollama_running: false,
        available_models: [],
        database_stats: {
          total_sessions: 0,
          total_messages: 0,
          most_used_model: null,
        },
      };
    }
  }

  // ------ NORMAL CHAT ------
  async sendMessage(request: ChatRequest): Promise<ChatResponse> {
    try {
      const response = await fetch(buildUrl("/chat"), { // FIX: Use buildUrl
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          message: request.message,
          model: request.model || "qwen2.5:0.5b-instruct-q4_K_M", // 🚀 FIX: Switched to qwen2.5:0.5b for speed
          conversation_history: request.conversation_history || [],
        }),
      });
      if (!response.ok) {
        const errorData = await response.json().catch(() => ({ error: "Unknown error" }));
        throw new Error(`Chat API error: ${errorData.error}`);
      }
      return await response.json();
    } catch (error) {
      console.error("Chat API failed:", error);
      throw error;
    }
  }

  // Convert to conversation history
  messagesToHistory(messages: ChatMessage[]): Array<{ role: "user" | "assistant"; content: string }> {
    return messages
      .filter((msg) => typeof msg.content === "string" && msg.content.trim())
      .map((msg) => ({ role: msg.sender, content: msg.content as string }));
  }

  // ------ SESSIONS ------
  async getSessions(): Promise<SessionResponse> {
    const resp = await fetch(buildUrl("/sessions")); // FIX: Use buildUrl
    if (!resp.ok) throw new Error(`Failed to get sessions: ${resp.status}`);
    return resp.json();
  }

  async createSession(title = "New Chat", model = "qwen2.5:0.5b-instruct-q4_K_M"): Promise<ChatSession> { // 🚀 FIX: Switched to qwen2.5:0.5b for speed
    const resp = await fetch(buildUrl("/sessions"), { // FIX: Use buildUrl
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ title, model }),
    });
    if (!resp.ok) throw new Error(`Failed to create session: ${resp.status}`);
    const data = await resp.json();
    return data.session;
  }

  async getSession(sessionId: string): Promise<{ session: ChatSession; messages: ChatMessage[] }> {
    const resp = await fetch(buildUrl(`/sessions/${sessionId}`)); // FIX: Use buildUrl
    if (!resp.ok) throw new Error(`Failed to get session: ${resp.status}`);
    return resp.json();
  }

  async sendSessionMessage(sessionId: string, message: string, opts: any = {}): Promise<SessionChatResponse & { source_documents: any[] }> {
    const resp = await fetch(buildUrl(`/sessions/${sessionId}/messages`), { // FIX: Use buildUrl
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message, ...opts }),
    });
    if (!resp.ok) {
      const errorData = await resp.json().catch(() => ({}));
      throw new Error(`Session chat error: ${errorData.error || resp.statusText}`);
    }
    return resp.json();
  }

  async deleteSession(sessionId: string) {
    const resp = await fetch(buildUrl(`/sessions/${sessionId}`), { method: "DELETE" }); // FIX: Use buildUrl
    if (!resp.ok) throw new Error(`Failed to delete session: ${resp.status}`);
    return resp.json();
  }

  async renameSession(sessionId: string, newTitle: string) {
    const resp = await fetch(buildUrl(`/sessions/${sessionId}/rename`), { // FIX: Use buildUrl
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ title: newTitle }),
    });
    if (!resp.ok) throw new Error(`Failed: ${resp.status}`);
    return resp.json();
  }

  async cleanupEmptySessions() {
    const resp = await fetch(buildUrl("/sessions/cleanup")); // FIX: Use buildUrl
    if (!resp.ok) throw new Error(`Cleanup failed: ${resp.status}`);
    return resp.json();
  }

      

  // ------ UPLOAD ------
  async uploadFiles(sessionId: string, files: File[]) {
    const fd = new FormData();
    files.forEach((f) => fd.append("files", f, f.name));
    const resp = await fetch(buildUrl(`/sessions/${sessionId}/upload`), { method: "POST", body: fd }); // FIX: Use buildUrl
    if (!resp.ok) throw new Error(`Upload failed: ${resp.status}`);
    return resp.json();
  }

  async indexDocuments(sessionId: string) {
    const resp = await fetch(buildUrl(`/sessions/${sessionId}/index`), { // FIX: Use buildUrl
      method: "POST",
      headers: { "Content-Type": "application/json" },
    });
    if (!resp.ok) throw new Error(`Index failed: ${resp.status}`);
    return resp.json();
  }

  // ------ MODELS ------
  async getModels(): Promise<ModelsResponse> {
    const resp = await fetch(buildUrl("/models")); // FIX: Use buildUrl
    if (!resp.ok) throw new Error(`Failed to fetch models: ${resp.status}`);
    return resp.json();
  }

  // ------ DOCUMENTS ------
  async getSessionDocuments(sessionId: string) {
    const resp = await fetch(buildUrl(`/sessions/${sessionId}/documents`)); // FIX: Use buildUrl
    if (!resp.ok) throw new Error(`Failed to fetch session documents: ${resp.status}`);
    return resp.json();
  }

// ------ FRONTEND-ONLY MESSAGE CREATION (NOT SENT TO BACKEND) ------
createMessage(content: string, sender: "user" | "assistant" = "assistant"): ChatMessage {
  return {
    id: generateUUID(),
    content,
    sender,
    timestamp: new Date().toISOString(),
  };
}


  // ------ INDEXING ------
  async createIndex(name: string, description?: string, metadata: any = {}) {
    const resp = await fetch(buildUrl("/indexes"), { // FIX: Use buildUrl
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ name, description, metadata }),
    });
    if (!resp.ok) throw new Error(`Create index error: ${resp.status}`);
    return resp.json();
  }

  async uploadFilesToIndex(indexId: string, files: File[]) {
    const fd = new FormData();
    files.forEach((f) => fd.append("files", f));
    const resp = await fetch(buildUrl(`/indexes/${indexId}/upload`), { method: "POST", body: fd }); // FIX: Use buildUrl
    if (!resp.ok) throw new Error(`Upload failed: ${resp.status}`);
    return resp.json();
  }

  async buildIndex(indexId: string, opts: any = {}) {
    const resp = await fetch(buildUrl(`/indexes/${indexId}/build`), { // FIX: Use buildUrl
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(opts),
    });
    if (!resp.ok) throw new Error(`Build index failed: ${resp.status}`);
    return resp.json();
  }

  async linkIndexToSession(sessionId: string, indexId: string) {
    const resp = await fetch(buildUrl(`/sessions/${sessionId}/indexes/${indexId}`), { method: "POST" }); // FIX: Use buildUrl
    if (!resp.ok) throw new Error(`Link index failed: ${resp.status}`);
    return resp.json();
  }

  async listIndexes() {
    const resp = await fetch(buildUrl("/indexes")); // FIX: Use buildUrl
    if (!resp.ok) throw new Error(`Failed: ${resp.status}`);
    return resp.json();
  }

  async getSessionIndexes(sessionId: string) {
    const resp = await fetch(buildUrl(`/sessions/${sessionId}/indexes`)); // FIX: Use buildUrl
    if (!resp.ok) throw new Error(`Failed: ${resp.status}`);
    return resp.json();
  }

  async deleteIndex(indexId: string) {
    const resp = await fetch(buildUrl(`/indexes/${indexId}`), { method: "DELETE" }); // FIX: Use buildUrl
    if (!resp.ok) throw new Error(`Failed: ${resp.status}`);
    return resp.json();
  }

  // -------------------- STREAMING FIXED --------------------
  async streamSessionMessage(params: any, onEvent: (e: any) => void) {
    const resp = await fetch(`${RAG_API_BASE_URL}/chat/stream`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(params),
    });
    if (!resp.ok || !resp.body) {
      throw new Error(`Stream request failed: ${resp.status}`);
    }

    const reader = resp.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";
    let doneReading = false;

    while (!doneReading) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      const parts = buffer.split("\n\n");
      buffer = parts.pop() || "";

      for (const part of parts) {
        const line = part.trim();
        if (!line.startsWith("data:")) continue;
        const json = line.replace("data:", "").trim();
        try {
          const evt = JSON.parse(json);
          onEvent(evt);
          if (evt.type === "complete") {
            doneReading = true;
            break;
          }
        } catch {}
      }
    }
  }
}

// -------------------- EXPORT --------------------
export const chatAPI = new ChatAPI();