import { useState, useRef, useEffect } from "react";
import "./App.css";

const API_URL = "https://rag-agent-platform-production.up.railway.app";

interface Message {
  role: "user" | "assistant";
  content: string;
}

function App() {
  const [token, setToken] = useState<string | null>(() => localStorage.getItem("token"));
  const [authMode, setAuthMode] = useState<"login" | "register">("login");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [authError, setAuthError] = useState("");
  const [authLoading, setAuthLoading] = useState(false);

  const [messages, setMessages] = useState<Message[]>([]);
  const [question, setQuestion] = useState("");
  const [uploading, setUploading] = useState(false);
  const [uploadedFile, setUploadedFile] = useState<string | null>(null);
  const [documentIds, setDocumentIds] = useState<string[]>([]);
  const [loading, setLoading] = useState(false);
  const fileRef = useRef<HTMLInputElement>(null);

  const authFetch = (url: string, options: RequestInit = {}) =>
    fetch(url, {
      ...options,
      headers: {
        ...(options.headers || {}),
        Authorization: `Bearer ${token}`,
      },
    });

  const handleAuth = async () => {
    setAuthError("");
    setAuthLoading(true);
    try {
      let res: Response;
      if (authMode === "register") {
        res = await fetch(`${API_URL}/auth/register`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ email, password }),
        });
      } else {
        const form = new URLSearchParams({ username: email, password });
        res = await fetch(`${API_URL}/auth/token`, {
          method: "POST",
          headers: { "Content-Type": "application/x-www-form-urlencoded" },
          body: form.toString(),
        });
      }
      if (!res.ok) {
        const data = await res.json();
        setAuthError(data.detail || "Authentication failed");
        return;
      }
      const data = await res.json();
      localStorage.setItem("token", data.access_token);
      setToken(data.access_token);
    } catch {
      setAuthError("Could not reach the server.");
    } finally {
      setAuthLoading(false);
    }
  };

  const logout = () => {
    localStorage.removeItem("token");
    setToken(null);
    setMessages([]);
    setDocumentIds([]);
    setUploadedFile(null);
  };

  const uploadPDF = async (file: File) => {
    setUploading(true);
    const formData = new FormData();
    formData.append("file", file);
    try {
      const res = await authFetch(`${API_URL}/upload`, { method: "POST", body: formData });
      const data = await res.json();
      if (!res.ok) {
        setMessages(prev => [...prev, { role: "assistant", content: `Upload failed: ${data.detail}` }]);
        return;
      }
      setUploadedFile(file.name);
      setDocumentIds(prev => [...prev, data.document_id]);
      setMessages(prev => [...prev, {
        role: "assistant",
        content: `**${file.name}** uploaded and indexed. You can now ask questions about it.`,
      }]);
    } catch {
      setMessages(prev => [...prev, { role: "assistant", content: "Upload failed. Please check the server is running." }]);
    }
    setUploading(false);
  };

  const askQuestion = async () => {
    if (!question.trim()) return;
    const userMsg = question;
    setQuestion("");
    setMessages(prev => [...prev, { role: "user", content: userMsg }]);
    setLoading(true);
    setMessages(prev => [...prev, { role: "assistant", content: "Thinking..." }]);

    let assistantMsg = "";
    try {
      const res = await authFetch(`${API_URL}/query`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question: userMsg, document_ids: documentIds }),
      });

      const reader = res.body!.getReader();
      const decoder = new TextDecoder();

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        const chunk = decoder.decode(value, { stream: true });
        for (const line of chunk.split("\n")) {
          if (line.startsWith("data: ") && !line.includes("[DONE]")) {
            assistantMsg += line.substring(6);
          }
        }
        setMessages(prev => {
          const updated = [...prev];
          updated[updated.length - 1] = { role: "assistant", content: assistantMsg || "Thinking..." };
          return updated;
        });
      }
    } catch {
      setMessages(prev => {
        const updated = [...prev];
        updated[updated.length - 1] = { role: "assistant", content: "Error connecting to server." };
        return updated;
      });
    }
    setLoading(false);
  };

  if (!token) {
    return (
      <div className="app">
        <header className="header">
          <div className="header-inner">
            <div className="logo">
              <span className="logo-icon">🤖</span>
              <div>
                <div className="logo-title">RAG Agent Platform</div>
                <div className="logo-sub">Powered by Claude API + pgvector + FastAPI</div>
              </div>
            </div>
          </div>
        </header>
        <main className="chat-area" style={{ display: "flex", alignItems: "center", justifyContent: "center" }}>
          <div style={{ width: 360, padding: "2rem", background: "#1e1e2e", borderRadius: 12, border: "1px solid #333" }}>
            <h2 style={{ marginTop: 0 }}>{authMode === "login" ? "Sign in" : "Create account"}</h2>
            <input
              className="chat-input"
              type="email"
              placeholder="Email"
              value={email}
              onChange={e => setEmail(e.target.value)}
              style={{ marginBottom: 8, display: "block", width: "100%", boxSizing: "border-box" }}
            />
            <input
              className="chat-input"
              type="password"
              placeholder="Password"
              value={password}
              onChange={e => setPassword(e.target.value)}
              onKeyDown={e => e.key === "Enter" && !authLoading && handleAuth()}
              style={{ marginBottom: 8, display: "block", width: "100%", boxSizing: "border-box" }}
            />
            {authError && <p style={{ color: "#f87171", margin: "4px 0 8px" }}>{authError}</p>}
            <button className="send-btn" onClick={handleAuth} disabled={authLoading} style={{ width: "100%" }}>
              {authLoading ? "..." : authMode === "login" ? "Sign in" : "Register"}
            </button>
            <p style={{ textAlign: "center", marginBottom: 0 }}>
              {authMode === "login" ? "No account? " : "Already registered? "}
              <button
                style={{ background: "none", border: "none", color: "#7c6af7", cursor: "pointer", padding: 0 }}
                onClick={() => { setAuthMode(authMode === "login" ? "register" : "login"); setAuthError(""); }}
              >
                {authMode === "login" ? "Register" : "Sign in"}
              </button>
            </p>
          </div>
        </main>
      </div>
    );
  }

  return (
    <div className="app">
      <header className="header">
        <div className="header-inner">
          <div className="logo">
            <span className="logo-icon">🤖</span>
            <div>
              <div className="logo-title">RAG Agent Platform</div>
              <div className="logo-sub">Powered by Claude API + pgvector + FastAPI</div>
            </div>
          </div>
          <div className="upload-area">
            <input
              type="file"
              accept=".pdf"
              ref={fileRef}
              style={{ display: "none" }}
              onChange={e => e.target.files?.[0] && uploadPDF(e.target.files[0])}
            />
            <button className="upload-btn" onClick={() => fileRef.current?.click()} disabled={uploading}>
              {uploading ? "Processing..." : "Upload PDF"}
            </button>
            {uploadedFile && <span className="upload-badge">{uploadedFile}</span>}
            <button
              style={{ background: "none", border: "1px solid #555", color: "#aaa", borderRadius: 6, padding: "6px 12px", cursor: "pointer", marginLeft: 8 }}
              onClick={logout}
            >
              Sign out
            </button>
          </div>
        </div>
      </header>

      <main className="chat-area">
        {messages.length === 0 && (
          <div className="empty-state">
            <div className="empty-icon">💬</div>
            <div className="empty-title">Upload a PDF and start asking questions</div>
            <div className="empty-sub">Powered by Claude AI — ask anything about your documents</div>
            <div className="example-questions">
              <div className="eq-label">Try asking:</div>
              {["What is this document about?", "Summarize the key points", "What are the main findings?"].map(q => (
                <button key={q} className="eq-btn" onClick={() => setQuestion(q)}>{q}</button>
              ))}
            </div>
          </div>
        )}
        {messages.map((msg, i) => (
          <div key={i} className={`message ${msg.role}`}>
            <div className="message-avatar">{msg.role === "user" ? "👤" : "🤖"}</div>
            <div className="message-content">{msg.content}</div>
          </div>
        ))}
      </main>

      <footer className="input-area">
        <div className="input-inner">
          <input
            className="chat-input"
            type="text"
            placeholder="Ask a question about your document..."
            value={question}
            onChange={e => setQuestion(e.target.value)}
            onKeyDown={e => e.key === "Enter" && !loading && askQuestion()}
            disabled={loading}
          />
          <button className="send-btn" onClick={askQuestion} disabled={loading || !question.trim()}>
            {loading ? "..." : "Send →"}
          </button>
        </div>
      </footer>
    </div>
  );
}

export default App;
