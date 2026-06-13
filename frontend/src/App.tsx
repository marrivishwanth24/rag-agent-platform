import { useState, useRef } from "react";
import ReactMarkdown from "react-markdown";
import "./App.css";

const API_URL = "https://rag-agent-platform-production.up.railway.app";

function getSessionId(): string {
  let id = localStorage.getItem("rag_session_id");
  if (!id) {
    id = crypto.randomUUID();
    localStorage.setItem("rag_session_id", id);
  }
  return id;
}

const SESSION_ID = getSessionId();

interface Citation {
  filename: string;
  page_num: number;
  similarity: number;
}

interface Message {
  role: "user" | "assistant";
  content: string;
  citations?: Citation[];
}

function App() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [question, setQuestion] = useState("");
  const [uploading, setUploading] = useState(false);
  const [uploadedDocs, setUploadedDocs] = useState<{ name: string; id: string }[]>([]);
  const [loading, setLoading] = useState(false);
  const fileRef = useRef<HTMLInputElement>(null);

  const documentIds = uploadedDocs.map(d => d.id);

  const removeDoc = (id: string) => {
    setUploadedDocs(prev => prev.filter(d => d.id !== id));
  };

  const uploadPDFs = async (files: File[]) => {
    setUploading(true);
    for (const file of files) {
      const formData = new FormData();
      formData.append("file", file);
      try {
        const res = await fetch(`${API_URL}/upload`, {
          method: "POST",
          headers: { "X-Session-ID": SESSION_ID },
          body: formData,
        });
        const data = await res.json();
        if (!res.ok) {
          setMessages(prev => [...prev, { role: "assistant", content: `Upload failed for **${file.name}**: ${data.detail}` }]);
          continue;
        }
        setUploadedDocs(prev => [...prev, { name: file.name, id: data.document_id }]);
        setMessages(prev => [...prev, {
          role: "assistant",
          content: `**${file.name}** uploaded and indexed.`,
        }]);
      } catch {
        setMessages(prev => [...prev, { role: "assistant", content: `Upload failed for **${file.name}**. Please check the server is running.` }]);
      }
    }
    setUploading(false);
  };

  const askQuestion = async () => {
    if (!question.trim()) return;
    const userMsg = question;
    setQuestion("");
    setMessages(prev => [...prev, { role: "user", content: userMsg }]);
    setLoading(true);
    setMessages(prev => [...prev, { role: "assistant", content: "⏳ Thinking..." }]);

    let assistantMsg = "";
    let citations: Citation[] = [];
    try {
      const res = await fetch(`${API_URL}/query`, {
        method: "POST",
        headers: { "Content-Type": "application/json", "X-Session-ID": SESSION_ID },
        body: JSON.stringify({ question: userMsg, document_ids: documentIds }),
      });

      const reader = res.body!.getReader();
      const decoder = new TextDecoder();
      let buffer = "";

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });

        // Split on SSE event boundary (\n\n) so newlines inside data fields are preserved
        const events = buffer.split("\n\n");
        buffer = events.pop() ?? ""; // keep incomplete last event for next iteration

        for (const event of events) {
          if (!event.startsWith("data: ") || event.includes("[DONE]")) continue;
          const payload = event.substring(6);
          if (payload.startsWith("[CITATIONS]")) {
            citations = JSON.parse(payload.substring(11));
          } else {
            assistantMsg += payload;
          }
        }
        setMessages(prev => {
          const updated = [...prev];
          updated[updated.length - 1] = { role: "assistant", content: assistantMsg || "⏳ Thinking...", citations };
          return updated;
        });
      }
    } catch {
      setMessages(prev => {
        const updated = [...prev];
        updated[updated.length - 1] = { role: "assistant", content: "❌ Error connecting to server." };
        return updated;
      });
    }
    setLoading(false);
  };

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
              multiple
              ref={fileRef}
              style={{ display: "none" }}
              onChange={e => {
                const files = Array.from(e.target.files ?? []);
                if (files.length > 0) uploadPDFs(files);
                e.target.value = "";
              }}
            />
            <button
              className="upload-btn"
              onClick={() => fileRef.current?.click()}
              disabled={uploading}
            >
              {uploading ? "⏳ Processing..." : "📄 Upload PDF"}
            </button>
          </div>
        </div>
        {uploadedDocs.length > 0 && (
          <div className="doc-badges">
            {uploadedDocs.map(doc => (
              <span key={doc.id} className="upload-badge">
                ✅ {doc.name}
                <button className="badge-remove" onClick={() => removeDoc(doc.id)} title="Remove">×</button>
              </span>
            ))}
          </div>
        )}
      </header>

      <main className="chat-area">
        {messages.length === 0 && (
          <div className="empty-state">
            <div className="empty-icon">💬</div>
            <div className="empty-title">Upload a PDF and start asking questions</div>
            <div className="empty-sub">Powered by Claude AI — ask anything about your documents</div>
            <div className="example-questions">
              <div className="eq-label">Try asking:</div>
              {[
                "What is this document about?",
                "Summarize the key points",
                "What are the main findings?",
              ].map(q => (
                <button key={q} className="eq-btn" onClick={() => setQuestion(q)}>{q}</button>
              ))}
            </div>
          </div>
        )}
        {messages.map((msg, i) => (
          <div key={i} className={`message ${msg.role}`}>
            <div className="message-avatar">{msg.role === "user" ? "👤" : "🤖"}</div>
            <div className="message-body">
              <div className="message-content">
                {msg.role === "assistant"
                  ? <ReactMarkdown>{msg.content}</ReactMarkdown>
                  : msg.content}
              </div>
              {msg.citations && msg.citations.length > 0 && (
                <div className="citations">
                  <span className="citations-label">Sources</span>
                  {msg.citations.map((c, j) => (
                    <span key={j} className="citation-chip">
                      <span className="citation-icon">📄</span>
                      <span className="citation-name" title={c.filename}>{c.filename.replace(/\.pdf$/i, "")}</span>
                      <span className="citation-page">p.{c.page_num}</span>
                      <span className="citation-score">{Math.round(c.similarity * 100)}%</span>
                    </span>
                  ))}
                </div>
              )}
            </div>
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
            onKeyPress={e => e.key === "Enter" && !loading && askQuestion()}
            disabled={loading}
          />
          <button className="send-btn" onClick={askQuestion} disabled={loading || !question.trim()}>
            {loading ? "⏳" : "Send →"}
          </button>
        </div>
      </footer>
    </div>
  );
}

export default App;
