import { useState, useRef, useEffect } from "react";
import ReactMarkdown from "react-markdown";
import "./App.css";

// ── Eval Panel ────────────────────────────────────────────────────────────────

interface EvalScores {
  faithfulness: number;
  answer_relevance: number;
  context_quality: number;
  overall: number;
}

interface EvalResult {
  question: string;
  answer: string;
  scores: EvalScores;
  sources: { filename: string; page_num: number; similarity: number }[];
}

function ScoreBar({ label, value }: { label: string; value: number }) {
  const pct = Math.round(value * 100);
  const color = pct >= 75 ? "#CAFF33" : pct >= 50 ? "#FFD700" : "#FF6B6B";
  return (
    <div className="score-row">
      <span className="score-label">{label}</span>
      <div className="score-bar-track">
        <div className="score-bar-fill" style={{ width: `${pct}%`, background: color }} />
      </div>
      <span className="score-value" style={{ color }}>{pct}%</span>
    </div>
  );
}

function EvalPanel({
  documentIds,
  onClose,
}: {
  documentIds: string[];
  onClose: () => void;
}) {
  const [q, setQ] = useState("");
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<EvalResult | null>(null);
  const [error, setError] = useState<string | null>(null);

  const run = async () => {
    if (!q.trim()) return;
    setLoading(true);
    setResult(null);
    setError(null);
    try {
      const res = await fetch(`${API_URL}/eval`, {
        method: "POST",
        headers: { "Content-Type": "application/json", "X-Session-ID": SESSION_ID },
        body: JSON.stringify({ question: q, document_ids: documentIds }),
      });
      if (!res.ok) {
        const d = await res.json();
        setError(d.detail ?? "Eval failed");
      } else {
        setResult(await res.json());
      }
    } catch {
      setError("Could not reach server.");
    }
    setLoading(false);
  };

  return (
    <div className="eval-overlay" onClick={onClose}>
      <div className="eval-panel" onClick={e => e.stopPropagation()}>
        <div className="eval-header">
          <div>
            <div className="eval-title">Eval Harness</div>
            <div className="eval-sub">LLM-as-judge · measures faithfulness, relevance & context quality</div>
          </div>
          <button className="eval-close" onClick={onClose}>×</button>
        </div>

        <div className="eval-input-row">
          <input
            className="eval-input"
            placeholder="Enter a question to evaluate…"
            value={q}
            onChange={e => setQ(e.target.value)}
            onKeyDown={e => e.key === "Enter" && !loading && run()}
          />
          <button className="eval-run-btn" onClick={run} disabled={loading || !q.trim()}>
            {loading ? <span className="send-spinner" /> : "Run Eval"}
          </button>
        </div>

        {error && <div className="eval-error">{error}</div>}

        {result && (
          <div className="eval-results">
            <div className="eval-scores">
              <ScoreBar label="Faithfulness" value={result.scores.faithfulness} />
              <ScoreBar label="Answer Relevance" value={result.scores.answer_relevance} />
              <ScoreBar label="Context Quality" value={result.scores.context_quality} />
              <div className="eval-divider" />
              <ScoreBar label="Overall" value={result.scores.overall} />
            </div>

            <div className="eval-answer-label">Answer</div>
            <div className="eval-answer">
              <ReactMarkdown>{result.answer}</ReactMarkdown>
            </div>

            {result.sources.length > 0 && (
              <div className="eval-sources">
                <div className="eval-answer-label">Sources used</div>
                {result.sources.map((s, i) => (
                  <span key={i} className="citation-chip">
                    <span className="citation-icon">📄</span>
                    <span className="citation-name">{s.filename.replace(/\.pdf$/i, "")}</span>
                    <span className="citation-page">p.{s.page_num}</span>
                    <span className="citation-score">{Math.round(s.similarity * 100)}%</span>
                  </span>
                ))}
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}

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

const THINKING = "__THINKING__";

function TypingIndicator() {
  return (
    <div className="typing-indicator">
      <span /><span /><span />
    </div>
  );
}

function App() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [question, setQuestion] = useState("");
  const [uploading, setUploading] = useState(false);
  const [uploadedDocs, setUploadedDocs] = useState<{ name: string; id: string }[]>([]);
  const [loading, setLoading] = useState(false);
  const [dragging, setDragging] = useState(false);
  const [evalOpen, setEvalOpen] = useState(false);

  const fileRef = useRef<HTMLInputElement>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const bottomRef = useRef<HTMLDivElement>(null);

  const documentIds = uploadedDocs.map(d => d.id);

  // Auto-scroll to bottom whenever messages update
  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  const autoResize = () => {
    const ta = textareaRef.current;
    if (!ta) return;
    ta.style.height = "auto";
    ta.style.height = `${Math.min(ta.scrollHeight, 160)}px`;
  };

  const resetTextarea = () => {
    setQuestion("");
    if (textareaRef.current) textareaRef.current.style.height = "auto";
  };

  const newChat = () => {
    setMessages([]);
    setUploadedDocs([]);
  };

  const removeDoc = (id: string) => {
    setUploadedDocs(prev => prev.filter(d => d.id !== id));
  };

  const uploadPDFs = async (files: File[]) => {
    const pdfs = files.filter(f => f.name.toLowerCase().endsWith(".pdf"));
    if (!pdfs.length) return;
    setUploading(true);
    for (const file of pdfs) {
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
        setMessages(prev => [...prev, { role: "assistant", content: `**${file.name}** uploaded and indexed.` }]);
      } catch {
        setMessages(prev => [...prev, { role: "assistant", content: `Upload failed for **${file.name}**. Please check the server is running.` }]);
      }
    }
    setUploading(false);
  };

  const askQuestion = async () => {
    if (!question.trim() || loading) return;
    const userMsg = question;
    resetTextarea();
    setMessages(prev => [...prev, { role: "user", content: userMsg }]);
    setLoading(true);
    setMessages(prev => [...prev, { role: "assistant", content: THINKING }]);

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

        const events = buffer.split("\n\n");
        buffer = events.pop() ?? "";

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
          updated[updated.length - 1] = { role: "assistant", content: assistantMsg || THINKING, citations };
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

  const handleDragOver = (e: React.DragEvent) => { e.preventDefault(); setDragging(true); };
  const handleDragLeave = () => setDragging(false);
  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setDragging(false);
    uploadPDFs(Array.from(e.dataTransfer.files));
  };

  return (
    <div className="app">
      {evalOpen && (
        <EvalPanel documentIds={documentIds} onClose={() => setEvalOpen(false)} />
      )}
      <header className="header">
        <div className="header-inner">
          <div className="logo">
            <span className="logo-icon">🤖</span>
            <div>
              <div className="logo-title">RAG Agent Platform</div>
              <div className="logo-sub">Claude API · pgvector · FastAPI</div>
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
                uploadPDFs(Array.from(e.target.files ?? []));
                e.target.value = "";
              }}
            />
            {messages.length > 0 && (
              <button className="new-chat-btn" onClick={newChat} title="New chat">
                ✦ New Chat
              </button>
            )}
            <button className="eval-btn" onClick={() => setEvalOpen(true)} title="Open eval harness">
              ⚡ Eval
            </button>
            <button className="upload-btn" onClick={() => fileRef.current?.click()} disabled={uploading}>
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

      <main
        className={`chat-area${dragging ? " dragging" : ""}`}
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onDrop={handleDrop}
      >
        {messages.length === 0 && (
          <div className="empty-state">
            <div className="empty-icon">💬</div>
            <div className="empty-title">Upload a PDF and start asking questions</div>
            <div className="empty-sub">Drag a PDF anywhere onto this page, or click Upload PDF</div>
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
                {msg.content === THINKING
                  ? <TypingIndicator />
                  : msg.role === "assistant"
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

        {dragging && (
          <div className="drop-overlay">
            <div className="drop-label">📄 Drop PDFs to upload</div>
          </div>
        )}

        <div ref={bottomRef} />
      </main>

      <footer className="input-area">
        <div className="input-inner">
          <textarea
            ref={textareaRef}
            className="chat-input"
            rows={1}
            placeholder="Ask a question… (Shift+Enter for new line)"
            value={question}
            onChange={e => { setQuestion(e.target.value); autoResize(); }}
            onKeyDown={e => {
              if (e.key === "Enter" && !e.shiftKey) {
                e.preventDefault();
                askQuestion();
              }
            }}
            disabled={loading}
          />
          <button className="send-btn" onClick={askQuestion} disabled={loading || !question.trim()}>
            {loading ? <span className="send-spinner" /> : "↑"}
          </button>
        </div>
      </footer>
    </div>
  );
}

export default App;
