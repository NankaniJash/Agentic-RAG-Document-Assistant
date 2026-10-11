const API_BASE = "/api";
const SESSION_KEY = "synapse_session_id";

function getSessionId() {
  return localStorage.getItem(SESSION_KEY) || null;
}

function setSessionId(id) {
  localStorage.setItem(SESSION_KEY, id);
}

const fileInput = document.getElementById("file-input");
const uploadStatus = document.getElementById("upload-status");
const docList = document.getElementById("doc-list");
const messagesEl = document.getElementById("messages");
const chatForm = document.getElementById("chat-form");
const chatInput = document.getElementById("chat-input");
const sendBtn = document.getElementById("send-btn");

function escapeHtml(str) {
  const div = document.createElement("div");
  div.textContent = str;
  return div.innerHTML;
}

async function loadDocuments() {
  const res = await fetch(`${API_BASE}/documents`);
  const docs = await res.json();
  docList.innerHTML = "";
  if (docs.length === 0) {
    docList.innerHTML = `<li class="doc-item" style="color:var(--text-dim)">No documents yet</li>`;
    return;
  }
  for (const doc of docs) {
    const li = document.createElement("li");
    li.className = "doc-item";
    li.innerHTML = `<span class="doc-name" title="${escapeHtml(doc.filename)}">${escapeHtml(doc.filename)}</span>
      <button class="doc-remove" title="Remove">✕</button>`;
    li.querySelector(".doc-remove").addEventListener("click", async () => {
      await fetch(`${API_BASE}/documents/${doc.id}`, { method: "DELETE" });
      loadDocuments();
    });
    docList.appendChild(li);
  }
}

fileInput.addEventListener("change", async () => {
  const file = fileInput.files[0];
  if (!file) return;
  uploadStatus.textContent = `Ingesting ${file.name}...`;
  const formData = new FormData();
  formData.append("file", file);
  try {
    const res = await fetch(`${API_BASE}/documents/upload`, { method: "POST", body: formData });
    if (!res.ok) {
      const err = await res.json();
      throw new Error(err.detail || err.error || "Upload failed");
    }
    const result = await res.json();
    uploadStatus.textContent = `Indexed ${result.chunks} chunks from ${result.filename}`;
    loadDocuments();
  } catch (e) {
    uploadStatus.textContent = `Error: ${e.message}`;
  } finally {
    fileInput.value = "";
  }
});

function clearEmptyState() {
  const empty = messagesEl.querySelector(".empty-state");
  if (empty) empty.remove();
}

function addUserMessage(text) {
  clearEmptyState();
  const div = document.createElement("div");
  div.className = "msg user";
  div.innerHTML = `<div class="bubble">${escapeHtml(text)}</div>`;
  messagesEl.appendChild(div);
  messagesEl.scrollTop = messagesEl.scrollHeight;
}

function addAssistantMessage() {
  clearEmptyState();
  const div = document.createElement("div");
  div.className = "msg assistant";
  div.innerHTML = `
    <div class="status-line">Working...</div>
    <div class="bubble" style="display:none"></div>
    <div class="sources"></div>
    <button class="trace-toggle" style="display:none">Show reasoning trace ▾</button>
    <div class="trace-panel" style="display:none"></div>`;
  messagesEl.appendChild(div);
  messagesEl.scrollTop = messagesEl.scrollHeight;
  return div;
}

function renderTrace(trace) {
  const parts = [];

  if (trace.plan) {
    parts.push(`<div class="trace-section">
      <div class="trace-heading">Research plan</div>
      <ol class="trace-plan">${trace.plan.map((q) => `<li>${escapeHtml(q)}</li>`).join("")}</ol>
    </div>`);
  }

  if (trace.retrieval.length) {
    parts.push(`<div class="trace-section">
      <div class="trace-heading">Evidence retrieved</div>
      ${trace.retrieval
        .map(
          (r) => `<div class="trace-subq">
            <div class="trace-subq-title">${escapeHtml(r.sub_question)}</div>
            ${r.chunks
              .map(
                (c) =>
                  `<div class="trace-chunk"><b>${escapeHtml(c.filename)}</b> · p.${c.page}<br>${escapeHtml(c.snippet)}...</div>`
              )
              .join("") || `<div class="trace-chunk trace-empty">No relevant chunks found</div>`}
          </div>`
        )
        .join("")}
    </div>`);
  }

  if (trace.draft) {
    parts.push(`<div class="trace-section">
      <div class="trace-heading">Draft answer (pre-critique)</div>
      <div class="trace-text">${escapeHtml(trace.draft)}</div>
    </div>`);
  }

  if (trace.critique) {
    parts.push(`<div class="trace-section">
      <div class="trace-heading">Critic review</div>
      <div class="trace-text">${escapeHtml(trace.critique)}</div>
    </div>`);
  }

  return parts.join("");
}

async function sendMessage(query) {
  addUserMessage(query);
  const msgEl = addAssistantMessage();
  const statusEl = msgEl.querySelector(".status-line");
  const bubbleEl = msgEl.querySelector(".bubble");
  const sourcesEl = msgEl.querySelector(".sources");
  const traceToggle = msgEl.querySelector(".trace-toggle");
  const tracePanel = msgEl.querySelector(".trace-panel");

  sendBtn.disabled = true;
  let answerText = "";
  const trace = { plan: null, retrieval: [], draft: null, critique: null };
  let traceExpanded = false;

  traceToggle.addEventListener("click", () => {
    traceExpanded = !traceExpanded;
    tracePanel.style.display = traceExpanded ? "block" : "none";
    traceToggle.textContent = traceExpanded ? "Hide reasoning trace ▴" : "Show reasoning trace ▾";
  });

  try {
    const res = await fetch(`${API_BASE}/chat`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ query, session_id: getSessionId() }),
    });

    if (res.status === 429) {
      statusEl.textContent = "Rate limit reached — please wait a bit before asking again.";
      return;
    }

    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";

    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });

      const events = buffer.split("\n\n");
      buffer = events.pop();

      for (const raw of events) {
        const line = raw.trim();
        if (!line.startsWith("data:")) continue;
        const payload = JSON.parse(line.slice(5).trim());

        if (payload.type === "session") {
          setSessionId(payload.data);
        } else if (payload.type === "status") {
          statusEl.textContent = payload.data;
        } else if (payload.type === "token") {
          if (bubbleEl.style.display === "none") {
            statusEl.remove();
            bubbleEl.style.display = "block";
          }
          answerText += payload.data;
          bubbleEl.textContent = answerText;
        } else if (payload.type === "sources") {
          sourcesEl.innerHTML = payload.data
            .map(
              (s) =>
                `<span class="source-chip" title="${escapeHtml(s.snippet)}"><b>[${s.index}]</b> ${escapeHtml(s.filename)} · p.${s.page}</span>`
            )
            .join("");
        } else if (payload.type === "trace") {
          if (payload.stage === "plan") trace.plan = payload.data.sub_questions;
          else if (payload.stage === "retrieval") trace.retrieval.push(payload.data);
          else if (payload.stage === "draft") trace.draft = payload.data.text;
          else if (payload.stage === "critique") trace.critique = payload.data.text;

          traceToggle.style.display = "inline-block";
          tracePanel.innerHTML = renderTrace(trace);
        }
        messagesEl.scrollTop = messagesEl.scrollHeight;
      }
    }
  } catch (e) {
    statusEl.textContent = `Error: ${e.message}`;
  } finally {
    sendBtn.disabled = false;
  }
}

chatForm.addEventListener("submit", (e) => {
  e.preventDefault();
  const query = chatInput.value.trim();
  if (!query) return;
  chatInput.value = "";
  sendMessage(query);
});

loadDocuments();
