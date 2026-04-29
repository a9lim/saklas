// saklas web UI — vanilla-JS baseline.
// Drives the analytics dashboard against the native /saklas/v1/* API
// and the WS co-stream.  The webui/ Svelte source can replace this
// in-place by emitting to the same dist/ directory.

const SESSION = "default";
const WS_URL = `${location.protocol === "https:" ? "wss" : "ws"}://${location.host}/saklas/v1/sessions/${SESSION}/stream`;
const API = `/saklas/v1/sessions/${SESSION}`;

// ---------------------------------------------------------------------
// Tiny DOM helper.  Avoids innerHTML so user-controlled strings can't
// inject markup, even though the API responses are server-controlled.
// ---------------------------------------------------------------------

function el(tag, opts = {}, children = []) {
  const node = document.createElement(tag);
  if (opts.className) node.className = opts.className;
  if (opts.text != null) node.textContent = String(opts.text);
  if (opts.title != null) node.title = String(opts.title);
  if (opts.style) Object.assign(node.style, opts.style);
  if (opts.value != null) node.value = opts.value;
  for (const child of children) {
    if (child == null) continue;
    node.appendChild(child);
  }
  return node;
}
function clear(node) {
  while (node.firstChild) node.removeChild(node.firstChild);
}

// ---------------------------------------------------------------------
// Session info bootstrap.
// ---------------------------------------------------------------------

const statusBar = document.getElementById("status-bar");
async function refreshSession() {
  try {
    const r = await fetch(API);
    if (!r.ok) throw new Error(`session: ${r.status}`);
    const info = await r.json();
    statusBar.textContent = `${info.model_id} · ${info.device}/${info.dtype} · vectors=${info.vectors.length} probes=${info.probes.length}`;
    return info;
  } catch (e) {
    statusBar.textContent = `error: ${e.message}`;
    return null;
  }
}

// ---------------------------------------------------------------------
// Chat panel.
// ---------------------------------------------------------------------

const chatLog = document.getElementById("chat-log");
const chatForm = document.getElementById("chat-form");
const chatInput = document.getElementById("chat-input");
const chatStop = document.getElementById("chat-stop");

let ws = null;
let assistantBuffer = null; // <div> being filled by streamed tokens
let tokenColumns = []; // [{text, thinking, scores}, ...] for the inspector
let probeOrder = []; // probe names in column order

function appendMessage(role, text) {
  const div = el("div", { className: `msg ${role}`, text });
  chatLog.appendChild(div);
  chatLog.scrollTop = chatLog.scrollHeight;
  return div;
}

function connectWs() {
  ws = new WebSocket(WS_URL);
  ws.addEventListener("open", () => {
    statusBar.textContent += " · ws connected";
  });
  ws.addEventListener("close", () => {
    if (statusBar.textContent.includes("ws connected")) {
      statusBar.textContent = statusBar.textContent.replace(" · ws connected", "") + " · ws closed";
    }
  });
  ws.addEventListener("error", () => {
    statusBar.textContent += " · ws error";
  });
  ws.addEventListener("message", (ev) => {
    let msg;
    try { msg = JSON.parse(ev.data); } catch { return; }
    handleWsMessage(msg);
  });
}

function handleWsMessage(msg) {
  if (msg.type === "started") {
    assistantBuffer = appendMessage("assistant", "");
    tokenColumns = [];
    probeOrder = [];
    return;
  }
  if (msg.type === "token") {
    if (!assistantBuffer) return;
    const text = msg.text || "";
    if (msg.thinking) assistantBuffer.classList.add("thinking");
    assistantBuffer.textContent += text;
    chatLog.scrollTop = chatLog.scrollHeight;
    if (msg.per_layer_scores) {
      tokenColumns.push({
        text,
        thinking: !!msg.thinking,
        scores: msg.per_layer_scores,
      });
      if (probeOrder.length === 0) {
        const firstLayer = Object.keys(msg.per_layer_scores)[0];
        probeOrder = Object.keys(msg.per_layer_scores[firstLayer]).sort();
      }
      renderInspector();
    }
    return;
  }
  if (msg.type === "done") {
    assistantBuffer = null;
    refreshCorrelation();
    return;
  }
  if (msg.type === "error") {
    appendMessage("system", `error: ${msg.message}`);
    assistantBuffer = null;
    return;
  }
}

chatForm.addEventListener("submit", (e) => {
  e.preventDefault();
  const text = chatInput.value.trim();
  if (!text) return;
  appendMessage("user", text);
  if (!ws || ws.readyState !== WebSocket.OPEN) connectWs();
  const send = () => ws.send(JSON.stringify({
    type: "generate",
    input: text,
    sampling: { max_tokens: 256 },
  }));
  if (ws.readyState === WebSocket.OPEN) send();
  else ws.addEventListener("open", send, { once: true });
  chatInput.value = "";
});

chatStop.addEventListener("click", () => {
  if (ws && ws.readyState === WebSocket.OPEN) {
    ws.send(JSON.stringify({ type: "stop" }));
  }
});

// ---------------------------------------------------------------------
// Inspector: per-token × per-layer × per-probe heatmap.
// ---------------------------------------------------------------------

const inspectorGrid = document.getElementById("inspector-grid");
const inspectorEmpty = document.getElementById("inspector-empty");

function renderInspector() {
  if (tokenColumns.length === 0 || probeOrder.length === 0) {
    inspectorEmpty.style.display = "";
    inspectorGrid.style.display = "none";
    return;
  }
  inspectorEmpty.style.display = "none";
  inspectorGrid.style.display = "grid";

  const layerSet = new Set();
  for (const col of tokenColumns) {
    for (const layer of Object.keys(col.scores)) layerSet.add(parseInt(layer, 10));
  }
  const layers = [...layerSet].sort((a, b) => a - b);

  const cols = tokenColumns.length;
  inspectorGrid.style.gridTemplateColumns = `auto repeat(${cols * probeOrder.length}, 14px)`;

  clear(inspectorGrid);
  // Top label row.
  inspectorGrid.appendChild(el("div", { className: "row-label" }));
  for (let t = 0; t < cols; t++) {
    const tokenText = (tokenColumns[t].text || "·").replace(/\s/g, "·").slice(0, 4);
    for (let p = 0; p < probeOrder.length; p++) {
      const tip = `${tokenColumns[t].text} / ${probeOrder[p]}`;
      inspectorGrid.appendChild(el("div", {
        className: "col-label", text: tokenText, title: tip,
      }));
    }
  }
  // Body rows.
  for (const layer of layers) {
    inspectorGrid.appendChild(el("div", { className: "row-label", text: `L${layer}` }));
    for (let t = 0; t < cols; t++) {
      const layerScores = tokenColumns[t].scores[String(layer)] || {};
      for (let p = 0; p < probeOrder.length; p++) {
        const v = layerScores[probeOrder[p]] ?? 0;
        inspectorGrid.appendChild(el("div", {
          className: "cell",
          style: { background: scoreToRgb(v) },
          title: `L${layer} · ${probeOrder[p]} · ${v.toFixed(3)}`,
        }));
      }
    }
  }
}

function scoreToRgb(v) {
  // -1 → red, 0 → black, +1 → green
  const a = Math.max(-1, Math.min(1, v));
  if (a >= 0) return `rgb(0, ${Math.round(a * 200)}, 0)`;
  return `rgb(${Math.round(-a * 200)}, 0, 0)`;
}

// ---------------------------------------------------------------------
// Correlation matrix.
// ---------------------------------------------------------------------

const corrGrid = document.getElementById("correlation-grid");
const corrRefresh = document.getElementById("corr-refresh");

function showHint(node, text) {
  clear(node);
  node.appendChild(el("div", { className: "hint", text }));
}

async function refreshCorrelation() {
  try {
    const r = await fetch(`${API}/correlation`);
    if (!r.ok) {
      showHint(corrGrid, `${r.status}: no vectors loaded`);
      return;
    }
    const data = await r.json();
    renderCorrelation(data);
  } catch (e) {
    showHint(corrGrid, e.message);
  }
}

function renderCorrelation(data) {
  const names = data.names || [];
  if (!names.length) {
    showHint(corrGrid, "no vectors loaded");
    return;
  }
  const n = names.length;
  corrGrid.style.gridTemplateColumns = `auto repeat(${n}, 18px)`;
  clear(corrGrid);
  corrGrid.appendChild(el("div"));
  for (const name of names) {
    corrGrid.appendChild(el("div", { className: "col-label", text: name, title: name }));
  }
  for (const a of names) {
    corrGrid.appendChild(el("div", { className: "row-label", text: a, title: a }));
    for (const b of names) {
      const v = data.matrix[a] ? data.matrix[a][b] : null;
      const rgb = v === null || v === undefined ? "#161b22" : scoreToRgb(v);
      const label = v === null || v === undefined ? "—" : v.toFixed(2);
      corrGrid.appendChild(el("div", {
        className: "cell",
        style: { background: rgb },
        text: label,
        title: `${a} vs ${b}: ${label}`,
      }));
    }
  }
}

corrRefresh.addEventListener("click", refreshCorrelation);

// ---------------------------------------------------------------------
// Layer norms panel.
// ---------------------------------------------------------------------

const vectorSelect = document.getElementById("vector-select");
const layerNormsBox = document.getElementById("layer-norms");

async function refreshVectors() {
  try {
    const r = await fetch(`${API}/vectors`);
    if (!r.ok) return;
    const data = await r.json();
    const names = (data.vectors || []).map((v) => v.name).sort();
    const previous = vectorSelect.value;
    clear(vectorSelect);
    for (const n of names) {
      const opt = el("option", { value: n, text: n });
      vectorSelect.appendChild(opt);
    }
    if (names.includes(previous)) vectorSelect.value = previous;
    if (vectorSelect.value) renderLayerNorms(vectorSelect.value);
  } catch {/* ignore */}
}

async function renderLayerNorms(name) {
  showHint(layerNormsBox, "loading…");
  try {
    const r = await fetch(`${API}/vectors/${encodeURIComponent(name)}`);
    if (!r.ok) {
      showHint(layerNormsBox, `${r.status}`);
      return;
    }
    const data = await r.json();
    const norms = data.per_layer_norms || {};
    const layers = Object.keys(norms).map((k) => parseInt(k, 10)).sort((a, b) => a - b);
    if (!layers.length) {
      showHint(layerNormsBox, "no layers");
      return;
    }
    const max = Math.max(...layers.map((l) => norms[String(l)]));
    clear(layerNormsBox);
    for (const l of layers) {
      const v = norms[String(l)];
      const w = max > 0 ? Math.round((v / max) * 140) : 0;
      const row = el("div", { className: "row" }, [
        el("span", { className: "label", text: `L${l}` }),
        el("span", { className: "bar", style: { width: `${w}px` } }),
        el("span", { className: "value", text: v.toFixed(3) }),
      ]);
      layerNormsBox.appendChild(row);
    }
  } catch (e) {
    showHint(layerNormsBox, e.message);
  }
}

vectorSelect.addEventListener("change", () => {
  if (vectorSelect.value) renderLayerNorms(vectorSelect.value);
});

// ---------------------------------------------------------------------
// Boot.
// ---------------------------------------------------------------------

(async () => {
  await refreshSession();
  connectWs();
  refreshCorrelation();
  refreshVectors();
})();
