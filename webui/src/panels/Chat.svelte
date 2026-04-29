<script lang="ts">
  import { connectWs, type WsMessage } from "../lib/api";
  import { inspectorTokens, refreshCorrelation } from "../lib/stores";

  interface ChatTurn {
    role: "user" | "assistant" | "system";
    text: string;
    thinking?: boolean;
  }

  let log: ChatTurn[] = [];
  let pendingAssistant: ChatTurn | null = null;
  let input = "";
  let ws: WebSocket | null = null;

  function ensureWs(): WebSocket {
    if (!ws || ws.readyState === WebSocket.CLOSED) {
      ws = connectWs();
      ws.addEventListener("message", (ev) => {
        let msg: WsMessage;
        try { msg = JSON.parse(ev.data); } catch { return; }
        handleMsg(msg);
      });
    }
    return ws;
  }

  function handleMsg(msg: WsMessage) {
    if (msg.type === "started") {
      pendingAssistant = { role: "assistant", text: "" };
      log = [...log, pendingAssistant];
      inspectorTokens.set([]);
      return;
    }
    if (msg.type === "token") {
      if (!pendingAssistant) return;
      pendingAssistant.text += msg.text;
      if (msg.thinking) pendingAssistant.thinking = true;
      log = [...log];
      if (msg.per_layer_scores) {
        inspectorTokens.update((cur) => [
          ...cur,
          { text: msg.text, thinking: msg.thinking, scores: msg.per_layer_scores! },
        ]);
      }
      return;
    }
    if (msg.type === "done") {
      pendingAssistant = null;
      refreshCorrelation();
      return;
    }
    if (msg.type === "error") {
      log = [...log, { role: "system", text: `error: ${msg.message}` }];
      pendingAssistant = null;
      return;
    }
  }

  function send() {
    const text = input.trim();
    if (!text) return;
    log = [...log, { role: "user", text }];
    input = "";
    const sock = ensureWs();
    const payload = JSON.stringify({
      type: "generate",
      input: text,
      sampling: { max_tokens: 256 },
    });
    if (sock.readyState === WebSocket.OPEN) sock.send(payload);
    else sock.addEventListener("open", () => sock.send(payload), { once: true });
  }

  function stop() {
    if (ws && ws.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify({ type: "stop" }));
    }
  }
</script>

<div class="header">CHAT</div>
<div class="log">
  {#each log as turn}
    <div class="msg {turn.role}" class:thinking={turn.thinking}>{turn.text}</div>
  {/each}
</div>
<form class="input" on:submit|preventDefault={send}>
  <input bind:value={input} placeholder="message…" />
  <button type="submit">send</button>
  <button type="button" on:click={stop}>stop</button>
</form>

<style>
  .header { font-size: 0.85em; color: #7d8590; letter-spacing: 0.1em; border-bottom: 1px solid #21262d; padding-bottom: 0.3em; margin-bottom: 0.5em; text-transform: uppercase; }
  .log { flex: 1 1 auto; overflow-y: auto; display: flex; flex-direction: column; gap: 0.6em; min-height: 0; }
  .msg { white-space: pre-wrap; word-break: break-word; border-left: 2px solid #30363d; padding-left: 0.6em; }
  .msg.user { border-color: #58a6ff; color: #c9d1d9; }
  .msg.assistant { border-color: #7ee787; }
  .msg.system { border-color: #f85149; color: #ff7b72; font-size: 0.85em; }
  .msg.thinking { color: #8b949e; opacity: 0.75; font-style: italic; }
  .input { display: flex; gap: 0.4em; margin-top: 0.5em; }
  .input input { flex: 1; background: #161b22; color: #e6edf3; border: 1px solid #30363d; padding: 0.4em 0.6em; font: inherit; }
  .input button { background: #21262d; color: #c9d1d9; border: 1px solid #30363d; padding: 0.4em 0.8em; cursor: pointer; font: inherit; }
  .input button:hover { background: #30363d; }
  .input button[type="submit"] { color: #7ee787; }
</style>
