// Thin fetch helpers. All paths are relative so the app works under any URL prefix.

export class ApiError extends Error {
  constructor(status, message, data) {
    super(message);
    this.status = status;
    this.data = data;
  }
}

async function parse(res) {
  const text = await res.text();
  let data = null;
  try { data = text ? JSON.parse(text) : null; } catch { data = { error: text }; }
  if (!res.ok) {
    const msg = (data && (data.error || (data.detail && (data.detail.error || data.detail)))) || res.statusText;
    throw new ApiError(res.status, typeof msg === "string" ? msg : JSON.stringify(msg), data);
  }
  return data;
}

export const getJSON = (path) => fetch(path, { headers: { accept: "application/json" } }).then(parse);
export const postJSON = (path, body) => fetch(path, {
  method: "POST", headers: { "content-type": "application/json", accept: "application/json" },
  body: JSON.stringify(body ?? {}),
}).then(parse);
export const del = (path) => fetch(path, { method: "DELETE", headers: { accept: "application/json" } }).then(parse);

/**
 * POST the prompt and consume the SSE response incrementally.
 * onEvent(type, data) is called per frame; resolves when the stream ends.
 */
export async function streamChat(prompt, { signal, onEvent }) {
  const res = await fetch("api/chat", {
    method: "POST", signal,
    headers: { "content-type": "application/json", accept: "text/event-stream" },
    body: JSON.stringify({ prompt }),
  });
  if (!res.ok) await parse(res); // throws ApiError with the server's message
  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buf = "";
  const dispatch = (block) => {
    let type = "message";
    const dataLines = [];
    for (const line of block.split("\n")) {
      if (line.startsWith(":")) continue;             // keepalive comment
      if (line.startsWith("event:")) type = line.slice(6).trim();
      else if (line.startsWith("data:")) dataLines.push(line.slice(5).replace(/^ /, ""));
    }
    if (!dataLines.length) return;
    let data;
    try { data = JSON.parse(dataLines.join("\n")); } catch { return; }
    onEvent(type, data);
  };
  for (;;) {
    const { value, done } = await reader.read();
    if (done) break;
    buf += decoder.decode(value, { stream: true });
    let idx;
    while ((idx = buf.indexOf("\n\n")) >= 0) {
      dispatch(buf.slice(0, idx));
      buf = buf.slice(idx + 2);
    }
  }
  if (buf.trim()) dispatch(buf);
}
