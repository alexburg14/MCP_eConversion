// Markdown -> sanitized HTML, plus a render-throttle for streaming text.

const md = window.marked;
const purify = window.DOMPurify;
if (md) md.setOptions({ gfm: true, breaks: false });
if (purify) {
  // external links open in a new tab and never leak the referrer
  purify.addHook("afterSanitizeAttributes", (node) => {
    if (node.tagName === "A" && node.getAttribute("href")) {
      node.setAttribute("target", "_blank");
      node.setAttribute("rel", "noopener noreferrer");
    }
  });
}

export function render(text) {
  const raw = md ? md.parse(text || "") : escapeHtml(text || "").replace(/\n/g, "<br>");
  return purify ? purify.sanitize(raw, { USE_PROFILES: { html: true } }) : escapeHtml(text || "");
}

export function escapeHtml(s) {
  return String(s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}

/** Coalesces many small updates into one DOM write per animation frame. */
export class StreamRenderer {
  constructor(el, { cursor = true } = {}) {
    this.el = el;
    this.text = "";
    this.cursor = cursor;
    this._raf = 0;
    if (cursor) el.classList.add("cursor");
  }
  append(piece) {
    this.text += piece;
    if (!this._raf) this._raf = requestAnimationFrame(() => { this._raf = 0; this.el.innerHTML = render(this.text); });
  }
  finish(text) {
    if (this._raf) { cancelAnimationFrame(this._raf); this._raf = 0; }
    if (text != null) this.text = text;
    this.el.innerHTML = render(this.text);
    this.el.classList.remove("cursor");
  }
}
