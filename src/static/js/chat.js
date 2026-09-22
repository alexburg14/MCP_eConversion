// Chat view: history, streaming assistant turns with live tool steps, composer.
import { streamChat, postJSON } from "./api.js";
import { StreamRenderer } from "./markdown.js";
import { renderSettingsRow, toast } from "./settings.js";
import { refreshSession } from "./app.js";

const SEND_ICON = "➤";
const STOP_ICON = "■";

function el(tag, cls, text) {
  const e = document.createElement(tag);
  if (cls) e.className = cls;
  if (text != null) e.textContent = text;
  return e;
}

function shortArgs(args) {
  const s = JSON.stringify(args ?? {});
  return s.length > 90 ? s.slice(0, 87) + "…" : s;
}

function fmtSeconds(s) {
  return s == null ? "" : `researched for ${Number(s).toFixed(1)} s`;
}

function svgUse(id, cls = "ico") {
  const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
  svg.setAttribute("class", cls);
  svg.innerHTML = `<use href="#${id}"/>`;
  return svg;
}

function copyButton(getText) {
  const b = el("button", "icon-btn");
  b.type = "button";
  b.title = "Copy answer as markdown";
  b.append(svgUse("i-copy"));
  b.addEventListener("click", async () => {
    try {
      await navigator.clipboard.writeText(getText());
      b.replaceChildren(svgUse("i-check")); b.classList.add("ok");
    } catch { b.title = "copy failed"; }
    setTimeout(() => { b.replaceChildren(svgUse("i-copy")); b.classList.remove("ok"); }, 1500);
  });
  return b;
}

/** One assistant message: steps above, markdown body, footer with summary + copy. */
class AssistantMessage {
  constructor(list) {
    this.root = el("div", "msg assistant");
    // Perplexity-style: the "Researched …" line sits above the answer
    this.head = el("details", "researched live empty");
    this.headSummary = el("summary");
    this.headSummary.append(svgUse("i-research"), el("span", "researched-text", "Researching…"), el("span", "chev", "▾"));
    this.head.append(this.headSummary);
    this.steps = el("div", "steps");
    this.head.append(this.steps);
    this.root.append(this.head);
    // A real click on the disclosure triangle (not our own JS toggling it) means
    // the user has an opinion — stop auto-opening/closing this message's dropdown.
    this.userToggled = false;
    this.headSummary.addEventListener("click", () => { this.userToggled = true; });
    this.bubble = el("div", "bubble");
    this.md = el("div", "md");
    this.bubble.append(this.md);
    this.foot = el("div", "msg-foot");
    this.root.append(this.bubble, this.foot);
    list.append(this.root);
    this.renderer = new StreamRenderer(this.md);
    this.toolSteps = new Map();
    this.think = null;
    this.thinkText = "";
    this.text = "";
    this.toolCount = 0;
  }

  static fromHistory(list, msg) {
    const a = new AssistantMessage(list);
    a.renderer.finish(msg.content || "");
    a.text = msg.content || "";
    const meta = msg.meta || {};
    a.summary(meta.tool_calls || [], meta.elapsed);
    return a;
  }

  // Live steps live inside the collapsible "Researching…" dropdown; open it as
  // soon as there's something to show, so the user watches progress without
  // the page filling up with permanently-visible cards -- unless they've
  // already clicked it themselves, in which case that choice sticks.
  openHead() {
    this.head.classList.remove("empty");
    if (!this.userToggled) this.head.open = true;
  }

  ensureThink() {
    if (this.think) return this.think;
    this.openHead();
    const d = el("details", "step think running");
    d.open = true;
    const s = el("summary");
    s.append(el("span", "st-ico", "◌"), el("span", "st-name", "Thinking…"));
    const body = el("div", "st-body");
    d.append(s, body);
    this.steps.prepend(d);
    this.think = { root: d, body, summary: s };
    return this.think;
  }

  reasoning(piece) {
    const t = this.ensureThink();
    this.thinkText += piece;
    t.body.textContent = this.thinkText;
    t.body.scrollTop = t.body.scrollHeight;
  }

  closeThink() {
    if (!this.think || !this.think.root.classList.contains("running")) return;
    this.think.root.classList.remove("running");
    this.think.root.open = false;
    this.think.summary.querySelector(".st-name").textContent = "Thought process";
    this.think.summary.querySelector(".st-ico").textContent = "✓";
  }

  appendText(piece) {
    this.closeThink();
    this.text += piece;
    this.renderer.append(piece);
  }

  toolStart(data) {
    this.closeThink();
    this.openHead();
    this.toolCount += 1;
    const d = el("details", "step running");
    const s = el("summary");
    const ico = el("span", "st-ico", "◌");
    const name = el("span", "st-name", data.name);
    const args = el("span", "st-args", shortArgs(data.args));
    const ms = el("span", "st-ms", "…");
    s.append(ico, name, args, ms);
    const body = el("div", "st-body");
    const pre = el("pre", null, JSON.stringify(data.args ?? {}, null, 2));
    body.append(el("div", "small muted", "arguments"), pre);
    d.append(s, body);
    this.steps.append(d);
    this.toolSteps.set(data.id, { root: d, ico, ms, body });
  }

  toolEnd(data) {
    const st = this.toolSteps.get(data.id);
    if (!st) return;
    st.root.classList.remove("running");
    st.root.classList.add(data.ok ? "ok" : "bad");
    st.ico.textContent = data.ok ? "✓" : "✗";
    st.ms.textContent = `${data.ms} ms`;
    st.body.append(el("div", "small muted", "result (preview)"), el("pre", null, data.preview || ""));
  }

  summary(toolCalls, elapsed) {
    this.head.classList.remove("live");
    const n = toolCalls.length;
    const parts = [elapsed != null ? `Researched ${Number(elapsed).toFixed(1)} s` : "Answered"];
    if (n) parts.push(`${n} tool call${n === 1 ? "" : "s"}`);
    this.headSummary.querySelector(".researched-text").textContent = parts.join(" · ");
    if (n) {
      this.head.classList.remove("empty");
      // Loaded from history: no live step cards were ever built, so fall back
      // to a plain name list inside the same dropdown.
      if (!this.steps.children.length) {
        const ul = el("ul");
        for (const c of toolCalls) ul.append(el("li", null, c.replace(/^`|`$/g, "")));
        this.steps.append(ul);
      }
    } else {
      this.head.classList.add("empty");
    }
    // Collapse once the turn is done — the summary line stays, the detail is a click away.
    // Skip it if the user already toggled this dropdown themselves; their choice sticks.
    if (!this.userToggled) this.head.open = false;
    if (this.text) this.foot.append(copyButton(() => this.text));
  }

  done(data) {
    this.closeThink();
    this.renderer.finish(data.answer);
    this.text = data.answer;
    // live steps stay; the summary line mirrors the stored history entry
    this.summary(data.tool_calls || [], data.elapsed);
    if (data.error === "cancelled") this.note("Stopped.", "info");
    else if (data.error === "tool_call_limit_reached") this.note("Tool-call limit reached.", "info");
  }

  note(text, kind = "") {
    this.closeThink();
    for (const st of this.toolSteps.values()) if (st.root.classList.contains("running")) {
      st.root.classList.remove("running"); st.ico.textContent = "–"; st.ms.textContent = "";
    }
    // A raw "error"/abort event never calls summary() (that's the "done" path),
    // so stop the live spinner here or it spins forever on a failed turn.
    this.head.classList.remove("live");
    this.renderer.finish();
    this.bubble.append(el("div", "note " + kind, text));
  }
}

async function waitForIdle(attempts = 40) {
  for (let i = 0; i < attempts; i++) {
    const session = await refreshSession();
    if (!session.busy) return session;
    await new Promise((r) => setTimeout(r, 250));
  }
  return null;
}

export function chatView(store) {
  return {
    mount(container) {
      const root = el("div", "chat");
      const list = el("div", "messages");
      const wrap = el("div", "composer-wrap");
      const composer = el("div", "composer");
      const box = el("div", "composer-box");
      const ta = el("textarea");
      ta.rows = 1;
      ta.placeholder = store.config.placeholder;
      const send = el("button", "btn primary send", SEND_ICON);
      send.type = "button";
      send.title = "Send (Enter)";
      box.append(ta, send);
      const settingsRow = el("div", "settings-row");
      composer.append(box, settingsRow);
      wrap.append(composer);
      root.append(list, wrap);
      container.append(root);

      let renderedCount = -1;
      // Follow the stream until the reader scrolls up; resume when they return to the bottom.
      let autoScroll = true;
      const atBottom = () => (document.documentElement.scrollHeight - window.scrollY - window.innerHeight) < 6;
      const onIntent = (ev) => {
        if (ev.type === "wheel" && ev.deltaY >= 0) return;
        if (ev.type === "keydown" && !["ArrowUp", "PageUp", "Home"].includes(ev.key)) return;
        autoScroll = false;
      };
      const onScroll = () => { if (atBottom()) autoScroll = true; };
      addEventListener("wheel", onIntent, { passive: true });
      addEventListener("touchmove", onIntent, { passive: true });
      addEventListener("keydown", onIntent);
      addEventListener("scroll", onScroll, { passive: true });
      let scrollRaf = 0;
      const scrollDown = () => {
        if (!autoScroll || scrollRaf) return;
        scrollRaf = requestAnimationFrame(() => { scrollRaf = 0; if (autoScroll) window.scrollTo(0, document.documentElement.scrollHeight); });
      };

      function renderEmpty() {
        const e = el("div", "landing");
        const head = el("div", "landing-head");
        head.append(svgUse("logo-mark", "logo"), el("h1", null, "Ask a detailed question about the research in e-conversion"));
        e.append(head, el("h2", "ex-label", "Example questions"));
        const starters = el("div", "starters");
        for (const q of store.config.examples) {
          const b = el("button", "starter", q);
          b.type = "button";
          b.addEventListener("click", () => submit(q));
          starters.append(b);
        }
        e.append(starters);
        list.append(e);
      }

      function appendUser(text) {
        const m = el("div", "msg user");
        m.append(el("div", "bubble", text));
        list.append(m);
      }

      function renderHistory() {
        list.replaceChildren();
        const msgs = store.session?.messages || [];
        if (!msgs.length) renderEmpty();
        for (const m of msgs) {
          if (m.role === "user") appendUser(m.content);
          else AssistantMessage.fromHistory(list, m);
        }
        renderedCount = msgs.length;
        if (store.streaming) list.append(el("div", "note info", "A response is still being generated…"));
      }

      function setMode(streaming) {
        send.textContent = streaming ? STOP_ICON : SEND_ICON;
        send.classList.toggle("stop", streaming);
        send.title = streaming ? "Stop generating" : "Send (Enter)";
        ta.disabled = streaming;
      }

      function grow() {
        ta.style.height = "auto";
        ta.style.height = Math.min(ta.scrollHeight, 200) + "px";
      }

      async function submit(text) {
        text = (text || "").trim();
        if (!text || store.streaming) return;
        if (!store.session.messages.length && !list.querySelector(".msg")) list.replaceChildren();
        ta.value = ""; grow();
        appendUser(text);
        autoScroll = true;
        const a = new AssistantMessage(list);
        // the server will hold these two messages after the turn; keep the live
        // DOM (with its tool steps) instead of re-rendering from history
        renderedCount = (store.session?.messages?.length ?? 0) + 2;
        scrollDown();
        const controller = new AbortController();
        const stop = () => { postJSON("api/chat/stop").catch(() => {}); controller.abort(); };
        store.update({ streaming: true, stop });
        setMode(true);
        try {
          await streamChat(text, {
            signal: controller.signal,
            onEvent: (type, data) => {
              switch (type) {
                case "text_delta": a.appendText(data.text); break;
                case "reasoning_delta": a.reasoning(data.text); break;
                case "tool_call_start": a.toolStart(data); break;
                case "tool_call_end": a.toolEnd(data); break;
                case "done": a.done(data); break;
                case "error": a.note("Error: " + data.message); break;
                default: break;
              }
              scrollDown();
            },
          });
        } catch (e) {
          if (e.name === "AbortError") a.note("Stopped.", "info");
          else a.note(e.message || String(e));
        } finally {
          // After a stop the fetch rejects before the server has committed the
          // turn; stay in "streaming" (no history re-render) until it is idle.
          try { await waitForIdle(); } catch { /* keep the DOM as is */ }
          store.update({ streaming: false, stop: null });
          setMode(false);
          ta.focus();
        }
      }

      send.addEventListener("click", () => { if (store.streaming) store.stop?.(); else submit(ta.value); });
      ta.addEventListener("keydown", (ev) => {
        if (ev.key === "Enter" && !ev.shiftKey && !ev.isComposing) { ev.preventDefault(); submit(ta.value); }
      });
      ta.addEventListener("input", grow);

      const unsub = store.subscribe((_, patch) => {
        renderSettingsRow(store, settingsRow);
        if (!("session" in patch) && !("resetTick" in patch)) return;
        const n = store.session?.messages?.length ?? 0;
        if (!store.streaming && n !== renderedCount) renderHistory();
      });
      renderSettingsRow(store, settingsRow);
      renderHistory();
      setMode(store.streaming);
      ta.focus();

      return () => {
        unsub();
        removeEventListener("scroll", onScroll); removeEventListener("wheel", onIntent);
        removeEventListener("touchmove", onIntent); removeEventListener("keydown", onIntent);
      };
    },
  };
}
