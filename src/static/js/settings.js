// Settings row (sources, model, feedback) and the dialogs (connect, feedback, stats).
import { getJSON, postJSON, del } from "./api.js";
import { refreshSession } from "./app.js";
import { escapeHtml } from "./markdown.js";

let store = null;

export function toast(message, kind = "") {
  const t = document.createElement("div");
  t.className = "toast " + kind;
  t.textContent = message;
  document.getElementById("toasts").append(t);
  setTimeout(() => t.remove(), 3200);
}

// ---------- connect dialog ----------
let connectKind = null;

function openConnect(kind) {
  connectKind = kind;
  const src = store.config.sources[kind];
  const conn = store.session.connected[kind];
  document.getElementById("connect-title").textContent = `Connect ${src.label}`;
  document.getElementById("connect-label").textContent = `${src.label} token`;
  // Mirrored through the app: the upstream page refuses to be framed cross-origin.
  document.getElementById("connect-frame").src = `api/register/${kind}`;
  document.getElementById("connect-link").href = src.register_url;
  document.getElementById("connect-token").value = "";
  document.getElementById("connect-error").hidden = true;
  document.getElementById("connect-disconnect").hidden = !conn.active;
  document.getElementById("connect-hint").textContent = conn.active
    ? `Connected — ${conn.tools} tools available. Paste a new token to replace it.`
    : "Register below — you stay inside the app. Then paste the token.";
  document.getElementById("dlg-connect").showModal();
}

async function submitConnect() {
  const token = document.getElementById("connect-token").value.trim();
  const err = document.getElementById("connect-error");
  const btn = document.getElementById("connect-submit");
  if (!token) { err.textContent = "Paste the token first."; err.hidden = false; return; }
  btn.disabled = true; btn.textContent = "Connecting…";
  try {
    const r = await postJSON(`api/session/connect/${connectKind}`, { token });
    await refreshSession();
    if (r.active) {
      toast(`${store.config.sources[connectKind].label}: connected (${r.tools} tools)`);
      document.getElementById("dlg-connect").close();
    } else {
      err.textContent = r.error || "Token invalid or expired — register a new one above.";
      err.hidden = false;
    }
  } catch (e) { err.textContent = e.message; err.hidden = false; }
  finally { btn.disabled = false; btn.textContent = "Connect"; }
}

async function disconnect() {
  try {
    await del(`api/session/connect/${connectKind}`);
    await refreshSession();
    toast(`${store.config.sources[connectKind].label} disconnected`);
    document.getElementById("dlg-connect").close();
  } catch (e) { toast(e.message, "bad"); }
}

// ---------- feedback dialog ----------
function openFeedback() {
  document.getElementById("feedback-text").value = "";
  document.getElementById("feedback-error").hidden = true;
  document.getElementById("dlg-feedback").showModal();
  document.getElementById("feedback-text").focus();
}

async function submitFeedback() {
  const text = document.getElementById("feedback-text").value.trim();
  const category = document.querySelector("#feedback-category input:checked").value;
  const err = document.getElementById("feedback-error");
  if (!text) { err.textContent = "Add a note before submitting."; err.hidden = false; return; }
  try {
    await postJSON("api/feedback", { category, text });
    document.getElementById("dlg-feedback").close();
    toast("Thanks — recorded.");
  } catch (e) { err.textContent = e.message; err.hidden = false; }
}

// ---------- stats dialog ----------
export async function openStats() {
  const body = document.getElementById("stats-body");
  body.innerHTML = '<p class="muted">Loading…</p>';
  document.getElementById("dlg-stats").showModal();
  try {
    const s = await getJSON("api/stats");
    body.innerHTML = renderStats(s);
  } catch (e) { body.innerHTML = `<p class="error">${escapeHtml(e.message)}</p>`; }
}

function renderStats(s) {
  const build = s.build.git_sha + (s.build.build_time ? ` (${s.build.build_time})` : "");
  const u = s.usage;
  const tools = store.session.tools;
  const inventory = [`${tools.local} local`]
    .concat(tools.elab ? [`${tools.elab} eLabFTW`] : [], tools.dt ? [`${tools.dt} DataTagger`] : []).join(" · ");
  const kpi = (n, l) => `<div class="kpi"><div class="n">${escapeHtml(n)}</div><div class="l">${escapeHtml(l)}</div></div>`;
  const row = (cells, num = []) => `<tr>${cells.map((c, i) => `<td class="${num.includes(i) ? "num" : ""}">${escapeHtml(c ?? "—")}</td>`).join("")}</tr>`;
  return `
    <p class="muted small">Build <code>${escapeHtml(build)}</code> · providers: ${escapeHtml(s.providers.join(", "))} ·
      default model <code>${escapeHtml(s.default_model)}</code> · tools in this session: ${escapeHtml(inventory)}</p>
    <section><h3>Usage (from the server logs)</h3>
      <div class="kpis">${kpi(u.turns, "turns")}${kpi(u.sessions, "sessions")}${kpi(u.error_turns, "error turns")}${kpi(u.avg_latency_ms + " ms", "⌀ latency")}${kpi(u.feedback, "feedback")}</div>
    </section>
    <section><h3>Pipeline (sources → caches → tools)</h3>
      <table><thead><tr><th>stage</th><th class="num">entries</th><th>available</th><th>built</th></tr></thead><tbody>
      ${s.pipeline.map((p) => row([p.stage, p.entries == null ? (p.available ? "yes" : "no") : String(p.entries), p.available ? "yes" : "no", p.built], [1])).join("")}
      </tbody></table></section>
    <section><h3>Tools</h3>
      <table><thead><tr><th>tool</th><th class="num">calls</th><th class="num">errors</th><th class="num">avg</th></tr></thead><tbody>
      ${s.tools.length ? s.tools.map((t) => row([t.name, String(t.calls), String(t.errors), t.avg_ms + " ms"], [1, 2, 3])).join("") : row(["none yet", "", "", ""])}
      </tbody></table></section>
    <section><h3>Models</h3>
      <table><thead><tr><th>model</th><th class="num">turns</th></tr></thead><tbody>
      ${s.models.length ? s.models.map((m) => row([m.name, String(m.turns)], [1])).join("") : row(["none yet", ""])}
      </tbody></table></section>
    `;
}

// ---------- theme ----------
const THEME_KEY = "econverse-theme";

export function applyTheme(mode) {
  const root = document.documentElement;
  if (mode === "light" || mode === "dark") root.dataset.theme = mode; else delete root.dataset.theme;
  for (const b of document.querySelectorAll("#theme-seg button")) b.classList.toggle("on", b.dataset.theme === mode);
  for (const f of document.querySelectorAll("iframe")) propagateTheme(f);
  try { localStorage.setItem(THEME_KEY, mode); } catch { /* private mode */ }
}

export function currentTheme() {
  try { return localStorage.getItem(THEME_KEY) || "system"; } catch { return "system"; }
}

/** Same-origin iframes (the maps) carry their own stylesheet; hand them the choice. */
export function propagateTheme(iframe) {
  try {
    const root = iframe.contentDocument?.documentElement;
    if (!root) return;
    const mode = currentTheme();
    if (mode === "light" || mode === "dark") root.dataset.theme = mode; else delete root.dataset.theme;
  } catch { /* cross-origin (registration pages) */ }
}

export function initDialogs(s) {
  store = s;
  applyTheme(currentTheme());
  document.getElementById("theme-seg").addEventListener("click", (e) => {
    const b = e.target.closest("button[data-theme]");
    if (b) applyTheme(b.dataset.theme);
  });
  const menu = document.getElementById("menu");
  document.addEventListener("click", (e) => { if (menu.open && !menu.contains(e.target)) menu.open = false; });
  document.getElementById("stats-btn").addEventListener("click", () => { menu.open = false; openStats(); });
  const box = document.getElementById("pipeline-box");
  box.addEventListener("toggle", () => {
    const f = document.getElementById("pipeline-frame");
    if (box.open && !f.src) { f.src = f.dataset.src; f.addEventListener("load", () => propagateTheme(f), { once: true }); }
  });
  document.getElementById("connect-submit").addEventListener("click", submitConnect);
  document.getElementById("connect-disconnect").addEventListener("click", disconnect);
  document.getElementById("connect-token").addEventListener("keydown", (e) => { if (e.key === "Enter") { e.preventDefault(); submitConnect(); } });
  document.getElementById("feedback-submit").addEventListener("click", submitFeedback);
  document.getElementById("dlg-connect").addEventListener("close", () => { document.getElementById("connect-frame").src = "about:blank"; });
}

// ---------- settings row ----------
export function renderSettingsRow(store, el) {
  const session = store.session;
  const cfg = store.config;
  el.replaceChildren();

  for (const [kind, src] of Object.entries(cfg.sources)) {
    const conn = session.connected[kind] || { active: false, tools: 0 };
    const b = document.createElement("button");
    b.type = "button";
    b.className = "chip" + (conn.active ? " on" : "");
    b.title = conn.active ? `connected — ${conn.tools} tools available` : "not connected — click to register and paste a token";
    b.innerHTML = `<span class="dot"></span>${escapeHtml(src.label)}${conn.active ? ` <span class="muted">· ${conn.tools}</span>` : ""}`;
    b.addEventListener("click", () => openConnect(kind));
    el.append(b);
  }

  const picker = document.createElement("details");
  picker.className = "picker";
  const label = session.auto_model ? `${session.provider} / auto (cheapest)` : `${session.provider} / ${session.model}`;
  picker.innerHTML = `<summary class="chip" title="Switch provider or model">${escapeHtml(label)} ▾</summary><div class="picker-menu"></div>`;
  const menu = picker.querySelector(".picker-menu");
  for (const [name, prov] of Object.entries(cfg.providers)) {
    const h = document.createElement("h4"); h.textContent = name; menu.append(h);
    const models = prov.openrouter ? ["", ...prov.models] : (prov.models.length ? prov.models : [prov.default_model]);
    for (const m of models) {
      const b = document.createElement("button");
      b.type = "button";
      const picked = name === session.provider && (m ? m === session.model : session.auto_model);
      b.className = picked ? "on" : "";
      b.textContent = m || "auto (cheapest eligible)";
      b.addEventListener("click", async () => {
        try { await postJSON("api/session/model", { provider: name, model: m }); await refreshSession(); }
        catch (e) { toast(e.message, "bad"); }
        picker.open = false;
      });
      menu.append(b);
    }
  }
  if (Object.values(cfg.providers).some((p) => p.openrouter)) {
    const hint = document.createElement("div"); hint.className = "hint";
    hint.textContent = "OpenRouter: without a pick, the cheapest model allowed by the account guardrails is used automatically.";
    menu.append(hint);
  }
  const tools = session.tools;
  const inv = document.createElement("div"); inv.className = "hint";
  inv.textContent = "Tools: " + [`${tools.local} local`]
    .concat(tools.elab ? [`${tools.elab} eLabFTW`] : [], tools.dt ? [`${tools.dt} DataTagger`] : []).join(" · ");
  menu.append(inv);
  el.append(picker);
  document.addEventListener("click", (ev) => { if (picker.open && !picker.contains(ev.target)) picker.open = false; });

  const spacer = document.createElement("span"); spacer.className = "spacer"; el.append(spacer);
  const fb = document.createElement("button");
  fb.type = "button"; fb.className = "chip"; fb.textContent = "Feedback";
  fb.title = "Bug report or note about the last answer";
  fb.addEventListener("click", openFeedback);
  el.append(fb);
}
