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
  document.getElementById("connect-base-label").textContent = `${src.label} base URL`;
  document.getElementById("connect-base").value = src.base_url_default || "";
  document.getElementById("connect-key-label").textContent = src.key_label || "API key";
  document.getElementById("connect-key").value = "";
  document.getElementById("connect-token").value = "";
  document.getElementById("connect-label").textContent = `${src.label} token`;
  document.getElementById("connect-error").hidden = true;

  const profiles = Array.isArray(src.profiles) ? src.profiles : [];
  document.getElementById("connect-profile-field").hidden = profiles.length === 0;
  const select = document.getElementById("connect-profile");
  select.replaceChildren();
  for (const p of profiles) {
    const option = document.createElement("option");
    option.value = p.value;
    option.textContent = p.label;
    select.append(option);
  }

  document.getElementById("connect-disconnect").hidden = !conn.active;
  const testBtn = document.getElementById("connect-test");
  testBtn.hidden = !src.test_user;
  if (src.test_user) testBtn.textContent = `Sign in as ${src.test_user.label}`;
  document.getElementById("connect-hint").textContent = conn.active
    ? `Connected — ${conn.tools} tools available. Signing in again replaces the key.`
    : "Sign in with your account — the token is created for you and stays on the server.";
  document.getElementById("dlg-connect").showModal();
}

async function submitRegister() {
  const err = document.getElementById("connect-error");
  const btn = document.getElementById("connect-register");
  const label = btn.textContent;
  const apiKey = document.getElementById("connect-key").value.trim();
  if (!apiKey) { err.textContent = "Enter your API key first."; err.hidden = false; return; }
  btn.disabled = true; btn.textContent = "Connecting…";
  try {
    const r = await postJSON(`api/session/register/${connectKind}`, {
      base_url: document.getElementById("connect-base").value.trim(),
      api_key: apiKey,
      profile: document.getElementById("connect-profile").value,
    });
    await refreshSession();
    document.getElementById("connect-key").value = "";
    toast(`${store.config.sources[connectKind].label}: connected (${r.tools} tools)`);
    document.getElementById("dlg-connect").close();
  } catch (e) { err.textContent = e.message; err.hidden = false; }
  finally { btn.disabled = false; btn.textContent = label; }
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

async function useTestAccount() {
  const err = document.getElementById("connect-error");
  const btn = document.getElementById("connect-test");
  const label = btn.textContent;
  btn.disabled = true; btn.textContent = "Signing in…";
  try {
    const r = await postJSON(`api/session/connect/${connectKind}/test`, {});
    await refreshSession();
    if (r.active) {
      toast(`${store.config.sources[connectKind].label}: connected (${r.tools} tools)`);
      document.getElementById("dlg-connect").close();
    } else {
      err.textContent = r.error || "Test account unavailable — paste a token instead.";
      err.hidden = false;
    }
  } catch (e) { err.textContent = e.message; err.hidden = false; }
  finally { btn.disabled = false; btn.textContent = label; }
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

// ---------- parameters dialog ----------
/** Fields the session changed away from the config.toml defaults. */
function changedParams(store) {
  const defaults = store.config.parameters.defaults;
  const effective = store.session.params || {};
  return Object.keys(defaults).filter((k) => effective[k] !== defaults[k]);
}

function optionLabel(spec, value) {
  if (value === "") return (spec.option_labels || {})[""] || "default";
  return (spec.option_labels || {})[value] || value;
}

function openParams() {
  const store0 = store;
  const spec = store0.config.parameters.spec;
  const effective = store0.session.params || {};
  const defaults = store0.config.parameters.defaults;
  const body = document.getElementById("params-body");
  document.getElementById("params-error").hidden = true;
  body.replaceChildren();

  for (const field of spec.filter((f) => !f.hidden)) {
    const wrap = document.createElement("div");
    const label = document.createElement("label");
    label.className = "field";
    const name = document.createElement("span");
    name.textContent = field.label;
    if (effective[field.key] !== defaults[field.key]) {
      const changed = document.createElement("b");
      changed.className = "muted small";
      changed.textContent = " · changed";
      name.append(changed);
    }
    label.append(name);

    if (field.type === "bool") {
      const seg = document.createElement("div");
      seg.className = "seg";
      seg.dataset.key = field.key;
      for (const val of [true, false]) {
        const b = document.createElement("button");
        b.type = "button";
        b.dataset.val = String(val);
        b.textContent = val ? "on" : "off";
        b.className = effective[field.key] === val ? "on" : "";
        b.addEventListener("click", () => {
          for (const other of seg.querySelectorAll("button")) other.classList.toggle("on", other === b);
        });
        seg.append(b);
      }
      label.append(seg);
    } else if (field.type === "number") {
      const input = document.createElement("input");
      input.type = "number";
      input.dataset.key = field.key;
      input.min = String(field.min);
      input.max = String(field.max);
      input.step = String(field.step || 1);
      input.placeholder = "model default";
      input.value = effective[field.key] || "";
      label.append(input);
    } else {
      const select = document.createElement("select");
      select.dataset.key = field.key;
      for (const opt of field.options) {
        const o = document.createElement("option");
        o.value = opt;
        o.textContent = optionLabel(field, opt);
        if (opt === effective[field.key]) o.selected = true;
        select.append(o);
      }
      label.append(select);
    }
    wrap.append(label);
    if (field.help) {
      const help = document.createElement("p");
      help.className = "hint";
      help.textContent = field.help;
      wrap.append(help);
    }
    body.append(wrap);
  }
  document.getElementById("dlg-params").showModal();
}

function readParams() {
  const out = {};
  for (const el of document.querySelectorAll("#params-body [data-key]")) {
    if (el.classList.contains("seg")) {
      const on = el.querySelector("button.on");
      if (on) out[el.dataset.key] = on.dataset.val === "true";
    } else {
      out[el.dataset.key] = el.value.trim();
    }
  }
  return out;
}

async function applyParams() {
  const err = document.getElementById("params-error");
  try {
    await postJSON("api/session/params", { params: readParams() });
    await refreshSession();
    document.getElementById("dlg-params").close();
    toast("Parameters applied");
  } catch (e) { err.textContent = e.message; err.hidden = false; }
}

async function resetParams() {
  const err = document.getElementById("params-error");
  try {
    await del("api/session/params");
    await refreshSession();
    openParams();  // re-render with the defaults
    toast("Parameters reset to defaults");
  } catch (e) { err.textContent = e.message; err.hidden = false; }
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

/** Native dialogs only close on Esc or the ✕; make the backdrop dismiss them too.
 *  The mousedown check keeps a selection drag that ends on the backdrop from closing. */
function closeOnBackdropClick(dlg) {
  let fromBackdrop = false;
  dlg.addEventListener("mousedown", (e) => { fromBackdrop = e.target === dlg; });
  dlg.addEventListener("click", (e) => { if (fromBackdrop && e.target === dlg) dlg.close("cancel"); });
}

export function initDialogs(s) {
  store = s;
  for (const dlg of document.querySelectorAll("dialog.dlg")) closeOnBackdropClick(dlg);
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
  document.getElementById("params-apply").addEventListener("click", applyParams);
  document.getElementById("params-reset").addEventListener("click", resetParams);
  document.getElementById("connect-register").addEventListener("click", submitRegister);
  document.getElementById("connect-test").addEventListener("click", useTestAccount);
  document.getElementById("connect-submit").addEventListener("click", submitConnect);
  document.getElementById("connect-disconnect").addEventListener("click", disconnect);
  document.getElementById("connect-token").addEventListener("keydown", (e) => { if (e.key === "Enter") { e.preventDefault(); submitConnect(); } });
  document.getElementById("feedback-submit").addEventListener("click", submitFeedback);
  document.getElementById("dlg-connect").addEventListener("close", () => {
    // never leave a key or a token in the DOM after the dialog closes
    document.getElementById("connect-key").value = "";
    document.getElementById("connect-token").value = "";
  });
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

  const shortModel = (id) => (id || "").replace(/^.*\//, "");
  // OpenRouter is the default, so it comes first; GWDG sits at the bottom as the
  // fallback it is.
  const providers = Object.entries(cfg.providers)
    .sort((a, b) => (b[1].openrouter ? 1 : 0) - (a[1].openrouter ? 1 : 0));
  const picker = document.createElement("details");
  picker.className = "picker";
  const label = session.auto_model
    ? `${session.provider} / ${session.route_label || "cheapest"}${session.resolved_model ? " \u00b7 " + shortModel(session.resolved_model) : ""}`
    : `${session.provider} / ${session.model}`;
  picker.innerHTML = `<summary class="chip" title="Switch model or routing">${escapeHtml(label)} \u25be</summary><div class="picker-menu"></div>`;
  const menu = picker.querySelector(".picker-menu");
  for (const [name, prov] of providers) {
    const h = document.createElement("h4");
    h.textContent = prov.note ? `${name} \u00b7 ${prov.note}` : name;
    menu.append(h);

    if (prov.openrouter) {
      // The routing choices are the model choices: every one uses the cheapest
      // eligible model, they differ in how the upstream provider is picked.
      const model = session.resolved_model || "cheapest eligible";
      for (const route of cfg.routes || []) {
        const active = name === session.provider && session.auto_model && session.sort === route.value;
        const card = document.createElement("button");
        card.type = "button";
        card.className = "route" + (active ? " on" : "");

        const title = document.createElement("span");
        title.className = "title";
        if (active) {
          const mark = document.createElement("span");
          mark.className = "mark";
          mark.textContent = "\u2713";
          title.append(mark);
        }
        const routeName = document.createElement("b");
        routeName.textContent = route.label;
        title.append(routeName);
        if (route.value === "price") {
          const badge = document.createElement("span");
          badge.className = "badge";
          badge.textContent = "default";
          title.append(badge);
        }
        card.append(title);

        const desc = document.createElement("span");
        desc.className = "desc";
        desc.textContent = route.help;
        card.append(desc);

        const meta = document.createElement("span");
        meta.className = "meta";
        const code = document.createElement("code");
        code.textContent = model;
        meta.append(code);
        const cap = document.createElement("span");
        cap.className = "cap";
        cap.textContent = "\u2264 1 \u20ac/M tokens";
        meta.append(cap);
        card.append(meta);

        card.addEventListener("click", async () => {
          try {
            await postJSON("api/session/model", { provider: name, model: "", sort: route.value });
            await refreshSession();
          } catch (e) { toast(e.message, "bad"); }
          picker.open = false;
        });
        menu.append(card);
      }
      continue;
    }

    const models = prov.models.length ? prov.models : [prov.default_model];
    for (const m of models) {
      const b = document.createElement("button");
      b.type = "button";
      b.className = (name === session.provider && !session.auto_model && m === session.model) ? "on" : "";
      b.textContent = m;
      b.title = "Ask GWDG directly with " + m;
      b.addEventListener("click", async () => {
        try { await postJSON("api/session/model", { provider: name, model: m }); await refreshSession(); }
        catch (e) { toast(e.message, "bad"); }
        picker.open = false;
      });
      menu.append(b);
    }
  }
  const tools = session.tools;
  const inv = document.createElement("div"); inv.className = "hint";
  inv.textContent = "Tools: " + [`${tools.local} local`]
    .concat(tools.elab ? [`${tools.elab} eLabFTW`] : [], tools.dt ? [`${tools.dt} DataTagger`] : []).join(" · ");
  menu.append(inv);
  el.append(picker);
  document.addEventListener("click", (ev) => { if (picker.open && !picker.contains(ev.target)) picker.open = false; });

  const params = document.createElement("button");
  params.type = "button";
  params.id = "params-chip";
  params.className = "chip";
  const changed = changedParams(store);
  params.title = "LLM parameters for this session: reasoning effort, temperature, provider routing, data retention";
  params.textContent = changed.length
    ? `\u2699 ${changed.length} changed: ` + changed.slice(0, 2)
        .map((k) => `${k.replace(/_/g, " ")} ${store.session.params[k] === "" ? "off" : store.session.params[k]}`).join(", ")
        + (changed.length > 2 ? " \u2026" : "")
    : "\u2699 parameters: defaults";
  params.addEventListener("click", openParams);
  el.append(params);

  const spacer = document.createElement("span"); spacer.className = "spacer"; el.append(spacer);
  const fb = document.createElement("button");
  fb.type = "button"; fb.className = "chip"; fb.textContent = "Feedback";
  fb.title = "Bug report or note about the last answer";
  fb.addEventListener("click", openFeedback);
  el.append(fb);
}
