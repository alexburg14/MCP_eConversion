// Bootstrap: load config + session, then start the router.
import { getJSON, postJSON } from "./api.js";
import { createRouter } from "./router.js";
import { chatView } from "./chat.js";
import { initDialogs, toast } from "./settings.js";
import { corpusMapView } from "./views/corpus-map.js";
import { collaborationView } from "./views/collaboration.js";

export const store = {
  config: null,
  session: null,
  streaming: false,     // a chat turn is in flight in this tab
  stop: null,           // function that aborts it
  listeners: new Set(),
  update(patch) { Object.assign(this, patch); for (const fn of this.listeners) fn(this, patch); },
  subscribe(fn) { this.listeners.add(fn); return () => this.listeners.delete(fn); },
};

export async function refreshSession() {
  const session = await getJSON("api/session");
  store.update({ session });
  return session;
}

async function boot() {
  const view = document.getElementById("view");
  try {
    const [config, session] = await Promise.all([getJSON("api/config"), getJSON("api/session")]);
    store.update({ config, session });
    document.title = config.title;
  } catch (e) {
    view.innerHTML = `<div class="page"><p class="note">Could not reach the server: ${e.message}</p></div>`;
    return;
  }
  const userLabel = document.getElementById("user-label");
  if (store.session.user) { userLabel.textContent = store.session.user; userLabel.hidden = false; }

  initDialogs(store);
  document.getElementById("new-chat").addEventListener("click", async () => {
    if (store.streaming && store.stop) { store.stop(); await new Promise((r) => setTimeout(r, 150)); }
    try {
      await postJSON("api/chat/reset");
      await refreshSession();
      location.hash = "#/chat";
      store.update({ resetTick: (store.resetTick || 0) + 1 });
    } catch (e) { toast(e.message, "bad"); }
  });

  const router = createRouter({
    "chat": chatView(store),
    "corpus-map": corpusMapView(store),
    "collaboration": collaborationView(store),
  }, { container: view, nav: document.getElementById("nav") });
  router.start();
}

boot();
