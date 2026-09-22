// Hash router: "#/chat" (default), "#/corpus-map", "#/collaboration".
// A view is { mount(container) -> unmount() }.

export function createRouter(routes, { container, nav, defaultRoute = "chat" }) {
  let current = null;
  let unmount = null;

  function routeName() {
    const h = location.hash.replace(/^#\/?/, "").split("?")[0];
    return routes[h] ? h : defaultRoute;
  }

  function go() {
    const name = routeName();
    if (name === current) return;
    if (unmount) { try { unmount(); } catch (e) { console.error(e); } }
    container.replaceChildren();
    current = name;
    unmount = routes[name].mount(container) || null;
    for (const a of nav.querySelectorAll("[data-route]")) a.classList.toggle("active", a.dataset.route === name);
    window.scrollTo(0, 0);
  }

  addEventListener("hashchange", go);
  return { start: go, current: () => current };
}
