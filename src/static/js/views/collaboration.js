// PI co-authorship network — the self-contained d3 page, framed.
import { propagateTheme } from "../settings.js";

export function collaborationView() {
  return {
    mount(container) {
      container.innerHTML = `
        <div class="page" style="padding-bottom:0">
          <h1>Collaboration</h1>
          <p class="lede">PI co-authorship network from the cluster's publications — nodes are PIs (sized by number
            of collaborators, colored by institution), edges are shared papers (thicker = more). Click a PI to isolate
            who they publish with.</p>
        </div>
        <div class="frame-fill"><iframe src="static/collaboration_map.html" title="Collaboration network" style="min-height:720px"></iframe></div>`;
      const f = container.querySelector("iframe");
      f.addEventListener("load", () => propagateTheme(f), { once: true });
    },
  };
}
