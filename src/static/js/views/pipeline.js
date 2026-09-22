// Data pipeline map — the self-contained board, framed.
import { propagateTheme } from "../settings.js";

export function pipelineView() {
  return {
    mount(container) {
      container.innerHTML = `
        <div class="page" style="padding-bottom:0">
          <h1>Pipeline</h1>
          <p class="lede">Every source, build script, cache and tool behind the assistant, source to delivery.
            Click a box to trace what it is built from, plus the one thing it directly produces.</p>
        </div>
        <div class="frame-fill"><iframe src="static/pipeline_map.html" title="Data pipeline map" style="min-height:900px"></iframe></div>`;
      const f = container.querySelector("iframe");
      f.addEventListener("load", () => propagateTheme(f), { once: true });
    },
  };
}
