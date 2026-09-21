// Corpus map: UMAP layout of the paper embeddings rendered with deck.gl.
import { getJSON } from "../api.js";

const DECK_URL = "https://cdn.jsdelivr.net/npm/deck.gl@9.0.38/dist.min.js";
let deckLoading = null;
function loadDeck() {
  if (window.deck) return Promise.resolve();
  if (!deckLoading) deckLoading = new Promise((resolve, reject) => {
    const s = document.createElement("script");
    s.src = DECK_URL; s.onload = resolve; s.onerror = () => reject(new Error("deck.gl failed to load"));
    document.head.append(s);
  });
  return deckLoading;
}

const cache = new Map(); // clusters -> payload (server memoises too; this saves the round-trip)

export function corpusMapView() {
  return {
    mount(container) {
      container.innerHTML = `
        <div class="page">
          <h1>Corpus Map</h1>
          <p class="lede">UMAP layout of the paper embeddings; KMeans clusters (computed in the full 384-d space)
            labeled with their top title keywords. Scroll to zoom, drag to pan, hover a point for its title —
            a visual answer to “which papers are near the one I'm reading?”</p>
          <div class="map-toolbar">
            <label>Clusters <input type="range" id="clusters" min="2" max="20" value="8"> <b id="clusters-n">8</b></label>
            <label>Find a paper <input type="search" id="paper-search" list="paper-titles" placeholder="Search or pick from the paper list…"><datalist id="paper-titles"></datalist></label>
          </div>
          <div class="map-frame"><canvas id="deck-canvas"></canvas><div class="map-status" id="map-status">Loading…</div></div>
          <div class="legend" id="legend"></div>
        </div>`;
      const $ = (id) => container.querySelector("#" + id);
      const status = $("map-status");
      let deckInst = null;
      let data = [];
      let disposed = false;

      function fit(matches) {
        const wrap = container.querySelector(".map-frame");
        const xs = data.map((d) => d.x), ys = data.map((d) => d.y);
        const minx = Math.min(...xs), maxx = Math.max(...xs), miny = Math.min(...ys), maxy = Math.max(...ys);
        const W = wrap.clientWidth || 700, H = wrap.clientHeight || 540;
        const zoom = Math.log2(0.85 * Math.min(W / Math.max(maxx - minx, 1e-6), H / Math.max(maxy - miny, 1e-6)));
        // A search rings every matching paper and frames the hits: a single hit gets
        // its ~30-neighbor surroundings; several get a bounding-box fit.
        let target = [(minx + maxx) / 2, (miny + maxy) / 2, 0];
        let z = zoom;
        const hits = data.filter((d) => matches.has(d.title));
        if (hits.length === 1) {
          const sel = hits[0];
          const dists = data.map((d) => Math.hypot(d.x - sel.x, d.y - sel.y)).sort((a, b) => a - b);
          const R = dists[Math.min(30, dists.length - 1)] || 1;
          const zin = Math.log2(0.35 * Math.min(W, H) / Math.max(R, 1e-6));
          target = [sel.x, sel.y, 0];
          z = Math.min(Math.max(zin, zoom + 1), zoom + 5);
        } else if (hits.length > 1) {
          const hx = hits.map((d) => d.x), hy = hits.map((d) => d.y);
          const nx = Math.min(...hx), Xx = Math.max(...hx), ny = Math.min(...hy), Xy = Math.max(...hy);
          target = [(nx + Xx) / 2, (ny + Xy) / 2, 0];
          const zfit = Math.log2(0.8 * Math.min(W / Math.max(Xx - nx, 1e-6), H / Math.max(Xy - ny, 1e-6)));
          z = Math.min(Math.max(zfit, zoom), zoom + 6);
        }
        return { hits, viewState: { target, zoom: z } };
      }

      function layers(hits) {
        const { ScatterplotLayer } = window.deck;
        const L = [new ScatterplotLayer({
          id: "points", data,
          getPosition: (d) => [d.x, d.y], getFillColor: (d) => d.color,
          getRadius: 4, radiusUnits: "pixels", radiusMinPixels: 2.5, radiusMaxPixels: 9,
          opacity: 0.85, pickable: true, autoHighlight: true, highlightColor: [255, 255, 255, 140],
        })];
        if (hits.length) L.push(new ScatterplotLayer({
          id: "matched", data: hits, getPosition: (d) => [d.x, d.y],
          filled: false, stroked: true, getLineColor: [255, 255, 255], lineWidthUnits: "pixels",
          getLineWidth: 2, lineWidthMinPixels: 2, lineWidthMaxPixels: 2,
          getRadius: 12, radiusUnits: "pixels", radiusMinPixels: 12, radiusMaxPixels: 12, pickable: false,
        }));
        return L;
      }

      function matchesFor(q) {
        q = (q || "").trim();
        if (!q) return new Set();
        const exact = data.find((d) => d.title === q);
        if (exact) return new Set([exact.title]);
        const lq = q.toLowerCase();
        return new Set(data.filter((d) => d.title.toLowerCase().includes(lq)).slice(0, 100).map((d) => d.title));
      }

      function draw() {
        const { hits, viewState } = fit(matchesFor($("paper-search").value));
        const { Deck, OrthographicView } = window.deck;
        if (!deckInst) {
          deckInst = new Deck({
            canvas: $("deck-canvas"), views: new OrthographicView({}),
            controller: { scrollZoom: true, dragPan: true, doubleClickZoom: true },
            initialViewState: viewState, layers: layers(hits),
            getTooltip: ({ object }) => object && { html: `<b>${object.title}</b><br/>${object.year} · ${object.cluster}`, className: "dk-tip" },
          });
        } else {
          deckInst.setProps({ layers: layers(hits), initialViewState: viewState });
        }
      }

      function renderLegend(legend) {
        const el = $("legend");
        el.innerHTML = '<div class="t">Cluster (top title keywords)</div>' + legend.map((e) =>
          `<span><i style="background:rgb(${e.color.join(",")})"></i>${e.cluster}</span>`).join("");
      }

      async function load(n) {
        status.textContent = cache.has(n) ? "" : "Computing corpus map (UMAP + clustering)…";
        status.hidden = cache.has(n);
        try {
          let payload = cache.get(n);
          if (!payload) { payload = await getJSON(`api/corpus-map?clusters=${n}`); if (payload.available) cache.set(n, payload); }
          if (disposed) return;
          if (!payload.available) { status.textContent = payload.hint || "Corpus map not available."; status.hidden = false; return; }
          await loadDeck();
          if (disposed) return;
          data = payload.points;
          if (!$("paper-titles").children.length) {
            const dl = $("paper-titles");
            for (const t of [...new Set(data.map((d) => d.title))].sort()) { const o = document.createElement("option"); o.value = t; dl.append(o); }
          }
          renderLegend(payload.legend);
          status.hidden = true;
          draw();
        } catch (e) { status.textContent = e.message; status.hidden = false; }
      }

      const slider = $("clusters");
      let timer = 0;
      slider.addEventListener("input", () => {
        $("clusters-n").textContent = slider.value;
        clearTimeout(timer);
        timer = setTimeout(() => load(Number(slider.value)), 250);
      });
      let searchTimer = 0;
      $("paper-search").addEventListener("input", () => { clearTimeout(searchTimer); searchTimer = setTimeout(() => data.length && draw(), 200); });
      $("paper-search").addEventListener("change", () => data.length && draw());
      load(8);

      return () => { disposed = true; if (deckInst) { deckInst.finalize(); deckInst = null; } };
    },
  };
}
