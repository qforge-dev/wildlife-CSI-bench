import "@fontsource/dm-sans/400.css";
import "@fontsource/dm-sans/500.css";
import "@fontsource/dm-sans/600.css";
import "@fontsource/dm-sans/700.css";
import "@fontsource/ibm-plex-mono/400.css";
import "@fontsource/ibm-plex-mono/500.css";
import "./style.css";
import {
  deriveGraph,
  displayName,
  modelNames,
  selectedModels,
  speciesTotals,
  sum,
} from "./graph-data";
import type { Edge, Filters, GraphView, Network } from "./graph-data";
import { NetworkCanvas, palette } from "./network-canvas";
import {
  defaultFilters,
  matchingPreset,
  presets,
  presetState,
  readState,
  writeState,
} from "./url-state";
import type { ExplorerState } from "./url-state";

const $ = <T extends HTMLElement = HTMLElement>(selector: string) =>
  document.querySelector<T>(selector)!;
const esc = (text: unknown) =>
  String(text).replace(
    /[&<>"']/g,
    (c) =>
      ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[
        c
      ]!,
  );
const num = (n: number) => n.toLocaleString();
const percent = (n: number, d: number) =>
  d ? `${((100 * n) / d).toFixed(1)}%` : "—";
const root = $("#app");

async function start() {
  const response = await fetch(`${import.meta.env.BASE_URL}network.json`);
  if (!response.ok)
    throw new Error(
      `Graph data could not be loaded (${response.status}). Run npm run data in explorer/.`,
    );
  const data = (await response.json()) as Network;
  const defaults = defaultFilters(data);
  const initial = readState(new URLSearchParams(location.search), data);
  let filters = initial.filters,
    selected = initial.selected,
    pair = initial.pair,
    camera = initial.camera;
  let graph: GraphView;
  root.innerHTML = `<div class="shell">
    <header class="topbar"><div class="identity"><span class="brand-mark" aria-hidden="true"><i></i><i></i><i></i><i></i></span><span class="brand">Wildlife CSI</span><span class="product">Confusion Explorer</span></div>
    <div class="top-meta"><button class="share-button" id="copy-link">Copy link</button><button class="mobile-toggle" id="filters-toggle" aria-expanded="false">Filters</button><button class="mobile-toggle" id="inspector-toggle" aria-expanded="false">Inspect</button><a href="https://github.com/qforge-dev/wildlife-CSI-bench" target="_blank" rel="noreferrer">Repository ↗</a></div></header>
    <main class="workspace"><aside class="sidebar" id="filters-panel" aria-label="Graph filters">
      <div class="section"><div class="section-title"><h2>Explore the graph</h2><button class="reset" id="reset">Reset</button></div><label class="preset-field"><span class="sr-only">Finding preset</span><select id="preset">${presets.map((p) => `<option value="${p.id}">${esc(p.label)}</option>`).join("")}<option value="custom" disabled>Custom view</option></select></label><div class="mode-switch" aria-label="Graph view"><button data-mode="direct" aria-pressed="true">Confusions</button><button data-mode="profile" aria-pressed="false">k-nearest</button></div><p class="note" id="view-description"></p></div>
      <div class="section"><div class="section-title"><h2>Models <span id="model-count"></span></h2><div class="model-actions"><button id="models-all">All</button><button id="models-none">None</button></div></div><fieldset id="model-fieldset"><legend class="sr-only">Models included in the graph</legend><div class="models">${data.models.map((m, i) => `<label class="check"><input type="checkbox" data-model="${i}" checked><span>${esc(modelNames[m.id])}</span>${m.provisional ? '<small title="Scoring has unresolved extractor reviews">*</small>' : ""}</label>`).join("")}</div></fieldset><p class="note" id="model-note"></p></div>
      <div class="section"><div class="section-title"><h2>Connections</h2></div><div id="direct-controls"><label class="field"><span class="field-row">Minimum mistakes <output id="count-value">1</output></span><input aria-label="Minimum mistakes" id="min-count" type="range" min="1" max="30" value="1"></label><label class="field"><span class="field-row">Shared by models <output id="shared-value">1</output></span><input aria-label="Minimum models sharing a pair" id="min-shared" type="range" min="1" max="8" value="1"></label></div>
      <div id="profile-controls" hidden><label class="field"><span class="field-row">Mutual neighbors <output id="k-value">10</output></span><select id="k" aria-label="Number of mutual nearest neighbors"><option>3</option><option>5</option><option selected>10</option><option>20</option></select></label></div>
      <label class="field">Species support<select id="support"><option value="0">All sample sizes</option><option value="3">At least 3 photos</option><option value="5">At least 5 photos</option><option value="10">At least 10 photos</option></select></label>
      <label class="field">Community<select id="group"><option value="">All communities</option></select></label><label class="check" style="margin-top:18px"><input type="checkbox" id="predicted" checked><span>Include predicted-only species</span></label><p class="note">Groups and positions stay fixed to the full graph across all eight models.</p></div>
      <details class="method"><summary>How to read this</summary><p>Arrows run from the actual species to the model’s guess. Larger nodes participate in more errors. Hollow nodes appear only in predictions.</p><p>Confusion communities use error rates per true species, averaged over models. k-nearest compares outgoing error profiles with cosine similarity, separately for each model, and keeps mutual matches. Its layout uses k = 10.</p><p>Correct answers and non-species outcomes stay in the inspector, outside the graph. The same 2,000 photos recur across models; these are not independent samples. Groups are exploratory, especially for species with few photos.</p></details>
    </aside>
    <section class="graph-panel" aria-label="Species confusion network"><div class="graph-top"><div class="search-wrap"><label class="sr-only" for="search">Find a species</label><input id="search" type="search" placeholder="Find a species…" autocomplete="off" aria-controls="search-results" aria-expanded="false"><div id="search-results" class="search-results"></div></div><span class="view-tag" id="view-tag">ALL MODELS / DIRECTED</span></div>
      <div class="focus-badge" id="focus-badge" hidden></div><div class="canvas-wrap"><canvas aria-label="Interactive species network. Search for a species to inspect connections using the keyboard." role="img"></canvas><div class="graph-tooltip" id="tooltip" hidden></div></div>
      <div class="empty" id="empty" hidden><strong>No connections match</strong>Lower the thresholds or select more models.</div><div class="graph-stat" id="graph-stat" aria-live="polite"></div>
      <div class="graph-footer"><div class="legend"><span class="legend-label">COMMUNITIES</span>${palette.map((color, i) => `<span><i style="--color:${color}"></i>${i + 1}</span>`).join("")}<span><i style="--color:#727c7d"></i>Others</span></div><div class="graph-controls"><button id="zoom-out" aria-label="Zoom out">−</button><span class="zoom-level" id="zoom-level">100%</span><button id="zoom-in" aria-label="Zoom in">+</button><button id="fit" class="fit">Fit graph</button></div></div>
    </section>
    <aside class="inspector" id="inspector" aria-label="Selection details"><div class="inspector-head"><span class="eyebrow" id="inspector-label">NETWORK OVERVIEW</span><button class="close" id="clear-selection" aria-label="Clear selection and close inspector">×</button></div><div id="inspector-body"></div></aside>
    </main></div><dialog id="share-dialog"><h2>Share this view</h2><label for="share-url">Link</label><input id="share-url" readonly><p class="note">Select and copy this link.</p><form method="dialog"><button class="secondary">Close</button></form></dialog>`;

  const renderer = new NetworkCanvas(
    $("canvas"),
    data,
    (id) => select(id),
    (edge) => selectEdge(edge),
    (text, x, y) => {
      const tooltip = $("#tooltip");
      tooltip.hidden = !text;
      if (text) {
        tooltip.textContent = text;
        tooltip.style.left = `${Math.max(8, Math.min(x + 15, tooltip.parentElement!.clientWidth - tooltip.offsetWidth - 8))}px`;
        tooltip.style.top = `${Math.max(8, Math.min(y + 16, tooltip.parentElement!.clientHeight - tooltip.offsetHeight - 8))}px`;
      }
    },
    (level) => {
      $("#zoom-level").textContent = `${Math.round(level * 100)}%`;
    },
    (viewCamera) => {
      camera = viewCamera;
      syncUrl();
    },
  );

  function state(): ExplorerState {
    return { filters, selected, pair, camera };
  }
  function syncUrl(push = false) {
    $<HTMLSelectElement>("#preset").value = matchingPreset(state(), data);
    const url = new URL(location.href);
    url.search = writeState(state(), data).toString();
    if (url.href !== location.href) {
      if (push) history.pushState(null, "", url);
      else history.replaceState(null, "", url);
    }
  }
  function frameSelection(animate = true) {
    const edge = graph.edges.find((e) => e.id === pair);
    const ids = edge
      ? new Set([edge.a, edge.b])
      : selected !== null
        ? new Set([selected, ...(graph.neighbors.get(selected) || [])])
        : undefined;
    renderer.fit(animate, ids);
  }
  function applyState(next: ExplorerState, push = false) {
    filters = next.filters;
    selected = next.selected;
    pair = next.pair;
    camera = next.camera;
    const restoreCamera = camera;
    if (push) syncUrl(true);
    refresh();
    $<HTMLInputElement>("#search").value =
      selected === null ? "" : displayName(data.nodes[selected]);
    $("#search-results").replaceChildren();
    if (restoreCamera) renderer.setCamera(restoreCamera);
    else frameSelection();
  }

  function controls() {
    $<HTMLSelectElement>("#preset").value = matchingPreset(state(), data);
    const profile = filters.view === "profile";
    document
      .querySelectorAll<HTMLButtonElement>("[data-mode]")
      .forEach((b) =>
        b.setAttribute("aria-pressed", String(b.dataset.mode === filters.view)),
      );
    $("#view-description").textContent = profile
      ? "Species connected by similar mistakes across all eight models."
      : "Every observed species mix-up, pooled across the selected models.";
    $("#model-count").textContent =
      `· ${profile ? data.models.length : filters.models.length}`;
    $<HTMLFieldSetElement>("#model-fieldset").disabled = profile;
    $<HTMLButtonElement>("#models-all").disabled = profile;
    $<HTMLButtonElement>("#models-none").disabled = profile;
    document.querySelectorAll<HTMLInputElement>("[data-model]").forEach((c) => {
      c.checked = profile || filters.models.includes(Number(c.dataset.model));
    });
    $("#model-note").textContent = profile
      ? "k-nearest profiles use all eight models."
      : "* Provisional scoring. Per-model evidence stays separate.";
    $("#direct-controls").hidden = profile;
    $("#profile-controls").hidden = !profile;
    $<HTMLInputElement>("#min-count").value = String(filters.minCount);
    $("#count-value").textContent = String(filters.minCount);
    $<HTMLInputElement>("#min-shared").max = String(
      Math.max(1, filters.models.length),
    );
    $<HTMLInputElement>("#min-shared").value = String(filters.minShared);
    $("#shared-value").textContent =
      `${filters.minShared} / ${filters.models.length}`;
    $<HTMLSelectElement>("#support").value = String(filters.minSupport);
    $<HTMLSelectElement>("#k").value = String(filters.k);
    $("#k-value").textContent = String(filters.k);
    $<HTMLInputElement>("#predicted").checked = filters.predictedOnly;
    $<HTMLInputElement>("#predicted").disabled = profile;
    const counts = new Map<number, number>();
    for (const node of data.nodes) {
      const p = node[profile ? "profile" : "direct"];
      if (p) counts.set(p[2], (counts.get(p[2]) || 0) + 1);
    }
    $<HTMLSelectElement>("#group").innerHTML =
      '<option value="">All communities</option>' +
      [...counts]
        .sort((a, b) => a[0] - b[0])
        .map(
          ([g, n]) =>
            `<option value="${g}">Group ${g + 1} · ${n} species</option>`,
        )
        .join("");
    $<HTMLSelectElement>("#group").value =
      filters.group === null ? "" : String(filters.group);
    $("#view-tag").textContent = profile
      ? "ALL MODELS / MUTUAL kNN"
      : `${filters.models.length === data.models.length ? "ALL" : filters.models.length} MODELS / DIRECTED`;
  }
  function refresh(fit = false) {
    graph = deriveGraph(data, filters);
    if (selected !== null && !graph.nodes.includes(selected)) {
      selected = null;
      pair = null;
    }
    if (pair && !graph.edges.some((e) => e.id === pair)) pair = null;
    controls();
    renderer.setGraph(graph, filters.view, selected, pair);
    const stat =
      filters.view === "profile"
        ? `<b>${num(graph.nodes.length)}</b> species · <b>${num(graph.edges.length)}</b> mutual links`
        : `<b>${num(graph.nodes.length)}</b> species · <b>${num(graph.edges.length)}</b> directed pairs<br><b>${num(graph.errors)}</b> / ${num(graph.totalErrors)} wrong-species answers shown`;
    $("#graph-stat").innerHTML =
      stat +
      `<div class="graph-caption">${filters.view === "profile" ? "Cosine similarity · mutual top-" + filters.k : "Drag nodes to arrange · drag background to pan · scroll to zoom"}</div>`;
    $("#empty").hidden = graph.edges.length > 0;
    const focus = $("#focus-badge");
    focus.hidden = filters.focus === null;
    if (filters.focus !== null) {
      focus.innerHTML = `Neighborhood: ${esc(displayName(data.nodes[filters.focus]))} <button aria-label="Show whole network">×</button>`;
      focus.querySelector("button")!.onclick = () => {
        filters.focus = null;
        refresh(true);
      };
    }
    inspect();
    syncUrl();
    if (fit) renderer.fit();
  }
  function openInspector() {
    if (window.innerWidth <= 1100) {
      $("#inspector").classList.add("open");
      $("#inspector-toggle").setAttribute("aria-expanded", "true");
      $("#filters-panel").classList.remove("open");
      $("#filters-toggle").setAttribute("aria-expanded", "false");
    }
  }
  function select(id: number, zoom = false) {
    if (!graph.nodes.includes(id)) return;
    selected = id;
    pair = null;
    renderer.setGraph(graph, filters.view, selected, pair);
    inspect();
    openInspector();
    syncUrl();
    if (zoom)
      renderer.fit(true, new Set([id, ...(graph.neighbors.get(id) || [])]));
  }
  function selectEdge(edge: Edge) {
    pair = edge.id;
    selected = null;
    renderer.setGraph(graph, filters.view, selected, pair);
    renderer.fit(true, new Set([edge.a, edge.b]));
    inspect();
    openInspector();
    syncUrl();
  }
  function pairButton(edge: Edge) {
    return `<button class="pair-button" data-pair="${edge.id}"><span class="pair-names">${esc(displayName(data.nodes[edge.a]))}<br><em>${filters.view === "profile" ? "↔" : "→"}</em> ${esc(displayName(data.nodes[edge.b]))}</span><span class="pair-value">${filters.view === "profile" ? edge.similarity.toFixed(2) : edge.count}<small>${filters.view === "profile" ? "similarity" : edge.shared + " models"}</small></span></button>`;
  }
  function wireInspector() {
    document.querySelectorAll<HTMLButtonElement>("[data-pair]").forEach(
      (button) =>
        (button.onclick = () => {
          const edge = graph.edges.find((e) => e.id === button.dataset.pair);
          if (edge) selectEdge(edge);
        }),
    );
    document
      .querySelectorAll<HTMLButtonElement>("[data-node]")
      .forEach(
        (button) =>
          (button.onclick = () => select(Number(button.dataset.node), true)),
      );
    const focus = $<HTMLButtonElement>("#focus");
    if (focus)
      focus.onclick = () => {
        filters.focus = filters.focus === selected ? null : selected;
        refresh(true);
      };
    const recenter = $<HTMLButtonElement>("#recenter");
    if (recenter)
      recenter.onclick = () => {
        if (selected !== null)
          renderer.fit(
            true,
            new Set([selected, ...(graph.neighbors.get(selected) || [])]),
          );
      };
  }
  function inspect() {
    const body = $("#inspector-body"),
      models = selectedModels(data, filters),
      edge = graph.edges.find((e) => e.id === pair),
      profile = filters.view === "profile";
    if (edge) {
      $("#inspector-label").textContent = profile
        ? "SIMILAR ERROR PROFILES"
        : "CONFUSION PAIR";
      body.innerHTML =
        `<h1 class="pair-header">${esc(displayName(data.nodes[edge.a]))}<span>${profile ? "↔" : "→"}</span>${esc(displayName(data.nodes[edge.b]))}</h1><div class="inspector-actions"><button class="secondary" data-node="${edge.a}">Inspect ${profile ? "first" : "actual"}</button><button class="secondary" data-node="${edge.b}">Inspect ${profile ? "second" : "guessed"}</button></div>` +
        (profile
          ? `<ul class="fact-list"><li>Cosine similarity <b>${edge.similarity.toFixed(3)}</b></li><li>Mutual ranks <b>${edge.ranks![0]} / ${edge.ranks![1]}</b></li><li>Photos per species <b>${data.nodes[edge.a].support} / ${data.nodes[edge.b].support}</b></li></ul><p class="intro">These species receive similar incorrect predictions from the same models. This does not require models to confuse them with each other.</p>`
          : `<ul class="fact-list"><li>Wrong answers <b>${num(edge.count)}</b></li><li>Models sharing this pair <b>${edge.shared} / ${models.length}</b></li><li>Actual species photos <b>${data.nodes[edge.a].support}</b></li></ul><h3>By model</h3><table class="breakdown"><thead><tr><th>Model</th><th>Count / rate</th></tr></thead><tbody>${models.map((i) => `<tr><td>${esc(modelNames[data.models[i].id])}${data.models[i].provisional ? " *" : ""}</td><td>${edge.counts[i]} · ${percent(edge.counts[i], data.nodes[edge.a].support)}</td></tr>`).join("")}</tbody></table><p class="note">Rates use this actual species’ photo count. Multiple models can make the same mistake on the same photo.</p>`);
    } else if (selected !== null) {
      const n = data.nodes[selected],
        t = speciesTotals(data, selected, models),
        group = n[profile ? "profile" : "direct"]![2];
      $("#inspector-label").textContent = "SPECIES INSPECTOR";
      const links = graph.edges
        .filter((e) => e.a === selected || e.b === selected)
        .sort((a, b) => (b.count || b.similarity) - (a.count || a.similarity));
      body.innerHTML =
        `<h1>${esc(displayName(n))}</h1>${displayName(n) !== n.name ? `<p class="scientific">${esc(n.name)}</p>` : ""}<span class="group-label"><i style="--color:${palette[group] || "#727c7d"}"></i>Community ${group + 1}${!n.support ? " · predicted only" : ""}</span><div class="inspector-actions"><button class="primary" id="focus">${filters.focus === selected ? "Show full graph" : "Isolate neighborhood"}</button><button class="secondary" id="recenter">Center</button></div><ul class="fact-list"><li>Actual photos per model <b>${n.support}</b></li><li>Correct answers <b>${t.exact} / ${t.opportunities}</b></li><li>Wrong guesses of other species <b>${sum(models.map((i) => t.outgoing[i]))}</b></li><li>Incorrect guesses of this species <b>${sum(models.map((i) => t.incoming[i]))}</b></li>${t.excluded ? `<li>Other outcomes <b>${t.excluded}</b></li>` : ""}</ul>` +
        (n.support
          ? `<h3>Exact accuracy by model</h3>${models.map((i) => `<div class="accuracy-row"><div class="accuracy-label"><span>${esc(modelNames[data.models[i].id])}${data.models[i].provisional ? " *" : ""}</span><span>${n.correct[i]}/${n.support}</span></div><div class="bar"><i style="width:${(100 * n.correct[i]) / n.support}%"></i></div></div>`).join("")}`
          : `<p class="intro">This species appears only in model predictions, so it has no measured recognition accuracy in this dataset.</p>`) +
        (n.support > 0 && n.support < 5
          ? '<p class="note">Fewer than five photos: this species’ pattern may be unstable.</p>'
          : "") +
        `<h3>${profile ? "Closest error profiles" : "Strongest visible connections"}</h3><div class="pair-list">${links.slice(0, 8).map(pairButton).join("") || '<p class="note">No connections under these filters.</p>'}</div>`;
    } else {
      $("#inspector-label").textContent = "NETWORK OVERVIEW";
      const strongest = graph.edges
        .slice()
        .sort((a, b) => (b.count || b.similarity) - (a.count || a.similarity));
      body.innerHTML = `<h1>${profile ? "Shared error patterns." : "Explore the network."}</h1><p class="intro">${profile ? "Explore species with similar error profiles across all eight models. A connection must rank among both species’ nearest neighbors." : "One network of actual species and incorrect guesses. Select a node or connection to see the evidence behind it."}</p><ul class="fact-list"><li>Unique benchmark photos <b>${num(data.photos)}</b></li><li>Models included <b>${models.length}</b></li><li>True species <b>${data.nodes.filter((n) => n.support).length}</b></li><li>${profile ? "Mutual connections" : "Visible mistakes"} <b>${num(profile ? graph.edges.length : graph.errors)}</b></li></ul><h3>${profile ? "Closest error profiles" : "Most repeated visible pairs"}</h3><div class="pair-list">${strongest.slice(0, 7).map(pairButton).join("") || '<p class="note">No pairs match the current filters.</p>'}</div>`;
    }
    if (models.some((i) => data.models[i].provisional))
      body.insertAdjacentHTML(
        "beforeend",
        '<p class="provisional">* Some scores retain extractor-review flags. See the saved matrices for their status.</p>',
      );
    body.insertAdjacentHTML(
      "beforeend",
      `<a class="source-link" href="https://github.com/qforge-dev/wildlife-CSI-bench/tree/main/runs" target="_blank" rel="noreferrer">View source runs ↗</a>`,
    );
    wireInspector();
  }
  document.querySelectorAll<HTMLButtonElement>("[data-mode]").forEach(
    (b) =>
      (b.onclick = () => {
        filters.view = b.dataset.mode as Filters["view"];
        filters.group = null;
        filters.focus = null;
        selected = null;
        pair = null;
        refresh(true);
      }),
  );
  document.querySelectorAll<HTMLInputElement>("[data-model]").forEach(
    (c) =>
      (c.onchange = () => {
        const i = Number(c.dataset.model);
        filters.models = c.checked
          ? [...filters.models, i].sort((a, b) => a - b)
          : filters.models.filter((m) => m !== i);
        filters.minShared = Math.min(
          filters.minShared,
          Math.max(1, filters.models.length),
        );
        refresh();
      }),
  );
  $("#models-all").onclick = () => {
    filters.models = [...defaults.models];
    refresh();
  };
  $("#models-none").onclick = () => {
    filters.models = [];
    filters.minShared = 1;
    refresh();
  };
  $<HTMLInputElement>("#min-count").oninput = () => {
    filters.minCount = +$<HTMLInputElement>("#min-count").value;
    refresh();
  };
  $<HTMLInputElement>("#min-shared").oninput = () => {
    filters.minShared = +$<HTMLInputElement>("#min-shared").value;
    refresh();
  };
  $<HTMLSelectElement>("#k").onchange = () => {
    filters.k = +$<HTMLSelectElement>("#k").value;
    refresh();
  };
  $<HTMLSelectElement>("#support").onchange = () => {
    filters.minSupport = +$<HTMLSelectElement>("#support").value;
    refresh(true);
  };
  $<HTMLSelectElement>("#group").onchange = () => {
    filters.group =
      $<HTMLSelectElement>("#group").value === ""
        ? null
        : +$<HTMLSelectElement>("#group").value;
    filters.focus = null;
    refresh(true);
  };
  $<HTMLInputElement>("#predicted").onchange = () => {
    filters.predictedOnly = $<HTMLInputElement>("#predicted").checked;
    refresh();
  };
  $("#reset").onclick = () => {
    applyState(presetState("shared", data), true);
  };
  $<HTMLSelectElement>("#preset").onchange = () =>
    applyState(presetState($<HTMLSelectElement>("#preset").value, data), true);
  window.addEventListener("popstate", () =>
    applyState(readState(new URLSearchParams(location.search), data)),
  );
  $("#copy-link").onclick = async () => {
    camera = renderer.getCamera();
    syncUrl();
    try {
      await navigator.clipboard.writeText(location.href);
      $("#copy-link").textContent = "Copied";
      window.setTimeout(() => {
        $("#copy-link").textContent = "Copy link";
      }, 1800);
    } catch {
      const input = $<HTMLInputElement>("#share-url");
      input.value = location.href;
      $<HTMLDialogElement>("#share-dialog").showModal();
      input.select();
    }
  };
  $("#clear-selection").onclick = () => {
    selected = null;
    pair = null;
    filters.focus = null;
    refresh();
    $("#inspector").classList.remove("open");
    $("#inspector-toggle").setAttribute("aria-expanded", "false");
  };
  $("#zoom-in").onclick = () => renderer.zoomBy(1.5);
  $("#zoom-out").onclick = () => renderer.zoomBy(1 / 1.5);
  $("#fit").onclick = () => renderer.fit();
  const search = $<HTMLInputElement>("#search"),
    results = $("#search-results");
  function searchResults() {
    const query = search.value.trim().toLowerCase();
    results.replaceChildren();
    search.setAttribute("aria-expanded", String(!!query));
    if (!query) return;
    const visible = new Set(graph.nodes),
      found = data.nodes
        .map((n, i) => ({ n, i }))
        .filter(
          ({ n, i }) =>
            visible.has(i) &&
            (n.name.toLowerCase().includes(query) ||
              displayName(n).toLowerCase().includes(query)),
        )
        .sort((a, b) => b.n.support - a.n.support)
        .slice(0, 9);
    results.innerHTML =
      found
        .map(
          ({ n, i }) =>
            `<button data-result="${i}"><strong>${esc(displayName(n))}</strong><small>${esc(n.name)} · ${n.support} photos</small></button>`,
        )
        .join("") || "<p>No species match the current filters.</p>";
    results.querySelectorAll<HTMLButtonElement>("button").forEach(
      (b) =>
        (b.onclick = () => {
          const id = Number(b.dataset.result);
          search.value = displayName(data.nodes[id]);
          results.replaceChildren();
          search.setAttribute("aria-expanded", "false");
          select(id, true);
        }),
    );
  }
  search.oninput = searchResults;
  search.onfocus = () => {
    if (search.value) searchResults();
  };
  search.onkeydown = (e) => {
    if (e.key === "ArrowDown") {
      results.querySelector<HTMLButtonElement>("button")?.focus();
      e.preventDefault();
    }
    if (e.key === "Enter")
      results.querySelector<HTMLButtonElement>("button")?.click();
    if (e.key === "Escape") {
      results.replaceChildren();
      search.setAttribute("aria-expanded", "false");
    }
  };
  document.addEventListener("click", (e) => {
    if (!(e.target as HTMLElement).closest(".search-wrap")) {
      results.replaceChildren();
      search.setAttribute("aria-expanded", "false");
    }
  });
  $("#filters-toggle").onclick = () => {
    const opened = $("#filters-panel").classList.toggle("open");
    $("#filters-toggle").setAttribute("aria-expanded", String(opened));
    $("#inspector").classList.remove("open");
    $("#inspector-toggle").setAttribute("aria-expanded", "false");
  };
  $("#inspector-toggle").onclick = () => {
    const opened = $("#inspector").classList.toggle("open");
    $("#inspector-toggle").setAttribute("aria-expanded", String(opened));
    $("#filters-panel").classList.remove("open");
    $("#filters-toggle").setAttribute("aria-expanded", "false");
  };
  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape") {
      $("#filters-panel").classList.remove("open");
      $("#inspector").classList.remove("open");
      $("#filters-toggle").setAttribute("aria-expanded", "false");
      $("#inspector-toggle").setAttribute("aria-expanded", "false");
    }
  });
  refresh();
  if (initial.camera) renderer.setCamera(initial.camera);
  else frameSelection(false);
  $<HTMLInputElement>("#search").value =
    selected === null ? "" : displayName(data.nodes[selected]);
}
start().catch((error) => {
  root.innerHTML = `<div class="loading"><h1>Unable to load the graph</h1><p>${esc(error.message)}</p></div>`;
  console.error(error);
});
