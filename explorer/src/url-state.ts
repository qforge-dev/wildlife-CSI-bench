import { deriveGraph, displayName } from "./graph-data.ts";
import type { Filters, Network } from "./graph-data.ts";

export interface CameraState {
  x: number;
  y: number;
  zoom: number;
}
export interface ExplorerState {
  filters: Filters;
  selected: number | null;
  pair: string | null;
  camera: CameraState | null;
}
export const presets = [
  { id: "shared", label: "Mistakes shared across models" },
  { id: "raccoon", label: "Guessed as raccoon" },
  { id: "similar", label: "Similar errors: bobcat & gray fox" },
  { id: "all", label: "All species and pairs" },
];
export function defaultFilters(data: Network): Filters {
  return {
    view: "direct",
    models: data.models.map((_, i) => i),
    minCount: 1,
    minShared: 1,
    minSupport: 0,
    group: null,
    predictedOnly: true,
    k: 10,
    focus: null,
  };
}
function speciesIndex(value: string | null, data: Network): number | null {
  if (!value) return null;
  const q = value.toLowerCase();
  const index = data.nodes.findIndex(
    (n) =>
      n.id === value ||
      n.id === `species:${value}` ||
      n.name.toLowerCase() === q ||
      displayName(n).toLowerCase() === q,
  );
  return index < 0 ? null : index;
}
export function presetState(id: string, data: Network): ExplorerState {
  const filters = defaultFilters(data);
  let selected: number | null = null,
    pair: string | null = null;
  if (id === "shared" || id === "raccoon") {
    filters.minShared = Math.min(4, data.models.length);
    filters.minSupport = 5;
  }
  if (id === "raccoon") {
    selected = speciesIndex("Procyon lotor", data);
    filters.focus = selected;
  }
  if (id === "similar") {
    filters.view = "profile";
    filters.minSupport = 5;
    filters.k = 5;
    const a = speciesIndex("Lynx rufus", data),
      b = speciesIndex("Urocyon cinereoargenteus", data);
    if (a !== null && b !== null) pair = `${Math.min(a, b)}:${Math.max(a, b)}`;
  }
  return { filters, selected, pair, camera: null };
}
const parameters = [
  "preset",
  "view",
  "models",
  "min",
  "shared",
  "support",
  "group",
  "predicted",
  "k",
  "species",
  "from",
  "to",
  "focus",
  "camera",
];
function integer(
  value: string | null,
  fallback: number,
  min: number,
  max: number,
): number {
  if (value === null || !/^\d+$/.test(value)) return fallback;
  const n = Number(value);
  return Number.isSafeInteger(n) && n >= min && n <= max ? n : fallback;
}
export function readState(
  params: URLSearchParams,
  data: Network,
): ExplorerState {
  const recognized = parameters.some((key) => params.has(key));
  const preset = params.get("preset") || (recognized ? "all" : "shared");
  const state = presetState(preset, data),
    f = state.filters;
  if (params.get("view") === "profile" || params.get("view") === "direct")
    f.view = params.get("view") as Filters["view"];
  if (params.has("models")) {
    const requested = params.get("models")!.split(",");
    const models = data.models.flatMap((m, i) =>
      requested.includes(m.id) ? [i] : [],
    );
    if (
      models.length ||
      params.get("models") === "none" ||
      params.get("models") === ""
    )
      f.models = models;
  }
  f.minCount = integer(params.get("min"), f.minCount, 1, 30);
  f.minShared = Math.min(
    integer(params.get("shared"), f.minShared, 1, data.models.length),
    Math.max(1, f.models.length),
  );
  const support = integer(params.get("support"), f.minSupport, 0, 10);
  if ([0, 3, 5, 10].includes(support)) f.minSupport = support;
  const k = integer(params.get("k"), f.k, 3, 20);
  if ([3, 5, 10, 20].includes(k)) f.k = k;
  if (params.get("predicted") === "0" || params.get("predicted") === "1")
    f.predictedOnly = params.get("predicted") === "1";
  const group = integer(params.get("group"), 0, 1, data.nodes.length) - 1;
  if (
    group >= 0 &&
    data.nodes.some(
      (n) => n[f.view === "profile" ? "profile" : "direct"]?.[2] === group,
    )
  )
    f.group = group;
  if (params.has("focus")) f.focus = speciesIndex(params.get("focus"), data);
  if (params.has("species")) {
    state.selected = speciesIndex(params.get("species"), data);
    state.pair = null;
  }
  const a = speciesIndex(params.get("from"), data),
    b = speciesIndex(params.get("to"), data);
  if (a !== null && b !== null) {
    state.pair =
      f.view === "profile"
        ? `${Math.min(a, b)}:${Math.max(a, b)}`
        : `${a}:${b}`;
    state.selected = null;
  }
  const graph = deriveGraph(data, f);
  if (state.selected !== null && !graph.nodes.includes(state.selected))
    state.selected = null;
  if (state.pair && !graph.edges.some((e) => e.id === state.pair))
    state.pair = null;
  const raw = params.get("camera");
  if (raw && raw.split(",").every((v) => v.trim() !== "")) {
    const [x, y, zoom, ...extra] = raw.split(",").map(Number);
    if (
      !extra.length &&
      [x, y, zoom].every(Number.isFinite) &&
      Math.abs(x) <= 100_000 &&
      Math.abs(y) <= 100_000 &&
      zoom >= 0.01 &&
      zoom <= 100
    )
      state.camera = { x, y, zoom };
  }
  return state;
}
export function writeState(
  state: ExplorerState,
  data: Network,
): URLSearchParams {
  const f = state.filters,
    params = new URLSearchParams({ view: f.view });
  if (f.models.length !== data.models.length)
    params.set(
      "models",
      f.models.map((i) => data.models[i].id).join(",") || "none",
    );
  if (f.minCount !== 1) params.set("min", String(f.minCount));
  if (f.minShared !== 1) params.set("shared", String(f.minShared));
  if (f.minSupport !== 0) params.set("support", String(f.minSupport));
  if (f.group !== null) params.set("group", String(f.group + 1));
  if (!f.predictedOnly) params.set("predicted", "0");
  if (f.k !== 10) params.set("k", String(f.k));
  const key = (i: number) => data.nodes[i].id.replace(/^species:/, "");
  if (f.focus !== null) params.set("focus", key(f.focus));
  if (state.selected !== null) params.set("species", key(state.selected));
  if (state.pair) {
    const [a, b] = state.pair.split(":").map(Number);
    params.set("from", key(a));
    params.set("to", key(b));
  }
  if (state.camera)
    params.set(
      "camera",
      [state.camera.x, state.camera.y, state.camera.zoom]
        .map((v) => Number(v.toFixed(4)))
        .join(","),
    );
  return params;
}
export function matchingPreset(state: ExplorerState, data: Network): string {
  const withoutCamera = { ...state, camera: null },
    key = writeState(withoutCamera, data).toString();
  return (
    presets.find(
      (p) => writeState(presetState(p.id, data), data).toString() === key,
    )?.id || "custom"
  );
}
