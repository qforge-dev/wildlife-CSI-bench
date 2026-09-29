import { test } from "node:test";
import assert from "node:assert/strict";
import {
  defaultFilters,
  matchingPreset,
  presetState,
  readState,
  writeState,
} from "./url-state.ts";
import type { Network } from "./graph-data.ts";
import type { ExplorerState } from "./url-state.ts";

const data: Network = {
  models: [
    "opus",
    "sonnet",
    "astra",
    "glm",
    "muse",
    "luna",
    "deepseek",
    "gpt-6-sol",
  ].map((id) => ({ id, provisional: false })),
  photos: 50,
  tasks_hash: "fixture",
  nodes: ["Procyon lotor", "Lynx rufus", "Urocyon cinereoargenteus"].map(
    (name, i) => ({
      id: `species:${100 + i}`,
      name,
      support: i ? 20 : 10,
      correct: Array(8).fill(0),
      excluded: Array(8).fill(0),
      direct: [i, i, 0],
      profile: [i, i, 0],
    }),
  ),
  edges: [
    [0, 1, Array(8).fill(1)],
    [1, 2, Array(8).fill(2)],
  ],
  knn: [[1, 2, 0.75, 1, 1]],
  summary: { errors: 24, top10_pair_share: 1, shared_by_all_models: 2 },
};
test("new visitors get shared mistakes while explicit direct links can show all pairs", () => {
  assert.equal(readState(new URLSearchParams(), data).filters.minShared, 4);
  assert.equal(
    readState(new URLSearchParams("view=direct"), data).filters.minShared,
    1,
  );
});
test("every filter, focused neighborhood, selected pair, and camera round-trips", () => {
  const state: ExplorerState = {
    filters: {
      ...defaultFilters(data),
      models: [0, 1],
      minCount: 2,
      minShared: 2,
      minSupport: 5,
      group: 0,
      predictedOnly: false,
      k: 5,
      focus: 1,
    },
    selected: null,
    pair: "1:2",
    camera: { x: 123.4, y: -22.5, zoom: 2.5 },
  };
  assert.deepEqual(readState(writeState(state, data), data), state);
  const query = writeState(state, data).toString();
  assert.match(query, /models=opus%2Csonnet/);
  assert.match(query, /from=101/);
  assert.match(query, /group=1/);
});
test("species links resolve stable IDs after node order changes", () => {
  const state = {
    filters: defaultFilters(data),
    selected: 2,
    pair: null,
    camera: null,
  };
  const query = writeState(state, data);
  const reordered = {
    ...data,
    nodes: [data.nodes[2], data.nodes[0], data.nodes[1]],
    edges: [],
  };
  assert.equal(readState(query, reordered).selected, 0);
});
test("empty model selections survive sharing", () => {
  const state = {
    filters: { ...defaultFilters(data), models: [] },
    selected: null,
    pair: null,
    camera: null,
  };
  assert.deepEqual(readState(writeState(state, data), data).filters.models, []);
});
test("invalid URL values do not create out-of-range controls or selections", () => {
  const state = readState(
    new URLSearchParams(
      "view=bad&models=unknown&min=NaN&shared=999&support=4&group=-5&k=19&species=missing&camera=NaN,0,2",
    ),
    data,
  );
  assert.deepEqual(state, {
    filters: defaultFilters(data),
    selected: null,
    pair: null,
    camera: null,
  });
});
test("presets are overridable and mutual pairs accept reverse endpoints", () => {
  const override = readState(
    new URLSearchParams("preset=shared&models=opus&shared=1&support=0"),
    data,
  );
  assert.deepEqual(override.filters.models, [0]);
  assert.equal(override.filters.minSupport, 0);
  const similar = readState(
    new URLSearchParams("preset=similar&from=102&to=101"),
    data,
  );
  assert.equal(similar.pair, "1:2");
  assert.equal(matchingPreset(similar, data), "similar");
  assert.equal(presetState("raccoon", data).filters.focus, 0);
});
test("hidden selections are dropped without weakening requested filters", () => {
  const state = readState(
    new URLSearchParams("view=direct&min=30&species=100&from=101&to=102"),
    data,
  );
  assert.equal(state.selected, null);
  assert.equal(state.pair, null);
  assert.equal(state.filters.minCount, 30);
});
