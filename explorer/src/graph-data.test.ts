import { test } from "node:test";
import assert from "node:assert/strict";
import { deriveGraph, speciesTotals } from "./graph-data.ts";
import type { Filters, Network } from "./graph-data.ts";

const data: Network = {
  models: [
    { id: "a", provisional: false },
    { id: "b", provisional: false },
  ],
  photos: 5,
  tasks_hash: "fixture",
  nodes: [
    {
      id: "a",
      name: "A",
      support: 3,
      correct: [1, 2],
      excluded: [0, 0],
      direct: [0, 0, 0],
      profile: [0, 0, 0],
    },
    {
      id: "b",
      name: "B",
      support: 2,
      correct: [1, 0],
      excluded: [0, 1],
      direct: [1, 1, 0],
      profile: [1, 1, 0],
    },
    {
      id: "c",
      name: "C",
      support: 0,
      correct: [0, 0],
      excluded: [0, 0],
      direct: [2, 2, 1],
      profile: null,
    },
  ],
  edges: [
    [0, 1, [2, 0]],
    [0, 2, [0, 1]],
    [1, 0, [1, 1]],
  ],
  knn: [[0, 1, 0.8, 2, 6]],
  summary: { errors: 5, top10_pair_share: 1, shared_by_all_models: 1 },
};
const filters: Filters = {
  view: "direct",
  models: [0, 1],
  minCount: 1,
  minShared: 1,
  minSupport: 0,
  group: null,
  predictedOnly: true,
  k: 10,
  focus: null,
};
test("all model counts reconcile and retain predicted-only species", () => {
  const graph = deriveGraph(data, filters);
  assert.equal(graph.errors, 5);
  assert.equal(graph.nodes.length, 3);
  const totals = speciesTotals(data, 0, [0, 1]);
  assert.equal(
    totals.exact + totals.outgoing.reduce((a, b) => a + b, 0) + totals.excluded,
    totals.opportunities,
  );
});
test("model selection changes both weights and sharing thresholds", () => {
  const graph = deriveGraph(data, { ...filters, models: [0] });
  assert.equal(graph.totalErrors, 3);
  assert.equal(graph.edges.length, 2);
  assert.equal(deriveGraph(data, { ...filters, minShared: 2 }).errors, 2);
  assert.equal(deriveGraph(data, { ...filters, models: [] }).nodes.length, 0);
});
test("mutual nearest neighbors require both ranks to satisfy k", () => {
  assert.equal(
    deriveGraph(data, { ...filters, view: "profile", k: 5 }).edges.length,
    0,
  );
  const graph = deriveGraph(data, {
    ...filters,
    view: "profile",
    k: 10,
    models: [],
  });
  assert.equal(graph.edges.length, 1);
  assert.deepEqual(graph.nodes, [0, 1]);
});
test("support filters remove incident edges but retain the full denominator", () => {
  const graph = deriveGraph(data, { ...filters, minSupport: 1 });
  assert.equal(graph.errors, 4);
  assert.equal(graph.totalErrors, 5);
  assert.deepEqual(graph.nodes, [0, 1]);
});
