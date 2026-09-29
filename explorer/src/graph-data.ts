export interface Species {
  id: string;
  name: string;
  support: number;
  correct: number[];
  excluded: number[];
  direct: [number, number, number];
  profile: [number, number, number] | null;
}
export interface Network {
  models: { id: string; provisional: boolean }[];
  photos: number;
  tasks_hash: string;
  nodes: Species[];
  edges: [number, number, number[]][];
  knn: [number, number, number, number, number][];
  summary: {
    errors: number;
    top10_pair_share: number;
    shared_by_all_models: number;
  };
}
export interface Filters {
  view: "direct" | "profile";
  models: number[];
  minCount: number;
  minShared: number;
  minSupport: number;
  group: number | null;
  predictedOnly: boolean;
  k: number;
  focus: number | null;
}
export interface Edge {
  id: string;
  a: number;
  b: number;
  count: number;
  shared: number;
  counts: number[];
  similarity: number;
  ranks?: [number, number];
}
export interface GraphView {
  nodes: number[];
  edges: Edge[];
  errors: number;
  totalErrors: number;
  neighbors: Map<number, Set<number>>;
}
export const modelNames: Record<string, string> = {
  astra: "GPT-6 Astra",
  deepseek: "DeepSeek V4.1 Flash",
  glm: "GLM 5.3 Flash",
  "gpt-6-sol": "GPT-6 Sol",
  luna: "GPT-6 Luna",
  muse: "Muse Spark 1.3",
  opus: "Opus 5.5",
  sonnet: "Sonnet 5.5",
};
export const commonNames: Record<string, string> = {
  "Procyon lotor": "Raccoon",
  "Canis latrans": "Coyote",
  "Vulpes vulpes": "Red fox",
  "Urocyon cinereoargenteus": "Gray fox",
  "Urocyon littoralis": "Island fox",
  "Didelphis virginiana": "Virginia opossum",
  "Lontra canadensis": "River otter",
  "Lynx rufus": "Bobcat",
  "Puma concolor": "Cougar",
  "Felis catus": "Domestic cat",
  "Castor canadensis": "North American beaver",
  "Neogale vison": "American mink",
  "Myocastor coypus": "Coypu",
  "Otospermophilus beecheyi": "California ground squirrel",
  "Odocoileus hemionus": "Mule deer",
  "Odocoileus virginianus": "White-tailed deer",
  "Cervus canadensis": "Elk",
  "Capreolus capreolus": "Roe deer",
  "Ovis canadensis": "Bighorn sheep",
  "Sus scrofa": "Wild boar",
  "Meles meles": "European badger",
  "Tyto furcata": "American barn owl",
  "Tyto alba": "Western barn owl",
  "Bubo virginianus": "Great horned owl",
  "Ursus americanus": "Black bear",
  "Panthera pardus": "Leopard",
  "Crocuta crocuta": "Spotted hyena",
  "Hippopotamus amphibius": "Hippopotamus",
  "Loxodonta africana": "African elephant",
  "Lepus californicus": "Black-tailed jackrabbit",
  "Sylvilagus audubonii": "Desert cottontail",
  "Melanerpes carolinus": "Red-bellied woodpecker",
  "Dryobates pubescens": "Downy woodpecker",
};
export const displayName = (node: Species) =>
  commonNames[node.name] || node.name;
export const sum = (values: number[]) => values.reduce((a, b) => a + b, 0);
export const selectedModels = (data: Network, filters: Filters) =>
  filters.view === "profile" ? data.models.map((_, i) => i) : filters.models;

export function deriveGraph(data: Network, filters: Filters): GraphView {
  const modelIds = selectedModels(data, filters);
  const profile = filters.view === "profile";
  const allEdges: Edge[] = profile
    ? data.knn
        .filter(([, , , a, b]) => Math.max(a, b) <= filters.k)
        .map(([a, b, similarity, ra, rb]) => ({
          id: `${a}:${b}`,
          a,
          b,
          similarity,
          ranks: [ra, rb],
          count: 0,
          shared: 0,
          counts: [],
        }))
    : data.edges.map(([a, b, counts]) => ({
        id: `${a}:${b}`,
        a,
        b,
        counts,
        count: sum(modelIds.map((i) => counts[i])),
        shared: modelIds.filter((i) => counts[i] > 0).length,
        similarity: 0,
      }));
  const totalErrors = profile
    ? data.summary.errors
    : sum(allEdges.map((e) => e.count));
  const eligible = new Set(
    data.nodes.flatMap((node, i) => {
      const position = node[profile ? "profile" : "direct"];
      return modelIds.length &&
        position &&
        node.support >= filters.minSupport &&
        (filters.predictedOnly || node.support > 0) &&
        (filters.group === null || position[2] === filters.group)
        ? [i]
        : [];
    }),
  );
  let edges = allEdges.filter(
    (e) =>
      eligible.has(e.a) &&
      eligible.has(e.b) &&
      (profile ||
        (e.count >= filters.minCount && e.shared >= filters.minShared)),
  );
  const neighbors = new Map<number, Set<number>>();
  for (const e of edges) {
    if (!neighbors.has(e.a)) neighbors.set(e.a, new Set());
    if (!neighbors.has(e.b)) neighbors.set(e.b, new Set());
    neighbors.get(e.a)!.add(e.b);
    neighbors.get(e.b)!.add(e.a);
  }
  const restricted =
    filters.minCount > 1 ||
    filters.minShared > 1 ||
    filters.models.length < data.models.length;
  let nodes = [...eligible].filter(
    (i) => profile || !restricted || neighbors.has(i),
  );
  if (filters.focus !== null) {
    const neighborhood = new Set([
      filters.focus,
      ...(neighbors.get(filters.focus) || []),
    ]);
    nodes = nodes.filter((i) => neighborhood.has(i));
    const ids = new Set(nodes);
    edges = edges.filter((e) => ids.has(e.a) && ids.has(e.b));
  }
  return {
    nodes,
    edges,
    neighbors,
    errors: sum(edges.map((e) => e.count)),
    totalErrors,
  };
}

export function speciesTotals(data: Network, index: number, models: number[]) {
  const incoming = data.models.map(() => 0),
    outgoing = data.models.map(() => 0);
  for (const [a, b, counts] of data.edges) {
    if (a === index) counts.forEach((v, i) => (outgoing[i] += v));
    if (b === index) counts.forEach((v, i) => (incoming[i] += v));
  }
  const n = data.nodes[index];
  return {
    incoming,
    outgoing,
    exact: sum(models.map((i) => n.correct[i])),
    excluded: sum(models.map((i) => n.excluded[i])),
    opportunities: n.support * models.length,
  };
}
