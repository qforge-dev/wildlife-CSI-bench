# /// script
# requires-python = ">=3.12"
# dependencies = ["networkx>=3.4,<4", "numpy>=2,<3", "scipy>=1.14,<2"]
# ///
"""Measure error concentration and graph communities in saved confusion matrices."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import networkx as nx
import numpy as np


def analyze(path: Path) -> dict:
    raw = path.read_bytes()
    matrix = json.loads(raw)
    overall, labels = matrix["overall"], matrix["labels"]
    edges, excluded = [], Counter()
    for actual, predictions in overall["counts"].items():
        for predicted, count in predictions.items():
            if actual == predicted:
                continue
            if labels[predicted]["kind"] == "species":
                edges.append({"source": actual, "target": predicted, "count": count})
            else:
                excluded[predicted] += count
    edges.sort(key=lambda edge: (-edge["count"], edge["source"], edge["target"]))
    total_errors = sum(edge["count"] for edge in edges)
    assert total_errors + sum(excluded.values()) + overall["exact"] == overall["total"]

    graph = nx.Graph()
    outgoing, incoming = Counter(), Counter()
    for edge in edges:
        a, b, count = edge["source"], edge["target"], edge["count"]
        outgoing[a] += count
        incoming[b] += count
        graph.add_edge(a, b, weight=graph.get_edge_data(a, b, {}).get("weight", 0) + count)
    assert graph.size(weight="weight") == total_errors

    def communities(seed: int) -> list[dict]:
        if not graph:
            return []
        groups = nx.community.louvain_communities(graph, seed=seed, weight="weight")
        rows = []
        for members in groups:
            internal = [
                edge for edge in edges if edge["source"] in members and edge["target"] in members
            ]
            internal_errors = sum(edge["count"] for edge in internal)
            rows.append(
                {
                    "members": sorted(members),
                    "species": len(members),
                    "internal_errors": internal_errors,
                    "share_of_wrong_species_answers": internal_errors / total_errors,
                    "top_pairs": internal[:5],
                }
            )
        return sorted(rows, key=lambda row: (-row["internal_errors"], row["members"]))

    groups = communities(42)
    memberships = {node: index for index, group in enumerate(groups) for node in group["members"]}
    nodes = []
    for key in sorted(graph):
        nodes.append(
            {
                "id": key,
                "name": labels[key]["name"] or f"Species ID {labels[key]['taxon_id']}",
                "observed_names": labels[key].get("observed_names", []),
                "community": memberships[key],
                "actual_photos": overall["row_totals"].get(key, 0),
                "correct": overall["counts"].get(key, {}).get(key, 0),
                "outgoing_errors": outgoing[key],
                "incoming_errors": incoming[key],
            }
        )

    def concentration(values) -> dict:
        ordered = sorted(values, reverse=True)
        cumulative, curve = 0, []
        for rank, count in enumerate(ordered, 1):
            cumulative += count
            curve.append([rank, cumulative])
        return {
            "distinct": len(ordered),
            "top_10_errors": sum(ordered[:10]),
            "top_10_share": sum(ordered[:10]) / total_errors if total_errors else 0,
            "items_for_half_of_errors": next(
                (rank for rank, count in curve if count >= total_errors / 2), 0
            ),
            "cumulative_counts": curve,
        }

    common = sorted(
        matrix["actual_labels"], key=lambda key: (-overall["row_totals"][key], labels[key]["name"])
    )[:15]

    def subset_accuracy(data: dict) -> dict:
        n = sum(data["row_totals"].get(key, 0) for key in common)
        exact = sum(data["counts"].get(key, {}).get(key, 0) for key in common)
        rest_n = data["total"] - n
        return {
            "common_photos": n,
            "common_exact": exact,
            "common_accuracy": exact / n if n else None,
            "remaining_photos": rest_n,
            "remaining_exact": data["exact"] - exact,
            "remaining_accuracy": (data["exact"] - exact) / rest_n if rest_n else None,
        }

    seed_checks = []
    for seed in (0, 1, 2, 42, 99):
        found = groups if seed == 42 else communities(seed)
        seed_checks.append(
            {"seed": seed, "largest_by_internal_errors": found[0] if found else None}
        )
    return {
        "model_id": matrix["model_id"],
        "source_matrix_sha256": hashlib.sha256(raw).hexdigest(),
        "tasks_hash": matrix["tasks_hash"],
        "score_complete": matrix["score_complete"],
        "provisional_due_to_extractor": matrix["provisional_due_to_extractor"],
        "photos": overall["total"],
        "exact": overall["exact"],
        "all_nonexact_outcomes": overall["total"] - overall["exact"],
        "wrong_species_answers": total_errors,
        "excluded_non_species_outcomes": dict(excluded),
        "single_occurrence_pairs": sum(edge["count"] == 1 for edge in edges),
        "pair_concentration": concentration(edge["count"] for edge in edges),
        "actual_species_concentration": concentration(outgoing.values()),
        "predicted_species_concentration": concentration(incoming.values()),
        "common_species": common,
        "common_species_comparison": subset_accuracy(overall),
        "common_species_by_trace": {
            kind: subset_accuracy(data) for kind, data in matrix["per_clue_type"].items()
        },
        "communities": groups,
        "seed_checks": seed_checks,
        "nodes": nodes,
        "edges": edges,
    }


def explorer_payload(results: list[dict]) -> list[dict]:
    """Keep repeat confusions for the interactive view; all metrics use the full graph."""
    output = []
    for result in results:
        edges = [edge for edge in result["edges"] if edge["count"] >= 2]
        visible = {edge[key] for edge in edges for key in ("source", "target")}
        output.append(
            {
                "model": result["model_id"],
                "errors": result["wrong_species_answers"],
                "excluded": sum(result["excluded_non_species_outcomes"].values()),
                "total": result["photos"],
                "exact": result["exact"],
                "provisional": result["provisional_due_to_extractor"],
                "common": result["common_species_comparison"],
                "pairs": {
                    key: value
                    for key, value in result["pair_concentration"].items()
                    if key != "cumulative_counts"
                },
                "one_off": result["single_occurrence_pairs"],
                "nodes": [
                    {key: value for key, value in node.items() if key != "observed_names"}
                    for node in result["nodes"]
                    if node["id"] in visible
                ],
                "edges": edges,
            }
        )
    return output


def combined_graph(paths: list[Path]) -> dict:
    """Pool directed errors and compare model-specific outgoing error profiles."""
    matrices = [json.loads(path.read_text()) for path in paths]
    if len({m["tasks_hash"] for m in matrices}) != 1:
        raise ValueError("Combined graphs require runs on the same task set")
    if len({m["model_id"] for m in matrices}) != len(matrices):
        raise ValueError("Supply only one matrix per model")
    reference = matrices[0]["overall"]["row_totals"]
    if any(m["overall"]["row_totals"] != reference for m in matrices):
        raise ValueError("Combined graphs require identical true-label support")
    labels = {
        key: label
        for matrix in matrices
        for key, label in matrix["labels"].items()
        if label["kind"] == "species"
    }
    keys = sorted(labels)
    index = {key: i for i, key in enumerate(keys)}
    size, model_count = len(keys), len(matrices)
    pairs = {}
    nodes = []
    profiles = np.zeros((size, size * model_count), dtype=np.float32)
    for key in keys:
        label = labels[key]
        nodes.append(
            {
                "id": key,
                "name": label["name"] or f"Species ID {label['taxon_id']}",
                "support": reference.get(key, 0),
                "correct": [m["overall"]["counts"].get(key, {}).get(key, 0) for m in matrices],
                "excluded": [0] * model_count,
            }
        )
    for model_index, matrix in enumerate(matrices):
        for actual, row in matrix["overall"]["counts"].items():
            a = index[actual]
            for predicted, count in row.items():
                if actual == predicted:
                    continue
                if matrix["labels"][predicted]["kind"] != "species":
                    nodes[a]["excluded"][model_index] += count
                    continue
                b = index[predicted]
                counts = pairs.setdefault((a, b), [0] * model_count)
                counts[model_index] += count
                profiles[a, model_index * size + b] = count / nodes[a]["support"]
    edges = [[a, b, counts] for (a, b), counts in sorted(pairs.items())]
    direct = nx.Graph()
    direct.add_nodes_from(range(size))
    for a, b, counts in edges:
        weight = sum(counts) / (nodes[a]["support"] * model_count)
        direct.add_edge(a, b, weight=direct.get_edge_data(a, b, {}).get("weight", 0) + weight)

    def layout(graph, seed=42):
        if not graph:
            return {}, []
        positions = nx.spring_layout(graph, seed=seed, weight="weight", iterations=150)
        groups = (
            nx.community.louvain_communities(graph, seed=seed, weight="weight")
            if graph.number_of_edges()
            else [{node} for node in graph]
        )
        groups.sort(key=lambda group: (-graph.subgraph(group).size(weight="weight"), sorted(group)))
        membership = {node: i for i, group in enumerate(groups) for node in group}
        return (
            {
                i: [round(float(x), 5), round(float(y), 5), membership[i]]
                for i, (x, y) in positions.items()
            },
            groups,
        )

    direct_positions, direct_groups = layout(direct)
    norms = np.linalg.norm(profiles, axis=1)
    valid = np.flatnonzero(norms > 0)
    normalized = profiles[valid] / norms[valid, None]
    similarity = np.clip(normalized @ normalized.T, 0, 1)
    np.fill_diagonal(similarity, 0)
    rankings = {}
    for local, node in enumerate(valid):
        neighbors = sorted(
            (j for j in range(len(valid)) if similarity[local, j] > 1e-7),
            key=lambda j: (-float(similarity[local, j]), int(valid[j])),
        )[:20]
        for rank, other in enumerate(neighbors, 1):
            rankings[int(node), int(valid[other])] = (rank, float(similarity[local, other]))
    knn = []
    for (a, b), (rank, cosine) in sorted(rankings.items()):
        reverse = rankings.get((b, a))
        if a < b and reverse:
            knn.append([a, b, round(cosine, 6), rank, reverse[0]])
    profile_graph = nx.Graph()
    profile_graph.add_nodes_from(map(int, valid))
    profile_graph.add_weighted_edges_from((a, b, c) for a, b, c, r, s in knn if max(r, s) <= 10)
    profile_positions, _ = layout(profile_graph)
    for i, node in enumerate(nodes):
        node["direct"] = direct_positions[i]
        node["profile"] = profile_positions.get(i)
    pair_totals = sorted((sum(counts) for _, _, counts in edges), reverse=True)
    errors = sum(pair_totals)
    group_summaries = []
    for group in direct_groups:
        internal = [edge for edge in edges if edge[0] in group and edge[1] in group]
        group_summaries.append(
            {
                "species": len(group),
                "errors": sum(sum(e[2]) for e in internal),
                "top_pairs": sorted(internal, key=lambda e: -sum(e[2]))[:5],
            }
        )
    return {
        "models": [
            {"id": m["model_id"], "provisional": m["provisional_due_to_extractor"]}
            for m in matrices
        ],
        "photos": matrices[0]["overall"]["total"],
        "tasks_hash": matrices[0]["tasks_hash"],
        "source_hashes": [hashlib.sha256(path.read_bytes()).hexdigest() for path in paths],
        "nodes": nodes,
        "edges": edges,
        "knn": knn,
        "summary": {
            "errors": errors,
            "top10_pair_share": sum(pair_totals[:10]) / errors if errors else 0,
            "groups": group_summaries,
            "shared_by_all_models": sum(all(counts) for _, _, counts in edges),
            "shared_by_at_least_four": sum(
                sum(c > 0 for c in counts) >= 4 for _, _, counts in edges
            ),
        },
        "method": {
            "direct": "All off-diagonal species pairs; counts preserved for every model. Correct and non-species outcomes retained on nodes. Communities/layout weighted by mean per-true-species error rate; both directions summed.",
            "knn": "Cosine similarity of outgoing error-rate profiles over (model, predicted species), excluding correct/non-species outcomes. Positive similarities only; mutual top-k neighbors; ties broken by species ID. Coordinates and communities use k=10, seed=42. No-error profiles have no similarity edges.",
            "interpretation": "Repeated answers concern the same photos, not independent observations. Similar error profiles do not imply direct confusion or visual similarity; low-support species are unstable. Communities are exploratory.",
        },
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("matrices", type=Path, nargs="+")
    parser.add_argument("--output-dir", type=Path, default=Path("data/work/confusion-clusters"))
    parser.add_argument(
        "--combined",
        action="store_true",
        help="Also build a pooled graph and mutual-neighbor profiles",
    )
    parser.add_argument(
        "--graph-output", type=Path, help="Write the combined graph to this path as well"
    )
    args = parser.parse_args()
    results = [analyze(path) for path in args.matrices]
    report = {
        "format_version": 1,
        "networkx_version": nx.__version__,
        "method": {
            "edges": "Directed actual-to-predicted species pairs, weighted by photo count; exact answers and non-species outcomes excluded.",
            "communities": "Louvain on the full undirected graph, combining both directions; resolution=1, seed=42.",
            "seed_checks": [0, 1, 2, 42, 99],
            "interpretation": "Exploratory communities are not proof of visual similarity or deployment reliability. Common-species comparison uses the 15 most frequent true species, not the 15 best scores.",
        },
        "models": results,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "analysis.json").write_text(json.dumps(report, indent=2) + "\n")
    (args.output_dir / "explorer-data.json").write_text(
        json.dumps(explorer_payload(results), separators=(",", ":")) + "\n"
    )
    if args.combined or args.graph_output:
        combined = combined_graph(args.matrices)
        encoded = json.dumps(combined, separators=(",", ":")) + "\n"
        (args.output_dir / "combined-data.json").write_text(encoded)
        if args.graph_output:
            args.graph_output.parent.mkdir(parents=True, exist_ok=True)
            args.graph_output.write_text(encoded)
    for result in results:
        largest = (
            result["communities"][0]
            if result["communities"]
            else {"species": 0, "internal_errors": 0}
        )
        print(
            json.dumps(
                {
                    "model": result["model_id"],
                    "species_errors": result["wrong_species_answers"],
                    "top10_pair_share": result["pair_concentration"]["top_10_share"],
                    "pairs_for_half": result["pair_concentration"]["items_for_half_of_errors"],
                    "one_off_pairs": result["single_occurrence_pairs"],
                    "largest_group_species": largest["species"],
                    "largest_group_errors": largest["internal_errors"],
                    "common15_accuracy": result["common_species_comparison"]["common_accuracy"],
                }
            )
        )
