# /// script
# requires-python = ">=3.12"
# dependencies = ["matplotlib>=3.9,<4"]
# ///
"""Render a top-species confusion chart from a saved matrix; no model calls."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle
from matplotlib.ticker import PercentFormatter

COMMON_NAMES = {
    "Alces alces": "Moose",
    "Bonasa umbellus": "Ruffed grouse",
    "Branta canadensis": "Canada goose",
    "Canis latrans": "Coyote",
    "Canis lupus": "Gray wolf",
    "Lontra canadensis": "River otter",
    "Lynx rufus": "Bobcat",
    "Meleagris gallopavo": "Wild turkey",
    "Odocoileus hemionus": "Mule deer",
    "Odocoileus virginianus": "White-tailed deer",
    "Procyon lotor": "Raccoon",
    "Puma concolor": "Cougar",
    "Sus scrofa": "Wild boar",
    "Urocyon cinereoargenteus": "Gray fox",
    "Ursus americanus": "American black bear",
}


def render(matrix_path: Path, output_dir: Path, title: str | None, top: int):
    report = json.loads(matrix_path.read_text())
    title = title or f"{report['model_id']} · Species confusion"
    if top < 1 or report["normalization"] != "raw_counts":
        raise ValueError("top must be positive and the matrix must contain raw counts")
    overall = report["overall"]
    selected = sorted(
        report["actual_labels"],
        key=lambda key: (-overall["row_totals"][key], report["labels"][key]["name"]),
    )[:top]
    names = [report["labels"][key]["name"] for key in selected]
    display = [COMMON_NAMES.get(name, name) for name in names]
    support = np.array([overall["row_totals"][key] for key in selected])
    counts = np.array(
        [
            [overall["counts"][key].get(predicted, 0) for predicted in selected]
            + [
                sum(
                    count
                    for predicted, count in overall["counts"][key].items()
                    if predicted not in selected
                )
            ]
            for key in selected
        ]
    )
    assert np.array_equal(counts.sum(axis=1), support)
    fractions = counts / support[:, None]
    n = len(selected)
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"confusion-{report['model_id']}-top{n}"
    description = (
        f"{title}. The {n} most frequent actual species cover {int(support.sum())} of "
        f"{overall['total']} photos. Rows are actual species; columns are predictions. "
        "Numbers are photo counts, and color shows each cell's share of its entire row. "
        "The diagonal contains correct species identifications. Other predictions includes "
        "every output outside the displayed species. Species are selected by photo count, "
        "with alphabetical scientific-name tie breaking."
    )
    matplotlib.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "svg.hashsalt": "wildlife-csi-confusion-v1",
        }
    )
    for theme in ("light", "dark"):
        dark = theme == "dark"
        background = "#09090b" if dark else "#ffffff"
        foreground = "#f4f4f5" if dark else "#27272a"
        muted = "#a1a1aa" if dark else "#65656e"
        edge = "#3e7464" if dark else "#a5c7bb"
        cmap = LinearSegmentedColormap.from_list(
            theme,
            ["#18181b", "#245b49", "#3f9273", "#7bd6ad"]
            if dark
            else ["#f5f7f6", "#b4dac9", "#4b9e7d", "#155b41"],
        )
        fig = plt.figure(figsize=(12.8, 11), facecolor=background)
        fig.text(0.025, 0.963, title, fontsize=21, color=foreground)
        fig.text(
            0.025,
            0.933,
            f"{n} most common species · {int(support.sum())} of {overall['total']:,} photos · All trace types",
            fontsize=11,
            color=muted,
        )
        fig.text(0.025, 0.865, "ACTUAL SPECIES", color=muted, fontsize=9)
        fig.text(0.62, 0.865, "PREDICTED SPECIES →", color=muted, fontsize=9, ha="center")
        ax = fig.add_axes((0.265, 0.27, 0.71, 0.575))
        plot = ax.imshow(fractions, cmap=cmap, vmin=0, vmax=1, aspect="auto")
        ax.set_xticks(range(n + 1), display + ["Other predictions"])
        plt.setp(ax.get_xticklabels(), rotation=58, ha="right", rotation_mode="anchor")
        ax.tick_params(axis="x", length=0, pad=8, labelsize=10, colors=foreground)
        ax.set_yticks([])
        for r, (name, common, total) in enumerate(zip(names, display, support, strict=True)):
            ax.text(
                -0.022,
                r - 0.12,
                common,
                transform=ax.get_yaxis_transform(),
                ha="right",
                va="center",
                color=foreground,
                fontsize=10.3,
            )
            ax.text(
                -0.022,
                r + 0.22,
                f"{name} · n={total}",
                transform=ax.get_yaxis_transform(),
                ha="right",
                va="center",
                color=muted,
                fontsize=8,
                style="italic",
            )
        ax.set_xticks(np.arange(-0.5, n + 1, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, n, 1), minor=True)
        ax.grid(which="minor", color=background, linewidth=1.5)
        ax.tick_params(which="minor", bottom=False, left=False)
        for spine in ax.spines.values():
            spine.set_visible(False)
        # Separate the aggregate column from the 15 species-to-species columns.
        ax.axvline(n - 0.5, color=muted, linewidth=1, alpha=0.65, zorder=3)
        for r in range(n):
            ax.add_patch(
                Rectangle(
                    (r - 0.5, r - 0.5), 1, 1, fill=False, edgecolor=edge, linewidth=1, zorder=3
                )
            )
            for c in range(n + 1):
                value = int(counts[r, c])
                if not value and r != c:
                    continue
                fraction = fractions[r, c]
                color = ("#10251c" if dark else "#ffffff") if fraction >= 0.6 else foreground
                ax.text(
                    c,
                    r,
                    str(value),
                    ha="center",
                    va="center",
                    color=color,
                    fontsize=10.5,
                    weight="bold" if r == c else "normal",
                )
        cax = fig.add_axes((0.66, 0.064, 0.315, 0.012))
        colorbar = fig.colorbar(
            plot, cax=cax, orientation="horizontal", ticks=[0, 0.25, 0.5, 0.75, 1]
        )
        colorbar.ax.xaxis.set_major_formatter(PercentFormatter(1))
        colorbar.ax.tick_params(length=0, labelsize=8.5, colors=muted, pad=5)
        colorbar.outline.set_visible(False)
        fig.text(0.66, 0.09, "Color: share of each actual-species row", fontsize=9, color=muted)
        fig.text(
            0.025,
            0.092,
            "Numbers = photos · Outlined diagonal = correct species",
            fontsize=9,
            color=muted,
        )
        fig.text(
            0.025,
            0.069,
            "Other predictions keeps every answer outside the species shown.",
            fontsize=9,
            color=muted,
        )
        fig.text(
            0.025,
            0.046,
            "Rows selected by frequency; ties sorted by scientific name.",
            fontsize=9,
            color=muted,
        )
        path = output_dir / f"{stem}-{theme}.svg"
        fig.savefig(
            path,
            facecolor=background,
            metadata={
                "Date": None,
                "Creator": "Wildlife CSI",
                "Title": title,
                "Description": description,
            },
        )
        path.write_text("\n".join(line.rstrip() for line in path.read_text().splitlines()) + "\n")
        plt.close(fig)
        print(path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("matrix", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("assets/charts"))
    parser.add_argument("--title", help="Chart title; defaults to the model ID and chart type.")
    parser.add_argument("--top", type=int, default=15)
    args = parser.parse_args()
    render(args.matrix, args.output_dir, args.title, args.top)
