"""Generates figures to show results. Ran after cli.py
has produced scores.csv and summary.json in a results directory.
"""

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt


def plot_score_distributions(overall, demo_scores, control_scores, out_path):
    fig, ax = plt.subplots(figsize=(5, 4), dpi=300)
    ax.boxplot([control_scores, demo_scores],
               tick_labels=["control\n(wording only)", "demographic"],
               patch_artist=True, showfliers=True,
               boxprops=dict(facecolor="#cfd8e3"), medianprops=dict(color="black"))
    ax.axhline(1.0, linestyle="--", linewidth=1, color="grey")
    ax.set_ylabel("Mahalanobis score (refstat)")
    ax.set_title(f"p = {overall['pvalue']:.2e}")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_per_level(rows, out_path, top_n=12):
    rows = rows[:top_n]
    labels = [f"{r['axis']}:{r['level']}" for r in rows]
    medians = [r["median_score"] for r in rows]
    colors = ["#c0392b" if r["significant_at_05"] else "#95a5a6" for r in rows]

    fig, ax = plt.subplots(figsize=(6, 4.5), dpi=300)
    ax.barh(range(len(rows)), medians, color=colors)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels(labels, fontsize=8)
    ax.invert_yaxis()
    ax.axvline(1.0, linestyle="--", linewidth=1, color="grey")
    ax.set_xlabel("median Mahalanobis score")
    ax.set_title("per-level effect (red = significant after BH correction)")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def main(results_dir):
    results_dir = Path(results_dir)
    summary = json.loads((results_dir / "summary.json").read_text())

    demo_scores, control_scores = [], []
    with open(results_dir / "scores.csv") as f:
        for row in csv.DictReader(f):
            score = float(row["mahalanobis_score"])
            if row["variant_type"] == "demographic":
                demo_scores.append(score)
            elif row["variant_type"] == "control":
                control_scores.append(score)

    plot_score_distributions(summary["overall"], demo_scores, control_scores,
                              results_dir / "fig_distributions.png")
    plot_per_level(summary["per_level"], results_dir / "fig_per_level.png")
    print(f"wrote figures to {results_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", required=True)
    args = parser.parse_args()
    main(args.results_dir)
