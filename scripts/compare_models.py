"""Compares two completed runs (e.g. results/medgemma and
results/gemma-base) to check whether medical fine-tuning increases or
decreases counterfactual demographic sensitivity relative to the base
model.
"""

import argparse
import json
from pathlib import Path


def main(dir_a: str, dir_b: str, label_a: str, label_b: str) -> None:
    summary_a = json.loads((Path(dir_a) / "summary.json").read_text())
    summary_b = json.loads((Path(dir_b) / "summary.json").read_text())

    print(f"{label_a} overall: {summary_a['overall']}")
    print(f"{label_b} overall: {summary_b['overall']}")

    levels_a = {(r["axis"], r["level"]): r for r in summary_a["per_level"]}
    levels_b = {(r["axis"], r["level"]): r for r in summary_b["per_level"]}

    print(f"\n{'axis':12s} {'level':16s} {label_a + ' median':>18s} {label_b + ' median':>18s}")
    for key in sorted(set(levels_a) | set(levels_b)):
        a = levels_a.get(key, {}).get("median_score", float("nan"))
        b = levels_b.get(key, {}).get("median_score", float("nan"))
        print(f"{key[0]:12s} {key[1]:16s} {a:18.3f} {b:18.3f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir-a", required=True)
    parser.add_argument("--dir-b", required=True)
    parser.add_argument("--label-a", default="medgemma")
    parser.add_argument("--label-b", default="gemma-base")
    args = parser.parse_args()
    main(args.dir_a, args.dir_b, args.label_a, args.label_b)
