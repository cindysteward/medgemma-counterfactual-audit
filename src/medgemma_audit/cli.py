"""Runs the full audit over a directory of images for one model with
checkpointing and writes a summary."""

import argparse
import csv
import json
import subprocess
from pathlib import Path

from PIL import Image

from medgemma_audit.vignettes import build_cases
from medgemma_audit.model_runner import VLMRunner
from medgemma_audit.stats_pipeline import (
    compute_diff_vectors, score_all_cases, overall_test,
    per_level_breakdown, intersectional_check,
)

RAW_FIELDS = ["case_id", "variant_type", "axis", "level", "phrasing_id", "vector"]


def _git_backup(paths: list[str], message: str) -> None:
    try:
        subprocess.run(["git", "add", *paths], check=True, capture_output=True)
        subprocess.run(["git", "commit", "-m", message], check=False, capture_output=True)
        subprocess.run(["git", "push"], check=False, capture_output=True)
    except Exception as e:
        print(f"backup push failed (continuing anyway, results are still on local disk): {e}")


def load_done(raw_path: Path) -> dict:
    done = {}
    if not raw_path.exists():
        return done
    with open(raw_path) as f:
        for row in csv.DictReader(f):
            key = (row["case_id"], row["variant_type"], row["axis"], row["level"], int(row["phrasing_id"]))
            done[key] = [float(x) for x in row["vector"].split(",")]
    return done


def run(image_dir: str, out_dir: str, model_id: str) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_path = out_dir / "raw_outputs.csv"

    done = load_done(raw_path)
    print(f"resuming with {len(done)} calls already completed" if done else "starting fresh")

    runner = VLMRunner(model_id)
    write_header = not raw_path.exists()
    raw_file = open(raw_path, "a", newline="")
    writer = csv.DictWriter(raw_file, fieldnames=RAW_FIELDS)
    if write_header:
        writer.writeheader()

    image_paths = sorted(Path(image_dir).glob("*.png"))
    for img_path in image_paths:
        case_id = img_path.stem
        image = Image.open(img_path).convert("RGB")

        for case in build_cases(case_id, str(img_path)):
            key = (case.case_id, case.variant_type, case.axis, case.level, case.phrasing_id)
            if key in done:
                continue
            vec = runner.run_case(image, case.text)
            writer.writerow({
                "case_id": case.case_id, "variant_type": case.variant_type,
                "axis": case.axis, "level": case.level, "phrasing_id": case.phrasing_id,
                "vector": ",".join(str(v) for v in vec),
            })
            raw_file.flush()
            done[key] = vec
        print(f"done: {case_id}")
        _git_backup(["results/"], f"checkpoint after {case_id}")

    raw_file.close()
    _summarise(raw_path, out_dir)
    _git_backup([str(out_dir)], f"final results for {out_dir}")


def _summarise(raw_path: Path, out_dir: Path) -> None:
    vectors_by_case = {}
    with open(raw_path) as f:
        for row in csv.DictReader(f):
            case_id = row["case_id"]
            vec = [float(x) for x in row["vector"].split(",")]
            entry = vectors_by_case.setdefault(
                case_id, {"baseline": None, "demographic": [], "intersectional": [], "control": []},
            )
            if row["variant_type"] == "baseline":
                entry["baseline"] = vec
            else:
                entry[row["variant_type"]].append((row["axis"], row["level"], int(row["phrasing_id"]), vec))

    vectors_by_case = {k: v for k, v in vectors_by_case.items() if v["baseline"] is not None}
    if not vectors_by_case:
        raise RuntimeError(
            f"no completed cases found in {raw_path}, refusing to write a summary. "
            "Check --image-dir actually contains images before rerunning."
        )

    diffs = compute_diff_vectors(vectors_by_case)
    results = score_all_cases(diffs)

    scores_path = out_dir / "scores.csv"
    with open(scores_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["case_id", "variant_type", "axis", "level", "phrasing_id", "mahalanobis_score"])
        for r in results:
            w.writerow([r.case_id, r.variant_type, r.axis, r.level, r.phrasing_id, r.mahalanobis_score])

    summary = {
        "method": "Mahalanobis distance via refstat.MahalanobisScorer (Ledoit-Wolf shrinkage), "
                  "fit per-case on control-condition difference vectors",
        "overall": overall_test(results),
        "per_level": per_level_breakdown(results),
        "intersectional": intersectional_check(results),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nwrote {scores_path} and {out_dir / 'summary.json'}")
    print(json.dumps(summary["overall"], indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-dir", required=True)
    parser.add_argument("--out-dir", required=True, help="e.g. results/medgemma or results/gemma-base")
    parser.add_argument("--model-id", default="google/medgemma-4b-it")
    args = parser.parse_args()
    run(args.image_dir, args.out_dir, args.model_id)
