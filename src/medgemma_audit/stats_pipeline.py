"""Aggregates per-case model outputs into a fairness audit summary.

For each fixed image, the baseline vector is subtracted from every variant
vector. A refstat.MahalanobisScorer is fit per case on that case's own
control difference vectors (falling back to a pooled reference if a case
has fewer than 3 control variants), then used to score that case's
demographic difference vectors.
Mann-Whitney U test checks whether demographic scores are larger than control.
Benjamini-Hochberg correction across per-level significance tests.
Includes a descriptive (not a hypothesis test) intersectional check.
"""

from dataclasses import dataclass
import numpy as np
from scipy.stats import mannwhitneyu
from refstat import MahalanobisScorer


@dataclass
class CaseResult:
    case_id: str
    variant_type: str
    axis: str
    level: str
    phrasing_id: int
    diff_vector: np.ndarray
    mahalanobis_score: float


def benjamini_hochberg(pvalues: list[float]) -> np.ndarray:
    pvalues = np.asarray(pvalues, dtype=float)
    n = len(pvalues)
    order = np.argsort(pvalues)
    ranked = pvalues[order]
    adjusted = ranked * n / (np.arange(n) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    adjusted = np.clip(adjusted, 0, 1)
    out = np.empty(n)
    out[order] = adjusted
    return out


def compute_diff_vectors(vectors_by_case: dict) -> dict:
    out = {}
    for case_id, entry in vectors_by_case.items():
        baseline = np.asarray(entry["baseline"], dtype=float)
        out[case_id] = {
            key: [(a, l, p, np.asarray(v, dtype=float) - baseline)
                  for a, l, p, v in entry[key]]
            for key in ("demographic", "intersectional", "control")
        }
    return out


def score_all_cases(diff_vectors_by_case: dict) -> list[CaseResult]:
    all_control_diffs = [v for entry in diff_vectors_by_case.values()
                          for _, _, _, v in entry["control"]]
    pooled_reference = np.array(all_control_diffs)

    results = []
    for case_id, entry in diff_vectors_by_case.items():
        control_diffs = np.array([v for _, _, _, v in entry["control"]])

        scorer = MahalanobisScorer()
        scorer.fit(control_diffs if len(control_diffs) >= 3 else pooled_reference)

        for variant_type in ("control", "demographic", "intersectional"):
            for axis, level, phrasing_id, vec in entry[variant_type]:
                results.append(CaseResult(
                    case_id, variant_type, axis, level, phrasing_id,
                    vec, scorer.score(vec),
                ))
    return results


def overall_test(results: list[CaseResult]) -> dict:
    demo_scores = [r.mahalanobis_score for r in results if r.variant_type == "demographic"]
    control_scores = [r.mahalanobis_score for r in results if r.variant_type == "control"]
    stat, pvalue = mannwhitneyu(demo_scores, control_scores, alternative="greater")
    return {
        "n_demographic": len(demo_scores), "n_control": len(control_scores),
        "median_demographic": float(np.median(demo_scores)),
        "median_control": float(np.median(control_scores)),
        "mannwhitney_u": float(stat), "pvalue": float(pvalue),
    }


def per_level_breakdown(results: list[CaseResult], elevated_threshold: float = 1.0) -> list[dict]:
    control_scores = [r.mahalanobis_score for r in results if r.variant_type == "control"]

    levels = sorted({(r.axis, r.level) for r in results if r.variant_type in ("demographic", "intersectional")})
    rows = []
    for axis, level in levels:
        level_scores = [r.mahalanobis_score for r in results
                         if r.axis == axis and r.level == level]
        stat, pvalue = mannwhitneyu(level_scores, control_scores, alternative="greater")
        elevated_fraction = float(np.mean([s > elevated_threshold for s in level_scores]))
        rows.append({
            "axis": axis, "level": level, "n": len(level_scores),
            "median_score": float(np.median(level_scores)),
            "elevated_fraction": elevated_fraction,
            "pvalue_raw": float(pvalue),
        })

    adjusted = benjamini_hochberg([r["pvalue_raw"] for r in rows])
    for row, adj_p in zip(rows, adjusted):
        row["pvalue_bh_adjusted"] = float(adj_p)
        row["significant_at_05"] = bool(adj_p < 0.05)

    return sorted(rows, key=lambda r: r["pvalue_bh_adjusted"])


def intersectional_check(results: list[CaseResult]) -> list[dict]:
    """Descriptive, flags race x sex combinations whose median score
    exceeds the corresponding race-alone and sex-alone medians.
    """
    single_axis_medians = {}
    for r in results:
        if r.variant_type == "demographic":
            single_axis_medians.setdefault((r.axis, r.level), []).append(r.mahalanobis_score)
    single_axis_medians = {k: float(np.median(v)) for k, v in single_axis_medians.items()}

    grouped = {}
    for r in results:
        if r.variant_type != "intersectional":
            continue
        race, sex = r.level.split("_")
        grouped.setdefault((race, sex), []).append(r.mahalanobis_score)

    out = []
    for (race, sex), scores in grouped.items():
        race_median = single_axis_medians.get(("race", race), float("nan"))
        sex_median = single_axis_medians.get(("sex", sex), float("nan"))
        intersectional_median = float(np.median(scores))
        out.append({
            "race": race, "sex": sex,
            "intersectional_median": intersectional_median,
            "race_alone_median": race_median,
            "sex_alone_median": sex_median,
            "exceeds_both_single_axes": bool(
                intersectional_median > race_median and intersectional_median > sex_median
            ),
        })
    return out