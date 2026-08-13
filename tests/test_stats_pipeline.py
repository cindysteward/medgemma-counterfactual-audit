import numpy as np
from medgemma_audit.stats_pipeline import (
    compute_diff_vectors, score_all_cases, overall_test,
    per_level_breakdown, intersectional_check, benjamini_hochberg,
)


def _build_synthetic(rng, n_cases=10):
    demo_axes = {"sex": ["male", "female"], "race": ["Black", "White", "Asian", "Hispanic"]}
    inter_combos = [(r, s) for r in ["Black", "White", "Asian", "Hispanic"] for s in ["male", "female"]]
    vectors_by_case = {}
    for i in range(n_cases):
        baseline = rng.normal(2, 0.5, size=6)
        demo = []
        for axis, levels in demo_axes.items():
            for level in levels:
                for p in range(3):
                    vec = baseline + rng.normal(0, 0.15, size=6)
                    if axis == "race" and level == "Black":
                        vec = vec + np.array([-0.9, -0.7, 0, 0, 0, 0])
                    demo.append((axis, level, p, vec))
        inter = []
        for race, sex in inter_combos:
            vec = baseline + rng.normal(0, 0.15, size=6)
            if race == "Black" and sex == "female":
                vec = vec + np.array([-2.2, -1.8, 0, 0, 0, 0])
            elif race == "Black":
                vec = vec + np.array([-0.9, -0.7, 0, 0, 0, 0])
            inter.append(("race_x_sex", f"{race}_{sex}", 0, vec))
        control = []
        for axis, levels in [("referral_source", ["a", "b", "c"]), ("weekday", ["mon", "fri"])]:
            for level in levels:
                for p in range(3):
                    control.append((axis, level, p, baseline + rng.normal(0, 0.15, size=6)))
        vectors_by_case[f"case{i}"] = {
            "baseline": baseline, "demographic": demo, "intersectional": inter, "control": control,
        }
    return vectors_by_case


def test_benjamini_hochberg_monotonic():
    adj = benjamini_hochberg([0.001, 0.04, 0.2, 0.9])
    assert all(adj[i] <= adj[i + 1] + 1e-9 for i in range(len(adj) - 1))


def test_injected_effect_is_top_ranked_after_correction():
    rng = np.random.default_rng(11)
    vectors_by_case = _build_synthetic(rng)
    diffs = compute_diff_vectors(vectors_by_case)
    results = score_all_cases(diffs)

    breakdown = per_level_breakdown(results)
    assert breakdown[0]["axis"] == "race" and breakdown[0]["level"] == "Black"
    assert breakdown[0]["significant_at_05"]
    assert breakdown[0]["elevated_fraction"] == 1.0


def test_intersectional_superadditivity_detected():
    rng = np.random.default_rng(11)
    vectors_by_case = _build_synthetic(rng)
    diffs = compute_diff_vectors(vectors_by_case)
    results = score_all_cases(diffs)

    inter = intersectional_check(results)
    black_female = next(r for r in inter if r["race"] == "Black" and r["sex"] == "female")
    assert black_female["exceeds_both_single_axes"]
