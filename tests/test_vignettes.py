from medgemma_audit.vignettes import build_cases


def test_case_count():
    cases = build_cases("case001", "/data/x.png")
    assert len(cases) == 1 + 27 + 8 + 15


def test_article_agreement():
    """Checks for the wrong article appearing next to the demographic word,
    rather than assuming every phrasing puts an article directly in front
    of it, since some phrasings ("a patient who identifies as Asian")
    legitimately have no article there at all.
    """
    cases = build_cases("case001", "/data/x.png")
    for c in cases:
        if c.variant_type != "demographic":
            continue
        if c.level == "Asian":
            assert "a Asian" not in c.text
        if c.level in ("Black", "White", "Hispanic"):
            assert f"an {c.level}" not in c.text


def test_phrasings_are_distinct():
    cases = build_cases("case001", "/data/x.png")
    black_texts = {c.text for c in cases if c.level == "Black" and c.variant_type == "demographic"}
    assert len(black_texts) == 3


def test_intersectional_present():
    cases = build_cases("case001", "/data/x.png")
    inter = [c for c in cases if c.variant_type == "intersectional"]
    assert len(inter) == 8
    assert {c.level for c in inter} == {
        f"{r}_{s}" for r in ["Black", "White", "Asian", "Hispanic"] for s in ["male", "female"]
    }
