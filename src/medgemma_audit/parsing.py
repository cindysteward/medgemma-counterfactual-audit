"""Parses MedGemma's free-text response into a fixed-length numeric vector.

MedGemma is prompted to answer in a strict JSON block. Real generations
sometimes wrap that JSON in markdown fences or add a short prose lead-in,
so parsing tolerates both rather than assuming a clean json.loads() works.
"""

import json
import re

PATHOLOGY_KEYS = [
    "atelectasis", "cardiomegaly", "consolidation", "edema",
    "effusion", "mass_or_nodule",
]

VECTOR_LABELS = ["severity", "urgency"] + PATHOLOGY_KEYS

JSON_BLOCK_RE = re.compile(r"\{.*\}", re.DOTALL)


class ParseError(Exception):
    pass


def extract_json(raw_text: str) -> dict:
    """Pulls the first JSON object out of a model response, tolerating
    markdown code fences and any prose before or after it.
    """
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw_text, re.DOTALL)
    candidate = fenced.group(1) if fenced else None

    if candidate is None:
        match = JSON_BLOCK_RE.search(raw_text)
        candidate = match.group(0) if match else None

    if candidate is None:
        raise ParseError(f"no JSON object found in response: {raw_text[:200]!r}")

    try:
        return json.loads(candidate)
    except json.JSONDecodeError as exc:
        raise ParseError(f"malformed JSON block: {exc}") from exc


def to_vector(parsed: dict) -> list[float]:
    """Converts a parsed response into [severity, urgency, <pathology flags>].

    Missing fields default to 0 rather than raising, since a missing flag
    is itself a meaningful signal, not a parse failure.
    """
    severity = float(parsed.get("severity_1_to_5", 0))
    urgency = float(parsed.get("recommended_urgency_1_to_5", 0))

    differential = parsed.get("differential", [])
    if isinstance(differential, str):
        differential = [differential]
    differential_lower = {str(d).lower().replace(" ", "_") for d in differential}

    flags = [1.0 if key in differential_lower else 0.0 for key in PATHOLOGY_KEYS]
    return [severity, urgency] + flags
