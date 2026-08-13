from medgemma_audit.parsing import extract_json, to_vector, ParseError
import pytest


def test_clean_json():
    raw = '{"severity_1_to_5": 3, "recommended_urgency_1_to_5": 2, "differential": ["Cardiomegaly"]}'
    vec = to_vector(extract_json(raw))
    assert vec[0] == 3.0 and vec[1] == 2.0


def test_fenced_json():
    raw = 'Here is my assessment:\n```json\n{"severity_1_to_5": 4, "recommended_urgency_1_to_5": 4, "differential": []}\n```'
    assert extract_json(raw)["severity_1_to_5"] == 4


def test_json_with_surrounding_prose():
    raw = 'Based on the image:\n\n{"severity_1_to_5": 1, "recommended_urgency_1_to_5": 1, "differential": []}\nLet me know.'
    assert extract_json(raw)["severity_1_to_5"] == 1


def test_unparseable_raises():
    with pytest.raises(ParseError):
        extract_json("I can't provide a diagnosis without more context.")
