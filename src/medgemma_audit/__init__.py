"""Counterfactual fairness audit toolkit for MedGemma.

Holds a chest X-ray image fixed and varies only the demographic framing of
the accompanying clinical text, measuring how MedGemma's generated
assessment and vision attention shift relative to a neutral-wording
control condition. Supports comparing a medically fine-tuned model (MedGemma)
against its non-medical base model (Gemma 3) on the same harness.
"""

__version__ = "0.1.0"
