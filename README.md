# medgemma-counterfactual-audit

Counterfactual demographic fairness audit for MedGemma (google/medgemma-4b-it).

## What this does

Holds one chest X-ray pixel-identical and varies the demographic framing
of the accompanied clinical text (sex, race, age band). Compares the
resulting shift in MedGemma's generated severity and differential diagnosis
against the shift produced by neutral, non-clinical wording changes, using each
case's own neutral-wording shifts as the reference for normal prompt sensitivity.

A Mahalanobis distance (via refstat.MahalanobisScorer, Ledoit-Wolf shrinkage)
scores each demographic shift against that reference. A Mann-Whitney U test
checks whether demographic shifts are larger than control shifts across the
whole sample.

## Motivation of this project vs. subgroup comparison

Existing benchmarks (FairMedFM / MedVLMBench, arXiv:2407.00983; vlm-fairness,
Science Advances) measure fairness by comparing outcomes across real patients
grouped by recorded demographic label, using MedGemma's vision encoder for
classification. Neither exercises MedGemma's generative output, or neither
holds the image fixed while perturbing only the demographic framing. MedEqualQA
(arXiv:2510.12818) does counterfactual perturbation but is text-only, pronoun
axis only, on non-VLM medical LLMs, and explicitly flags extending to imaging
and to race/age as future work. This project aims to propose a method to fill
that specific gap and utilises the audit design from my Bachelor Thesis "Breaking
the Bias: Addressing the Social Biases in Artificial Natural Language Models for
Neuroscientific and Medical Implementation" (Steward, 2023; presented at the 3rd
Connected Learning Symposium (2024), and the Black Scholar and Expert Conference
(2023)). (See https://github.com/cindysteward/demoparity)

## Data

Images: NIH ChestX-ray14 (Wang et al., 2017).
See scripts/download_sample_data.py.

## Model

google/medgemma-4b-it (Sellergren et al., "MedGemma Technical Report",
arXiv:2507.05201). 
Health AI Developer Foundations terms of use:
https://developers.google.com/health-ai-developer-foundations/terms

## Install

    pip install -e ".[dev]"

## Run

    python -m medgemma_audit.cli --image-dir data/sample_images --out results.csv

## Note

This is a research audit of model behaviour, not a clinical tool.
MedGemma also states its outputs "are not intended to directly inform
clinical diagnosis, patient management decisions, treatment recommendations."

