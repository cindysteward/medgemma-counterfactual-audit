"""Diffs where the model's attention lands on the image between two runs
on the same image.

MedGemma encodes each image to 256 tokens (16x16 patch grid), per its model
card. Run probe_attention_shapes() first, before trusting extract_attention_map()
on a full sweep.
"""

import numpy as np
import torch

EXPECTED_IMAGE_TOKEN_COUNT = 256
GRID_SIDE = 16


def probe_attention_shapes(model, processor, image, user_text: str) -> dict:
    """Diagnostic step. Runs one real forward pass and reports what's
    actually available, so you confirm before building on assumptions:
    whether processor exposes image_token_id, how many image-token
    positions actually appear in input_ids, and the shape of one
    attention layer. Run this in Colab as its own cell first.
    """
    messages = [{"role": "user", "content": [
        {"type": "text", "text": user_text}, {"type": "image", "image": image},
    ]}]
    inputs = processor.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=True,
        return_dict=True, return_tensors="pt",
    ).to(model.device, dtype=torch.bfloat16)

    with torch.inference_mode():
        out = model(**inputs, output_attentions=True)

    report = {
        "has_image_token_id_attr": hasattr(processor, "image_token_id"),
        "image_token_id": getattr(processor, "image_token_id", None),
        "n_attention_layers": len(out.attentions),
        "one_layer_shape": tuple(out.attentions[-1].shape),
        "input_ids_length": int(inputs["input_ids"].shape[-1]),
    }
    if report["image_token_id"] is not None:
        positions = (inputs["input_ids"][0] == report["image_token_id"]).nonzero(as_tuple=True)[0]
        report["n_image_token_positions_found"] = int(len(positions))
        report["matches_expected_256"] = int(len(positions)) == EXPECTED_IMAGE_TOKEN_COUNT
    return report


def find_image_token_span(processor, input_ids: torch.Tensor) -> tuple[int, int]:
    image_token_id = getattr(processor, "image_token_id", None)
    if image_token_id is None:
        raise RuntimeError(
            "processor has no image_token_id attribute, run probe_attention_shapes() "
            "first and inspect processor.tokenizer.special_tokens_map manually"
        )
    positions = (input_ids[0] == image_token_id).nonzero(as_tuple=True)[0]
    if len(positions) != EXPECTED_IMAGE_TOKEN_COUNT:
        raise RuntimeError(
            f"expected {EXPECTED_IMAGE_TOKEN_COUNT} image tokens, found {len(positions)}, "
            "run probe_attention_shapes() to see what's actually happening"
        )
    return int(positions[0]), int(positions[-1]) + 1


def extract_attention_map(model, processor, inputs: dict, last_n_layers: int = 4) -> np.ndarray:
    with torch.inference_mode():
        out = model(**inputs, output_attentions=True)

    start, end = find_image_token_span(processor, inputs["input_ids"])
    layers = out.attentions[-last_n_layers:]

    maps = [layer_attn[0, :, -1, start:end].mean(dim=0).float().cpu().numpy()
            for layer_attn in layers]
    return np.mean(maps, axis=0).reshape(GRID_SIDE, GRID_SIDE)


def attention_shift(map_a: np.ndarray, map_b: np.ndarray) -> float:
    a, b = map_a.flatten(), map_b.flatten()
    cos_sim = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12)
    return float(1 - cos_sim)