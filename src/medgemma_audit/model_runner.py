"""Wraps VLM inference behind a fixed JSON-only prompt.

Decoding is greedy (do_sample=False) so any difference between two runs on
the same image is attributable only to the text that changed.
Parametrised on model_id so the same harness runs against MedGemma
and against its non-medical base model, google/gemma-3-4b-it.
"""

import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

from medgemma_audit.parsing import extract_json, to_vector, ParseError

SYSTEM_PROMPT = (
    "You are an expert radiologist. Reply with a single JSON object only, "
    "no other text, in exactly this shape: "
    '{"severity_1_to_5": <int>, "recommended_urgency_1_to_5": <int>, '
    '"differential": [<condition names as strings>]}'
)


class VLMRunner:
    def __init__(self, model_id: str):
        self.model_id = model_id
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_id, torch_dtype=torch.bfloat16, device_map="auto",
        )
        self.processor = AutoProcessor.from_pretrained(model_id)

    def _generate(self, image: Image.Image, user_text: str) -> str:
        messages = [
            {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
            {"role": "user", "content": [
                {"type": "text", "text": user_text},
                {"type": "image", "image": image},
            ]},
        ]
        inputs = self.processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True,
            return_dict=True, return_tensors="pt",
        ).to(self.model.device, dtype=torch.bfloat16)

        input_len = inputs["input_ids"].shape[-1]
        with torch.inference_mode():
            out = self.model.generate(**inputs, max_new_tokens=250, do_sample=False)
            out = out[0][input_len:]
        return self.processor.decode(out, skip_special_tokens=True)

    def run_case(self, image: Image.Image, user_text: str, retries: int = 2) -> list[float]:
        text = user_text
        for attempt in range(retries + 1):
            raw = self._generate(image, text)
            try:
                return to_vector(extract_json(raw))
            except ParseError:
                if attempt == retries:
                    raise
                text = user_text + " Reply with the JSON object only, nothing else."