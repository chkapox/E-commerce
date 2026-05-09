from __future__ import annotations

from typing import Sequence

import torch
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration

from .config import DEFAULT_PRODUCT_PROMPT, pick_device

from peft import PeftModel


def strip_prompt_echo(text: str, prompt: str | None) -> str:
    text = " ".join(text.split()).strip()
    prompt = " ".join((prompt or "").split()).strip()
    if prompt and text.lower().startswith(prompt.lower()):
        text = text[len(prompt) :].lstrip(" ,:;-")
    return text


class BlipCaptioner:
    def __init__(self, model_name: str, adapter_path: str | None = None):
        self.model_name = model_name
        self.device, self.amp_dtype = pick_device()

        self.processor = BlipProcessor.from_pretrained(model_name)
        self.model = BlipForConditionalGeneration.from_pretrained(model_name, use_safetensors=True)
        if adapter_path:
            self.model = PeftModel.from_pretrained(self.model, adapter_path)
        self.model.to(self.device)
        self.model.eval()

    @torch.no_grad()
    def caption(
        self,
        image_path: str,
        max_new_tokens: int = 40,
        num_beams: int = 3,
        prompt: str | None = DEFAULT_PRODUCT_PROMPT,
    ) -> str:
        return self.caption_batch(
            [image_path],
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
            prompt=prompt,
        )[0]

    @torch.no_grad()
    def caption_batch(
        self,
        image_paths: Sequence[str],
        max_new_tokens: int = 40,
        num_beams: int = 3,
        prompt: str | None = DEFAULT_PRODUCT_PROMPT,
    ) -> list[str]:
        if not image_paths:
            return []

        images = [Image.open(image_path).convert("RGB") for image_path in image_paths]
        processor_kwargs = {
            "images": images,
            "return_tensors": "pt",
            "padding": True,
        }
        if prompt:
            processor_kwargs["text"] = [prompt] * len(images)

        inputs = self.processor(**processor_kwargs)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        use_amp = self.device in {"cuda", "mps"} and self.amp_dtype == torch.float16
        if use_amp:
            with torch.autocast(device_type=self.device, dtype=self.amp_dtype):
                out = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    num_beams=num_beams,
                )
        else:
            out = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                num_beams=num_beams,
            )

        decoded = self.processor.batch_decode(out, skip_special_tokens=True)
        return [strip_prompt_echo(text, prompt) for text in decoded]
