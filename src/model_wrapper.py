from __future__ import annotations

import re
from typing import Sequence

import torch
from PIL import Image, ImageFile
from transformers import BlipProcessor, BlipForConditionalGeneration

from .config import DEFAULT_PRODUCT_PROMPT, configure_hf_offline_mode, pick_device

from peft import PeftModel


ImageFile.LOAD_TRUNCATED_IMAGES = True


INLINE_GENERATION_ARTIFACTS = (
    " creaked ",
    " creaking ",
)

TRAILING_GENERATION_ARTIFACTS = (
    " person",
    " people",
    " it",
)


def strip_prompt_echo(text: str, prompt: str | None) -> str:
    text = " ".join(text.split()).strip()
    marker_matches = list(re.finditer(r"marketplace\s+description\s*:", text, flags=re.IGNORECASE))
    if marker_matches:
        return text[marker_matches[-1].end() :].lstrip(" ,:;-")

    prompt = " ".join((prompt or "").split()).strip()
    if prompt and text.lower().startswith(prompt.lower()):
        text = text[len(prompt) :].lstrip(" ,:;-")
    return text


def prompt_for_index(prompt: str | Sequence[str] | None, idx: int) -> str | None:
    if isinstance(prompt, str) or prompt is None:
        return prompt
    return prompt[idx] if idx < len(prompt) else None


def clean_generation_artifacts(text: str) -> str:
    text = " ".join(text.replace(" - ", "-").split()).strip()
    text = re.sub(r"\b([a-zA-Z][\w'-]*)(?:\s+\1\b)+", r"\1", text, flags=re.IGNORECASE)
    padded = f" {text} "
    for artifact in INLINE_GENERATION_ARTIFACTS:
        padded = padded.replace(artifact, " ")
    text = " ".join(padded.split()).strip()
    lowered = text.lower()
    for artifact in TRAILING_GENERATION_ARTIFACTS:
        if lowered.endswith(artifact):
            text = text[: -len(artifact)].rstrip(" ,.;")
            lowered = text.lower()
    return text


class BlipCaptioner:
    def __init__(self, model_name: str, adapter_path: str | None = None, local_files_only: bool = False):
        if local_files_only:
            configure_hf_offline_mode()
        self.model_name = model_name
        self.device, self.amp_dtype = pick_device()

        self.processor = BlipProcessor.from_pretrained(model_name, local_files_only=local_files_only, use_fast=False)
        try:
            self.model = BlipForConditionalGeneration.from_pretrained(
                model_name,
                use_safetensors=True,
                local_files_only=local_files_only,
            )
        except OSError:
            self.model = BlipForConditionalGeneration.from_pretrained(
                model_name,
                use_safetensors=False,
                local_files_only=local_files_only,
            )
        if adapter_path:
            self.model = PeftModel.from_pretrained(self.model, adapter_path, local_files_only=local_files_only)
        self.model.to(self.device)
        self.model.eval()

    @torch.no_grad()
    def caption(
        self,
        image_path: str,
        max_new_tokens: int = 40,
        num_beams: int = 3,
        prompt: str | Sequence[str] | None = DEFAULT_PRODUCT_PROMPT,
        no_repeat_ngram_size: int = 3,
        repetition_penalty: float = 1.15,
        clean_output: bool = True,
    ) -> str:
        return self.caption_batch(
            [image_path],
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
            prompt=prompt,
            no_repeat_ngram_size=no_repeat_ngram_size,
            repetition_penalty=repetition_penalty,
            clean_output=clean_output,
        )[0]

    @torch.no_grad()
    def caption_batch(
        self,
        image_paths: Sequence[str],
        max_new_tokens: int = 40,
        num_beams: int = 3,
        prompt: str | Sequence[str] | None = DEFAULT_PRODUCT_PROMPT,
        no_repeat_ngram_size: int = 3,
        repetition_penalty: float = 1.15,
        clean_output: bool = True,
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
            processor_kwargs["text"] = [prompt] * len(images) if isinstance(prompt, str) else list(prompt)

        inputs = self.processor(**processor_kwargs)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        use_amp = self.device in {"cuda", "mps"} and self.amp_dtype == torch.float16
        if use_amp:
            with torch.autocast(device_type=self.device, dtype=self.amp_dtype):
                out = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    num_beams=num_beams,
                    no_repeat_ngram_size=no_repeat_ngram_size,
                    repetition_penalty=repetition_penalty,
                )
        else:
            out = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                num_beams=num_beams,
                no_repeat_ngram_size=no_repeat_ngram_size,
                repetition_penalty=repetition_penalty,
            )

        decoded = self.processor.batch_decode(out, skip_special_tokens=True)
        if not clean_output:
            return [" ".join(text.split()).strip() for text in decoded]
        return [
            clean_generation_artifacts(strip_prompt_echo(text, prompt_for_index(prompt, idx)))
            for idx, text in enumerate(decoded)
        ]
