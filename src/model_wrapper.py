from __future__ import annotations

import torch
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration

from .config import pick_device

from peft import PeftModel

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
    ) -> str:
        image = Image.open(image_path).convert("RGB")
        inputs = self.processor(images=image, return_tensors="pt")
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

        return self.processor.decode(out[0], skip_special_tokens=True).strip()