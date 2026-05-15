"""Qwen2.5-VL-7B HF loader. Native function-calling; exposes gradients for attacks."""

from __future__ import annotations

from typing import Any, cast

import torch
from PIL import Image

from ..types import AttackInputs
from .base import VLMBase, VLMGenerateResult

_MIN_TEMP: float = 1e-5  # temperature floor for generation kwargs


class QwenVL(VLMBase):
    family = "qwen_vl"
    supports_gradients = True

    def __init__(
        self,
        hf_id: str = "Qwen/Qwen2.5-VL-7B-Instruct",
        torch_dtype: torch.dtype = torch.float16,
        device_map: str = "auto",
        revision: str = "main",
    ) -> None:
        # Deferred import: avoid loading transformers at package import time so
        # CI + lightweight tests don't pay the cost.
        from pathlib import Path

        from transformers import AutoModelForVision2Seq, AutoProcessor

        self.model_id = hf_id
        from_pretrained_kwargs: dict[str, Any] = {}
        if not Path(hf_id).is_dir():
            from_pretrained_kwargs["revision"] = revision
        self.processor = AutoProcessor.from_pretrained(hf_id, **from_pretrained_kwargs)
        self.model = AutoModelForVision2Seq.from_pretrained(
            hf_id,
            torch_dtype=torch_dtype,
            device_map=device_map,
            **from_pretrained_kwargs,
        )
        # Set inference mode (equivalent to `.eval()` but avoids ambiguity).
        self.model.train(False)
        eos_id = self.processor.tokenizer.eos_token_id
        if self.model.generation_config.pad_token_id is None:
            self.model.generation_config.pad_token_id = eos_id

    def preprocess_image(self, image: Image.Image) -> Any:
        # Qwen2.5-VL processor expects PIL → tensor via the processor pipeline.
        return self.processor(images=image, return_tensors="pt")

    def _build_processor_inputs(
        self,
        image: Image.Image,
        prompt: str,
        tools_schema: list[dict] | None,
    ) -> Any:
        """Apply Qwen2.5-VL chat template and run the processor; returns inputs
        moved to ``self.model.device``. ``tools_schema`` injects native JSON
        function-calling when supplied.
        """
        messages: list[dict] = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            tools=tools_schema,
        )
        return self.processor(
            text=[text],
            images=[image],
            return_tensors="pt",
            padding=True,
        ).to(self.model.device)

    def _decode_generated(self, out: Any, prompt_len: int) -> VLMGenerateResult:
        """Slice the generated suffix, batch-decode, and wrap as VLMGenerateResult."""
        generated_ids = out.sequences[:, prompt_len:]
        text_out = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return VLMGenerateResult(
            text=text_out,
            tokens=generated_ids[0].tolist(),
            finish_reason="stop",
        )

    def generate(
        self,
        image: Image.Image,
        prompt: str,
        *,
        max_new_tokens: int = 512,
        temperature: float = 0.0,
        seed: int | None = None,
        tools_schema: list[dict] | None = None,
    ) -> VLMGenerateResult:
        if seed is not None:
            torch.manual_seed(seed)

        inputs = self._build_processor_inputs(image, prompt, tools_schema)

        with torch.no_grad():
            out = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=temperature > 0.0,
                temperature=max(temperature, _MIN_TEMP),
                return_dict_in_generate=True,
                output_scores=False,
            )

        return self._decode_generated(out, prompt_len=inputs.input_ids.shape[1])

    def forward_with_logits(  # type: ignore[override]
        self,
        image_tensor: torch.Tensor,
        input_ids: torch.Tensor,
        **forward_kwargs: Any,
    ) -> torch.Tensor:
        """Differentiable forward pass used by gradient-based attacks.

        `image_tensor` must be preprocessed (pixel_values, normalized).
        Qwen2.5-VL additionally requires `image_grid_thw` describing the
        (T, H, W) patch grid and `attention_mask` covering `input_ids`. The
        caller (attack harness) precomputes these via `prepare_attack_inputs`
        and passes them through `forward_kwargs` so this method remains a
        thin differentiable wrapper.
        """
        assert self.model.training is False
        outputs = self.model(
            pixel_values=image_tensor,
            input_ids=input_ids,
            output_hidden_states=False,
            return_dict=True,
            **forward_kwargs,
        )
        return cast(torch.Tensor, outputs.logits)

    def generate_from_pixel_values(  # type: ignore[override]
        self,
        pixel_values: torch.Tensor,
        prompt: str,
        *,
        template_image: Image.Image,
        image_grid_thw: torch.Tensor,
        max_new_tokens: int = 512,
        temperature: float = 0.0,
        seed: int | None = None,
        tools_schema: list[dict] | None = None,
    ) -> VLMGenerateResult:
        """Bypass image processor — feed pre-prepared (perturbed) pixel_values.

        Used by adversarial inference: PGD optimises ``pixel_values`` directly,
        and we sidestep re-running the image processor (which would re-quantise
        the perturbation) by handing the already-normalised tensor to the model.

        ``template_image`` must be the original (clean) PIL with the same H/W
        as the source of ``pixel_values`` so that ``apply_chat_template`` emits
        the matching number of ``<|image_pad|>`` tokens. The processor's
        pixel_values output is discarded — only its tokenized prompt is kept,
        and the perturbed tensor is substituted into ``model.generate``.
        """
        if seed is not None:
            torch.manual_seed(seed)

        # Run full processor so image-token count matches pixel_values; we then
        # discard processor's pixel_values and use the (perturbed) ones we got.
        proc_inputs = self._build_processor_inputs(template_image, prompt, tools_schema)

        with torch.no_grad():
            out = self.model.generate(
                pixel_values=pixel_values,
                image_grid_thw=image_grid_thw,
                input_ids=proc_inputs["input_ids"],
                attention_mask=proc_inputs["attention_mask"],
                max_new_tokens=max_new_tokens,
                do_sample=temperature > 0.0,
                temperature=max(temperature, _MIN_TEMP),
                return_dict_in_generate=True,
                output_scores=False,
            )

        return self._decode_generated(out, prompt_len=proc_inputs["input_ids"].shape[1])

    def prepare_attack_inputs(
        self,
        image: Image.Image,
        prompt: str,
        tools_schema: list[dict] | None = None,
    ) -> AttackInputs:
        """Precompute the tensors attacks need to invoke `forward_with_logits`.

        Returns a dict with at minimum:
          - `pixel_values` (B, C, H', W') — normalized image tensor
          - `input_ids` (B, T_prompt) — prompt token ids
          - `attention_mask` (B, T_prompt)
          - `image_grid_thw` (N, 3) — Qwen2.5-VL vision patch grid
        """
        messages: list[dict] = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            tools=tools_schema,
        )
        inputs = self.processor(
            text=[text],
            images=[image],
            return_tensors="pt",
            padding=True,
        ).to(self.model.device)
        return cast(AttackInputs, dict(inputs))
