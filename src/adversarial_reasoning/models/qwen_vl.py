"""Qwen2.5-VL-7B HF loader. Native function-calling; exposes gradients for attacks."""

from __future__ import annotations

from typing import Any

import torch
from PIL import Image

from .base import VLMBase, VLMGenerateResult


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
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

        self.model_id = hf_id
        self.processor = AutoProcessor.from_pretrained(hf_id, revision=revision)
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            hf_id,
            torch_dtype=torch_dtype,
            device_map=device_map,
            revision=revision,
        )
        # Set inference mode (equivalent to `.eval()` but avoids ambiguity).
        self.model.train(False)

    def preprocess_image(self, image: Image.Image) -> Any:
        # Qwen2.5-VL processor expects PIL → tensor via the processor pipeline.
        return self.processor(images=image, return_tensors="pt")

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

        # Qwen2.5-VL chat template. Tool schema injection via `tools` argument
        # when `tools_schema` is supplied — native JSON function calling.
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

        with torch.no_grad():
            out = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=temperature > 0.0,
                temperature=max(temperature, 1e-5),
                return_dict_in_generate=True,
                output_scores=False,
            )

        generated_ids = out.sequences[:, inputs.input_ids.shape[1]:]
        text_out = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return VLMGenerateResult(
            text=text_out,
            tokens=generated_ids[0].tolist(),
            finish_reason="stop",
        )

    def forward_with_logits(
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
        return outputs.logits

    def generate_from_pixel_values(
        self,
        pixel_values: torch.Tensor,
        image_grid_thw: torch.Tensor,
        prompt: str,
        *,
        template_image: Image.Image,
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

        messages: list[dict] = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": template_image},
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
        # Run full processor so image-token count matches pixel_values; we then
        # discard processor's pixel_values and use the (perturbed) ones we got.
        proc_inputs = self.processor(
            text=[text],
            images=[template_image],
            return_tensors="pt",
            padding=True,
        ).to(self.model.device)

        with torch.no_grad():
            out = self.model.generate(
                pixel_values=pixel_values,
                image_grid_thw=image_grid_thw,
                input_ids=proc_inputs["input_ids"],
                attention_mask=proc_inputs["attention_mask"],
                max_new_tokens=max_new_tokens,
                do_sample=temperature > 0.0,
                temperature=max(temperature, 1e-5),
                return_dict_in_generate=True,
                output_scores=False,
            )

        generated_ids = out.sequences[:, proc_inputs["input_ids"].shape[1] :]
        text_out = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return VLMGenerateResult(
            text=text_out,
            tokens=generated_ids[0].tolist(),
            finish_reason="stop",
        )

    def prepare_attack_inputs(
        self,
        image: Image.Image,
        prompt: str,
        tools_schema: list[dict] | None = None,
    ) -> dict[str, Any]:
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
        return dict(inputs)
