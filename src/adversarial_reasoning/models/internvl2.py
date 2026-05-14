"""InternVL2-8B (OpenGVLab) HF loader. Prompt-scaffolded tool calling via ReAct system prompt.

InternVL2 ships custom modeling code, so we load it via ``AutoModel`` with
``trust_remote_code=True``. There is no first-class HF ``Processor`` class — we
use ``AutoTokenizer`` plus an inline ``dynamic_preprocess`` (mirrors the
canonical OpenGVLab/InternVL2-8B model card recipe) to tile the image into
``num_patches`` 448x448 crops.

The wrapper exposes the same four methods as the LLaVA-Next and Qwen-VL
siblings (``generate``, ``forward_with_logits``, ``generate_from_pixel_values``,
``prepare_attack_inputs``), and threads ``num_patches_list`` through the
runner via the optional ``AttackInputs`` key of the same name so PGD can
perturb the already-tiled ``pixel_values`` tensor without losing the per-image
tile layout.
"""

from __future__ import annotations

from typing import Any, cast

import torch
import torchvision.transforms as T
from PIL import Image
from torchvision.transforms.functional import InterpolationMode

from ..types import AttackInputs
from .base import VLMBase, VLMGenerateResult

_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)
_IMG_START_TOKEN = "<img>"
_IMG_END_TOKEN = "</img>"
_IMG_CONTEXT_TOKEN = "<IMG_CONTEXT>"

_MIN_TEMP: float = 1e-5  # temperature floor for generation kwargs


def _build_transform(input_size: int) -> T.Compose:
    return T.Compose(
        [
            T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
            T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
        ]
    )


def _find_closest_aspect_ratio(
    aspect_ratio: float,
    target_ratios: list[tuple[int, int]],
    width: int,
    height: int,
    image_size: int,
) -> tuple[int, int]:
    best_ratio_diff = float("inf")
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif (
            ratio_diff == best_ratio_diff
            and area > 0.5 * image_size * image_size * ratio[0] * ratio[1]
        ):
            best_ratio = ratio
    return best_ratio


def _dynamic_preprocess(
    image: Image.Image,
    min_num: int = 1,
    max_num: int = 12,
    image_size: int = 448,
    use_thumbnail: bool = True,
) -> list[Image.Image]:
    """OpenGVLab dynamic-tile recipe: pick the closest aspect-ratio grid in
    [min_num, max_num] tile budget, tile, optionally append a global thumbnail.
    """
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height
    target_ratios = sorted(
        {
            (i, j)
            for n in range(min_num, max_num + 1)
            for i in range(1, n + 1)
            for j in range(1, n + 1)
            if min_num <= i * j <= max_num
        },
        key=lambda x: x[0] * x[1],
    )
    target_aspect_ratio = _find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size
    )
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]
    resized_img = image.resize((target_width, target_height))
    processed_images: list[Image.Image] = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size,
        )
        processed_images.append(resized_img.crop(box))
    if use_thumbnail and len(processed_images) != 1:
        processed_images.append(image.resize((image_size, image_size)))
    return processed_images


class InternVL2(VLMBase):
    family = "internvl2"
    supports_gradients = True

    def __init__(
        self,
        hf_id: str = "OpenGVLab/InternVL2-8B",
        torch_dtype: torch.dtype = torch.float16,
        device_map: str = "auto",
        revision: str = "main",
        max_tiles: int = 12,
        image_size: int = 448,
    ) -> None:
        from transformers import AutoModel, AutoTokenizer

        self.model_id = hf_id
        self.max_tiles = max_tiles
        self.image_size = image_size
        self.tokenizer = AutoTokenizer.from_pretrained(
            hf_id,
            revision=revision,
            trust_remote_code=True,
            use_fast=False,
        )
        self.model = AutoModel.from_pretrained(
            hf_id,
            torch_dtype=torch_dtype,
            device_map=device_map,
            revision=revision,
            trust_remote_code=True,
        )
        self.model.train(False)
        eos_id = self.tokenizer.eos_token_id
        if eos_id is not None and self.model.generation_config.pad_token_id is None:
            self.model.generation_config.pad_token_id = eos_id
        self._transform = _build_transform(image_size)
        self._img_context_token_id = self.tokenizer.convert_tokens_to_ids(_IMG_CONTEXT_TOKEN)
        self.model.img_context_token_id = self._img_context_token_id
        self._num_image_token = int(self.model.num_image_token)

    def preprocess_image(self, image: Image.Image) -> torch.Tensor:
        tiles = _dynamic_preprocess(
            image, min_num=1, max_num=self.max_tiles, image_size=self.image_size, use_thumbnail=True
        )
        pixel_values = torch.stack([self._transform(t) for t in tiles])
        return pixel_values

    def _format_prompt(self, prompt: str, tools_schema: list[dict] | None) -> str:
        """InternLM2-Chat conversation template with `<image>` placeholder.

        Mirrors LlavaNext._build_system_prompt — same ReAct tool-call scaffold,
        same JSON emission contract. Family-specific surrounding chat markers
        are applied later when we tokenize via ``model.chat``.
        """
        system_block = self._build_system_prompt(tools_schema)
        body = f"{system_block}\n\n{prompt}" if system_block else prompt
        return f"<image>\n{body}"

    def _build_system_prompt(self, tools_schema: list[dict] | None) -> str:
        if not tools_schema:
            return ""
        lines = [
            "You are a medical imaging agent. You have access to the following tools:",
            "",
        ]
        for tool in tools_schema:
            spec = tool.get("function", tool)
            lines.append(f"- {spec['name']}: {spec.get('description', '')}")
            lines.append(f"  args: {spec.get('parameters', {})}")
        lines += [
            "",
            "To use a tool, emit a single line of JSON exactly like:",
            '{"tool": "<tool_name>", "args": {...}}',
            "",
            "Reason step-by-step. Call tools as needed. Finish with a plain-text conclusion.",
        ]
        return "\n".join(lines)

    def _build_query(self, prompt: str, num_patches: int) -> str:
        """Replicate ``InternVLChatModel.chat`` query construction so we can
        tokenize manually (needed for gradient-enabled forward passes that
        bypass ``model.chat``).

        InternLM2-Chat conversation template — same role markers as Qwen
        (``<|im_start|>`` / ``<|im_end|>``). The system message defaults to
        the value carried by the loaded model, falling back to the
        InternVL2 model card default when the attribute is absent (e.g.
        when the wrapper is exercised in a unit test with a stub model).
        """
        system_message = getattr(self.model, "system_message", "You are a helpful assistant.")
        base_query = (
            f"<|im_start|>system\n{system_message}<|im_end|>\n"
            f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
        )
        image_tokens = (
            _IMG_START_TOKEN
            + _IMG_CONTEXT_TOKEN * self._num_image_token * num_patches
            + _IMG_END_TOKEN
        )
        return base_query.replace("<image>", image_tokens, 1)

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

        question = self._format_prompt(prompt, tools_schema)
        pixel_values = self.preprocess_image(image).to(
            self.model.device, dtype=next(self.model.parameters()).dtype
        )

        gen_cfg = {
            "max_new_tokens": max_new_tokens,
            "do_sample": temperature > 0.0,
            "temperature": max(temperature, 1e-5),
        }
        with torch.no_grad():
            response = self.model.chat(
                tokenizer=self.tokenizer,
                pixel_values=pixel_values,
                question=question,
                generation_config=gen_cfg,
            )
        return VLMGenerateResult(text=response, finish_reason="stop")

    def forward_with_logits(  # type: ignore[override]
        self,
        image_tensor: torch.Tensor,
        input_ids: torch.Tensor,
        **forward_kwargs: Any,
    ) -> torch.Tensor:
        """Differentiable forward pass.

        Caller threads ``attention_mask`` and ``num_patches_list`` (the latter
        already encoded in ``image_tensor.shape[0]``) via ``forward_kwargs``.
        InternVL2's modeling code expands the IMG_CONTEXT_TOKEN occurrences
        in ``input_ids`` against the patches in ``pixel_values``.
        """
        assert self.model.training is False
        # num_patches_list only meaningful for batched generate, not forward.
        forward_kwargs.pop("num_patches_list", None)
        # InternVL2 modeling code requires explicit image_flags: a 1-D
        # LongTensor of length pixel_values.shape[0] selecting which patches
        # contribute vit_embeds. Single-image attack/clean forward → all ones.
        if "image_flags" not in forward_kwargs:
            forward_kwargs["image_flags"] = torch.ones(
                image_tensor.shape[0],
                dtype=torch.long,
                device=image_tensor.device,
            )
        model_dtype = next(self.model.parameters()).dtype
        outputs = self.model(
            pixel_values=image_tensor.to(model_dtype),
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
        num_patches_list: torch.Tensor | None = None,
        max_new_tokens: int = 512,
        temperature: float = 0.0,
        seed: int | None = None,
        tools_schema: list[dict] | None = None,
    ) -> VLMGenerateResult:
        """Bypass image processor — feed pre-prepared (perturbed) pixel_values.

        ``template_image`` is used only to infer ``num_patches`` when the
        caller does not provide ``num_patches_list``. The number of patches
        encoded in ``pixel_values.shape[0]`` is the load-bearing quantity for
        IMG_CONTEXT_TOKEN expansion.
        """
        if seed is not None:
            torch.manual_seed(seed)

        question = self._format_prompt(prompt, tools_schema)
        if num_patches_list is None:
            num_patches = int(pixel_values.shape[0])
        else:
            num_patches = int(num_patches_list.sum().item())

        query = self._build_query(question, num_patches=num_patches)
        model_inputs = self.tokenizer(query, return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            output_ids = self.model.generate(
                pixel_values=pixel_values,
                input_ids=model_inputs["input_ids"],
                attention_mask=model_inputs["attention_mask"],
                max_new_tokens=max_new_tokens,
                do_sample=temperature > 0.0,
                temperature=max(temperature, _MIN_TEMP),
            )
        gen_ids = output_ids[0, model_inputs["input_ids"].shape[1] :]
        text_out = self.tokenizer.decode(gen_ids, skip_special_tokens=True)
        return VLMGenerateResult(text=text_out, finish_reason="stop")

    def prepare_attack_inputs(
        self,
        image: Image.Image,
        prompt: str,
        tools_schema: list[dict] | None = None,
    ) -> AttackInputs:
        """Tokenize prompt + tile image into tensors ready for attack.

        Returns dict with ``pixel_values``, ``input_ids``, ``attention_mask``,
        and ``num_patches_list`` (one-element tensor; multi-image batches not
        supported in the attack pipeline today).
        """
        question = self._format_prompt(prompt, tools_schema)
        pixel_values = self.preprocess_image(image).to(
            self.model.device, dtype=next(self.model.parameters()).dtype
        )
        num_patches = int(pixel_values.shape[0])
        query = self._build_query(question, num_patches=num_patches)
        tok = self.tokenizer(query, return_tensors="pt").to(self.model.device)
        out: dict[str, Any] = {
            "pixel_values": pixel_values,
            "input_ids": tok["input_ids"],
            "attention_mask": tok["attention_mask"],
            "num_patches_list": torch.tensor([num_patches], device=self.model.device),
        }
        return cast(AttackInputs, out)
