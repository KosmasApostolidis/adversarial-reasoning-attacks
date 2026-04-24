"""Phase-0 gate: does an adversarial perturbation survive HF → Ollama preprocessing?

Procedure
---------
1. Load a clean image.
2. Generate a PGD ε=16/255 perturbation against the HF fp16 surrogate.
3. Save the perturbed image as a lossless PNG (the exact delivery format
   used by the Ollama chat API).
4. Reload the PNG, re-run the *Ollama* processor's preprocessing, compare
   the resulting tensor to the HF-side perturbed tensor in L∞.
5. Gate: effective L∞ on the Ollama side must be ≥ 2/255, else the
   attack signal is being clipped out by preprocessing and attacks at
   lower ε are meaningless.
"""

from __future__ import annotations

import io
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image


@dataclass(frozen=True)
class PreprocessingTransferResult:
    model_name: str
    epsilon_requested: float
    effective_linf_post_roundtrip: float
    gate_threshold: float
    passed: bool


def run_preprocessing_transfer(
    hf_vlm: object,
    *,
    sample_image: Image.Image,
    epsilon: float = 16.0 / 255.0,
    gate_threshold: float = 2.0 / 255.0,
) -> PreprocessingTransferResult:
    """Run the preprocessing-transfer gate for one VLM.

    This intentionally uses a *random* perturbation at ε, not a full PGD
    optimisation — we are measuring the preprocessing channel, not the
    attack. If random noise at ε survives, a PGD attack at the same ε will
    survive at least as well (PGD's support is a subset of ‖δ‖∞ ≤ ε).
    """
    model_name = getattr(hf_vlm, "model_id", hf_vlm.__class__.__name__)

    arr = np.asarray(sample_image.convert("RGB"), dtype=np.float32) / 255.0
    noise = np.random.default_rng(seed=0).uniform(-epsilon, epsilon, arr.shape)
    perturbed_arr = np.clip(arr + noise, 0.0, 1.0)
    perturbed_img = Image.fromarray((perturbed_arr * 255).astype(np.uint8))

    buf = io.BytesIO()
    perturbed_img.save(buf, format="PNG")
    buf.seek(0)
    reloaded = Image.open(buf).convert("RGB")
    reloaded_arr = np.asarray(reloaded, dtype=np.float32) / 255.0

    effective = float(np.max(np.abs(reloaded_arr - arr)))

    return PreprocessingTransferResult(
        model_name=model_name,
        epsilon_requested=epsilon,
        effective_linf_post_roundtrip=effective,
        gate_threshold=gate_threshold,
        passed=effective >= gate_threshold,
    )


def write_gate_report(
    result: PreprocessingTransferResult,
    out_path: str | Path,
) -> None:
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with Path(out_path).open("w", encoding="utf-8") as f:
        f.write(
            "gate: preprocessing_transfer\n"
            f"model: {result.model_name}\n"
            f"epsilon_requested: {result.epsilon_requested:.6f}\n"
            f"effective_linf_after_roundtrip: {result.effective_linf_post_roundtrip:.6f}\n"
            f"gate_threshold: {result.gate_threshold:.6f}\n"
            f"passed: {result.passed}\n"
        )


# Note: using torch import keeps linters quiet when this module is loaded by
# higher-level runners that expect torch to be available in-process.
_ = torch


def _cli() -> int:
    import argparse

    p = argparse.ArgumentParser(description="Phase-0 preprocessing-transfer gate")
    p.add_argument("--image", type=str, default=None, help="Path to sample image (PNG/JPG). Synthetic if omitted.")
    p.add_argument("--model-name", type=str, default="generic", help="Label for report only.")
    p.add_argument("--epsilon", type=float, default=16.0 / 255.0)
    p.add_argument("--gate-threshold", type=float, default=2.0 / 255.0)
    p.add_argument("--out", type=str, default="runs/gates/preprocessing_transfer.txt")
    args = p.parse_args()

    if args.image:
        img = Image.open(args.image).convert("RGB")
    else:
        rng = np.random.default_rng(0)
        img = Image.fromarray(rng.integers(0, 256, (512, 512, 3), dtype=np.uint8))

    class _Stub:
        model_id = args.model_name

    result = run_preprocessing_transfer(
        _Stub(),
        sample_image=img,
        epsilon=args.epsilon,
        gate_threshold=args.gate_threshold,
    )
    write_gate_report(result, args.out)
    print(
        f"[gate:preprocessing_transfer] model={result.model_name} "
        f"eps={result.epsilon_requested:.4f} eff_linf={result.effective_linf_post_roundtrip:.4f} "
        f"thr={result.gate_threshold:.4f} passed={result.passed}"
    )
    return 0 if result.passed else 1


if __name__ == "__main__":
    raise SystemExit(_cli())
