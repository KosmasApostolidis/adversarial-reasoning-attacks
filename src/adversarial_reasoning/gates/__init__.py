"""Phase-0 blocking gates.

These are pre-registered calibration checks that MUST pass before any
attack results are considered meaningful:

- `preprocessing_transfer`: ε-budget preservation through HF → Ollama
  preprocessing round-trip
- `noise_floor`: intra-seed trajectory variability per model at T=0
- `gradient_masking`: Athalye-Carlini-Wagner 2018 four-check sanity
  list for gradient-based attacks
"""

__all__ = [
    "GradientMaskingResult",
    "NoiseFloorResult",
    "PreprocessingTransferResult",
    "run_gradient_masking",
    "run_noise_floor",
    "run_preprocessing_transfer",
]


def __getattr__(name: str) -> object:  # PEP 562 lazy import
    if name in ("PreprocessingTransferResult", "run_preprocessing_transfer"):
        from .preprocessing_transfer import (
            PreprocessingTransferResult,
            run_preprocessing_transfer,
        )

        return {
            "PreprocessingTransferResult": PreprocessingTransferResult,
            "run_preprocessing_transfer": run_preprocessing_transfer,
        }[name]
    if name in ("NoiseFloorResult", "run_noise_floor"):
        from .noise_floor import NoiseFloorResult, run_noise_floor

        return {"NoiseFloorResult": NoiseFloorResult, "run_noise_floor": run_noise_floor}[name]
    if name in ("GradientMaskingResult", "run_gradient_masking"):
        from .gradient_masking import GradientMaskingResult, run_gradient_masking

        return {
            "GradientMaskingResult": GradientMaskingResult,
            "run_gradient_masking": run_gradient_masking,
        }[name]
    raise AttributeError(name)
