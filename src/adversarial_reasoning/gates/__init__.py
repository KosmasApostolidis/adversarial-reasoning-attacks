"""Phase-0 blocking gates.

These are pre-registered calibration checks that MUST pass before any
attack results are considered meaningful:

- `preprocessing_transfer`: ε-budget preservation through HF → Ollama
  preprocessing round-trip
- `noise_floor`: intra-seed trajectory variability per model at T=0
"""

__all__ = [
    "PreprocessingTransferResult",
    "run_preprocessing_transfer",
    "NoiseFloorResult",
    "run_noise_floor",
]


def __getattr__(name: str):  # PEP 562 lazy import
    if name in ("PreprocessingTransferResult", "run_preprocessing_transfer"):
        from .preprocessing_transfer import (
            PreprocessingTransferResult,
            run_preprocessing_transfer,
        )
        return {"PreprocessingTransferResult": PreprocessingTransferResult,
                "run_preprocessing_transfer": run_preprocessing_transfer}[name]
    if name in ("NoiseFloorResult", "run_noise_floor"):
        from .noise_floor import NoiseFloorResult, run_noise_floor
        return {"NoiseFloorResult": NoiseFloorResult,
                "run_noise_floor": run_noise_floor}[name]
    raise AttributeError(name)
