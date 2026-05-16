"""Task/scenario loader. Reads configs/tasks.yaml; yields TaskSample tuples.

Real dataset files (when present) live under ``data/<dataset>/<split>/``.
Synthetic fallback is **opt-in** (``allow_synthetic_fallback=True`` or
``synthetic=True``) so partial / missing dataset shards do not silently
inflate benchmark numbers — see ``load_task`` for the exact semantics.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from PIL import Image


def _stable_seed(*parts: object) -> int:
    """Deterministic 32-bit seed from a tuple of parts.

    Replaces ``hash(tuple)`` because Python's built-in ``hash`` is randomised
    per-process unless ``PYTHONHASHSEED`` is pinned, which broke synthetic-image
    reproducibility across runs of the synthetic-fallback path.
    """
    payload = "|".join(str(p) for p in parts).encode("utf-8")
    digest = hashlib.blake2b(payload, digest_size=4).digest()
    return int.from_bytes(digest, "big")


@dataclass(frozen=True)
class TaskSample:
    task_id: str
    sample_id: str
    image: Image.Image
    prompt: str
    metadata: Mapping[str, Any] = field(default_factory=dict)


def load_task_config(task_id: str, config_path: str | Path = "configs/tasks.yaml") -> dict:
    with Path(config_path).open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if task_id not in cfg["tasks"]:
        raise KeyError(f"Task {task_id!r} not in {config_path}")
    return dict(cfg["tasks"][task_id])


def _synthetic_image(seed: int, size: tuple[int, int] = (512, 512)) -> Image.Image:
    rng = np.random.default_rng(seed)
    arr = rng.integers(0, 256, (size[1], size[0], 3), dtype=np.uint8)
    return Image.fromarray(arr)


def _resolve_image_dir(dataset: str, split: str) -> Path:
    return Path("data") / dataset / split


def _iter_split_files(data_dir: Path) -> list[Path]:
    if not data_dir.exists():
        return []
    return sorted([p for p in data_dir.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg"}])


# ----- ProstateX .npy volume loader ----------------------------------------

# Base path for ProstateX cv_folds (3D volumes), produced by
# `scripts/preprocess_prostatex2_dicom.py`. Override via
# AR_PROSTATEX_BHI_ROOT env var.
_BHI_IN_REPO = Path("data/prostatex/processed/cv_folds")


def _bhi_root() -> Path:
    import os

    override = os.environ.get("AR_PROSTATEX_BHI_ROOT")
    return Path(override) if override else _BHI_IN_REPO


def _normalize_slice_to_uint8(slice2d: np.ndarray) -> np.ndarray:
    """Z-scored float32 → [0, 255] uint8 via per-slice min-max stretch."""
    s = slice2d.astype(np.float32)
    lo, hi = float(s.min()), float(s.max())
    if hi - lo < 1e-6:
        return np.zeros_like(s, dtype=np.uint8)
    stretched = (s - lo) / (hi - lo)
    return (stretched * 255.0).astype(np.uint8)


def _best_slice_index(mask_volume: np.ndarray) -> int:
    """Pick the mask-positive slice with the largest lesion, else middle."""
    if mask_volume.ndim == 4:
        mask_volume = mask_volume[..., 0]
    pos_counts = mask_volume.astype(bool).sum(axis=(1, 2))
    if pos_counts.max() > 0:
        return int(pos_counts.argmax())
    return int(mask_volume.shape[0] // 2)


def _load_prostatex_bhi(split: str, n: int | None, fold: int = 1) -> list[tuple[str, Image.Image]]:
    """Load (sample_id, PIL.Image) list from one BHI cv_fold split."""
    fold_dir = _bhi_root() / f"fold_{fold}"
    x_path = fold_dir / f"fold_{fold}_X_{split}_3D.npy"
    y_path = fold_dir / f"fold_{fold}_y_{split}_3D.npy"
    if not x_path.exists():
        return []
    X = np.load(x_path, mmap_mode="r")
    Y = np.load(y_path, mmap_mode="r") if y_path.exists() else None
    count = X.shape[0] if n is None else min(int(n), X.shape[0])
    out: list[tuple[str, Image.Image]] = []
    for i in range(count):
        volume = np.asarray(X[i, ..., 0]) if X.ndim == 5 else np.asarray(X[i])
        mask_vol = np.asarray(Y[i]) if Y is not None else None
        k = _best_slice_index(mask_vol) if mask_vol is not None else int(volume.shape[0] // 2)
        slice2d = volume[k]
        arr8 = _normalize_slice_to_uint8(slice2d)
        rgb = np.stack([arr8, arr8, arr8], axis=-1)
        out.append((f"bhi_f{fold}_{split}_p{i:03d}_s{k:02d}", Image.fromarray(rgb)))
    return out


_BHI_DEFAULT_SPLIT_TO_FOLD = {"train": 1, "dev": 2, "test": 3, "val": 1}


def _dataset_images(
    dataset: str,
    split: str,
    n: int | None,
    *,
    cfg: dict | None = None,
) -> list[tuple[str, Image.Image]]:
    if dataset == "prostatex_bhi":
        bhi_map = (cfg or {}).get("bhi_split_to_fold", _BHI_DEFAULT_SPLIT_TO_FOLD)
        fold = int(bhi_map.get(split, 1))
        # All BHI logical splits read from filesystem split "val"; the
        # fold differentiates patient cohorts.
        return _load_prostatex_bhi(split="val", n=n, fold=fold)
    # Default file-based lookup.
    files = _iter_split_files(_resolve_image_dir(dataset, split))
    return [(p.stem, Image.open(p).convert("RGB")) for p in files[: n or len(files)]]


def load_task(
    task_id: str,
    *,
    split: str = "dev",
    n: int | None = None,
    synthetic: bool = False,
    allow_synthetic_fallback: bool = False,
    config_path: str | Path = "configs/tasks.yaml",
) -> Iterator[TaskSample]:
    """Yield TaskSample for a task + split.

    Parameters
    ----------
    split : ``dev`` or ``test`` (any key under ``dataset_split`` in tasks.yaml).
    n : cap on samples; defaults to the config's ``dataset_split[split]``.
    synthetic : if True, skip disk lookup — always emit synthetic images.
        Use only for smoke tests; metadata flags each sample so analysis
        scripts can filter.
    allow_synthetic_fallback : if True, silently pad missing real samples
        with deterministic synthetic images and emit a WARN. Defaults to
        False — the loader raises if the dataset returns fewer than
        ``count`` images. Silent padding hid missing dataset shards in
        published numbers, so the safe default is now to fail loud.

    Raises
    ------
    RuntimeError
        If ``synthetic=False`` and ``allow_synthetic_fallback=False`` and
        the configured dataset returns fewer than ``count`` images.
    """
    import sys

    cfg = load_task_config(task_id, config_path=config_path)
    prompt = cfg["prompt_template"].strip()
    dataset = cfg.get("dataset", "synthetic")
    n_cfg = int(cfg.get("dataset_split", {}).get(split, 0))
    count = n if n is not None else n_cfg
    if count <= 0:
        return

    if synthetic:
        real: list[tuple[str, Image.Image]] = []
    else:
        real = _dataset_images(dataset, split, count, cfg=cfg)
        if len(real) < count and not allow_synthetic_fallback:
            raise RuntimeError(
                f"Dataset {dataset!r} returned {len(real)} samples for "
                f"task={task_id!r} split={split!r} but {count} were requested. "
                "Refusing to silently pad with synthetic images — pass "
                "allow_synthetic_fallback=True (and accept the metadata "
                "marker) or set synthetic=True for an explicit smoke test."
            )
        if len(real) < count:
            print(
                f"[load_task] WARN: padding {count - len(real)} synthetic "
                f"samples for task={task_id!r} split={split!r}; results will "
                "be marked synthetic=True in metadata.",
                file=sys.stderr,
                flush=True,
            )

    for i in range(count):
        if i < len(real):
            sample_id, img = real[i]
            yield TaskSample(
                task_id=task_id, sample_id=sample_id, image=img, prompt=prompt
            )
        else:
            img = _synthetic_image(seed=_stable_seed(task_id, split, i))
            sample_id = f"synthetic_{split}_{i:04d}"
            yield TaskSample(
                task_id=task_id,
                sample_id=sample_id,
                image=img,
                prompt=prompt,
                metadata={"synthetic": True, "synthetic_reason": "padding"},
            )


def load_task_sample(
    task_id: str,
    index: int = 0,
    *,
    split: str = "dev",
    synthetic: bool = False,
    config_path: str | Path = "configs/tasks.yaml",
) -> TaskSample:
    """Load a single TaskSample by index. Convenience wrapper."""
    for i, sample in enumerate(
        load_task(task_id, split=split, n=index + 1, synthetic=synthetic, config_path=config_path)
    ):
        if i == index:
            return sample
    raise IndexError(f"No sample at index {index} for task {task_id}")
