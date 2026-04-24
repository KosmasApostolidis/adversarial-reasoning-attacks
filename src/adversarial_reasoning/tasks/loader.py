"""Task/scenario loader. Reads configs/tasks.yaml; yields TaskSample tuples.

Real dataset files (when present) live under ``data/<dataset>/<split>/``.
When dataset files are missing, falls back to deterministic synthetic RGB
images so smoke / CI can exercise the full pipeline without TCIA access.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np
import yaml
from PIL import Image


@dataclass(frozen=True)
class TaskSample:
    task_id: str
    sample_id: str
    image: Image.Image
    prompt: str


def load_task_config(
    task_id: str, config_path: str | Path = "configs/tasks.yaml"
) -> dict:
    with Path(config_path).open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if task_id not in cfg["tasks"]:
        raise KeyError(f"Task {task_id!r} not in {config_path}")
    return cfg["tasks"][task_id]


def _synthetic_image(seed: int, size: tuple[int, int] = (512, 512)) -> Image.Image:
    rng = np.random.default_rng(seed)
    arr = rng.integers(0, 256, (size[1], size[0], 3), dtype=np.uint8)
    return Image.fromarray(arr)


def _resolve_image_dir(dataset: str, split: str) -> Path:
    return Path("data") / dataset / split


def _iter_split_files(data_dir: Path) -> list[Path]:
    if not data_dir.exists():
        return []
    return sorted(
        [p for p in data_dir.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg"}]
    )


# ----- ProstateX-BHI .npy volume loader ------------------------------------

# Base path for user's local BHI ProstateX cv_folds (3D volumes).
# Override via AR_PROSTATEX_BHI_ROOT env var.
_BHI_ROOT_DEFAULT = Path("/home/medadmin/kosmasapostolidis/BHI/data/processed/cv_folds")


def _bhi_root() -> Path:
    import os

    return Path(os.environ.get("AR_PROSTATEX_BHI_ROOT", str(_BHI_ROOT_DEFAULT)))


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


def _load_prostatex_bhi(
    split: str, n: int | None, fold: int = 1
) -> list[tuple[str, Image.Image]]:
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


def _dataset_images(dataset: str, split: str, n: int | None) -> list[tuple[str, Image.Image]]:
    if dataset == "prostatex_bhi":
        return _load_prostatex_bhi(split=split, n=n)
    # Default file-based lookup.
    files = _iter_split_files(_resolve_image_dir(dataset, split))
    return [(p.stem, Image.open(p).convert("RGB")) for p in files[: n or len(files)]]


def load_task(
    task_id: str,
    *,
    split: str = "dev",
    n: int | None = None,
    synthetic: bool = False,
    config_path: str | Path = "configs/tasks.yaml",
) -> Iterator[TaskSample]:
    """Yield TaskSample for a task + split.

    Parameters
    ----------
    split : ``dev`` or ``test`` (any key under ``dataset_split`` in tasks.yaml).
    n : cap on samples; defaults to the config's ``dataset_split[split]``.
    synthetic : if True, skip disk lookup — always emit synthetic images.
    """
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
        real = _dataset_images(dataset, split, count)

    for i in range(count):
        if i < len(real):
            sample_id, img = real[i]
        else:
            img = _synthetic_image(seed=hash((task_id, split, i)) & 0xFFFFFFFF)
            sample_id = f"synthetic_{split}_{i:04d}"
        yield TaskSample(task_id=task_id, sample_id=sample_id, image=img, prompt=prompt)


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
