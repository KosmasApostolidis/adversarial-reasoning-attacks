"""DICOM volume I/O, resampling, crop/pad, and z-norm helpers."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import SimpleITK as sitk

LOG = logging.getLogger("prostatex2_preproc")


def read_volume(series_dir: Path) -> sitk.Image:
    reader = sitk.ImageSeriesReader()
    files = reader.GetGDCMSeriesFileNames(str(series_dir))
    if not files:
        raise RuntimeError(f"no DICOM files in {series_dir}")
    reader.SetFileNames(files)
    return reader.Execute()


def resample_to(reference: sitk.Image, moving: sitk.Image) -> sitk.Image:
    return sitk.Resample(
        moving,
        reference,
        sitk.Transform(),
        sitk.sitkLinear,
        0.0,
        moving.GetPixelID(),
    )


def crop_or_pad_to(
    arr: np.ndarray, target_z: int, target_h: int, target_w: int, pad_value: float = 0.0
) -> np.ndarray:
    """Centre crop/pad a (Z, H, W) array to (target_z, target_h, target_w)."""
    z, h, w = arr.shape
    out = np.full((target_z, target_h, target_w), pad_value, dtype=arr.dtype)

    sz = (z - target_z) // 2 if z > target_z else 0
    sh = (h - target_h) // 2 if h > target_h else 0
    sw = (w - target_w) // 2 if w > target_w else 0
    pz_z = (target_z - z) // 2 if z < target_z else 0
    pz_h = (target_h - h) // 2 if h < target_h else 0
    pz_w = (target_w - w) // 2 if w < target_w else 0

    src = arr[
        sz : sz + min(z, target_z),
        sh : sh + min(h, target_h),
        sw : sw + min(w, target_w),
    ]
    out[
        pz_z : pz_z + src.shape[0],
        pz_h : pz_h + src.shape[1],
        pz_w : pz_w + src.shape[2],
    ] = src
    return out


def znorm(volume: np.ndarray) -> np.ndarray:
    fg = volume[volume > 0]
    if fg.size == 0:
        return volume.astype(np.float32)
    mu = fg.mean()
    sd = fg.std() + 1e-6
    return ((volume - mu) / sd).astype(np.float32)
