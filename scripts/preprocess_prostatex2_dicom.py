#!/usr/bin/env python
"""ProstateX-2 DICOM → BHI-compatible NPY pipeline.

Input
-----
- ``data/prostatex/raw/<SeriesInstanceUID>/<*.dcm>`` (output of
  ``scripts/fetch_prostatex2_tcia.py``).
- ``data/prostatex/metadata/prostatex2_series_manifest.csv`` (the same fetcher
  emits this — series-level metadata with PatientID, Modality,
  SeriesDescription, SeriesInstanceUID).

Per patient (98 in the ProstateX-2 cohort) we build:

  * T2W axial volume (reference grid)
  * ADC volume resampled to T2 grid
  * Calculated DWI-b800 volume resampled to T2 grid
  * Lesion mask (binary) decoded from QIICR DICOM SEG series and resampled to
    T2 grid. Multiple SEG series for the same patient are OR-ed together so
    every annotated lesion appears in one mask.

The four volumes are cropped/padded to ``(Z=20, H=512, W=512)``, z-normalised
per channel, stacked into ``X[i] = (Z, H, W, 3)`` and ``y[i] = (Z, H, W)``,
then split 80/10/10 patient-level (val/test held out) and 3-fold cross-
validated inside the 80% train portion. NPY layout matches the existing BHI
loader contract (``src/adversarial_reasoning/tasks/loader.py``):

  data/prostatex/processed/
  ├── cv_folds/
  │   ├── fold_1/
  │   │   ├── fold_1_X_train_3D.npy
  │   │   ├── fold_1_y_train_3D.npy
  │   │   ├── fold_1_X_val_3D.npy
  │   │   └── fold_1_y_val_3D.npy
  │   ├── fold_2/...
  │   └── fold_3/...
  ├── holdout/
  │   ├── X_val_3D.npy   y_val_3D.npy
  │   └── X_test_3D.npy  y_test_3D.npy
  ├── manifest.json
  └── skipped.csv

Run
---
    python scripts/preprocess_prostatex2_dicom.py \
        --raw data/prostatex/raw \
        --metadata data/prostatex/metadata \
        --out data/prostatex/processed \
        --random-seed 42

Idempotent: skips patients whose intermediate volume cache already exists.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import SimpleITK as sitk
from sklearn.model_selection import KFold, train_test_split

LOG = logging.getLogger("prostatex2_preproc")

# ---- Series-description regexes ----------------------------------------------------
# T2: axial T2W only (drop sagittal / coronal / localizer).
T2_RE = re.compile(r"^t2_tse_tra(?!_loc)(?!_localizer)(?!_sag)(?!_cor)", re.IGNORECASE)
# ADC and calculated DWI-b800 cover three series-description families seen in
# the ProstateX cohort:
#   ep2d_diff_tra_DYNDIST*_ADC                       (standard Siemens 2014+)
#   ep2d_diff_tra2x2_Noise0_FS_DYNDIST*_ADC          (Philips Noise0/FS variant)
#   diffusie-3Scan-4bval_fs_ADC                      (Dutch RIVET/Philips)
ADC_RE = re.compile(r"(ep2d_diff_tra.*_ADC|diffusie.*_ADC)$", re.IGNORECASE)
DWI_B800_RE = re.compile(r"(ep2d_diff_tra.*CALC_BVAL|diffusie.*CALC_BVAL)$", re.IGNORECASE)

# ---- Canonical output shape --------------------------------------------------------
TARGET_Z = 20
TARGET_H = 512
TARGET_W = 512

# ---- Splits -----------------------------------------------------------------------
TRAIN_FRAC = 0.80
VAL_FRAC = 0.10  # of the original 100% (so test = 1 - train - val = 0.10)
N_CV_FOLDS = 3


@dataclass
class PatientBundle:
    patient_id: str
    t2_dir: Path
    adc_dir: Path
    dwi_dir: Path
    lesion_mask_files: list[Path]  # NIfTI lesion ROIs from Cuocolo et al.


CUOCOLO_DIR_DEFAULT = Path("data/prostatex/metadata/cuocolo_masks/Files/lesions/Masks/T2")


# ============================================================================ #
# Series discovery                                                             #
# ============================================================================ #
def load_series_manifest(metadata_dir: Path) -> pd.DataFrame:
    csv = metadata_dir / "prostatex2_series_manifest.csv"
    if not csv.exists():
        raise FileNotFoundError(
            f"missing series manifest: {csv}. Run fetch_prostatex2_tcia.py first."
        )
    df = pd.read_csv(csv)
    LOG.info("loaded series manifest: %d rows", len(df))
    return df


def best_series_dir(rows: pd.DataFrame, raw_root: Path) -> Path | None:
    """Return the on-disk dir for the first series whose folder exists."""
    for _, r in rows.iterrows():
        d = raw_root / str(r["SeriesInstanceUID"])
        if d.is_dir() and any(d.iterdir()):
            return d
    return None


def discover_patient_bundles(
    manifest: pd.DataFrame, raw_root: Path, cuocolo_dir: Path
) -> tuple[list[PatientBundle], list[dict]]:
    bundles: list[PatientBundle] = []
    skipped: list[dict] = []
    for pid, sub in manifest.groupby("PatientID"):
        # T2W axial — pick the smallest pixel-spacing series first (highest in-plane res)
        t2_rows = sub[
            sub["Modality"].eq("MR") & sub["SeriesDescription"].fillna("").str.match(T2_RE)
        ]
        adc_rows = sub[
            sub["Modality"].eq("MR") & sub["SeriesDescription"].fillna("").str.match(ADC_RE)
        ]
        dwi_rows = sub[
            sub["Modality"].eq("MR") & sub["SeriesDescription"].fillna("").str.match(DWI_B800_RE)
        ]

        t2_dir = best_series_dir(t2_rows, raw_root)
        adc_dir = best_series_dir(adc_rows, raw_root)
        dwi_dir = best_series_dir(dwi_rows, raw_root)

        # Cuocolo mask filename variants:
        #   ProstateX-XXXX-Finding{N}-t2_tse_tra_ROI.nii.gz
        #   ProstateX-XXXX-Finding{N}-t2_tse_tra0_ROI.nii.gz   (≥120)
        #   ProstateX-XXXX-Finding{N}-t2_tse_cor0_ROI.nii.gz   (rare, e.g. 0201)
        #   ProstateX-XXXX-Finding{N}-t2_tse_sag0_ROI.nii.gz   (rare)
        # Coronal/sagittal masks resample onto axial T2 grid via NIfTI affine.
        lesion_files = sorted(
            list(cuocolo_dir.glob(f"{pid}-Finding*-t2_tse_tra*_ROI.nii.gz"))
            + list(cuocolo_dir.glob(f"{pid}-Finding*-t2_tse_cor*_ROI.nii.gz"))
            + list(cuocolo_dir.glob(f"{pid}-Finding*-t2_tse_sag*_ROI.nii.gz"))
        )

        if not (t2_dir and adc_dir and dwi_dir and lesion_files):
            skipped.append(
                {
                    "patient": pid,
                    "t2": bool(t2_dir),
                    "adc": bool(adc_dir),
                    "dwi_b800": bool(dwi_dir),
                    "lesion_count": len(lesion_files),
                }
            )
            continue
        bundles.append(PatientBundle(pid, t2_dir, adc_dir, dwi_dir, lesion_files))

    LOG.info("discovered %d valid patient bundles (%d skipped)", len(bundles), len(skipped))
    return bundles, skipped


# ============================================================================ #
# DICOM volume loading                                                         #
# ============================================================================ #
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


# ============================================================================ #
# Lesion mask loading (Cuocolo et al. NIfTI ROIs)                              #
# ============================================================================ #
def load_lesion_mask_to_t2_grid(lesion_files: list[Path], t2: sitk.Image) -> np.ndarray:
    """Read each Cuocolo ``Finding{N}`` NIfTI lesion ROI and OR into a single mask
    resampled onto the T2 reference grid.

    The Cuocolo masks are saved on the T2 ``t2_tse_tra`` axial geometry — a
    nearest-neighbour resample to the patient's T2 voxel grid is sufficient
    (same orientation, spacing usually identical).
    """
    Z = t2.GetSize()[2]
    H = t2.GetSize()[1]
    W = t2.GetSize()[0]
    accumulator = sitk.Image(t2.GetSize(), sitk.sitkUInt8)
    accumulator.CopyInformation(t2)

    for path in lesion_files:
        try:
            mask_img = sitk.ReadImage(str(path))
        except Exception as e:
            LOG.warning("failed to read lesion NIfTI %s: %s", path, e)
            continue
        # Resample lesion mask onto T2 grid using nearest-neighbour (binary).
        resampled = sitk.Resample(
            mask_img,
            t2,
            sitk.Transform(),
            sitk.sitkNearestNeighbor,
            0,
            sitk.sitkUInt8,
        )
        accumulator = sitk.Or(
            accumulator, sitk.Cast(sitk.BinaryThreshold(resampled, 1, 65535, 1, 0), sitk.sitkUInt8)
        )

    arr = sitk.GetArrayFromImage(accumulator).astype(np.uint8)
    if arr.shape != (Z, H, W):
        LOG.warning(
            "mask shape %s != expected T2 %s — falling back to zero mask", arr.shape, (Z, H, W)
        )
        return np.zeros((Z, H, W), dtype=np.uint8)
    return arr


# ============================================================================ #
# Per-patient pipeline                                                          #
# ============================================================================ #
def process_patient(b: PatientBundle, cache_dir: Path) -> tuple[np.ndarray, np.ndarray] | None:
    cache_x = cache_dir / f"{b.patient_id}_X.npy"
    cache_y = cache_dir / f"{b.patient_id}_y.npy"
    if cache_x.exists() and cache_y.exists():
        return np.load(cache_x), np.load(cache_y)

    LOG.info("[%s] loading T2/ADC/DWI", b.patient_id)
    t2 = read_volume(b.t2_dir)
    adc = resample_to(t2, read_volume(b.adc_dir))
    dwi = resample_to(t2, read_volume(b.dwi_dir))

    t2_arr = sitk.GetArrayFromImage(t2).astype(np.float32)
    adc_arr = sitk.GetArrayFromImage(adc).astype(np.float32)
    dwi_arr = sitk.GetArrayFromImage(dwi).astype(np.float32)

    LOG.info("[%s] loading %d Cuocolo lesion mask(s)", b.patient_id, len(b.lesion_mask_files))
    mask = load_lesion_mask_to_t2_grid(b.lesion_mask_files, t2)

    if mask.sum() == 0:
        LOG.warning(
            "[%s] empty lesion mask after T2 resample — keeping with all-zero y", b.patient_id
        )

    # crop/pad
    t2_arr = crop_or_pad_to(t2_arr, TARGET_Z, TARGET_H, TARGET_W, 0.0)
    adc_arr = crop_or_pad_to(adc_arr, TARGET_Z, TARGET_H, TARGET_W, 0.0)
    dwi_arr = crop_or_pad_to(dwi_arr, TARGET_Z, TARGET_H, TARGET_W, 0.0)
    mask = crop_or_pad_to(mask, TARGET_Z, TARGET_H, TARGET_W, 0).astype(np.uint8)

    # z-norm per channel
    t2n = znorm(t2_arr)
    adcn = znorm(adc_arr)
    dwin = znorm(dwi_arr)

    X = np.stack([t2n, adcn, dwin], axis=-1)  # (Z, H, W, 3)
    y = mask  # (Z, H, W)

    cache_dir.mkdir(parents=True, exist_ok=True)
    np.save(cache_x, X)
    np.save(cache_y, y)
    return X, y


# ============================================================================ #
# Splits and saving                                                             #
# ============================================================================ #
def save_split(out_dir: Path, X: np.ndarray, y: np.ndarray, name: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / f"X_{name}_3D.npy", X)
    np.save(out_dir / f"y_{name}_3D.npy", y)
    LOG.info("wrote %s split: X%s  y%s", name, X.shape, y.shape)


def save_fold(out_dir: Path, fold: int, X_tr, y_tr, X_va, y_va) -> None:
    fd = out_dir / f"fold_{fold}"
    fd.mkdir(parents=True, exist_ok=True)
    np.save(fd / f"fold_{fold}_X_train_3D.npy", X_tr)
    np.save(fd / f"fold_{fold}_y_train_3D.npy", y_tr)
    np.save(fd / f"fold_{fold}_X_val_3D.npy", X_va)
    np.save(fd / f"fold_{fold}_y_val_3D.npy", y_va)
    LOG.info("fold_%d -> X_train%s X_val%s", fold, X_tr.shape, X_va.shape)


# ============================================================================ #
# Main                                                                          #
# ============================================================================ #
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--raw", type=Path, default=Path("data/prostatex/raw"))
    p.add_argument("--metadata", type=Path, default=Path("data/prostatex/metadata"))
    p.add_argument("--out", type=Path, default=Path("data/prostatex/processed"))
    p.add_argument("--random-seed", type=int, default=42)
    p.add_argument("--cache", type=Path, default=Path("data/prostatex/processed/_patient_cache"))
    p.add_argument(
        "--cuocolo-dir",
        type=Path,
        default=CUOCOLO_DIR_DEFAULT,
        help="Path to rcuocolo/PROSTATEx_masks lesion T2 ROI dir.",
    )
    return p.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()

    manifest = load_series_manifest(args.metadata)
    if not args.cuocolo_dir.exists():
        LOG.error(
            "missing Cuocolo lesion masks at %s — clone https://github.com/rcuocolo/PROSTATEx_masks first.",
            args.cuocolo_dir,
        )
        sys.exit(1)
    bundles, skipped = discover_patient_bundles(manifest, args.raw, args.cuocolo_dir)
    if not bundles:
        LOG.error("no patient bundles found — fetcher may not have completed yet.")
        sys.exit(1)

    Xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    pids: list[str] = []

    for b in bundles:
        try:
            res = process_patient(b, args.cache)
        except Exception as e:
            LOG.exception("[%s] failed: %s", b.patient_id, e)
            skipped.append({"patient": b.patient_id, "error": str(e)})
            continue
        if res is None:
            skipped.append({"patient": b.patient_id, "error": "empty_seg"})
            continue
        X, y = res
        Xs.append(X)
        ys.append(y)
        pids.append(b.patient_id)

    LOG.info("processed %d patients, %d skipped", len(Xs), len(skipped))
    if not Xs:
        LOG.error("nothing to save")
        sys.exit(1)

    X_all = np.stack(Xs, axis=0)  # (N, Z, H, W, 3) float32
    y_all = np.stack(ys, axis=0)  # (N, Z, H, W) uint8
    LOG.info("X_all %s, y_all %s", X_all.shape, y_all.shape)

    args.out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(skipped).to_csv(args.out / "skipped.csv", index=False)

    # 80/10/10 patient-level
    idx_all = np.arange(X_all.shape[0])
    idx_train, idx_holdout = train_test_split(
        idx_all,
        train_size=TRAIN_FRAC,
        random_state=args.random_seed,
        shuffle=True,
    )
    idx_val, idx_test = train_test_split(
        idx_holdout,
        test_size=0.5,
        random_state=args.random_seed,
        shuffle=True,
    )
    LOG.info("split: train=%d val=%d test=%d", len(idx_train), len(idx_val), len(idx_test))

    holdout_dir = args.out / "holdout"
    save_split(holdout_dir, X_all[idx_val], y_all[idx_val], "val")
    save_split(holdout_dir, X_all[idx_test], y_all[idx_test], "test")

    # 3-fold CV on train
    kf = KFold(n_splits=N_CV_FOLDS, shuffle=True, random_state=args.random_seed)
    cv_dir = args.out / "cv_folds"
    cv_dir.mkdir(parents=True, exist_ok=True)
    for fold, (tr_idx_local, va_idx_local) in enumerate(kf.split(idx_train), start=1):
        tr_global = idx_train[tr_idx_local]
        va_global = idx_train[va_idx_local]
        save_fold(
            cv_dir, fold, X_all[tr_global], y_all[tr_global], X_all[va_global], y_all[va_global]
        )

    manifest_out = {
        "cohort": "ProstateX-2 (segmented subset of TCIA PROSTATEx)",
        "channel_order": ["T2W", "ADC", "DWI_b800"],
        "shape_per_patient": [TARGET_Z, TARGET_H, TARGET_W, 3],
        "n_patients_total": int(X_all.shape[0]),
        "n_train": len(idx_train),
        "n_holdout_val": len(idx_val),
        "n_holdout_test": len(idx_test),
        "n_cv_folds": N_CV_FOLDS,
        "random_seed": int(args.random_seed),
        "patient_ids_train": [pids[i] for i in idx_train.tolist()],
        "patient_ids_val": [pids[i] for i in idx_val.tolist()],
        "patient_ids_test": [pids[i] for i in idx_test.tolist()],
    }
    (args.out / "manifest.json").write_text(json.dumps(manifest_out, indent=2))
    LOG.info("manifest written to %s", args.out / "manifest.json")


if __name__ == "__main__":
    main()
