"""Per-patient pipeline, splits, and CLI entrypoint."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import SimpleITK as sitk
from sklearn.model_selection import KFold, train_test_split

from .dicom_io import crop_or_pad_to, read_volume, resample_to, znorm
from .lesion_extract import (
    CUOCOLO_DIR_DEFAULT,
    PatientBundle,
    discover_patient_bundles,
    load_lesion_mask_to_t2_grid,
    load_series_manifest,
)

LOG = logging.getLogger("prostatex2_preproc")

# ---- Canonical output shape --------------------------------------------------------
TARGET_Z = 20
TARGET_H = 512
TARGET_W = 512

# ---- Splits -----------------------------------------------------------------------
TRAIN_FRAC = 0.80
VAL_FRAC = 0.10  # of the original 100% (so test = 1 - train - val = 0.10)
N_CV_FOLDS = 3


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

    t2_arr = crop_or_pad_to(t2_arr, TARGET_Z, TARGET_H, TARGET_W, 0.0)
    adc_arr = crop_or_pad_to(adc_arr, TARGET_Z, TARGET_H, TARGET_W, 0.0)
    dwi_arr = crop_or_pad_to(dwi_arr, TARGET_Z, TARGET_H, TARGET_W, 0.0)
    mask = crop_or_pad_to(mask, TARGET_Z, TARGET_H, TARGET_W, 0).astype(np.uint8)

    t2n = znorm(t2_arr)
    adcn = znorm(adc_arr)
    dwin = znorm(dwi_arr)

    X = np.stack([t2n, adcn, dwin], axis=-1)
    y = mask

    cache_dir.mkdir(parents=True, exist_ok=True)
    np.save(cache_x, X)
    np.save(cache_y, y)
    return X, y


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

    X_all = np.stack(Xs, axis=0)
    y_all = np.stack(ys, axis=0)
    LOG.info("X_all %s, y_all %s", X_all.shape, y_all.shape)

    args.out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(skipped).to_csv(args.out / "skipped.csv", index=False)

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
