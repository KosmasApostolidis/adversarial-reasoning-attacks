#!/usr/bin/env bash
# Dataset prep. ProstateX-2 (primary, with QIICR-segmented cohort + Cuocolo
# lesion masks), VQA-RAD + SLAKE (secondary).
#
# Pipeline
#   * VQA-RAD / SLAKE : pulled via HuggingFace datasets API (public).
#   * ProstateX-2     : raw DICOM (T2 + ADC + DWI-b800) fetched from TCIA
#                       (`fetch_prostatex2_tcia.py`); lesion ROI masks cloned
#                       from Cuocolo et al's GitHub mirror; 3D NPY volumes
#                       (T2/ADC/DWI stacked, cropped/padded to 20x512x512x3,
#                       split 80/10/10 + 3-fold CV) emitted by
#                       `preprocess_prostatex2_dicom.py`.
#   * Idempotent: each stage skips work that's already on disk.

set -euo pipefail

DATA_DIR="${DATA_DIR:-./data}"
mkdir -p "${DATA_DIR}"/{prostatex/raw,prostatex/processed,prostatex/metadata,vqa_rad,slake}

echo "[prepare_datasets] ProstateX-2: raw DICOM at ${DATA_DIR}/prostatex/raw/"
echo "                   Processed 3-channel volumes at ${DATA_DIR}/prostatex/processed/"

# ---------- VQA-RAD ----------
echo "[prepare_datasets] Pulling VQA-RAD via HF datasets..."
python -c "
from datasets import load_dataset
ds = load_dataset('flaviagiammarino/vqa-rad', cache_dir='${DATA_DIR}/vqa_rad')
ds.save_to_disk('${DATA_DIR}/vqa_rad/snapshot')
print('[prepare_datasets] VQA-RAD saved:', len(ds['train']), 'train samples')
"

# ---------- SLAKE ----------
echo "[prepare_datasets] Pulling SLAKE via HF datasets..."
python -c "
try:
    from datasets import load_dataset
    ds = load_dataset('BoKelvin/SLAKE', cache_dir='${DATA_DIR}/slake')
    ds.save_to_disk('${DATA_DIR}/slake/snapshot')
    print('[prepare_datasets] SLAKE saved:', len(ds['train']), 'train samples')
except Exception as e:
    print('[prepare_datasets] SLAKE pull failed (non-fatal):', e)
"

# ---------- ProstateX-2 (TCIA DICOM + Cuocolo lesion masks → 3D NPY) ----------
echo "[prepare_datasets] ProstateX-2: bootstrapping TCIA DICOM + lesion masks..."

# 1. Clone Cuocolo et al. lesion masks first (~2.4 MB) — list of patients drives cohort.
if [ ! -d "${DATA_DIR}/prostatex/metadata/cuocolo_masks" ]; then
  echo "[prepare_datasets]  cloning Cuocolo lesion masks (rcuocolo/PROSTATEx_masks)"
  git clone --depth 1 https://github.com/rcuocolo/PROSTATEx_masks.git \
    "${DATA_DIR}/prostatex/metadata/cuocolo_masks"
else
  echo "[prepare_datasets]  Cuocolo masks already present — skipping"
fi

# 2. Fetch raw DICOM from TCIA (full 200-patient Cuocolo cohort, ~11 GB)
if [ -z "$(ls -A "${DATA_DIR}/prostatex/raw" 2>/dev/null)" ]; then
  echo "[prepare_datasets]  raw/ empty — pulling 200-patient Cuocolo cohort (~11 GB)"
  python scripts/fetch_prostatex_cuocolo_cohort.py \
    --out "${DATA_DIR}/prostatex/raw" \
    --metadata "${DATA_DIR}/prostatex/metadata" \
    --cuocolo-dir "${DATA_DIR}/prostatex/metadata/cuocolo_masks/Files/lesions/Masks/T2"
else
  echo "[prepare_datasets]  raw/ populated — skipping TCIA fetch"
fi

# 3. Preprocess DICOM + masks → 3D NPY folds
if [ ! -f "${DATA_DIR}/prostatex/processed/manifest.json" ]; then
  echo "[prepare_datasets]  preprocessing DICOM → 3-channel NPY (T2+ADC+DWI-b800)"
  python scripts/preprocess_prostatex2_dicom.py \
    --raw "${DATA_DIR}/prostatex/raw" \
    --metadata "${DATA_DIR}/prostatex/metadata" \
    --out "${DATA_DIR}/prostatex/processed" \
    --random-seed 42
else
  echo "[prepare_datasets]  manifest.json present — skipping preprocess"
fi

# 4. Sanity check fold artifacts
python - <<PY
from pathlib import Path
import sys
folds = sorted(Path("${DATA_DIR}/prostatex/processed/cv_folds").glob("fold_*/fold_*_X_*_3D.npy"))
holdout = sorted(Path("${DATA_DIR}/prostatex/processed/holdout").glob("X_*_3D.npy"))
expected = 12  # 3 folds x (X_train, X_val, y_train, y_val) but we glob only X
got_folds = len(folds)
got_holdout = len(holdout)
print(f"[prepare_datasets]  cv_folds X NPYs: {got_folds}/6  holdout X NPYs: {got_holdout}/2")
if got_folds < 6 or got_holdout < 2:
    print("[prepare_datasets]  ProstateX-2 pipeline incomplete — check logs above.", file=sys.stderr)
    sys.exit(1)
PY

echo "[prepare_datasets] done."
