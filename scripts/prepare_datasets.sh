#!/usr/bin/env bash
# Dataset prep. ProstateX (primary), VQA-RAD + SLAKE (secondary).
#
# Notes:
#   - ProstateX requires TCIA account for bulk download via NBIA Data Retriever
#     (https://wiki.cancerimagingarchive.net/). This script only creates
#     directories; you must place raw DICOM files in data/prostatex/raw/.
#   - VQA-RAD is public on HuggingFace (osunlp/VQA-RAD).
#   - SLAKE is public (PHBenchmark/slake).

set -euo pipefail

DATA_DIR="${DATA_DIR:-./data}"
mkdir -p "${DATA_DIR}"/{prostatex/raw,prostatex/processed,vqa_rad,slake}

echo "[prepare_datasets] ProstateX: expects DICOM files at ${DATA_DIR}/prostatex/raw/"
echo "                   Processed bi-parametric slices will land in ${DATA_DIR}/prostatex/processed/"

# ---------- VQA-RAD ----------
echo "[prepare_datasets] Pulling VQA-RAD via HF datasets..."
python -c "
from datasets import load_dataset
ds = load_dataset('osunlp/VQA-RAD', cache_dir='${DATA_DIR}/vqa_rad')
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

echo "[prepare_datasets] done."
