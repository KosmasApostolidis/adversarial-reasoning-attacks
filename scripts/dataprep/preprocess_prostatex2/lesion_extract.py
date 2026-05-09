"""Series-manifest discovery and Cuocolo NIfTI lesion-mask loading."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import SimpleITK as sitk

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

CUOCOLO_DIR_DEFAULT = Path("data/prostatex/metadata/cuocolo_masks/Files/lesions/Masks/T2")


@dataclass
class PatientBundle:
    patient_id: str
    t2_dir: Path
    adc_dir: Path
    dwi_dir: Path
    lesion_mask_files: list[Path]


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


def load_lesion_mask_to_t2_grid(lesion_files: list[Path], t2: sitk.Image) -> np.ndarray:
    """Read each Cuocolo Finding{N} NIfTI lesion ROI and OR into a single mask
    resampled onto the T2 reference grid."""
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
