#!/usr/bin/env python
"""Fetch ProstateX-2 (the 98-patient subset of PROSTATEx with QIICR pixel-level lesion segmentations) from TCIA.

The "ProstateX-2" cohort is not its own TCIA collection — its segmentations live
inside the main `PROSTATEx` collection as 164 SEG-modality series across 98
patients. This script:

  1. Queries the TCIA REST API for all PROSTATEx series.
  2. Identifies the 98 patients that have at least one SEG series.
  3. Downloads every series (MR + SEG) for those patients into
     `data/prostatex/raw/`.
  4. Saves a CSV manifest of pulled series + the public Findings/Images CSVs.

Resume-safe: tcia-utils skips series whose SeriesInstanceUID directory already
exists on disk.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import pandas as pd
import requests
from tcia_utils import nbia

LOG = logging.getLogger("fetch_prostatex2")

COLLECTION = "PROSTATEx"
METADATA_FILES = {
    "ProstateX-Findings-Train.csv": "https://wiki.cancerimagingarchive.net/download/attachments/23691656/ProstateX-Findings-Train.csv",
    "ProstateX-Images-Train.csv": "https://wiki.cancerimagingarchive.net/download/attachments/23691656/ProstateX-Images-Train.csv",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--out",
        type=Path,
        default=Path("data/prostatex/raw"),
        help="DICOM target dir (default: data/prostatex/raw)",
    )
    p.add_argument(
        "--metadata",
        type=Path,
        default=Path("data/prostatex/metadata"),
        help="Metadata CSV target dir (default: data/prostatex/metadata)",
    )
    p.add_argument(
        "--limit-patients",
        type=int,
        default=None,
        help="Smoke-test cap: pull only the first N segmented patients.",
    )
    return p.parse_args()


def fetch_metadata_csvs(meta_dir: Path) -> None:
    meta_dir.mkdir(parents=True, exist_ok=True)
    for fname, url in METADATA_FILES.items():
        out = meta_dir / fname
        if out.exists():
            LOG.info("metadata cached: %s", out.name)
            continue
        LOG.info("downloading %s", fname)
        resp = requests.get(url, timeout=120)
        if resp.status_code == 200:
            out.write_bytes(resp.content)
        else:
            LOG.warning(
                "metadata fetch failed (%d) for %s — pipeline can still run",
                resp.status_code,
                fname,
            )


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    args.metadata.mkdir(parents=True, exist_ok=True)

    LOG.info("querying TCIA for PROSTATEx series ...")
    all_series = nbia.getSeries(collection=COLLECTION)
    LOG.info("collection has %d series", len(all_series))

    seg_patients = sorted({s["PatientID"] for s in all_series if s.get("Modality") == "SEG"})
    LOG.info("patients with SEG (= ProstateX-2 cohort): %d", len(seg_patients))

    if args.limit_patients:
        seg_patients = seg_patients[: args.limit_patients]
        LOG.info("smoke mode: clipping to %d patient(s)", len(seg_patients))

    seg_patient_set = set(seg_patients)
    target_series = [s for s in all_series if s.get("PatientID") in seg_patient_set]
    LOG.info("series to download: %d (MR + SEG)", len(target_series))

    fetch_metadata_csvs(args.metadata)

    series_csv = args.metadata / "prostatex2_series_manifest.csv"
    pd.DataFrame(target_series).to_csv(series_csv, index=False)
    LOG.info("wrote series manifest -> %s", series_csv)

    LOG.info("starting bulk download (resume-safe). Target: %s", args.out)
    series_uids = [s["SeriesInstanceUID"] for s in target_series]
    nbia.downloadSeries(
        series_data=series_uids,
        input_type="list",
        path=str(args.out),
        csv_filename=str(args.metadata / "prostatex2_download_log"),
    )

    summary = {
        "collection": COLLECTION,
        "cohort": "ProstateX-2 (segmented subset)",
        "patient_count": len(seg_patient_set),
        "series_count": len(target_series),
        "raw_dir": str(args.out.resolve()),
        "metadata_dir": str(args.metadata.resolve()),
    }
    (args.metadata / "prostatex2_fetch_summary.json").write_text(json.dumps(summary, indent=2))
    LOG.info("done. summary -> %s", args.metadata / "prostatex2_fetch_summary.json")


if __name__ == "__main__":
    main()
