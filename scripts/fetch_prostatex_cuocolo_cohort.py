#!/usr/bin/env python
"""Pull TCIA PROSTATEx MR series for every patient that has a Cuocolo lesion mask.

Use after `fetch_prostatex2_tcia.py` if you want to extend the cohort beyond
the 98 QIICR-segmented patients to the full 200-patient Cuocolo et al.
lesion-mask cohort. Resume-safe: tcia-utils skips series already on disk.
"""

from __future__ import annotations

import argparse
import logging
import re
from pathlib import Path

import pandas as pd
from tcia_utils import nbia

LOG = logging.getLogger("cuocolo_fetch")
COLLECTION = "PROSTATEx"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, default=Path("data/prostatex/raw"))
    p.add_argument("--metadata", type=Path, default=Path("data/prostatex/metadata"))
    p.add_argument(
        "--cuocolo-dir",
        type=Path,
        default=Path("data/prostatex/metadata/cuocolo_masks/Files/lesions/Masks/T2"),
    )
    return p.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()

    if not args.cuocolo_dir.exists():
        raise FileNotFoundError(f"missing Cuocolo dir at {args.cuocolo_dir}")

    cuocolo_pids = sorted(
        {
            m.group(1)
            for m in (
                re.match(r"(ProstateX-\d+)", f.name) for f in args.cuocolo_dir.glob("*.nii.gz")
            )
            if m
        }
    )
    LOG.info("Cuocolo cohort: %d patients", len(cuocolo_pids))

    LOG.info("querying TCIA series catalogue (collection=%s)", COLLECTION)
    all_series = nbia.getSeries(collection=COLLECTION)
    target = [s for s in all_series if s.get("PatientID") in set(cuocolo_pids)]
    LOG.info("series for Cuocolo patients: %d", len(target))

    # write/refresh manifest CSV (used by preprocessor)
    args.metadata.mkdir(parents=True, exist_ok=True)
    csv = args.metadata / "prostatex2_series_manifest.csv"
    pd.DataFrame(target).to_csv(csv, index=False)
    LOG.info("wrote refreshed series manifest -> %s (%d rows)", csv, len(target))

    args.out.mkdir(parents=True, exist_ok=True)
    series_uids = [s["SeriesInstanceUID"] for s in target]
    LOG.info("starting bulk download (resume-safe). Target: %s", args.out)
    nbia.downloadSeries(
        series_data=series_uids,
        input_type="list",
        path=str(args.out),
        csv_filename=str(args.metadata / "prostatex2_download_log"),
    )
    LOG.info("done.")


if __name__ == "__main__":
    main()
