#!/usr/bin/env python
"""ProstateX-2 DICOM → BHI NPY pipeline entrypoint.

Dispatches to scripts/dataprep/preprocess_prostatex2/ package. See package modules:
  - dicom_io        — volume read/resample/crop/znorm helpers
  - lesion_extract  — series-manifest discovery + Cuocolo NIfTI lesion masks
  - pipeline        — per-patient processing, splits, CLI main

Run
---
    python -m scripts.dataprep.preprocess_prostatex2_dicom \
        --raw data/prostatex/raw \
        --metadata data/prostatex/metadata \
        --out data/prostatex/processed \
        --random-seed 42

Idempotent: skips patients whose intermediate volume cache already exists.
"""

from __future__ import annotations

from .preprocess_prostatex2 import main

if __name__ == "__main__":
    main()
