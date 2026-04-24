"""Shared test fixtures."""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image


@pytest.fixture
def dummy_image() -> Image.Image:
    """Small deterministic PIL image suitable for cheap smoke tests."""
    rng = np.random.default_rng(seed=0)
    arr = rng.integers(0, 255, size=(64, 64, 3), dtype=np.uint8)
    return Image.fromarray(arr)


@pytest.fixture
def dummy_trajectory_pair():
    """A pair of benign / attack tool-name sequences for metric smoke tests."""
    benign = ["query_guidelines", "calculate_risk_score", "draft_report", "request_followup"]
    attack = ["query_guidelines", "calculate_risk_score", "escalate_to_specialist"]
    return benign, attack
