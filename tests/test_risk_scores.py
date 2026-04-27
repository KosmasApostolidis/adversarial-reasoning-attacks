"""Pure-logic tests for tools.risk_scores._compute / _pi_rads_like / _damico_like."""

from __future__ import annotations

import pytest

from adversarial_reasoning.tools.risk_scores import (
    _compute,
    _damico_like,
    _pi_rads_like,
    tool,
)


class TestPiRadsLike:
    def test_typical_input(self) -> None:
        score = _pi_rads_like(psa=4.0, volume_cc=40.0, lesion_grade=3)
        assert 0.0 <= score <= 5.0

    def test_zero_volume_raises(self) -> None:
        with pytest.raises(ValueError, match="volume_cc must be positive"):
            _pi_rads_like(psa=4.0, volume_cc=0.0, lesion_grade=3)

    def test_negative_volume_raises(self) -> None:
        with pytest.raises(ValueError):
            _pi_rads_like(psa=4.0, volume_cc=-1.0, lesion_grade=3)

    def test_grade_clamped_high(self) -> None:
        score = _pi_rads_like(psa=1.0, volume_cc=40.0, lesion_grade=99)
        score_max = _pi_rads_like(psa=1.0, volume_cc=40.0, lesion_grade=5)
        assert score == score_max

    def test_grade_clamped_low(self) -> None:
        score = _pi_rads_like(psa=1.0, volume_cc=40.0, lesion_grade=-5)
        score_min = _pi_rads_like(psa=1.0, volume_cc=40.0, lesion_grade=0)
        assert score == score_min

    def test_capped_at_five(self) -> None:
        # Massive PSAD pushes value above 5.0 -> must be clamped.
        score = _pi_rads_like(psa=10000.0, volume_cc=1.0, lesion_grade=5)
        assert score == 5.0


class TestDamicoLike:
    @pytest.mark.parametrize(
        "psa,gleason,t_stage,expected",
        [
            (5.0, 6, 1, 0.0),
            (8.0, 6, 1, 0.0),
            (15.0, 7, 2, 0.5),
            (25.0, 6, 1, 1.0),
            (5.0, 9, 1, 1.0),
            (5.0, 6, 3, 1.0),
        ],
    )
    def test_table(self, psa: float, gleason: int, t_stage: int, expected: float) -> None:
        assert _damico_like(psa=psa, gleason=gleason, t_stage=t_stage) == expected

    @pytest.mark.parametrize(
        "psa,gleason,t_stage",
        [
            (10.0, 6, 1),  # PSA exactly on the low/intermediate boundary
            (20.0, 6, 1),  # PSA exactly on the intermediate/high boundary
            (5.0, 7, 1),  # Gleason exactly on the low/intermediate boundary
            (5.0, 6, 2),  # T-stage exactly on the low/intermediate boundary
        ],
    )
    def test_boundary_values_fall_into_intermediate(
        self, psa: float, gleason: int, t_stage: int
    ) -> None:
        """Documented strict-inequality convention: boundary patients
        always land in the intermediate (0.5) bucket. Regression-locks
        the docstring contract added during the logic-error audit."""
        assert _damico_like(psa=psa, gleason=gleason, t_stage=t_stage) == 0.5


class TestComputeDispatch:
    def test_pi_rads_alias_pirads(self) -> None:
        out = _compute("pirads", {"psa": 4.0, "volume_cc": 40.0, "lesion_grade": 3})
        assert isinstance(out, float)

    def test_pi_rads_alias_dash(self) -> None:
        out = _compute("pi-rads", {"psa": 4.0, "volume_cc": 40.0, "lesion_grade": 3})
        assert isinstance(out, float)

    def test_pi_rads_default_lesion_grade(self) -> None:
        out = _compute("pi_rads", {"psa": 4.0, "volume_cc": 40.0})
        assert isinstance(out, float)

    def test_damico(self) -> None:
        out = _compute("damico", {"psa": 5.0, "gleason": 6, "t_stage": 1})
        assert out == 0.0

    def test_damico_defaults(self) -> None:
        out = _compute("damico", {"psa": 5.0})
        assert out == 0.0

    def test_unknown_score_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown risk score"):
            _compute("not_a_score", {"psa": 1.0})

    def test_case_insensitive(self) -> None:
        a = _compute("DaMiCo", {"psa": 5.0, "gleason": 6, "t_stage": 1})
        b = _compute("damico", {"psa": 5.0, "gleason": 6, "t_stage": 1})
        assert a == b


def test_tool_factory_returns_registered_tool() -> None:
    t = tool()
    assert t.name == "calculate_risk_score"
    assert "name" in t.parameters_schema["properties"]
    assert t.handler is _compute
