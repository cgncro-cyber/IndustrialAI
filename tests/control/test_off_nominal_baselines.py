"""Sanity tests on the persisted off-nominal C1/C0 baseline JSON files.

Avoids re-running the grids (each sweep takes ~20 min); instead
validates that the stored artifacts pass the contract documented in
``docs/kpis.md`` §2.2-§2.3.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_C1_JSON = _REPO_ROOT / "data" / "reference" / "c1_off_nominal_baseline.json"
_C0_JSON = _REPO_ROOT / "data" / "reference" / "c0_off_nominal_baseline.json"

_EXPECTED_OPS = {(F, zF) for F in (0.8, 0.9, 1.1, 1.2) for zF in (0.45, 0.475, 0.525, 0.55)}


@pytest.mark.skipif(
    not _C1_JSON.exists(),
    reason="c1_off_nominal_baseline.json not yet produced (run tools/run_c1_off_nominal_grid.py)",
)
class TestC1OffNominalBaseline:
    @pytest.fixture(scope="class")
    def data(self) -> dict:
        with _C1_JSON.open() as fh:
            return json.load(fh)

    def test_grid_covers_section_2_2(self, data: dict) -> None:
        ops = {(b["F"], b["zF"]) for b in data["per_op"]}
        assert ops == _EXPECTED_OPS

    def test_linearization_recompute_count_matches_grid(self, data: dict) -> None:
        assert data["linearization"]["linearization_recompute_count"] == 16

    def test_off_nominal_robustness_iae_is_p95(self, data: dict) -> None:
        assert data["aggregate"]["off_nominal_robustness_iae"] == pytest.approx(
            data["aggregate"]["p95"]
        )

    def test_all_ops_completed_without_solver_failure(self, data: dict) -> None:
        for block in data["per_op"]:
            assert not block["any_scenario_failed"], (
                f"OP F={block['F']} zF={block['zF']}: at least one scenario failed"
            )
            assert block["aggregate_iae_finite"]

    def test_max_wall_clock_within_supervisory_budget(self, data: dict) -> None:
        cadence_min = data["mpc_config"]["sampling_time_min"]
        budget_s = cadence_min * 60.0 * 0.2
        for block in data["per_op"]:
            assert block["max_cycle_wall_clock_seconds"] < budget_s


@pytest.mark.skipif(
    not _C0_JSON.exists(),
    reason="c0_off_nominal_baseline.json not yet produced (run tools/run_c0_off_nominal_grid.py)",
)
class TestC0OffNominalBaseline:
    @pytest.fixture(scope="class")
    def data(self) -> dict:
        with _C0_JSON.open() as fh:
            return json.load(fh)

    def test_grid_covers_section_2_2(self, data: dict) -> None:
        ops = {(b["F"], b["zF"]) for b in data["per_op"]}
        assert ops == _EXPECTED_OPS

    def test_no_retune_per_op(self, data: dict) -> None:
        assert data["c0_config"]["retune_per_op"] is False

    def test_infeasibility_count_is_non_negative_int(self, data: dict) -> None:
        total = data["aggregate"]["total_scenario_infeasibilities"]
        assert isinstance(total, int)
        assert total >= 0
        # Per-OP counts must sum to the aggregate
        per_op_sum = sum(b["infeasibility_count"] for b in data["per_op"])
        assert per_op_sum == total
