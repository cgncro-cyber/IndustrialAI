"""Phase-2 gate: C1 (Linear MPC) must outperform C0 on >= 3/5 scenarios.

This test reads the persisted ``c1_baseline_kpis.json`` rather than
re-running the scenarios — both because the full sweep is expensive
(~20 s) and because the audit JSON is the artifact downstream phases
(Phase 3 agent benchmark, Phase 5 paper figures) actually consume. If
the JSON is missing the test instructs the operator to run
``tools/run_c1_baseline_scenarios.py``.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_C1_BASELINE = _REPO_ROOT / "data" / "reference" / "c1_baseline_kpis.json"


pytestmark = pytest.mark.skipif(
    not _C1_BASELINE.exists(),
    reason="c1_baseline_kpis.json not yet produced (run tools/run_c1_baseline_scenarios.py)",
)


def test_c1_gate_passes_per_phase2_criterion() -> None:
    """The persisted JSON must record gate_passed=True (>= 3 of 5 wins)."""
    with _C1_BASELINE.open() as fh:
        data = json.load(fh)
    aggregate = data["aggregate"]
    assert aggregate["scenarios_won_by_c1"] >= 3, (
        f"C1 only won {aggregate['scenarios_won_by_c1']}/5 scenarios — Phase-2 gate requires >= 3"
    )
    assert aggregate["gate_passed"] is True


def test_aggregate_iae_ratio_at_least_2x() -> None:
    """A defensible C1 implementation beats C0 in aggregate IAE by >= 2x.

    Looser than the Day-3 result (6.8x), but tight enough to catch a
    regressed or misconfigured MPC.
    """
    with _C1_BASELINE.open() as fh:
        data = json.load(fh)
    ratio = data["aggregate"]["ratio_c0_over_c1"]
    assert ratio >= 2.0, f"aggregate IAE ratio {ratio:.2f}x below the 2x sanity bar"


def test_every_scenario_completes_without_solver_failure() -> None:
    """No scenario may have left a NaN sentinel in the persisted KPIs."""
    with _C1_BASELINE.open() as fh:
        data = json.load(fh)
    for name, block in data["scenarios"].items():
        assert block["success"] is True, f"{name}: simulation reported failure"
        assert block["constraint_violations"] >= 0, (
            f"{name}: constraint_violations sentinel -1 indicates solver bail-out"
        )


def test_max_supervisory_wall_clock_under_one_minute() -> None:
    """Per ADR 006, the supervisor's wall clock must stay << the cadence."""
    with _C1_BASELINE.open() as fh:
        data = json.load(fh)
    cadence_min = data["mpc_config"]["sampling_time_min"]
    for name, block in data["scenarios"].items():
        max_wall = block["max_cycle_wall_clock_seconds"]
        assert max_wall < cadence_min * 60.0 * 0.2, (
            f"{name}: MPC solve {max_wall:.1f}s exceeds 20 % of {cadence_min}-min cadence"
        )
