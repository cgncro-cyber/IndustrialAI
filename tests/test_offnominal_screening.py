"""Tests for the Schritt-B off-nominal screening driver + analyzer + smoke flags."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any

import pytest

from industrial_ai.io import atomic_write_json

_REPO_ROOT = Path(__file__).resolve().parent.parent


def _load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {name} from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def screening_driver() -> Any:
    return _load_module(
        "run_offnominal_screening", _REPO_ROOT / "tools" / "run_offnominal_screening.py"
    )


@pytest.fixture(scope="module")
def screening_analyzer() -> Any:
    return _load_module(
        "analyze_offnominal_screening", _REPO_ROOT / "tools" / "analyze_offnominal_screening.py"
    )


# ---------------------------------------------------------------------------
# Driver — cell enumeration + smoke command shape
# ---------------------------------------------------------------------------


def test_enumerate_cells_full_count(screening_driver: Any) -> None:
    """Per kpis.md §2.5 amendment 2026-06-02: 3-corner grid (LV-singular excluded)."""
    cells = screening_driver.enumerate_cells()
    assert len(cells) == 3 * 5 * 2 * 10 == 300


def test_enumerate_cells_have_unique_ids(screening_driver: Any) -> None:
    cells = screening_driver.enumerate_cells()
    ids = [c["cell_id"] for c in cells]
    assert len(set(ids)) == 300


def test_enumerate_cells_cover_3_corners_excluding_lv_singular(
    screening_driver: Any,
) -> None:
    """LV-singular (0.8, 0.45) corner is NOT in the grid (kpis.md §2.5)."""
    cells = screening_driver.enumerate_cells()
    ops_seen = {(c["op_F"], c["op_zF"]) for c in cells}
    assert ops_seen == {(1.2, 0.45), (1.2, 0.55), (0.8, 0.55)}
    assert (0.8, 0.45) not in ops_seen


def test_enumerate_cells_cover_5_scenarios_2_submetrics(screening_driver: Any) -> None:
    cells = screening_driver.enumerate_cells()
    assert {c["scenario"] for c in cells} == {
        "F_step_+20pct",
        "F_step_-20pct",
        "zF_step_+10pct",
        "zF_step_-10pct",
        "yD_setpoint_+0p5pct",
    }
    assert {c["submetric"] for c in cells} == {"target_acquisition", "disturbance_rejection"}
    assert {c["seed"] for c in cells} == set(range(10))


def test_smoke_command_threads_all_factors(screening_driver: Any) -> None:
    cell = {
        "cell_id": "F=0.8_zF=0.45_S=F_step_+20pct_M=target_acquisition_seed0",
        "op_F": 0.8,
        "op_zF": 0.45,
        "scenario": "F_step_+20pct",
        "submetric": "target_acquisition",
        "seed": 0,
    }
    cmd = screening_driver._smoke_command(cell, "test-model", Path("/tmp/cell"))
    assert "--scenario" in cmd
    assert "F_step_+20pct" in cmd
    assert "--op-F" in cmd and "0.8" in cmd
    assert "--op-zF" in cmd
    assert "--submetric" in cmd
    assert "target_acquisition" in cmd
    assert "--seed" in cmd and "0" in cmd


def test_classify_smoke_failure_recognises_infeasible_submetric(screening_driver: Any) -> None:
    err_class, _excerpt = screening_driver._classify_smoke_failure(
        returncode=1, stdout="", stderr="InfeasibleSubmetricError: ..."
    )
    assert err_class == "pre_stage_infeasible"


def test_dry_run_marks_all_cells_done(tmp_path: Path, screening_driver: Any) -> None:
    manifest = screening_driver._build_manifest("test-model")
    manifest_path = tmp_path / "sweep_manifest.json"
    atomic_write_json(manifest_path, manifest)
    pending = list(
        screening_driver._select_pending(manifest["cells"], retry_failed=False, max_cells=3)
    )
    assert len(pending) == 3
    for cell in pending:
        screening_driver._run_one_cell(
            cell,
            model="test-model",
            output_root=tmp_path,
            manifest_path=manifest_path,
            manifest=manifest,
            timeout_s=10.0,
            dry_run=True,
        )
    saved = json.loads(manifest_path.read_text())
    assert sum(1 for c in saved["cells"] if c["status"] == "done") == 3


def test_idempotent_restart_skips_done(screening_driver: Any) -> None:
    cells = screening_driver.enumerate_cells()
    cells[0]["status"] = "done"
    cells[5]["status"] = "failed"
    cells[10]["status"] = "pending"
    pending = list(screening_driver._select_pending(cells, retry_failed=False, max_cells=None))
    assert not any(c["status"] == "done" for c in pending)
    assert not any(c["status"] == "failed" for c in pending)


def test_retry_failed_includes_failed(screening_driver: Any) -> None:
    cells = screening_driver.enumerate_cells()
    cells[0]["status"] = "done"
    cells[5]["status"] = "failed"
    pending = list(screening_driver._select_pending(cells, retry_failed=True, max_cells=None))
    assert any(c["status"] == "failed" for c in pending)
    assert not any(c["status"] == "done" for c in pending)


# ---------------------------------------------------------------------------
# Analyzer — synthetic 400-cell input → expected aggregates + Bucket-B verdict
# ---------------------------------------------------------------------------


def _synth_done_cell(
    op_F: float,
    op_zF: float,
    scenario: str,
    submetric: str,
    seed: int,
    iae: float,
) -> dict[str, Any]:
    return {
        "cell_id": f"F={op_F:g}_zF={op_zF:g}_S={scenario}_M={submetric}_seed{seed}",
        "op_F": op_F,
        "op_zF": op_zF,
        "scenario": scenario,
        "submetric": submetric,
        "seed": seed,
        "status": "done",
        "canonical_iae": iae,
        "completion_tokens_p95": 200,
        "wall_clock_p95": 25.0,
        "completed_cycles": 12,
    }


def test_aggregate_cells_collapses_seeds(screening_analyzer: Any) -> None:
    """Use one of the 3 valid corners (LV-singular (0.8, 0.45) excluded)."""
    cells = [
        _synth_done_cell(1.2, 0.45, "F_step_+20pct", "target_acquisition", s, 0.5 + 0.01 * s)
        for s in range(10)
    ]
    aggregates = screening_analyzer._aggregate_cells(cells)
    assert len(aggregates) == 1
    key = (1.2, 0.45, "F_step_+20pct", "target_acquisition")
    assert aggregates[key]["n_seeds_used"] == 10
    assert aggregates[key]["mean_canonical_iae"] == pytest.approx(
        sum(0.5 + 0.01 * s for s in range(10)) / 10
    )


def test_per_op_p95_picks_highest_scenario_mean(screening_analyzer: Any) -> None:
    """5 scenarios at one valid OP (LV-singular (0.8, 0.45) excluded per §2.5)."""
    cells: list[dict[str, Any]] = []
    scenarios = [
        ("F_step_+20pct", 0.1),
        ("F_step_-20pct", 0.2),
        ("zF_step_+10pct", 0.3),
        ("zF_step_-10pct", 0.4),
        ("yD_setpoint_+0p5pct", 0.5),
    ]
    test_op = (1.2, 0.45)
    for scenario, base_iae in scenarios:
        for s in range(10):
            cells.append(
                _synth_done_cell(
                    test_op[0], test_op[1], scenario, "target_acquisition", s, base_iae
                )
            )
    aggregates = screening_analyzer._aggregate_cells(cells)
    per_op = screening_analyzer._per_op_p95(aggregates, "target_acquisition")
    assert test_op in per_op
    p95 = per_op[test_op]["p95_canonical_iae"]
    assert p95 == pytest.approx(0.5, abs=0.05)  # P95 lands at/near the max=0.5


def test_classify_submetric_strong_band_when_c2_much_lower(
    screening_analyzer: Any,
) -> None:
    c2_grid = {
        "p95_of_p95s": 0.1,
        "per_op_values": [0.1, 0.1, 0.1],
    }
    per_op_c2 = {
        (1.2, 0.45): {"ci_95_p95": [0.05, 0.15]},
        (1.2, 0.55): {"ci_95_p95": [0.05, 0.15]},
        (0.8, 0.55): {"ci_95_p95": [0.05, 0.15]},
    }
    c1_grid = {
        "grid_p95_of_p95s": 1.0,  # C1 worst is 10x C2's worst
        "grid_max_of_p95s": 1.0,
        "grid_mean_of_p95s": 1.0,
        "per_op_p95": [],
        "n_ops": 3,
    }
    verdict = screening_analyzer._classify_submetric(
        "target_acquisition", c2_grid, c1_grid, per_op_c2
    )
    assert verdict["bucket_b_threshold_clear"] is True
    assert verdict["evidence_band"] == "strong"


def test_classify_submetric_fails_band_when_c2_no_improvement(
    screening_analyzer: Any,
) -> None:
    c2_grid = {
        "p95_of_p95s": 1.0,
        "per_op_values": [1.0, 1.0, 1.0],
    }
    per_op_c2 = {
        (1.2, 0.45): {"ci_95_p95": [0.9, 1.1]},
        (1.2, 0.55): {"ci_95_p95": [0.9, 1.1]},
        (0.8, 0.55): {"ci_95_p95": [0.9, 1.1]},
    }
    c1_grid = {"grid_p95_of_p95s": 1.0, "per_op_p95": [], "n_ops": 3}
    verdict = screening_analyzer._classify_submetric(
        "target_acquisition", c2_grid, c1_grid, per_op_c2
    )
    assert verdict["bucket_b_threshold_clear"] is False
    assert verdict["evidence_band"] == "fails"


def test_decide_overall_both_clear(screening_analyzer: Any) -> None:
    a = {"bucket_b_threshold_clear": True, "evidence_band": "strong"}
    b = {"bucket_b_threshold_clear": True, "evidence_band": "moderate"}
    assert screening_analyzer._decide_overall(a, b) == "Bucket B (both)"


def test_decide_overall_does_not_clear(screening_analyzer: Any) -> None:
    a = {"bucket_b_threshold_clear": False, "evidence_band": "fails"}
    b = {"bucket_b_threshold_clear": False, "evidence_band": "fails"}
    assert screening_analyzer._decide_overall(a, b) == "does_not_clear"


def test_c1_per_op_p95_reads_baseline_files(screening_analyzer: Any) -> None:
    """End-to-end: real C1 baselines on disk produce a finite grid P95."""
    with open(_REPO_ROOT / "data/reference/c1_off_nominal_baseline.json") as fh:
        c1 = json.load(fh)
    # The 3-corner amended grid (LV-singular (0.8, 0.45) excluded).
    grid = screening_analyzer._c1_per_op_p95(c1, ((1.2, 0.45), (1.2, 0.55), (0.8, 0.55)))
    assert grid["n_ops"] == 3
    assert grid["grid_p95_of_p95s"] > 0
