"""Tests for ``tools/run_doe_confirmation.py``.

The confirmation driver shells out to ``run_c2_smoke.py``; tests
focus on the manifest construction, the result computation including
the kpis.md §1.1 pass/fail flag, and the fail-fast on missing
``confirmation_spec.json``.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any

import pytest

from industrial_ai.agents.errors import MissingConfirmationSpecError
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
def doe_confirm() -> Any:
    return _load_module("run_doe_confirmation", _REPO_ROOT / "tools" / "run_doe_confirmation.py")


def _spec(*, model: str = "test-model", reasoning_config: str = "off") -> dict[str, Any]:
    return {
        "model_identifier": model,
        "optimum_cell": {
            "temperature": 0.3,
            "top_p": 0.95,
            "reasoning_config": reasoning_config,
            "reasoning_budget": None if reasoning_config == "off" else 2048,
            "cell_id": f"T=0.3_p=0.95_R={reasoning_config}",
        },
        "screening_metrics_n5": {
            "mean_canonical_iae": 0.0,
            "ci_95_canonical_iae": [0.0, 0.0],
            "completion_tokens_p95": 197,
            "wall_clock_p95": 16.3,
            "seeds_used": [0, 1, 2, 3, 4],
        },
        "confirmation_seeds": [5, 6, 7, 8, 9],
        "confirmation_output_root": "/tmp/dummy",
        "selection_rationale": "test",
        "kpis_md_threshold_canonical_iae": 0.01,
    }


def test_build_manifest_creates_5_cells(doe_confirm: Any) -> None:
    spec = _spec()
    manifest = doe_confirm._build_manifest(spec)
    assert len(manifest["cells"]) == 5
    assert [c["seed"] for c in manifest["cells"]] == [5, 6, 7, 8, 9]
    assert all(c["status"] == "pending" for c in manifest["cells"])
    assert all(c["temperature"] == 0.3 for c in manifest["cells"])


def test_smoke_command_off_reasoning(doe_confirm: Any) -> None:
    cell = {
        "cell_id": "T=0.3_p=0.95_R=off_S=5",
        "temperature": 0.3,
        "top_p": 0.95,
        "reasoning_config": "off",
        "reasoning_budget": None,
        "seed": 5,
    }
    cmd = doe_confirm._smoke_command(cell, "x", Path("/tmp/cell"))
    assert "--reasoning-mode" in cmd
    assert cmd[cmd.index("--reasoning-mode") + 1] == "off"
    assert "--reasoning-budget" not in cmd


def test_smoke_command_on_reasoning_with_budget(doe_confirm: Any) -> None:
    cell = {
        "cell_id": "T=0.3_p=0.95_R=on_budget_2048_S=5",
        "temperature": 0.3,
        "top_p": 0.95,
        "reasoning_config": "on_budget_2048",
        "reasoning_budget": 2048,
        "seed": 5,
    }
    cmd = doe_confirm._smoke_command(cell, "x", Path("/tmp/cell"))
    assert cmd[cmd.index("--reasoning-mode") + 1] == "on"
    assert "2048" in cmd


def test_missing_confirmation_spec_raises_named_error(tmp_path: Path, doe_confirm: Any) -> None:
    """ADR 010 §2 fail-fast: no confirmation_spec.json → named exception."""
    import sys

    saved = sys.argv[:]
    sys.argv = ["run_doe_confirmation.py", "--analysis-root", str(tmp_path)]
    try:
        with pytest.raises(MissingConfirmationSpecError):
            doe_confirm.main()
    finally:
        sys.argv = saved


def test_write_confirmation_result_combines_screening_and_confirmation(
    tmp_path: Path, doe_confirm: Any
) -> None:
    """Result combines screening IAE (from on-disk smokes) + confirmation cells."""
    spec = _spec()
    spec["confirmation_output_root"] = str(tmp_path / "cf")
    # Write the screening smoke.jsons (seeds 0..4 all IAE 0.0).
    cell_id = spec["optimum_cell"]["cell_id"]
    for seed in [0, 1, 2, 3, 4]:
        d = tmp_path / f"{cell_id}_S={seed}"
        d.mkdir(parents=True)
        (d / "smoke.json").write_text(
            json.dumps(
                {
                    "aggregate": {
                        "iae_mole_fraction_min": 0.0,
                        "completion_tokens_p95": 150,
                        "cycle_wall_clock_seconds_p95": 12.0,
                        "completed_cycles": 12,
                    }
                }
            )
        )
    # Build a confirmation manifest with cells "done" + canonical_iae 0.0.
    manifest = doe_confirm._build_manifest(spec)
    for c in manifest["cells"]:
        c["status"] = "done"
        c["canonical_iae"] = 0.0
        c["completion_tokens_p95"] = 160
        c["wall_clock_p95"] = 11.0
    result_path = tmp_path / "confirmation_result.json"
    result = doe_confirm._write_confirmation_result(tmp_path, spec, manifest, result_path)
    assert result_path.exists()
    assert result["n_seeds_total"] == 10
    assert result["n10_metrics"]["mean_canonical_iae"] == pytest.approx(0.0)
    assert result["kpis_md_section_1_1_pass"] is True


def test_write_confirmation_result_threshold_failure_does_not_abort(
    tmp_path: Path, doe_confirm: Any
) -> None:
    """If N=10 IAE > 0.01 threshold, pass flag is False but no exception."""
    spec = _spec()
    # Screening seeds all 0.0; confirmation seeds all 0.5 (breach).
    cell_id = spec["optimum_cell"]["cell_id"]
    for seed in [0, 1, 2, 3, 4]:
        d = tmp_path / f"{cell_id}_S={seed}"
        d.mkdir(parents=True)
        (d / "smoke.json").write_text(
            json.dumps({"aggregate": {"iae_mole_fraction_min": 0.0, "completed_cycles": 12}})
        )
    manifest = doe_confirm._build_manifest(spec)
    for c in manifest["cells"]:
        c["status"] = "done"
        c["canonical_iae"] = 0.5
        c["completion_tokens_p95"] = 150
        c["wall_clock_p95"] = 12.0
    result_path = tmp_path / "confirmation_result.json"
    result = doe_confirm._write_confirmation_result(tmp_path, spec, manifest, result_path)
    assert result["kpis_md_section_1_1_pass"] is False
    assert result["n10_metrics"]["mean_canonical_iae"] > 0.0


def test_idempotent_restart_skips_done_cells(tmp_path: Path, doe_confirm: Any) -> None:
    """Re-reading an existing confirmation_manifest.json finds done cells."""
    spec = _spec()
    spec["confirmation_output_root"] = str(tmp_path)
    manifest = doe_confirm._build_manifest(spec)
    # Mark first two cells done.
    manifest["cells"][0]["status"] = "done"
    manifest["cells"][0]["canonical_iae"] = 0.0
    manifest["cells"][1]["status"] = "done"
    manifest["cells"][1]["canonical_iae"] = 0.0
    atomic_write_json(tmp_path / "confirmation_manifest.json", manifest)
    saved = json.loads((tmp_path / "confirmation_manifest.json").read_text())
    pending = [c for c in saved["cells"] if c["status"] != "done"]
    assert len(pending) == 3
