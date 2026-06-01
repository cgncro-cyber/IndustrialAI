"""Tests for ``tools/analyze_doe_sweep.py``."""

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
def doe_analyze() -> Any:
    return _load_module("analyze_doe_sweep", _REPO_ROOT / "tools" / "analyze_doe_sweep.py")


def _done_cell(
    *,
    temperature: float,
    top_p: float,
    reasoning_config: str,
    seed: int,
    iae: float,
    tokens_p95: int = 150,
    wall_p95: float = 12.0,
) -> dict[str, Any]:
    return {
        "cell_id": f"T={temperature:g}_p={top_p:g}_R={reasoning_config}_S={seed}",
        "temperature": temperature,
        "top_p": top_p,
        "reasoning_config": reasoning_config,
        "reasoning_budget": None if reasoning_config == "off" else 1024,
        "seed": seed,
        "status": "done",
        "canonical_iae": iae,
        "completion_tokens_p95": tokens_p95,
        "wall_clock_p95": wall_p95,
        "completed_cycles": 12,
    }


def test_aggregate_combines_seeds_per_factor_cell(doe_analyze: Any) -> None:
    cells = [
        _done_cell(temperature=0.6, top_p=0.95, reasoning_config="off", seed=s, iae=0.0)
        for s in range(5)
    ]
    aggregates = doe_analyze._aggregate_by_factor_cell(cells)
    assert len(aggregates) == 1
    a = aggregates[0]
    assert a["n_seeds_used"] == 5
    assert a["seeds_used"] == [0, 1, 2, 3, 4]
    assert a["mean_canonical_iae"] == pytest.approx(0.0)
    assert a["ci_95_canonical_iae"] == [0.0, 0.0]


def test_aggregate_skips_failed_cells(doe_analyze: Any) -> None:
    cells = [_done_cell(temperature=0.6, top_p=0.95, reasoning_config="off", seed=0, iae=0.0)]
    cells.append(
        {
            "cell_id": "T=0.6_p=0.95_R=off_S=1",
            "temperature": 0.6,
            "top_p": 0.95,
            "reasoning_config": "off",
            "seed": 1,
            "status": "failed",
            "canonical_iae": None,
        }
    )
    aggregates = doe_analyze._aggregate_by_factor_cell(cells)
    assert len(aggregates) == 1
    assert aggregates[0]["n_seeds_used"] == 1


def test_select_optimum_prefers_lowest_iae_passing_ci_filter(doe_analyze: Any) -> None:
    good_cells = [
        _done_cell(temperature=0.3, top_p=0.95, reasoning_config="off", seed=s, iae=0.0)
        for s in range(5)
    ]
    drift_cells = [
        _done_cell(temperature=1.0, top_p=0.95, reasoning_config="off", seed=s, iae=0.5)
        for s in range(5)
    ]
    aggregates = doe_analyze._aggregate_by_factor_cell(good_cells + drift_cells)
    optimum, rationale = doe_analyze._select_optimum(aggregates)
    assert optimum["temperature"] == 0.3
    assert "argmin" in rationale.lower()
    assert "ci-filtered set was empty" not in rationale.lower()


def test_select_optimum_falls_back_to_full_surface_when_no_cell_clears_ci(
    doe_analyze: Any,
) -> None:
    drift_cells = [
        _done_cell(temperature=t, top_p=0.95, reasoning_config="off", seed=s, iae=0.5 + 0.1 * t)
        for t in (0.0, 0.3)
        for s in range(5)
    ]
    aggregates = doe_analyze._aggregate_by_factor_cell(drift_cells)
    optimum, rationale = doe_analyze._select_optimum(aggregates)
    # Lowest mean wins despite failing CI threshold.
    assert optimum["temperature"] == 0.0
    assert "ci-filtered set was empty" in rationale.lower()


def test_select_optimum_respects_wall_clock_secondary_constraint(
    doe_analyze: Any,
) -> None:
    cells = [
        _done_cell(
            temperature=0.3,
            top_p=0.95,
            reasoning_config="off",
            seed=s,
            iae=0.0,
            wall_p95=12.0,
        )
        for s in range(5)
    ] + [
        _done_cell(
            temperature=0.0,
            top_p=0.95,
            reasoning_config="off",
            seed=s,
            iae=0.0,
            wall_p95=120.0,  # blows the 60s ceiling
        )
        for s in range(5)
    ]
    aggregates = doe_analyze._aggregate_by_factor_cell(cells)
    optimum, rationale = doe_analyze._select_optimum(aggregates)
    # Both have mean IAE 0.0 + clear CI; secondary wall constraint
    # eliminates the 120s cell.
    assert optimum["temperature"] == 0.3
    assert "wall_clock_p95" in rationale


def test_analyze_writes_artifacts(tmp_path: Path, doe_analyze: Any) -> None:
    """End-to-end: synthesize a manifest, run main(), confirm three artifacts land."""
    manifest = {
        "model_identifier": "test-model",
        "factor_space": {},
        "total_cells": 5,
        "cells": [
            _done_cell(temperature=0.3, top_p=0.95, reasoning_config="off", seed=s, iae=0.0)
            for s in range(5)
        ],
    }
    atomic_write_json(tmp_path / "sweep_manifest.json", manifest)
    rc = _invoke_main(doe_analyze, ["--output-root", str(tmp_path)])
    assert rc == 0
    assert (tmp_path / "analysis.json").exists()
    assert (tmp_path / "analysis_summary.md").exists()
    assert (tmp_path / "confirmation_spec.json").exists()
    spec = json.loads((tmp_path / "confirmation_spec.json").read_text())
    assert spec["optimum_cell"]["temperature"] == 0.3
    assert spec["confirmation_seeds"] == [5, 6, 7, 8, 9]


def _invoke_main(module: Any, argv: list[str]) -> int:
    import sys

    saved = sys.argv[:]
    sys.argv = [saved[0], *argv]
    try:
        return int(module.main())
    finally:
        sys.argv = saved
