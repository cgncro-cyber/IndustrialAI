"""Smoke tests for the ``max_wall_clock_seconds`` cap on the LV simulator."""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from industrial_ai.control.scenarios import build_scenario
from industrial_ai.twin.simulate import simulate_lv_closed_loop

_SS_FIXTURE = (
    Path(__file__).resolve().parent.parent.parent
    / "data"
    / "reference"
    / "skogestad_column_a_steady_state.json"
)


def _load_nominal_ss() -> np.ndarray:
    with _SS_FIXTURE.open() as fh:
        ss = json.load(fh)["steady_state"]
    return np.array(ss["compositions"] + ss["holdups_kmol"], dtype=np.float64)


def test_cap_triggers_clean_abort() -> None:
    """A sub-millisecond cap must abort the sim and surface a clear message."""
    X0 = _load_nominal_ss()
    scenario_fn, spec = build_scenario("F_step_+20pct")
    t0 = time.perf_counter()
    sim = simulate_lv_closed_loop(
        X0=X0,
        scenario=scenario_fn,
        duration_min=spec.horizon_min,
        tick_dt_min=0.05,
        max_wall_clock_seconds=0.001,
    )
    elapsed = time.perf_counter() - t0
    assert sim.success is False
    assert "wall-clock cap" in sim.message
    # Abort must not take orders of magnitude longer than the cap itself.
    assert elapsed < 2.0


def test_generous_cap_lets_sim_complete() -> None:
    """A cap larger than the nominal sim time must not interfere."""
    X0 = _load_nominal_ss()
    scenario_fn, _spec = build_scenario("F_step_+20pct")
    sim = simulate_lv_closed_loop(
        X0=X0,
        scenario=scenario_fn,
        duration_min=30.0,
        tick_dt_min=0.05,
        max_wall_clock_seconds=60.0,
    )
    assert sim.success is True
    assert "wall-clock" not in sim.message


def test_default_no_cap_unchanged_behavior() -> None:
    """Omitting the cap reproduces the pre-cap behavior (no early abort)."""
    X0 = _load_nominal_ss()
    scenario_fn, _spec = build_scenario("F_step_+20pct")
    sim = simulate_lv_closed_loop(
        X0=X0,
        scenario=scenario_fn,
        duration_min=10.0,
        tick_dt_min=0.05,
    )
    assert sim.success is True
    assert np.all(np.isfinite(sim.X))
