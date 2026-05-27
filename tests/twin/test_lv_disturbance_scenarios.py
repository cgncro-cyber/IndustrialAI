"""End-to-end LV disturbance scenarios with full data-logging-contract verification.

Phase 1 gate item: *"Three independent disturbance scenarios
(feed-rate step, feed-composition step, reflux step) run end-to-end
and write the full data-logging contract."*

Each test below runs ``simulate_lv_closed_loop`` against one
disturbance scenario, then opens every artifact written by
:class:`RunLogger` (timeseries / tray_profile / setpoints parquets,
kpis / latency / manifest JSON, config YAML) and asserts the
contract from ``docs/figures.md`` is fully populated.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pandas as pd
import pytest
import yaml

from industrial_ai.twin.column_a import DEFAULT_PARAMETERS
from industrial_ai.twin.data_logging import RunLogger
from industrial_ai.twin.simulate import (
    ScenarioStep,
    SimulationResult,
    simulate_lv_closed_loop,
)

# Conservative scenario horizon: enough ticks to populate every channel
# (especially the tray-profile every-20-ticks subsampling), but short
# enough to keep the test fast. With tick_dt=0.05 min and 30 min total
# the simulator runs 600 ticks; the tray profile gets ~30 snapshots.
_DURATION_MIN = 30.0
_TICK_DT_MIN = 0.05
_STEP_TIME_MIN = 5.0


def _make_step_scenario(
    *,
    base: ScenarioStep,
    field: str,
    step_value: float,
    step_time: float = _STEP_TIME_MIN,
) -> object:
    """Return a scenario function that applies a step in ``field`` at ``step_time``.

    ``field`` names the attribute of :class:`ScenarioStep` to step
    (``"F"``, ``"zF"``, or ``"y_D_setpoint"`` for the three scenarios).
    """

    def scenario(t: float) -> ScenarioStep:
        if t < step_time:
            return base
        kwargs = dict(
            y_D_setpoint=base.y_D_setpoint,
            x_B_setpoint=base.x_B_setpoint,
            F=base.F,
            zF=base.zF,
            qF=base.qF,
        )
        kwargs[field] = step_value
        return ScenarioStep(**kwargs)

    return scenario


def _nominal_scenario_step() -> ScenarioStep:
    return ScenarioStep(
        y_D_setpoint=0.99,
        x_B_setpoint=0.01,
        F=DEFAULT_PARAMETERS.nominal_feed_F_kmol_per_min,
        zF=0.5,
        qF=DEFAULT_PARAMETERS.nominal_feed_liquid_fraction_qF,
    )


def _run_with_logger(
    *,
    X0: npt.NDArray[np.float64],
    scenario: object,
    tmp_path: Path,
    scenario_name: str,
) -> tuple[SimulationResult, Path]:
    runs_root = tmp_path / "runs"
    logger = RunLogger.create(
        runs_root=runs_root,
        config="c0_pid",
        scenario=scenario_name,
        seed=0,
    )
    result = simulate_lv_closed_loop(
        X0=X0,
        scenario=scenario,
        duration_min=_DURATION_MIN,
        tick_dt_min=_TICK_DT_MIN,
        logger=logger,
        config_snapshot={
            "scenario": scenario_name,
            "duration_min": _DURATION_MIN,
            "tick_dt_min": _TICK_DT_MIN,
        },
    )
    # KPIs: y_D mean over the post-step window, IAE on y_D, max wall clock.
    nt = result.X.shape[1] // 2
    post_step_mask = result.t > _STEP_TIME_MIN
    y_D_post = result.X[post_step_mask, nt - 1]
    x_B_post = result.X[post_step_mask, 0]
    logger.set_kpis(
        {
            "y_D_post_step_mean": float(np.mean(y_D_post)),
            "x_B_post_step_mean": float(np.mean(x_B_post)),
            "iae_y_D": float(
                np.trapezoid(
                    np.abs(y_D_post - 0.99),
                    result.t[post_step_mask],
                )
            ),
            "max_cycle_wall_clock_seconds": float(np.max(result.cycle_wall_clock_seconds)),
            "success": result.success,
        }
    )
    paths = logger.finalize(input_hashes={"X0_sha256_proxy": "test_fixture"})
    return result, paths.root


def _assert_data_logging_contract(run_root: Path, *, expect_safety_log: bool) -> None:
    """Verify every contract artifact exists, parses, and is non-empty."""
    expected_paths = {
        "timeseries.parquet",
        "tray_profile.parquet",
        "setpoints.parquet",
        "kpis.json",
        "latency.json",
        "config.yaml",
        "manifest.json",
    }
    if expect_safety_log:
        expected_paths.add("safety_log.parquet")
    actual = {p.name for p in run_root.iterdir()}
    missing = expected_paths - actual
    assert not missing, f"data-logging contract incomplete; missing: {sorted(missing)}"

    timeseries = pd.read_parquet(run_root / "timeseries.parquet")
    assert len(timeseries) > 0
    for col in ("t", "y_D", "x_B", "L", "V", "D", "B", "F", "zF", "qF"):
        assert col in timeseries.columns, f"timeseries missing column {col!r}"

    tray = pd.read_parquet(run_root / "tray_profile.parquet")
    assert len(tray) > 0
    for col in ("t", "stage", "composition", "holdup_kmol"):
        assert col in tray.columns
    n_stages_in_tray = tray["stage"].nunique()
    assert n_stages_in_tray == DEFAULT_PARAMETERS.NT, (
        f"tray profile should have {DEFAULT_PARAMETERS.NT} stages, has {n_stages_in_tray}"
    )

    setpoints = pd.read_parquet(run_root / "setpoints.parquet")
    assert len(setpoints) > 0
    assert set(setpoints["channel"].unique()) >= {"y_D", "x_B"}

    with (run_root / "kpis.json").open() as fh:
        kpis = json.load(fh)
    assert kpis["success"] is True
    assert "iae_y_D" in kpis

    with (run_root / "latency.json").open() as fh:
        latency = json.load(fh)
    assert len(latency) > 0
    assert all("cycle_index" in row and "wall_clock_seconds" in row for row in latency)

    with (run_root / "config.yaml").open() as fh:
        config = yaml.safe_load(fh)
    assert "scenario" in config

    with (run_root / "manifest.json").open() as fh:
        manifest = json.load(fh)
    assert "artifact_hashes" in manifest
    assert "package_versions" in manifest
    assert manifest["artifact_hashes"], "manifest should hash at least one artifact"


@pytest.mark.parametrize(
    ("scenario_name", "step_field", "step_value"),
    [
        ("F_step_+20pct", "F", 1.2 * DEFAULT_PARAMETERS.nominal_feed_F_kmol_per_min),
        ("zF_step_-10pct", "zF", 0.4),
        ("yD_setpoint_step_+0p5pct", "y_D_setpoint", 0.995),
    ],
)
def test_lv_disturbance_scenario_end_to_end(
    skogestad_reference_state: npt.NDArray[np.float64],
    tmp_path: Path,
    scenario_name: str,
    step_field: str,
    step_value: float,
) -> None:
    """Run the three Phase-1-mandated disturbance scenarios and verify the contract."""
    base = _nominal_scenario_step()
    scenario = _make_step_scenario(base=base, field=step_field, step_value=step_value)

    result, run_root = _run_with_logger(
        X0=skogestad_reference_state,
        scenario=scenario,
        tmp_path=tmp_path,
        scenario_name=scenario_name,
    )

    assert result.success, result.message
    assert np.all(np.isfinite(result.X)), "state contains NaN/Inf"
    assert result.X.shape[0] == result.t.shape[0]

    # Verify that the scenario actually injected the disturbance into
    # the closed loop. Composition response to F/zF steps is governed by
    # tau_1 ~ 194 min (Skogestad 1997 Eq. 31), so in a 25-min post-step
    # window the y_D move can be tiny under tight regulatory control;
    # the meaningful invariant is that the disturbance entered the
    # plant-side input vector or the supervisor's requested setpoint.
    tick_times = result.t[1:]
    pre_mask = tick_times < _STEP_TIME_MIN
    post_mask = tick_times > _STEP_TIME_MIN
    if step_field == "y_D_setpoint":
        requested_pre = result.requested_setpoints[pre_mask, 0]
        requested_post = result.requested_setpoints[post_mask, 0]
        assert np.abs(requested_post.mean() - requested_pre.mean()) > 1e-4
    else:
        column = {"F": 4, "zF": 5}[step_field]
        plant_pre = result.inputs[pre_mask, column]
        plant_post = result.inputs[post_mask, column]
        assert np.abs(plant_post.mean() - plant_pre.mean()) > 1e-4, (
            f"{scenario_name}: {step_field} disturbance did not enter the input vector"
        )

    _assert_data_logging_contract(run_root, expect_safety_log=False)
