"""End-to-end smoke: run C0 over all 5 canonical scenarios, score KPIs.

This is the Day-2 baseline number. Day 2.5 (PID tuning shootout) will
have to beat it on aggregate IAE to qualify as the new C0. The test
also locks the pipeline that the Phase-2 benchmark notebook and the
Phase-5 paper Figure 4 will both reuse: scenario factory ->
:func:`simulate_lv_closed_loop` -> :func:`compute_kpis`.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import pytest

from industrial_ai.control.c0_pid_only import build_c0_pids
from industrial_ai.control.scenarios import (
    SCENARIO_NAMES,
    build_scenario,
)
from industrial_ai.evaluation.kpis import KPISet, compute_kpis
from industrial_ai.twin.column_a import DEFAULT_PARAMETERS
from industrial_ai.twin.simulate import simulate_lv_closed_loop


def _run_c0_on(name: str, X0: npt.NDArray[np.float64]) -> KPISet:
    p = DEFAULT_PARAMETERS
    scenario_fn, spec = build_scenario(name)
    top, bottom = build_c0_pids(
        LT_initial=p.nominal_reflux_L0_kmol_per_min,
        VB_initial=p.nominal_boilup_V0_kmol_per_min,
    )
    sim = simulate_lv_closed_loop(
        X0=X0,
        scenario=scenario_fn,
        duration_min=spec.horizon_min,
        tick_dt_min=0.05,
        pid_top=top,
        pid_bottom=bottom,
    )
    assert sim.success, f"{name}: C0 closed loop failed: {sim.message}"
    return compute_kpis(sim)


@pytest.mark.parametrize("name", SCENARIO_NAMES)
def test_c0_completes_each_scenario(
    name: str, skogestad_reference_state: npt.NDArray[np.float64]
) -> None:
    """C0 must finish all 5 scenarios without solver divergence."""
    kpis = _run_c0_on(name, skogestad_reference_state)
    # Sanity floors — these are the bar the Day-2.5 shootout must clear,
    # not a quality gate. Any tightening belongs to the shootout.
    assert kpis.constraint_violations >= 0  # not the failure sentinel
    assert np.isfinite(kpis.specific_energy_kmol_per_kmol)
    # Transient yield can exceed 1.0 on a F- or zF-step because the
    # column releases stored light holdup faster than the new lower
    # feed rate replenishes it — physically correct over a finite
    # window, asymptotically <= 1 at the new SS. The loose upper bound
    # only guards against gross broken pipelines.
    assert 0.0 <= kpis.light_yield <= 1.5
    assert kpis.mv_activity_kmol_per_min >= 0.0
    assert kpis.iae_mole_fraction_min >= 0.0


def test_c0_aggregate_iae_across_scenarios(
    skogestad_reference_state: npt.NDArray[np.float64],
) -> None:
    """Aggregated IAE across all 5 scenarios is finite and small in absolute terms.

    The threshold is intentionally loose at this stage — the Day-2.5
    shootout will tighten it once the best PID variant is picked. The
    point of the test is to lock the *pipeline*, not the absolute
    quality bar.
    """
    total = 0.0
    for name in SCENARIO_NAMES:
        kpis = _run_c0_on(name, skogestad_reference_state)
        total += kpis.iae_mole_fraction_min
    # IAE is in mole-fraction-minutes over 60-min horizons; 1.0 is a
    # generous ceiling that catches catastrophically broken tunings.
    assert total < 1.0, f"C0 aggregate IAE = {total:.4f} unexpectedly large"
