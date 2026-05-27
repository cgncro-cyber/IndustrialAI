"""Tests for the C1 Linear MPC supervisor."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import pytest

from industrial_ai.control.c1_linear_mpc import (
    C1MPCConfig,
    _supervisor_outputs,
    build_c1_mpc,
    simulate_lv_with_mpc,
)
from industrial_ai.control.scenarios import build_scenario
from industrial_ai.twin.column_a import DEFAULT_PARAMETERS
from industrial_ai.twin.column_a.linearize import linearize_lv


@pytest.fixture(scope="module")
def nominal_linearization(
    skogestad_reference_state: npt.NDArray[np.float64],
):
    """Linearize once per module — the MPC build is cheap but linearization is too."""
    p = DEFAULT_PARAMETERS
    return linearize_lv(
        X_ss=skogestad_reference_state,
        L_ss=p.nominal_reflux_L0_kmol_per_min,
        V_ss=p.nominal_boilup_V0_kmol_per_min,
        F_ss=p.nominal_feed_F_kmol_per_min,
        zF_ss=0.5,
        backend="casadi",
    )


def test_build_c1_mpc_returns_configured_controller(nominal_linearization) -> None:
    """Smoke check that the do-mpc controller builds without errors."""
    mpc, model = build_c1_mpc(nominal_linearization)
    assert mpc.settings.n_horizon == C1MPCConfig().n_horizon
    assert mpc.settings.t_step == C1MPCConfig().sampling_time_min
    assert model.n_x == 2 * DEFAULT_PARAMETERS.NT
    assert model.n_u == 2


def test_mpc_at_nominal_holds_steady_state(
    nominal_linearization,
    skogestad_reference_state: npt.NDArray[np.float64],
) -> None:
    """At the linearization point with nominal setpoints, MPC must hold the bias MVs."""
    mpc, _ = build_c1_mpc(nominal_linearization)
    LT, VB = _supervisor_outputs(
        mpc,
        X_current=skogestad_reference_state,
        linearized=nominal_linearization,
        F=DEFAULT_PARAMETERS.nominal_feed_F_kmol_per_min,
        zF=0.5,
        y_D_sp=0.99,
        x_B_sp=0.01,
    )
    assert LT == pytest.approx(DEFAULT_PARAMETERS.nominal_reflux_L0_kmol_per_min, abs=5e-3)
    assert VB == pytest.approx(DEFAULT_PARAMETERS.nominal_boilup_V0_kmol_per_min, abs=5e-3)


def test_mpc_outputs_respect_bounds(
    nominal_linearization,
    skogestad_reference_state: npt.NDArray[np.float64],
) -> None:
    """MPC-commanded LT, VB must stay inside the configured [0, 10] kmol/min box even on a large perturbation."""
    mpc, _ = build_c1_mpc(nominal_linearization)
    LT, VB = _supervisor_outputs(
        mpc,
        X_current=skogestad_reference_state,
        linearized=nominal_linearization,
        F=1.2 * DEFAULT_PARAMETERS.nominal_feed_F_kmol_per_min,
        zF=0.7,
        y_D_sp=0.99,
        x_B_sp=0.01,
    )
    cfg = C1MPCConfig()
    assert cfg.lt_min <= LT <= cfg.lt_max
    assert cfg.vb_min <= VB <= cfg.vb_max


def test_c1_runs_zf_step_end_to_end(
    nominal_linearization,
    skogestad_reference_state: npt.NDArray[np.float64],
) -> None:
    """A canonical zF step must complete without solver failures and stay finite."""
    mpc, _ = build_c1_mpc(nominal_linearization)
    scenario_fn, spec = build_scenario("zF_step_+10pct")
    sim = simulate_lv_with_mpc(
        X0=skogestad_reference_state,
        scenario=scenario_fn,
        mpc=mpc,
        linearized=nominal_linearization,
        duration_min=spec.horizon_min,
        tick_dt_min=0.05,
    )
    assert sim.success, sim.message
    assert np.all(np.isfinite(sim.X))
    # Compositions must remain physical.
    NT = DEFAULT_PARAMETERS.NT
    assert (sim.X[:, NT - 1] >= 0.0).all() and (sim.X[:, NT - 1] <= 1.0).all()
    assert (sim.X[:, 0] >= 0.0).all() and (sim.X[:, 0] <= 1.0).all()


def test_c1_beats_c0_on_y_d_setpoint_step(
    nominal_linearization,
    skogestad_reference_state: npt.NDArray[np.float64],
) -> None:
    """Sanity: C1 should outperform the relay-tuned C0 on a clean setpoint step.

    Quantitative win threshold matches the published Day-3 result
    (C1 IAE ~ 0.07 vs C0 IAE ~ 0.17 — gives ~2x); we require >= 1.5x
    here to leave headroom for solver-tolerance drift.
    """
    from industrial_ai.control.c0_pid_only import build_c0_pids
    from industrial_ai.evaluation.kpis import compute_kpis
    from industrial_ai.twin.simulate import simulate_lv_closed_loop

    p = DEFAULT_PARAMETERS
    L0, V0 = p.nominal_reflux_L0_kmol_per_min, p.nominal_boilup_V0_kmol_per_min
    scenario_fn, spec = build_scenario("yD_setpoint_+0p5pct")

    # C0
    top, bottom = build_c0_pids(LT_initial=L0, VB_initial=V0)
    sim_c0 = simulate_lv_closed_loop(
        X0=skogestad_reference_state,
        scenario=scenario_fn,
        duration_min=spec.horizon_min,
        tick_dt_min=0.05,
        pid_top=top,
        pid_bottom=bottom,
    )
    c0_iae = compute_kpis(sim_c0).iae_mole_fraction_min

    # C1
    mpc, _ = build_c1_mpc(nominal_linearization)
    sim_c1 = simulate_lv_with_mpc(
        X0=skogestad_reference_state,
        scenario=scenario_fn,
        mpc=mpc,
        linearized=nominal_linearization,
        duration_min=spec.horizon_min,
        tick_dt_min=0.05,
    )
    c1_iae = compute_kpis(sim_c1).iae_mole_fraction_min
    assert c1_iae < c0_iae / 1.5, (
        f"C1 IAE {c1_iae:.4f} must beat C0 IAE {c0_iae:.4f} by at least 1.5x"
    )
