"""SIMC 2DoF tracking-filter sanity check.

The 2DoF variant of SIMC differs from 1DoF only by a first-order
setpoint filter ahead of the PID. On a pure-tracking scenario the
filter must visibly soften the controller's initial kick (smaller
peak MV move) and reduce overshoot relative to 1DoF, even when
aggregate IAE across the full mixed scenario set looks similar.
This test exercises that contrast on the yD_setpoint_+0p5pct
scenario alone — a clean tracking test with no disturbance.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from industrial_ai.control.c0_variants import build_pids_for_variant
from industrial_ai.control.scenarios import build_scenario
from industrial_ai.control.simc import simc_tunings_from_linearization
from industrial_ai.twin.column_a import DEFAULT_PARAMETERS
from industrial_ai.twin.column_a.linearize import linearize_lv
from industrial_ai.twin.simulate import simulate_lv_closed_loop


def _build_simc_variant_from_lin(
    *, model: object, variant: str
) -> tuple[float, float, float, float]:
    """Convenience: return (Kp_top, Ti_top, Kp_bottom, Ti_bottom) from SIMC."""
    top, bottom = simc_tunings_from_linearization(model, variant=variant)
    return top.Kp, top.Ti, bottom.Kp, bottom.Ti


def test_2dof_filter_softens_pure_tracking_step(
    skogestad_reference_state: npt.NDArray[np.float64],
) -> None:
    """On a pure y_D setpoint step, SIMC-2DoF must apply less aggressive LT swings.

    The filter time constant tau_c moves the *requested* setpoint
    through a first-order lag, so the PID sees a slower-rising target
    and produces a smaller initial MV deviation than the 1DoF variant.
    A direct equality of aggregate IAE on the mixed 5-scenario set is
    not informative because 4 of 5 scenarios are disturbance-dominated
    (the filter only affects setpoint tracking). This isolated
    tracking test gives the unambiguous signal.
    """
    p = DEFAULT_PARAMETERS
    L0 = p.nominal_reflux_L0_kmol_per_min
    V0 = p.nominal_boilup_V0_kmol_per_min
    lin = linearize_lv(
        X_ss=skogestad_reference_state,
        L_ss=L0,
        V_ss=V0,
        F_ss=p.nominal_feed_F_kmol_per_min,
        zF_ss=0.5,
        backend="casadi",
    )
    # Both variants share PI gains; only the filter differs.
    top_t, bottom_t = simc_tunings_from_linearization(lin, variant="1dof")
    Kp_top, Ti_top = top_t.Kp, top_t.Ti
    Kp_bottom, Ti_bottom = bottom_t.Kp, bottom_t.Ti

    scenario_fn, spec = build_scenario("yD_setpoint_+0p5pct")

    from industrial_ai.control.c0_variants import C0Variant
    from industrial_ai.control.decoupler import identity_decoupler

    variant_1dof = C0Variant(
        name="probe_1dof",
        tuning_method="SIMC-1DoF",
        Kp_top=Kp_top,
        Ti_top_min=Ti_top,
        Kp_bottom=Kp_bottom,
        Ti_bottom_min=Ti_bottom,
        decoupler=identity_decoupler(),
        setpoint_filter_tau_min=None,
        reference="test",
    )
    variant_2dof = C0Variant(
        name="probe_2dof",
        tuning_method="SIMC-2DoF",
        Kp_top=Kp_top,
        Ti_top_min=Ti_top,
        Kp_bottom=Kp_bottom,
        Ti_bottom_min=Ti_bottom,
        decoupler=identity_decoupler(),
        setpoint_filter_tau_min=top_t.tau_c,
        reference="test",
    )

    def run(variant: C0Variant) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        top, bottom = build_pids_for_variant(variant, LT_initial=L0, VB_initial=V0)
        sim = simulate_lv_closed_loop(
            X0=skogestad_reference_state,
            scenario=scenario_fn,
            duration_min=spec.horizon_min,
            tick_dt_min=0.05,
            pid_top=top,
            pid_bottom=bottom,
            mv_decoupler=None,
            setpoint_filter_tau_min=variant.setpoint_filter_tau_min,
        )
        return sim.inputs[:, 0], sim.y_D  # LT trajectory and measured y_D

    LT_1dof, y_D_1dof = run(variant_1dof)
    LT_2dof, y_D_2dof = run(variant_2dof)

    # Maximum LT deviation from the bias right after the setpoint step.
    peak_dev_1dof = float(np.max(np.abs(LT_1dof - L0)))
    peak_dev_2dof = float(np.max(np.abs(LT_2dof - L0)))
    assert peak_dev_2dof < peak_dev_1dof * 0.95, (
        f"2DoF must produce a smaller peak MV swing than 1DoF on a "
        f"tracking step. Got 1DoF={peak_dev_1dof:.4f}, 2DoF={peak_dev_2dof:.4f}"
    )

    # y_D peak overshoot: 2DoF must not overshoot the new setpoint as
    # hard as 1DoF. (For this small step + slow dynamics, both stay
    # below the new setpoint, but the 2DoF stays *further* below at
    # the peak rise rate.)
    sp_new = 0.995
    peak_y_D_1dof = float(np.max(y_D_1dof))
    peak_y_D_2dof = float(np.max(y_D_2dof))
    # Either both undershoot (still rising) or both overshoot — the
    # invariant is that 2DoF's rise is slower than 1DoF's, i.e. at
    # the same horizon the 2DoF y_D is closer to the initial value
    # than 1DoF's.
    rise_1dof = peak_y_D_1dof - skogestad_reference_state[DEFAULT_PARAMETERS.NT - 1]
    rise_2dof = peak_y_D_2dof - skogestad_reference_state[DEFAULT_PARAMETERS.NT - 1]
    assert rise_2dof < rise_1dof, (
        f"2DoF should yield a slower y_D rise than 1DoF; got "
        f"1DoF rise={rise_1dof:.5f}, 2DoF rise={rise_2dof:.5f}, sp={sp_new}"
    )
