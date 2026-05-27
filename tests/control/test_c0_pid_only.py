"""Tests for the C0 PID-only configuration."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pytest

from industrial_ai.control.c0_pid_only import (
    C0PIDTuning,
    build_c0_pids,
    load_c0_tuning,
)
from industrial_ai.twin.column_a import DEFAULT_PARAMETERS
from industrial_ai.twin.simulate import (
    ScenarioStep,
    build_skogestad_phase1_pids,
    simulate_lv_closed_loop,
)


def test_load_c0_tuning_finds_canonical_file() -> None:
    """The default tuning file ships at data/reference/c0_pid_tuning.json."""
    tuning = load_c0_tuning()
    assert isinstance(tuning, C0PIDTuning)
    assert tuning.Kp_top > 0.0
    assert tuning.Ti_top_min > 0.0
    assert tuning.Kp_bottom > 0.0
    assert tuning.Ti_bottom_min > 0.0
    assert tuning.source.exists()


def test_load_c0_tuning_from_explicit_path(tmp_path: Path) -> None:
    """A custom JSON path is honored — useful for tests and re-tuning experiments."""
    payload = {
        "schema_version": 1,
        "loops": {
            "top": {"tyreus_luyben": {"Kp": 12.0, "Ti_min": 18.0, "Ki_per_min": 0.6667}},
            "bottom": {"tyreus_luyben": {"Kp": 15.0, "Ti_min": 9.0, "Ki_per_min": 1.6667}},
        },
    }
    custom = tmp_path / "custom_tuning.json"
    with custom.open("w") as fh:
        json.dump(payload, fh)
    tuning = load_c0_tuning(custom)
    assert tuning.Kp_top == 12.0
    assert tuning.Ti_top_min == 18.0
    assert tuning.Kp_bottom == 15.0
    assert tuning.Ti_bottom_min == 9.0
    assert tuning.source == custom


def test_build_c0_pids_directions_match_physics() -> None:
    """Top loop is direct-acting (more LT -> y_D rises); bottom is reverse-acting."""
    p = DEFAULT_PARAMETERS
    top, bottom = build_c0_pids(
        LT_initial=p.nominal_reflux_L0_kmol_per_min,
        VB_initial=p.nominal_boilup_V0_kmol_per_min,
    )
    assert top.direct_acting is True
    assert bottom.direct_acting is False


def test_build_c0_pids_seeds_integral_for_bias() -> None:
    """At zero error, the seeded controller output equals the bias MV (LT_initial)."""
    p = DEFAULT_PARAMETERS
    L0 = p.nominal_reflux_L0_kmol_per_min
    V0 = p.nominal_boilup_V0_kmol_per_min
    top, bottom = build_c0_pids(LT_initial=L0, VB_initial=V0)
    # One step at zero error must yield the bias to within numerical noise.
    u_top = top.step(measurement=0.5, setpoint=0.5, dt=0.05)
    u_bottom = bottom.step(measurement=0.5, setpoint=0.5, dt=0.05)
    assert u_top == pytest.approx(L0, abs=1e-9)
    assert u_bottom == pytest.approx(V0, abs=1e-9)


def test_detune_factor_scales_kp_only() -> None:
    """detune_factor scales Kp on both loops but leaves Ti unchanged."""
    p = DEFAULT_PARAMETERS
    top_nominal, _ = build_c0_pids(
        LT_initial=p.nominal_reflux_L0_kmol_per_min,
        VB_initial=p.nominal_boilup_V0_kmol_per_min,
        detune_factor=1.0,
    )
    top_detuned, _ = build_c0_pids(
        LT_initial=p.nominal_reflux_L0_kmol_per_min,
        VB_initial=p.nominal_boilup_V0_kmol_per_min,
        detune_factor=0.5,
    )
    assert top_detuned.Kp == pytest.approx(top_nominal.Kp * 0.5)
    # Ti = Kp / Ki, so Ti should be unchanged: both Kp and Ki scale by 0.5.
    Ti_nominal = top_nominal.Kp / top_nominal.Ki
    Ti_detuned = top_detuned.Kp / top_detuned.Ki
    assert Ti_detuned == pytest.approx(Ti_nominal, rel=1e-12)


def test_detune_factor_must_be_positive() -> None:
    """Zero or negative detune is rejected to fail fast."""
    with pytest.raises(ValueError, match="detune_factor"):
        build_c0_pids(LT_initial=2.7, VB_initial=3.2, detune_factor=0.0)


def test_c0_outperforms_phase1_placeholder_on_yD_setpoint_step(
    skogestad_reference_state: npt.NDArray[np.float64],
) -> None:
    """C0 Tyreus-Luyben must track a y_D setpoint step closer than the Phase-1 placeholder.

    Both controllers are stable; the gate is that the relay-tuned C0 closes the
    setpoint gap measurably faster within a 60 min window (one half tau_2).
    """
    p = DEFAULT_PARAMETERS
    L0 = p.nominal_reflux_L0_kmol_per_min
    V0 = p.nominal_boilup_V0_kmol_per_min

    def yD_step(t: float) -> ScenarioStep:
        return ScenarioStep(
            y_D_setpoint=0.995 if t >= 5.0 else 0.99,
            x_B_setpoint=0.01,
            F=p.nominal_feed_F_kmol_per_min,
            zF=0.5,
            qF=p.nominal_feed_liquid_fraction_qF,
        )

    phase1_top, phase1_bottom = build_skogestad_phase1_pids(LT_initial=L0, VB_initial=V0)
    c0_top, c0_bottom = build_c0_pids(LT_initial=L0, VB_initial=V0)

    sim_phase1 = simulate_lv_closed_loop(
        X0=skogestad_reference_state,
        scenario=yD_step,
        duration_min=60.0,
        tick_dt_min=0.05,
        pid_top=phase1_top,
        pid_bottom=phase1_bottom,
    )
    sim_c0 = simulate_lv_closed_loop(
        X0=skogestad_reference_state,
        scenario=yD_step,
        duration_min=60.0,
        tick_dt_min=0.05,
        pid_top=c0_top,
        pid_bottom=c0_bottom,
    )
    assert sim_phase1.success and sim_c0.success
    # Setpoint = 0.995. Distance to setpoint at end.
    gap_phase1 = abs(sim_phase1.y_D[-1] - 0.995)
    gap_c0 = abs(sim_c0.y_D[-1] - 0.995)
    assert gap_c0 < gap_phase1, (
        f"C0 should close the gap faster: gap_C0={gap_c0:.5f} vs gap_phase1={gap_phase1:.5f}"
    )
    # Compositions must remain physically valid.
    assert np.all(sim_c0.y_D >= 0.0) and np.all(sim_c0.y_D <= 1.0)
    assert np.all(sim_c0.x_B >= 0.0) and np.all(sim_c0.x_B <= 1.0)


def test_c0_closed_loop_stays_at_steady_state_under_nominal_inputs(
    skogestad_reference_state: npt.NDArray[np.float64],
) -> None:
    """At nominal SS with nominal setpoints, the C0 closed loop must hold for an hour."""
    p = DEFAULT_PARAMETERS
    L0 = p.nominal_reflux_L0_kmol_per_min
    V0 = p.nominal_boilup_V0_kmol_per_min

    def nominal_scenario(t: float) -> ScenarioStep:
        return ScenarioStep(
            y_D_setpoint=0.99,
            x_B_setpoint=0.01,
            F=p.nominal_feed_F_kmol_per_min,
            zF=0.5,
            qF=p.nominal_feed_liquid_fraction_qF,
        )

    top, bottom = build_c0_pids(LT_initial=L0, VB_initial=V0)
    sim = simulate_lv_closed_loop(
        X0=skogestad_reference_state,
        scenario=nominal_scenario,
        duration_min=60.0,
        tick_dt_min=0.05,
        pid_top=top,
        pid_bottom=bottom,
    )
    assert sim.success
    # State should barely drift from the published SS.
    drift = float(np.max(np.abs(sim.X[-1] - skogestad_reference_state)))
    assert drift < 5e-4, f"C0 nominal closed-loop drift {drift:.3e} too large"
