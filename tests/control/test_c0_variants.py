"""Tests for the C0 variant builder + PID construction."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import pytest

from industrial_ai.control.c0_variants import (
    C0Variant,
    build_pids_for_variant,
    build_six_variants,
)
from industrial_ai.control.relay_tuning import relay_test
from industrial_ai.twin.column_a import DEFAULT_PARAMETERS
from industrial_ai.twin.column_a.linearize import linearize_lv


@pytest.fixture(scope="module")
def shootout_inputs(
    skogestad_reference_state: npt.NDArray[np.float64],
) -> tuple:
    """Linearization + relay tests at the published nominal SS.

    Scoped to *module* so the expensive relay tests run once per file.
    """
    p = DEFAULT_PARAMETERS
    lin = linearize_lv(
        X_ss=skogestad_reference_state,
        L_ss=p.nominal_reflux_L0_kmol_per_min,
        V_ss=p.nominal_boilup_V0_kmol_per_min,
        F_ss=p.nominal_feed_F_kmol_per_min,
        zF_ss=0.5,
        backend="casadi",
    )
    rt = relay_test(
        loop="top",
        X0=skogestad_reference_state,
        setpoint=float(skogestad_reference_state[p.NT - 1]),
        relay_amplitude_d=0.5,
        hysteresis=5e-3,
        duration_min=400.0,
    )
    rb = relay_test(
        loop="bottom",
        X0=skogestad_reference_state,
        setpoint=float(skogestad_reference_state[0]),
        relay_amplitude_d=0.5,
        hysteresis=5e-3,
        duration_min=400.0,
    )
    return lin, rt, rb


def test_six_variants_are_built(shootout_inputs: tuple) -> None:
    """build_six_variants returns exactly six distinct candidates."""
    lin, rt, rb = shootout_inputs
    variants = build_six_variants(linearized=lin, relay_top=rt, relay_bottom=rb)
    assert len(variants) == 6
    names = {v.name for v in variants}
    assert len(names) == 6


def test_three_tuning_methods_each_with_and_without_decoupler(
    shootout_inputs: tuple,
) -> None:
    """The matrix is {TL, SIMC-1DoF, SIMC-2DoF} x {no decoupler, with decoupler}."""
    lin, rt, rb = shootout_inputs
    variants = build_six_variants(linearized=lin, relay_top=rt, relay_bottom=rb)
    methods = {v.tuning_method for v in variants}
    assert methods == {"Tyreus-Luyben", "SIMC-1DoF", "SIMC-2DoF"}
    no_dec = [v for v in variants if v.decoupler.rga_11 == 1.0]
    with_dec = [v for v in variants if v.decoupler.rga_11 != 1.0]
    assert len(no_dec) == 3
    assert len(with_dec) == 3


def test_2dof_variants_carry_a_setpoint_filter(shootout_inputs: tuple) -> None:
    """Only the SIMC-2DoF variants must specify a setpoint-filter time constant."""
    lin, rt, rb = shootout_inputs
    variants = build_six_variants(linearized=lin, relay_top=rt, relay_bottom=rb)
    for v in variants:
        if v.tuning_method == "SIMC-2DoF":
            assert v.setpoint_filter_tau_min is not None
            assert v.setpoint_filter_tau_min > 0.0
        else:
            assert v.setpoint_filter_tau_min is None


def test_decoupled_simc_has_larger_kp_than_undecoupled(
    shootout_inputs: tuple,
) -> None:
    """SIMC-with-decoupler must compensate for the shrunk effective gain."""
    lin, rt, rb = shootout_inputs
    variants = build_six_variants(linearized=lin, relay_top=rt, relay_bottom=rb)
    by_name = {v.name: v for v in variants}
    simc_no_dec = by_name["SIMC_1DoF_no_decoupler"]
    simc_with_dec = by_name["SIMC_1DoF_with_decoupler"]
    # Effective gain is ~ 36x smaller, so Kp must be ~ 36x larger to
    # achieve the same closed-loop bandwidth.
    assert simc_with_dec.Kp_top > 10.0 * simc_no_dec.Kp_top
    assert simc_with_dec.Kp_bottom > 10.0 * simc_no_dec.Kp_bottom


def test_pid_builder_directions_match_loop_physics(shootout_inputs: tuple) -> None:
    """Top loop is direct-acting; bottom loop is reverse-acting, on every variant."""
    lin, rt, rb = shootout_inputs
    variants = build_six_variants(linearized=lin, relay_top=rt, relay_bottom=rb)
    p = DEFAULT_PARAMETERS
    for v in variants:
        top, bottom = build_pids_for_variant(
            v,
            LT_initial=p.nominal_reflux_L0_kmol_per_min,
            VB_initial=p.nominal_boilup_V0_kmol_per_min,
        )
        assert top.direct_acting is True
        assert bottom.direct_acting is False


def test_pid_builder_seeds_integrals_for_bias(shootout_inputs: tuple) -> None:
    """At zero error, the built PID outputs the bias MV — the seeded-integral contract."""
    lin, rt, rb = shootout_inputs
    variants = build_six_variants(linearized=lin, relay_top=rt, relay_bottom=rb)
    p = DEFAULT_PARAMETERS
    L0 = p.nominal_reflux_L0_kmol_per_min
    V0 = p.nominal_boilup_V0_kmol_per_min
    for v in variants:
        top, bottom = build_pids_for_variant(v, LT_initial=L0, VB_initial=V0)
        u_top = top.step(measurement=0.5, setpoint=0.5, dt=0.05)
        u_bottom = bottom.step(measurement=0.5, setpoint=0.5, dt=0.05)
        assert u_top == pytest.approx(L0, abs=1e-9), v.name
        assert u_bottom == pytest.approx(V0, abs=1e-9), v.name


def test_variant_is_json_serializable(shootout_inputs: tuple) -> None:
    """to_serializable() returns dict-compatible content (round-trip via JSON)."""
    import json

    lin, rt, rb = shootout_inputs
    variants = build_six_variants(linearized=lin, relay_top=rt, relay_bottom=rb)
    blob = json.dumps([v.to_serializable() for v in variants])
    back = json.loads(blob)
    assert len(back) == 6
    assert all("tuning_method" in d for d in back)


def test_variant_dataclass_is_frozen() -> None:
    """C0Variant is immutable (frozen dataclass)."""
    from industrial_ai.control.decoupler import identity_decoupler

    v = C0Variant(
        name="probe",
        tuning_method="SIMC-1DoF",
        Kp_top=1.0,
        Ti_top_min=10.0,
        Kp_bottom=1.0,
        Ti_bottom_min=10.0,
        decoupler=identity_decoupler(),
        setpoint_filter_tau_min=None,
        reference="test",
    )
    with pytest.raises(Exception):  # noqa: B017 — frozen dataclasses raise dataclasses.FrozenInstanceError
        v.name = "other"  # type: ignore[misc]
