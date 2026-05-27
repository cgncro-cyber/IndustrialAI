"""Tests for the L/D-V/B (double-ratio) configuration."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import pytest

from industrial_ai.twin.column_a import (
    DEFAULT_PARAMETERS,
    integrate_open_loop,
)
from industrial_ai.twin.column_a.configurations.ldvb import (
    LDVBConfiguration,
    assemble_inputs_ldvb,
    nominal_ratios,
)
from industrial_ai.twin.column_a.configurations.lv import assemble_inputs_lv


def test_nominal_ratios_match_published_ss() -> None:
    """nominal_ratios must reproduce LR=L0/Ds and VR=V0/Bs."""
    LR, VR = nominal_ratios()
    expected_LR = DEFAULT_PARAMETERS.nominal_reflux_L0_kmol_per_min / 0.5
    expected_VR = DEFAULT_PARAMETERS.nominal_boilup_V0_kmol_per_min / 0.5
    np.testing.assert_allclose(LR, expected_LR, atol=1e-12)
    np.testing.assert_allclose(VR, expected_VR, atol=1e-12)


def test_input_assembly_at_nominal_state(
    skogestad_reference_state: npt.NDArray[np.float64],
) -> None:
    """At the published SS, the LDVB closure reproduces nominal LT, VB, D and B."""
    LR, VR = nominal_ratios()
    U = assemble_inputs_ldvb(
        state=skogestad_reference_state,
        LR=LR,
        VR=VR,
        F=DEFAULT_PARAMETERS.nominal_feed_F_kmol_per_min,
        zF=0.5,
        qF=DEFAULT_PARAMETERS.nominal_feed_liquid_fraction_qF,
    )
    assert U.shape == (7,)
    assert U[0] == pytest.approx(DEFAULT_PARAMETERS.nominal_reflux_L0_kmol_per_min, abs=1e-12), (
        "reflux LT should equal nominal L0"
    )
    assert U[1] == pytest.approx(DEFAULT_PARAMETERS.nominal_boilup_V0_kmol_per_min, abs=1e-12), (
        "boilup VB should equal nominal V0"
    )
    assert U[2] == pytest.approx(0.5, abs=1e-12), "distillate D should fall back to Ds"
    assert U[3] == pytest.approx(0.5, abs=1e-12), "bottoms B should fall back to Bs"


def test_ldvb_matches_lv_at_nominal_steady_state(
    skogestad_reference_state: npt.NDArray[np.float64],
) -> None:
    """LDVB and LV closures must produce identical U at the published SS.

    Both configurations reduce to the same nominal operating point when
    fed their nominal supervisor commands — this is the cross-consistency
    check that guards against unit/ratio bugs in the LDVB mapping.
    """
    LR, VR = nominal_ratios()
    U_ldvb = assemble_inputs_ldvb(
        state=skogestad_reference_state,
        LR=LR,
        VR=VR,
        F=DEFAULT_PARAMETERS.nominal_feed_F_kmol_per_min,
        zF=0.5,
        qF=DEFAULT_PARAMETERS.nominal_feed_liquid_fraction_qF,
    )
    U_lv = assemble_inputs_lv(
        state=skogestad_reference_state,
        LT=DEFAULT_PARAMETERS.nominal_reflux_L0_kmol_per_min,
        VB=DEFAULT_PARAMETERS.nominal_boilup_V0_kmol_per_min,
        F=DEFAULT_PARAMETERS.nominal_feed_F_kmol_per_min,
        zF=0.5,
        qF=DEFAULT_PARAMETERS.nominal_feed_liquid_fraction_qF,
    )
    np.testing.assert_allclose(U_ldvb, U_lv, atol=1e-12)


def test_condenser_holdup_perturbation_propagates_into_reflux() -> None:
    """Bumping MD must increase D (P-controller) AND scale LT through LR."""
    p = DEFAULT_PARAMETERS
    state = np.empty(p.n_states, dtype=np.float64)
    state[: p.NT] = 0.5
    state[p.NT :] = p.nominal_holdup_kmol
    state[2 * p.NT - 1] += 0.1  # bump condenser holdup by +0.1 kmol
    LR, VR = nominal_ratios()
    U = assemble_inputs_ldvb(state=state, LR=LR, VR=VR, F=1.0, zF=0.5, qF=1.0)
    cfg = LDVBConfiguration()
    expected_D = cfg.Ds + 0.1 * cfg.Kc_D
    expected_LT = LR * expected_D
    assert U[2] == pytest.approx(expected_D, abs=1e-12)
    assert U[0] == pytest.approx(expected_LT, abs=1e-12)


def test_reboiler_holdup_perturbation_propagates_into_boilup() -> None:
    """Bumping MB must increase B (P-controller) AND scale VB through VR."""
    p = DEFAULT_PARAMETERS
    state = np.empty(p.n_states, dtype=np.float64)
    state[: p.NT] = 0.5
    state[p.NT :] = p.nominal_holdup_kmol
    state[p.NT] += 0.05  # bump reboiler holdup
    LR, VR = nominal_ratios()
    U = assemble_inputs_ldvb(state=state, LR=LR, VR=VR, F=1.0, zF=0.5, qF=1.0)
    cfg = LDVBConfiguration()
    expected_B = cfg.Bs + 0.05 * cfg.Kc_B
    expected_VB = VR * expected_B
    assert U[3] == pytest.approx(expected_B, abs=1e-12)
    assert U[1] == pytest.approx(expected_VB, abs=1e-12)


def test_custom_gains_apply() -> None:
    """A custom LDVBConfiguration replaces the default gains."""
    p = DEFAULT_PARAMETERS
    state = np.empty(p.n_states, dtype=np.float64)
    state[: p.NT] = 0.5
    state[p.NT :] = p.nominal_holdup_kmol
    state[2 * p.NT - 1] += 0.1
    cfg = LDVBConfiguration(Kc_D=20.0)
    LR, VR = nominal_ratios()
    U = assemble_inputs_ldvb(state=state, LR=LR, VR=VR, F=1.0, zF=0.5, qF=1.0, config=cfg)
    expected_D = cfg.Ds + 0.1 * 20.0
    assert U[2] == pytest.approx(expected_D, abs=1e-12)
    assert U[0] == pytest.approx(LR * expected_D, abs=1e-12)


def test_ldvb_closed_loop_stays_at_steady_state(
    skogestad_reference_state: npt.NDArray[np.float64],
) -> None:
    """Closing the LDVB loops at the published SS must not drift away."""
    p = DEFAULT_PARAMETERS
    LR, VR = nominal_ratios()

    def inputs_fn(t: float, X: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        return assemble_inputs_ldvb(
            state=X,
            LR=LR,
            VR=VR,
            F=p.nominal_feed_F_kmol_per_min,
            zF=0.5,
            qF=p.nominal_feed_liquid_fraction_qF,
        )

    result = integrate_open_loop(
        X0=skogestad_reference_state,
        t_span=(0.0, 5.0),
        inputs_fn=inputs_fn,
    )
    assert result.success, result.message
    drift = np.max(np.abs(result.X[-1] - skogestad_reference_state))
    assert drift < 1e-4, f"LDVB closed loop drifted {drift:.3e} from SS in 5 min"
