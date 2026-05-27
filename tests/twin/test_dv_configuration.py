"""Tests for the DV configuration."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import pytest

from industrial_ai.twin.column_a import (
    DEFAULT_PARAMETERS,
    integrate_open_loop,
)
from industrial_ai.twin.column_a.configurations.dv import (
    DVConfiguration,
    assemble_inputs_dv,
)


def test_input_assembly_at_nominal_state(
    skogestad_reference_state: npt.NDArray[np.float64],
) -> None:
    """At the published steady state, the DV closure reproduces nominal LT and B."""
    U = assemble_inputs_dv(
        state=skogestad_reference_state,
        D=0.5,
        VB=DEFAULT_PARAMETERS.nominal_boilup_V0_kmol_per_min,
        F=DEFAULT_PARAMETERS.nominal_feed_F_kmol_per_min,
        zF=0.5,
        qF=DEFAULT_PARAMETERS.nominal_feed_liquid_fraction_qF,
    )
    # At the nominal SS, MD == MDs and MB == MBs, so LT = Ls and B = Bs.
    assert U.shape == (7,)
    assert U[0] == pytest.approx(DEFAULT_PARAMETERS.nominal_reflux_L0_kmol_per_min, abs=1e-12), (
        "reflux LT should fall back to the nominal L0 bias"
    )
    assert U[3] == pytest.approx(0.5, abs=1e-12), "bottoms B should fall back to Bs"


def test_condenser_holdup_perturbation_increases_reflux() -> None:
    """A condenser holdup above setpoint must increase reflux (P-controller)."""
    p = DEFAULT_PARAMETERS
    state = np.empty(p.n_states, dtype=np.float64)
    state[: p.NT] = 0.5
    state[p.NT :] = p.nominal_holdup_kmol
    # Bump the condenser holdup by +0.1 kmol.
    state[2 * p.NT - 1] += 0.1
    U = assemble_inputs_dv(state=state, D=0.5, VB=3.2, F=1.0, zF=0.5, qF=1.0)
    cfg = DVConfiguration()
    expected_LT = cfg.Ls + 0.1 * cfg.Kc_L
    assert U[0] == pytest.approx(expected_LT, abs=1e-12)


def test_reboiler_holdup_perturbation_increases_bottoms() -> None:
    """A reboiler holdup above setpoint must increase bottoms drawoff."""
    p = DEFAULT_PARAMETERS
    state = np.empty(p.n_states, dtype=np.float64)
    state[: p.NT] = 0.5
    state[p.NT :] = p.nominal_holdup_kmol
    state[p.NT] += 0.05  # bump reboiler holdup
    U = assemble_inputs_dv(state=state, D=0.5, VB=3.2, F=1.0, zF=0.5, qF=1.0)
    cfg = DVConfiguration()
    expected_B = cfg.Bs + 0.05 * cfg.Kc_B
    assert U[3] == pytest.approx(expected_B, abs=1e-12)


def test_custom_gains_apply() -> None:
    """A custom DVConfiguration replaces the default gains."""
    p = DEFAULT_PARAMETERS
    state = np.empty(p.n_states, dtype=np.float64)
    state[: p.NT] = 0.5
    state[p.NT :] = p.nominal_holdup_kmol
    state[2 * p.NT - 1] += 0.1  # bump condenser holdup
    cfg = DVConfiguration(Kc_L=20.0)
    U = assemble_inputs_dv(state=state, D=0.5, VB=3.2, F=1.0, zF=0.5, qF=1.0, config=cfg)
    expected_LT = cfg.Ls + 0.1 * 20.0
    assert U[0] == pytest.approx(expected_LT, abs=1e-12)


def test_dv_closed_loop_stays_at_steady_state(
    skogestad_reference_state: npt.NDArray[np.float64],
) -> None:
    """Closing the DV level loops at the published SS must not drift away."""
    p = DEFAULT_PARAMETERS

    def inputs_fn(t: float, X: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        return assemble_inputs_dv(
            state=X,
            D=0.5,
            VB=p.nominal_boilup_V0_kmol_per_min,
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
    assert drift < 1e-4, f"DV closed loop drifted {drift:.3e} from SS in 5 min"
