"""Tests for the LV configuration."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import pytest

from industrial_ai.twin.column_a import DEFAULT_PARAMETERS
from industrial_ai.twin.column_a.configurations.lv import (
    LVConfiguration,
    assemble_inputs_lv,
)


def test_input_assembly_at_nominal_state(
    skogestad_reference_state: npt.NDArray[np.float64],
) -> None:
    """At the published steady state, the LV closure reproduces nominal D and B."""
    U = assemble_inputs_lv(
        state=skogestad_reference_state,
        LT=DEFAULT_PARAMETERS.nominal_reflux_L0_kmol_per_min,
        VB=DEFAULT_PARAMETERS.nominal_boilup_V0_kmol_per_min,
        F=DEFAULT_PARAMETERS.nominal_feed_F_kmol_per_min,
        zF=0.5,
        qF=DEFAULT_PARAMETERS.nominal_feed_liquid_fraction_qF,
    )
    # At the nominal SS, MD == MDs and MB == MBs, so D = Ds and B = Bs.
    assert U.shape == (7,)
    assert U[2] == pytest.approx(0.5, abs=1e-12), "distillate D should fall back to Ds"
    assert U[3] == pytest.approx(0.5, abs=1e-12), "bottoms B should fall back to Bs"


def test_holdup_perturbation_increases_drawoff() -> None:
    """A condenser holdup above setpoint must increase distillate (P-controller)."""
    p = DEFAULT_PARAMETERS
    state = np.empty(p.n_states, dtype=np.float64)
    state[: p.NT] = 0.5
    state[p.NT :] = p.nominal_holdup_kmol
    # Bump the condenser holdup by +0.1 kmol.
    state[2 * p.NT - 1] += 0.1
    U = assemble_inputs_lv(state=state, LT=2.7, VB=3.2, F=1.0, zF=0.5, qF=1.0)
    cfg = LVConfiguration()
    expected_D = cfg.Ds + 0.1 * cfg.Kc_D
    assert U[2] == pytest.approx(expected_D, abs=1e-12)


def test_custom_gains_apply() -> None:
    """A custom LVConfiguration replaces the default gains."""
    p = DEFAULT_PARAMETERS
    state = np.empty(p.n_states, dtype=np.float64)
    state[: p.NT] = 0.5
    state[p.NT :] = p.nominal_holdup_kmol
    state[p.NT] += 0.05  # bump reboiler holdup
    config = LVConfiguration(Kc_B=20.0)
    U = assemble_inputs_lv(state=state, LT=2.7, VB=3.2, F=1.0, zF=0.5, qF=1.0, config=config)
    assert U[3] == pytest.approx(0.5 + 0.05 * 20.0, abs=1e-12)
