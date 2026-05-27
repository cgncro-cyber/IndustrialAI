"""Tests for the SIMC PI tuning rules."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import pytest

from industrial_ai.control.simc import (
    SIMCTuning,
    simc_pi_1dof,
    simc_pi_2dof,
    simc_tunings_from_linearization,
)
from industrial_ai.twin.column_a import DEFAULT_PARAMETERS
from industrial_ai.twin.column_a.linearize import linearize_lv


def test_simc_1dof_formula_no_deadtime() -> None:
    """Without deadtime, Kp = tau / (|k| * tau_c), Ti = min(tau, 4 tau_c)."""
    t = simc_pi_1dof(plant_gain=0.875, plant_tau=194.0, tau_c=12.0)
    assert t.Kp == pytest.approx(194.0 / (0.875 * 12.0), rel=1e-12)
    assert t.Ti == pytest.approx(min(194.0, 4.0 * 12.0), rel=1e-12)
    assert t.method == "SIMC-1DoF"


def test_simc_2dof_carries_same_pi_and_extra_tau_c() -> None:
    """The 2DoF variant has the same PI as 1DoF; the filter time constant
    comes from tau_c and is recorded on the dataclass.
    """
    t = simc_pi_2dof(plant_gain=0.875, plant_tau=194.0, tau_c=12.0)
    sibling = simc_pi_1dof(plant_gain=0.875, plant_tau=194.0, tau_c=12.0)
    assert t.Kp == sibling.Kp
    assert t.Ti == sibling.Ti
    assert t.method == "SIMC-2DoF"
    assert t.tau_c == 12.0


def test_simc_handles_negative_gain_via_absolute_value() -> None:
    """Reverse-acting plant (g < 0) yields the same |Kp| as +|g|."""
    t_pos = simc_pi_1dof(plant_gain=1.10, plant_tau=194.0, tau_c=12.0)
    t_neg = simc_pi_1dof(plant_gain=-1.10, plant_tau=194.0, tau_c=12.0)
    assert t_pos.Kp == pytest.approx(t_neg.Kp, rel=1e-12)


def test_simc_rejects_zero_gain_and_nonpositive_taus() -> None:
    """Invalid inputs fail fast."""
    with pytest.raises(ValueError, match="non-zero"):
        simc_pi_1dof(plant_gain=0.0, plant_tau=194.0, tau_c=12.0)
    with pytest.raises(ValueError, match="strictly positive"):
        simc_pi_1dof(plant_gain=1.0, plant_tau=0.0, tau_c=12.0)
    with pytest.raises(ValueError, match="strictly positive"):
        simc_pi_1dof(plant_gain=1.0, plant_tau=194.0, tau_c=0.0)


def test_simc_tunings_from_linearization_uses_diagonal_gains(
    skogestad_reference_state: npt.NDArray[np.float64],
) -> None:
    """The convenience reads G(0)[0,0] and G(0)[1,1] from a real linearization."""
    p = DEFAULT_PARAMETERS
    lin = linearize_lv(
        X_ss=skogestad_reference_state,
        L_ss=p.nominal_reflux_L0_kmol_per_min,
        V_ss=p.nominal_boilup_V0_kmol_per_min,
        F_ss=p.nominal_feed_F_kmol_per_min,
        zF_ss=0.5,
        backend="casadi",
    )
    top, bottom = simc_tunings_from_linearization(lin, variant="1dof")
    assert isinstance(top, SIMCTuning)
    assert isinstance(bottom, SIMCTuning)
    # Skogestad 1997 Eq. 31 puts g_top ~ 0.87 and g_bottom ~ -1.10.
    assert top.plant_gain == pytest.approx(0.875, rel=0.02)
    assert bottom.plant_gain == pytest.approx(-1.098, rel=0.02)
    # tau_1 ~ 194 min per Skogestad 1997 Section 4.4.
    assert top.plant_tau == pytest.approx(193.9, rel=0.02)


def test_effective_gain_override_changes_kp_by_the_rga_factor(
    skogestad_reference_state: npt.NDArray[np.float64],
) -> None:
    """Passing effective_gain_diag re-derives Kp without re-computing G(0)."""
    p = DEFAULT_PARAMETERS
    lin = linearize_lv(
        X_ss=skogestad_reference_state,
        L_ss=p.nominal_reflux_L0_kmol_per_min,
        V_ss=p.nominal_boilup_V0_kmol_per_min,
        F_ss=p.nominal_feed_F_kmol_per_min,
        zF_ss=0.5,
        backend="casadi",
    )
    base_top, _ = simc_tunings_from_linearization(lin, variant="1dof")
    shrunk = (base_top.plant_gain / 36.0, -base_top.plant_gain / 36.0)
    rt, _ = simc_tunings_from_linearization(lin, variant="1dof", effective_gain_diag=shrunk)
    # |Kp| scales inversely with |g|. Sign of g doesn't matter (uses |g|).
    expected = base_top.Kp * 36.0
    assert rt.Kp == pytest.approx(expected, rel=1e-9)


def test_unknown_variant_raises() -> None:
    """Bad variant name fails fast."""
    p = DEFAULT_PARAMETERS
    # Doesn't need a real linearization — the ValueError fires before any
    # plant access.
    fake = type(
        "_FakeLin",
        (),
        {"A": np.zeros((2, 2)), "B": np.zeros((2, 2)), "C": np.zeros((2, 2))},
    )()
    with pytest.raises(ValueError, match="variant"):
        simc_tunings_from_linearization(fake, variant="3dof")  # type: ignore[arg-type]
    del p
