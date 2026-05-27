"""Parity tests: ``linearize_lv(backend="casadi")`` vs the finite-difference default.

The CasADi backend must (a) produce the same numerical Jacobians as
the central-difference backend to a tight tolerance and (b) keep
passing the Skogestad 1997 published-invariant checks that the
Phase 1 mini-gate uses. Both are the contract the Phase 2 Linear MPC
baseline (``do-mpc``) will rely on.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import pytest

from industrial_ai.twin.column_a import DEFAULT_PARAMETERS
from industrial_ai.twin.column_a.linearize import (
    LinearizedLVModel,
    dominant_time_constants_min,
    linearize_lv,
    steady_state_gain,
)

_TAU_1_PUBLISHED_MIN = 193.9
_TAU_2_PUBLISHED_MIN = 12.0
_TAU_3_PUBLISHED_MIN = 3.5
_G_LV_PUBLISHED = np.array(
    [
        [0.8754, -0.8618],
        [1.0846, -1.0982],
    ],
    dtype=np.float64,
)


def _linearize(backend: str, X_ss: npt.NDArray[np.float64]) -> LinearizedLVModel:
    p = DEFAULT_PARAMETERS
    return linearize_lv(
        X_ss=X_ss,
        L_ss=p.nominal_reflux_L0_kmol_per_min,
        V_ss=p.nominal_boilup_V0_kmol_per_min,
        F_ss=p.nominal_feed_F_kmol_per_min,
        zF_ss=0.5,
        backend=backend,
    )


def test_casadi_and_finite_difference_jacobians_agree(
    skogestad_reference_state: npt.NDArray[np.float64],
) -> None:
    """A_fd ~= A_cs and B_fd ~= B_cs at the published SS to 1e-5."""
    fd = _linearize("finite_difference", skogestad_reference_state)
    cs = _linearize("casadi", skogestad_reference_state)
    np.testing.assert_allclose(cs.A, fd.A, atol=1e-5, rtol=1e-5)
    np.testing.assert_allclose(cs.B, fd.B, atol=1e-5, rtol=1e-5)


def test_casadi_backend_passes_skogestad_1997_eq_31(
    skogestad_reference_state: npt.NDArray[np.float64],
) -> None:
    """G^LV(0) under the CasADi backend stays within ±5 % of the published values."""
    cs = _linearize("casadi", skogestad_reference_state)
    G_LV = steady_state_gain(cs)[:, :2]
    rel_err = np.abs(G_LV - _G_LV_PUBLISHED) / np.abs(_G_LV_PUBLISHED)
    assert np.max(rel_err) < 0.05, (
        f"casadi G^LV(0) deviates from Skogestad 1997 Eq. (31) by up to "
        f"{np.max(rel_err) * 100:.2f} %"
    )


def test_casadi_backend_passes_skogestad_1997_section_4_4(
    skogestad_reference_state: npt.NDArray[np.float64],
) -> None:
    """tau_1, tau_2, tau_3 under the CasADi backend stay within ±2 % of the published values."""
    cs = _linearize("casadi", skogestad_reference_state)
    taus = dominant_time_constants_min(cs, n=3)
    rel_errs = [
        abs(taus[0] - _TAU_1_PUBLISHED_MIN) / _TAU_1_PUBLISHED_MIN,
        abs(taus[1] - _TAU_2_PUBLISHED_MIN) / _TAU_2_PUBLISHED_MIN,
        abs(taus[2] - _TAU_3_PUBLISHED_MIN) / _TAU_3_PUBLISHED_MIN,
    ]
    assert max(rel_errs) < 0.02, (
        f"casadi time constants {taus} deviate up to {max(rel_errs) * 100:.2f} % from "
        f"Skogestad 1997 (193.9, 12.0, 3.5 min)"
    )


def test_unknown_backend_raises(
    skogestad_reference_state: npt.NDArray[np.float64],
) -> None:
    """Bad backend names fail fast with a clear message."""
    with pytest.raises(ValueError, match="unknown backend"):
        linearize_lv(
            X_ss=skogestad_reference_state,
            L_ss=2.7,
            V_ss=3.2,
            F_ss=1.0,
            zF_ss=0.5,
            backend="not_a_backend",  # type: ignore[arg-type]
        )
