"""Tests for the simplified RGA-aware steady-state decoupler."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import pytest

from industrial_ai.control.decoupler import (
    DecouplerSpec,
    identity_decoupler,
    rga,
    simplified_decoupler,
)
from industrial_ai.twin.column_a import DEFAULT_PARAMETERS
from industrial_ai.twin.column_a.linearize import linearize_lv


def test_identity_decoupler_is_a_noop() -> None:
    """The fallback decoupler is the 2x2 identity matrix."""
    spec = identity_decoupler()
    np.testing.assert_allclose(spec.matrix, np.eye(2), atol=1e-15)
    assert spec.rga_11 == 1.0


def test_rga_of_diagonal_matrix_is_identity() -> None:
    """RGA of a diagonal plant is the identity (loops are independent)."""
    G = np.array([[2.0, 0.0], [0.0, -3.0]])
    np.testing.assert_allclose(rga(G), np.eye(2), atol=1e-15)


def test_rga_raises_on_non_2x2() -> None:
    """Helper is 2x2 only — guard against misuse."""
    with pytest.raises(ValueError, match="2x2"):
        rga(np.eye(3))


def test_simplified_decoupler_diagonal_is_unit_off_diagonal_is_signed_ratio(
    skogestad_reference_state: npt.NDArray[np.float64],
) -> None:
    """D = [[1, -G12/G11], [-G21/G22, 1]] at the published SS."""
    p = DEFAULT_PARAMETERS
    lin = linearize_lv(
        X_ss=skogestad_reference_state,
        L_ss=p.nominal_reflux_L0_kmol_per_min,
        V_ss=p.nominal_boilup_V0_kmol_per_min,
        F_ss=p.nominal_feed_F_kmol_per_min,
        zF_ss=0.5,
        backend="casadi",
    )
    spec = simplified_decoupler(lin)
    D = spec.matrix
    assert D.shape == (2, 2)
    assert D[0, 0] == 1.0
    assert D[1, 1] == 1.0
    # Off-diagonals near +1 because G12, G22 share sign (negative) and
    # G11, G21 share sign (positive) for Column A LV.
    assert 0.9 < D[0, 1] < 1.0
    assert 0.9 < D[1, 0] < 1.0


def test_simplified_decoupler_carries_correct_rga_diagonal(
    skogestad_reference_state: npt.NDArray[np.float64],
) -> None:
    """Skogestad Column A LV has RGA(1,1) ~ 36 — the decoupler must surface this."""
    p = DEFAULT_PARAMETERS
    lin = linearize_lv(
        X_ss=skogestad_reference_state,
        L_ss=p.nominal_reflux_L0_kmol_per_min,
        V_ss=p.nominal_boilup_V0_kmol_per_min,
        F_ss=p.nominal_feed_F_kmol_per_min,
        zF_ss=0.5,
        backend="casadi",
    )
    spec = simplified_decoupler(lin)
    assert 30.0 < spec.rga_11 < 45.0, (
        f"RGA(1,1) = {spec.rga_11:.2f} is outside the expected Column A LV band"
    )


def test_effective_diagonal_gain_shrinks_by_rga_factor(
    skogestad_reference_state: npt.NDArray[np.float64],
) -> None:
    """G(0) @ D diagonal ~ g_ii / lambda_ii."""
    p = DEFAULT_PARAMETERS
    lin = linearize_lv(
        X_ss=skogestad_reference_state,
        L_ss=p.nominal_reflux_L0_kmol_per_min,
        V_ss=p.nominal_boilup_V0_kmol_per_min,
        F_ss=p.nominal_feed_F_kmol_per_min,
        zF_ss=0.5,
        backend="casadi",
    )
    spec = simplified_decoupler(lin)
    # Effective top gain ~ g11 / lambda_11 ~ 0.875 / 36 ~ 0.024.
    assert 0.01 < abs(spec.g_effective_diag[0]) < 0.05
    assert 0.01 < abs(spec.g_effective_diag[1]) < 0.05


def test_simplified_decoupler_is_correctly_typed() -> None:
    """Sanity check the dataclass wiring."""
    spec = identity_decoupler()
    assert isinstance(spec, DecouplerSpec)
    assert spec.matrix.dtype == np.float64
