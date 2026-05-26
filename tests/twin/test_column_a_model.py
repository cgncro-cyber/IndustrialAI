"""Tests for the Column A nonlinear ODE and steady-state initialization.

These tests anchor the port against the published Skogestad reference
in two layers:

1. **Dimensional / smoke contracts.** State vectors are length 2 * NT,
   input vectors are length 7, the RHS preserves both, all
   compositions stay in [0, 1] at the reference state, holdups are
   strictly positive.
2. **Numerical validation against Skogestad cola_init.mat.** Both
   steady-state paths (long-time integration and Newton-Krylov) must
   reproduce the published reference state. The reference data lives in
   ``data/reference/skogestad_column_a_steady_state.json`` (extracted
   from Skogestad's published MATLAB output file, NTNU URL recorded in
   the JSON).
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import pytest

from industrial_ai.twin.column_a import (
    DEFAULT_PARAMETERS,
    column_a_rhs,
    compute_steady_state_by_integration,
    compute_steady_state_by_newton,
)
from industrial_ai.twin.column_a.steady_state import (
    flat_initial_state,
    nominal_inputs,
)

# Tolerances. The long-time integration path reproduces the published
# reference to machine precision; the Newton-Krylov path uses an inexact-
# Newton scheme that converges to its own internal f_tol (much tighter
# than ±1 %), but the resulting state may differ from the published
# reference by a small numerical offset. PROJECT_PLAN Phase 1 gate
# tolerates ±1 % on steady-state values.
INTEGRATION_MAX_ABS_DIFF = 1e-10
NEWTON_MAX_ABS_DIFF = 1e-2  # ±1 % per PROJECT_PLAN Phase 1 gate
STEADY_STATE_RESIDUAL_TOL = 1e-10
COMPOSITION_ONE_PERCENT_TOL = 1e-2


# ---------------------------------------------------------------------------
# Dimensional / smoke contracts.
# ---------------------------------------------------------------------------


def test_default_parameters_match_skogestad_column_a() -> None:
    p = DEFAULT_PARAMETERS
    assert p.NT == 41
    assert p.NF == 21
    assert p.alpha == pytest.approx(1.5)
    assert p.nominal_holdup_kmol == pytest.approx(0.5)
    assert p.feed_stage_idx == 20
    assert p.n_states == 82


def test_flat_initial_state_has_correct_layout() -> None:
    X0 = flat_initial_state()
    assert X0.shape == (82,)
    assert np.all(X0[:41] == 0.5), "compositions should be flat at 0.5"
    assert np.all(X0[41:] == 0.5), "holdups should be at nominal 0.5 kmol"


def test_nominal_inputs_shape_and_balance() -> None:
    U = nominal_inputs()
    assert U.shape == (7,)
    # Overall material balance at nominal: F = D + B.
    assert U[4] == pytest.approx(U[2] + U[3], abs=1e-12)


def test_rhs_preserves_dimensions() -> None:
    X0 = flat_initial_state()
    U = nominal_inputs()
    dXdt = column_a_rhs(0.0, X0, U)
    assert dXdt.shape == X0.shape
    assert dXdt.dtype == np.float64


# ---------------------------------------------------------------------------
# Numerical validation vs Skogestad reference.
# ---------------------------------------------------------------------------


def test_steady_state_integration_matches_skogestad_reference(
    skogestad_reference_state: npt.NDArray[np.float64],
) -> None:
    """Long-time integration converges to Skogestad's published cola_init state."""
    result = compute_steady_state_by_integration(
        residual_tol=STEADY_STATE_RESIDUAL_TOL,
    )
    assert result.success, f"integration did not converge: {result.message}"
    max_abs_diff = float(np.max(np.abs(result.X - skogestad_reference_state)))
    assert max_abs_diff < INTEGRATION_MAX_ABS_DIFF, (
        f"max |our - skogestad| = {max_abs_diff:.3e} exceeds tolerance "
        f"{INTEGRATION_MAX_ABS_DIFF:.1e}"
    )


def test_steady_state_newton_matches_skogestad_reference(
    skogestad_reference_state: npt.NDArray[np.float64],
) -> None:
    """Newton-Krylov from a slightly perturbed reference returns to the SS."""
    rng = np.random.default_rng(seed=0)
    X_guess = skogestad_reference_state * (1.0 + 1e-3 * rng.standard_normal(82))
    # Holdups must stay strictly positive; clamp to be safe.
    X_guess[41:] = np.clip(X_guess[41:], 0.1, None)

    result = compute_steady_state_by_newton(
        X0=X_guess,
        residual_tol=STEADY_STATE_RESIDUAL_TOL,
    )
    assert result.success, f"newton did not converge: {result.message}"
    max_abs_diff = float(np.max(np.abs(result.X - skogestad_reference_state)))
    assert max_abs_diff < NEWTON_MAX_ABS_DIFF, (
        f"max |our - skogestad| = {max_abs_diff:.3e} exceeds tolerance {NEWTON_MAX_ABS_DIFF:.1e}"
    )


def test_skogestad_reference_satisfies_rhs(
    skogestad_reference_state: npt.NDArray[np.float64],
    skogestad_reference_inputs: npt.NDArray[np.float64],
) -> None:
    """Sanity: the published reference state is a true zero of the RHS."""
    dXdt = column_a_rhs(
        0.0,
        skogestad_reference_state,
        skogestad_reference_inputs,
    )
    residual_norm = float(np.linalg.norm(dXdt, ord=np.inf))
    assert residual_norm < STEADY_STATE_RESIDUAL_TOL, (
        f"|f(X*, U*)|_inf = {residual_norm:.3e} exceeds tolerance {STEADY_STATE_RESIDUAL_TOL:.1e}"
    )


# ---------------------------------------------------------------------------
# Physical-plausibility checks on the steady state.
# ---------------------------------------------------------------------------


def test_steady_state_compositions_monotonically_increase(
    skogestad_reference_state: npt.NDArray[np.float64],
) -> None:
    """Light component is enriched towards the top — strictly monotonic."""
    compositions = skogestad_reference_state[:41]
    diffs = np.diff(compositions)
    assert np.all(diffs > 0), (
        "stage compositions must increase monotonically from reboiler to "
        f"condenser; min diff = {diffs.min():.3e}"
    )


def test_steady_state_compositions_in_unit_interval(
    skogestad_reference_state: npt.NDArray[np.float64],
) -> None:
    """Mole fractions stay in [0, 1] throughout the column."""
    compositions = skogestad_reference_state[:41]
    assert compositions.min() >= 0.0
    assert compositions.max() <= 1.0


def test_steady_state_holdups_match_nominal(
    skogestad_reference_state: npt.NDArray[np.float64],
) -> None:
    """At the canonical steady state, all stage holdups sit at the nominal 0.5 kmol."""
    holdups = skogestad_reference_state[41:]
    assert np.allclose(holdups, DEFAULT_PARAMETERS.nominal_holdup_kmol, atol=1e-12)


def test_steady_state_meets_skogestad_purity_specs(
    skogestad_reference_state: npt.NDArray[np.float64],
) -> None:
    """Reboiler xB and condenser yD match the canonical 99 % purity targets."""
    xB = skogestad_reference_state[0]
    yD = skogestad_reference_state[40]
    assert xB == pytest.approx(0.01, abs=COMPOSITION_ONE_PERCENT_TOL)
    assert yD == pytest.approx(0.99, abs=COMPOSITION_ONE_PERCENT_TOL)
