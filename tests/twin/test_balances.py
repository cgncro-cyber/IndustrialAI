"""Tests for the mass-balance closure check.

The Phase 1 gate requires that overall and light-component balances
close within 0.1 % at any converged steady state. These tests verify
the published SS and a re-converged perturbed SS against that bound.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import pytest

from industrial_ai.twin.column_a import (
    DEFAULT_PARAMETERS,
    compute_steady_state_by_newton,
)
from industrial_ai.twin.column_a.balances import (
    BalanceResiduals,
    assert_balances_close,
    check_balances,
)

_PHASE_1_GATE_TOL = 1.0e-3  # 0.1 % per PROJECT_PLAN.md Phase 1 gate.


def test_balance_closure_at_published_steady_state(
    skogestad_reference_state: npt.NDArray[np.float64],
    skogestad_reference_inputs: npt.NDArray[np.float64],
) -> None:
    """At the published Skogestad SS, both balances must close to <= 0.1 %."""
    residuals = check_balances(
        state=skogestad_reference_state,
        inputs=skogestad_reference_inputs,
    )
    assert residuals.max_abs() < _PHASE_1_GATE_TOL, (
        f"published SS violates Phase 1 gate: residuals={residuals}"
    )
    # No exception expected.
    assert_balances_close(residuals, tol=_PHASE_1_GATE_TOL)


def test_balance_closure_after_zF_perturbation(
    skogestad_reference_state: npt.NDArray[np.float64],
    skogestad_reference_inputs: npt.NDArray[np.float64],
) -> None:
    """A re-converged SS at zF = 0.6 must also close both balances to <= 0.1 %.

    This is the harder test: at a perturbed feed composition the
    P-only level loops carry a steady-state offset, so the holdups do
    not equal their setpoints. The algebraic balances must still close
    because the system is fully steady.
    """
    perturbed_inputs = skogestad_reference_inputs.copy()
    perturbed_inputs[5] = 0.6  # zF = 0.6 instead of 0.5

    result = compute_steady_state_by_newton(
        X0=skogestad_reference_state,
        inputs=perturbed_inputs,
    )
    assert result.success, result.message

    # D and B in the inputs vector are still the original 0.5 values,
    # but in a self-consistent closure under a perturbed zF they would
    # differ. The level-loop level-bias scheme (cola_lv.m) is the way
    # those flows track at SS. For the algebraic balance check we use
    # the consistent (D, B) that come out of the LV closure at the new
    # state — recompute them inline from the level-loop equations.
    NT = DEFAULT_PARAMETERS.NT
    M_B = result.X[NT]
    M_D = result.X[2 * NT - 1]
    Kc, MDs, MBs, Ds, Bs = 10.0, 0.5, 0.5, 0.5, 0.5
    consistent_inputs = perturbed_inputs.copy()
    consistent_inputs[2] = Ds + Kc * (M_D - MDs)
    consistent_inputs[3] = Bs + Kc * (M_B - MBs)

    residuals = check_balances(state=result.X, inputs=consistent_inputs)
    assert residuals.max_abs() < _PHASE_1_GATE_TOL, (
        f"perturbed SS violates Phase 1 gate: residuals={residuals}"
    )


def test_assert_balances_close_raises_on_violation() -> None:
    """Bad residuals must raise AssertionError with a diagnostic message."""
    bad = BalanceResiduals(
        overall_relative=5.0e-2,  # 5 %
        light_relative=0.0,
        y_D=0.99,
        x_B=0.01,
    )
    with pytest.raises(AssertionError, match="overall mass balance"):
        assert_balances_close(bad, tol=_PHASE_1_GATE_TOL)

    bad_light = BalanceResiduals(
        overall_relative=0.0,
        light_relative=-2.0e-2,
        y_D=0.99,
        x_B=0.01,
    )
    with pytest.raises(AssertionError, match="light-component balance"):
        assert_balances_close(bad_light, tol=_PHASE_1_GATE_TOL)


def test_max_abs_returns_larger_residual() -> None:
    """The convenience accessor returns the larger of the two relative magnitudes."""
    residuals = BalanceResiduals(
        overall_relative=-1.0e-4,
        light_relative=3.0e-4,
        y_D=0.99,
        x_B=0.01,
    )
    assert residuals.max_abs() == pytest.approx(3.0e-4, abs=1e-15)


def test_balance_check_handles_zero_flow_safely() -> None:
    """Zero feed flow returns infinity rather than dividing by zero."""
    p = DEFAULT_PARAMETERS
    state = np.empty(p.n_states, dtype=np.float64)
    state[: p.NT] = 0.5
    state[p.NT :] = p.nominal_holdup_kmol
    inputs = np.array([2.7, 3.2, 0.0, 0.0, 0.0, 0.5, 1.0], dtype=np.float64)
    residuals = check_balances(state=state, inputs=inputs)
    assert np.isinf(residuals.overall_relative)
    assert np.isinf(residuals.light_relative)
