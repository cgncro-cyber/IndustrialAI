"""Algebraic mass-balance closure checks for Column A.

At any steady state, the constant-molar-overflow assumption (see
``assumptions.md``) requires two algebraic balances to close:

- Overall material balance:    ``F == D + B``
- Light-component balance:     ``F * zF == D * y_D + B * x_B``

where ``y_D = x[NT - 1]`` is the condenser (distillate) composition
under the total-condenser assumption and ``x_B = x[0]`` is the
reboiler (bottoms) composition.

These balances are reported as *relative* residuals so the same
threshold (0.1 % per the Phase 1 gate) applies regardless of the
absolute flow magnitude. They are the cleanest end-to-end sanity check
on a converged steady state: if either balance is far from zero, the
state is not actually at steady state, or the solver settled on a
spurious fixed point.

Energy balance is implicitly satisfied by the constant-molar-overflow
assumption (Lewis approximation) and is therefore not checked
explicitly — see ``assumptions.md`` §3.11.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from industrial_ai.twin.column_a.parameters import (
    DEFAULT_PARAMETERS,
    ColumnAParameters,
)

__all__ = ["BalanceResiduals", "assert_balances_close", "check_balances"]

StateVector = npt.NDArray[np.float64]
InputVector = npt.NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class BalanceResiduals:
    """Relative residuals of the two steady-state material balances.

    Attributes
    ----------
    overall_relative : float
        ``(F - D - B) / F``. Zero at a converged SS.
    light_relative : float
        ``(F * zF - D * y_D - B * x_B) / (F * zF)``. Zero at a
        converged SS, where ``y_D`` and ``x_B`` are the distillate and
        bottoms compositions.
    y_D : float
        Distillate composition used in the light-component balance.
    x_B : float
        Bottoms composition used in the light-component balance.
    """

    overall_relative: float
    light_relative: float
    y_D: float
    x_B: float

    def max_abs(self) -> float:
        """Return the larger of the two absolute relative residuals."""
        return max(abs(self.overall_relative), abs(self.light_relative))


def check_balances(
    *,
    state: StateVector,
    inputs: InputVector,
    parameters: ColumnAParameters = DEFAULT_PARAMETERS,
) -> BalanceResiduals:
    """Compute the steady-state mass-balance residuals.

    Parameters
    ----------
    state : numpy.ndarray of shape (2 * NT,)
        State vector ``[x_0 ... x_{NT-1}, M_0 ... M_{NT-1}]``.
    inputs : numpy.ndarray of shape (7,)
        Input vector ``[LT, VB, D, B, F, zF, qF]``.
    parameters : ColumnAParameters, optional
        Column specification.

    Returns
    -------
    BalanceResiduals
    """
    NT = parameters.NT
    x_B = float(state[0])
    y_D = float(state[NT - 1])

    D = float(inputs[2])
    B = float(inputs[3])
    F = float(inputs[4])
    zF = float(inputs[5])

    overall = (F - D - B) / F if F != 0.0 else float("inf")
    light_lhs = F * zF
    light = (light_lhs - D * y_D - B * x_B) / light_lhs if light_lhs != 0.0 else float("inf")

    return BalanceResiduals(
        overall_relative=float(overall),
        light_relative=float(light),
        y_D=y_D,
        x_B=x_B,
    )


def assert_balances_close(
    residuals: BalanceResiduals,
    *,
    tol: float = 1.0e-3,
) -> None:
    """Raise ``AssertionError`` if either balance exceeds ``tol`` in relative magnitude.

    Parameters
    ----------
    residuals : BalanceResiduals
        Residuals returned by :func:`check_balances`.
    tol : float, optional
        Threshold for the absolute relative residual. Default 1e-3
        (0.1 %), matching the Phase 1 gate.

    Raises
    ------
    AssertionError
        If either balance is further from zero than ``tol``.
    """
    if abs(residuals.overall_relative) > tol:
        raise AssertionError(
            f"overall mass balance violates {tol:.1e}: residual {residuals.overall_relative:.3e}"
        )
    if abs(residuals.light_relative) > tol:
        raise AssertionError(
            f"light-component balance violates {tol:.1e}: residual {residuals.light_relative:.3e}"
        )
