"""Time integration of the Column A nonlinear ODE.

Thin wrapper around :func:`scipy.integrate.solve_ivp` configured for
stiff ODE integration. The wrapper exists so that downstream code
(steady-state initialization, LV/DV/L/D-V/B configurations, regulatory
PID layer, supervisory loops) shares a single point of solver
parameterization.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

import numpy as np
import numpy.typing as npt
from scipy.integrate import solve_ivp

from industrial_ai.twin.column_a.model import column_a_rhs
from industrial_ai.twin.column_a.parameters import (
    DEFAULT_PARAMETERS,
    ColumnAParameters,
)

__all__ = ["InputFunction", "IntegrationResult", "integrate_open_loop"]

StateVector = npt.NDArray[np.float64]
InputVector = npt.NDArray[np.float64]

#: Callable signature for time-varying inputs and disturbances. The
#: function is invoked at every solver evaluation point with the
#: current time and state and must return the seven-element input
#: vector ``U`` expected by :func:`column_a_rhs`.
InputFunction = Callable[[float, StateVector], InputVector]


@dataclass(frozen=True, slots=True)
class IntegrationResult:
    """Result of a Column A integration.

    Attributes
    ----------
    t : numpy.ndarray of shape (n_timesteps,)
        Time grid (min).
    X : numpy.ndarray of shape (n_timesteps, n_states)
        State trajectory; one row per timestep. Column layout matches
        :mod:`industrial_ai.twin.column_a.model` (compositions followed
        by holdups).
    success : bool
        ``True`` if the solver reported successful termination.
    message : str
        Solver status message.
    """

    t: npt.NDArray[np.float64]
    X: npt.NDArray[np.float64]
    success: bool
    message: str


def integrate_open_loop(
    *,
    X0: StateVector,
    t_span: tuple[float, float],
    inputs_fn: InputFunction,
    parameters: ColumnAParameters = DEFAULT_PARAMETERS,
    method: Literal["LSODA", "Radau", "BDF"] = "LSODA",
    rtol: float = 1e-8,
    atol: float = 1e-10,
    max_step: float | None = None,
    t_eval: npt.NDArray[np.float64] | None = None,
) -> IntegrationResult:
    """Integrate the Column A ODE forward in time.

    Parameters
    ----------
    X0 : numpy.ndarray of shape (2 * NT,)
        Initial state vector.
    t_span : tuple of (float, float)
        Start and stop time (min).
    inputs_fn : InputFunction
        Function mapping ``(t, X) -> U``. Closures over this argument
        implement open-loop runs, level-loop closures (LV/DV/L/D-V/B),
        and regulatory-PID layering without modifying this integrator.
    parameters : ColumnAParameters, optional
        Column specification.
    method : {"LSODA", "Radau", "BDF"}, optional
        SciPy solver name. LSODA (default) handles automatic
        stiff/non-stiff switching; Radau and BDF are fully implicit
        fallbacks for difficult cases.
    rtol, atol : float, optional
        Relative and absolute tolerances.
    max_step : float, optional
        Maximum permitted internal step size (min). Useful when input
        trajectories contain abrupt step changes the adaptive stepper
        might otherwise overshoot.
    t_eval : numpy.ndarray, optional
        Specific time points at which to return solution values.

    Returns
    -------
    IntegrationResult
    """

    def rhs(t: float, X: StateVector) -> StateVector:
        U = inputs_fn(t, X)
        return column_a_rhs(t, X, U, parameters)

    kwargs: dict[str, object] = {
        "method": method,
        "rtol": rtol,
        "atol": atol,
        "dense_output": False,
    }
    if max_step is not None:
        kwargs["max_step"] = max_step
    if t_eval is not None:
        kwargs["t_eval"] = t_eval

    sol = solve_ivp(rhs, t_span, X0, **kwargs)

    return IntegrationResult(
        t=np.asarray(sol.t, dtype=np.float64),
        X=np.asarray(sol.y, dtype=np.float64).T,
        success=bool(sol.success),
        message=str(sol.message),
    )
