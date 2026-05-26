"""Steady-state initialization for Column A.

Two complementary paths are offered:

1. :func:`compute_steady_state_by_integration` runs a long-time
   open-loop integration from a flat composition profile (``x = 0.5``
   on every stage, holdups at nominal) under the nominal inputs. After
   roughly 20 000 minutes the system has settled to its numerical
   steady state. This matches the procedure in ``cola_init.m`` from
   Skogestad's MATLAB suite.
2. :func:`compute_steady_state_by_newton` solves the algebraic
   condition ``f(X*, U) = 0`` via SciPy's Newton-Krylov. Faster than
   integration, but requires a good initial guess. Typical use is
   re-initialization at a perturbed operating point starting from a
   previously computed steady state.

Both functions return the same :class:`SteadyStateResult` container.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from scipy.optimize import NoConvergence, newton_krylov

from industrial_ai.twin.column_a.integrator import integrate_open_loop
from industrial_ai.twin.column_a.model import column_a_rhs
from industrial_ai.twin.column_a.parameters import (
    DEFAULT_PARAMETERS,
    ColumnAParameters,
)

__all__ = [
    "SteadyStateResult",
    "compute_steady_state_by_integration",
    "compute_steady_state_by_newton",
    "flat_initial_state",
    "nominal_inputs",
]

StateVector = npt.NDArray[np.float64]
InputVector = npt.NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class SteadyStateResult:
    """Container for a Column A steady-state computation.

    Attributes
    ----------
    X : numpy.ndarray of shape (2 * NT,)
        Steady-state state vector.
    U : numpy.ndarray of shape (7,)
        Input vector at which the steady state was computed.
    residual_norm : float
        Infinity norm of ``f(X*, U)``. Should be near machine epsilon
        for a converged steady state.
    method : str
        Either ``"integration"`` or ``"newton"``.
    success : bool
        ``True`` if the underlying solver reported success and the
        residual norm meets the requested tolerance.
    message : str
        Solver status message.
    """

    X: npt.NDArray[np.float64]
    U: npt.NDArray[np.float64]
    residual_norm: float
    method: str
    success: bool
    message: str


def flat_initial_state(
    parameters: ColumnAParameters = DEFAULT_PARAMETERS,
    composition: float = 0.5,
) -> StateVector:
    """Return a flat initial state — uniform composition, nominal holdups.

    Parameters
    ----------
    parameters : ColumnAParameters, optional
        Column specification.
    composition : float, optional
        Uniform composition applied to every stage. The Skogestad
        canonical initialization uses 0.5.

    Returns
    -------
    numpy.ndarray of shape (2 * NT,)
    """
    NT = parameters.NT
    X = np.empty(2 * NT, dtype=np.float64)
    X[:NT] = composition
    X[NT:] = parameters.nominal_holdup_kmol
    return X


def nominal_inputs(
    parameters: ColumnAParameters = DEFAULT_PARAMETERS,
) -> InputVector:
    """Return the seven-element nominal input vector for Column A.

    The values match Skogestad's published canonical case: balanced
    distillate and bottoms (``D = B = 0.5``), feed rate 1.0 kmol/min,
    feed composition 0.5, saturated liquid feed (``qF = 1``), and the
    published ``L0 = 2.70629`` / ``V0 = 3.20629`` reflux and boilup.
    """
    return np.array(
        [
            parameters.nominal_reflux_L0_kmol_per_min,
            parameters.nominal_boilup_V0_kmol_per_min,
            0.5,
            0.5,
            parameters.nominal_feed_F_kmol_per_min,
            0.5,
            parameters.nominal_feed_liquid_fraction_qF,
        ],
        dtype=np.float64,
    )


def compute_steady_state_by_integration(
    *,
    parameters: ColumnAParameters = DEFAULT_PARAMETERS,
    inputs: InputVector | None = None,
    X0: StateVector | None = None,
    t_end_min: float = 20_000.0,
    residual_tol: float = 1.0e-6,
    rtol: float = 1e-9,
    atol: float = 1e-11,
) -> SteadyStateResult:
    """Compute a Column A steady state via long-time integration.

    Parameters
    ----------
    parameters : ColumnAParameters, optional
        Column specification.
    inputs : numpy.ndarray of shape (7,), optional
        Input vector held constant during the integration. Defaults to
        :func:`nominal_inputs`.
    X0 : numpy.ndarray of shape (2 * NT,), optional
        Initial state. Defaults to a flat ``x = 0.5`` profile from
        :func:`flat_initial_state`.
    t_end_min : float, optional
        Integration horizon in minutes. The Skogestad canonical
        procedure uses 20 000 min.
    residual_tol : float, optional
        Acceptance threshold for the infinity norm of ``f(X*, U)``.
        Convergence is reported successful only if this threshold is
        met.
    rtol, atol : float, optional
        SciPy integrator tolerances. Tightened relative to the
        defaults of :func:`integrate_open_loop` because steady-state
        identification is more demanding than transient simulation.

    Returns
    -------
    SteadyStateResult
    """
    if inputs is None:
        inputs = nominal_inputs(parameters)
    if X0 is None:
        X0 = flat_initial_state(parameters)

    def _const_inputs(t: float, X: StateVector) -> InputVector:
        return inputs

    end_time = np.array([t_end_min], dtype=np.float64)
    result = integrate_open_loop(
        X0=X0,
        t_span=(0.0, t_end_min),
        inputs_fn=_const_inputs,
        parameters=parameters,
        rtol=rtol,
        atol=atol,
        t_eval=end_time,
    )

    if not result.success or result.X.shape[0] == 0:
        return SteadyStateResult(
            X=X0.copy(),
            U=inputs.copy(),
            residual_norm=float("inf"),
            method="integration",
            success=False,
            message=f"integration failed: {result.message}",
        )

    X_star = result.X[-1]
    residual = column_a_rhs(t_end_min, X_star, inputs, parameters)
    residual_norm = float(np.linalg.norm(residual, ord=np.inf))

    return SteadyStateResult(
        X=X_star,
        U=inputs.copy(),
        residual_norm=residual_norm,
        method="integration",
        success=residual_norm <= residual_tol,
        message=(
            f"converged after {t_end_min:.0f} min of integration"
            if residual_norm <= residual_tol
            else f"integration completed but residual {residual_norm:.3e} exceeds tol {residual_tol:.1e}"
        ),
    )


def compute_steady_state_by_newton(
    *,
    parameters: ColumnAParameters = DEFAULT_PARAMETERS,
    inputs: InputVector | None = None,
    X0: StateVector,
    residual_tol: float = 1.0e-8,
    max_iter: int = 200,
) -> SteadyStateResult:
    """Solve ``f(X*, U) = 0`` via SciPy Newton-Krylov.

    Parameters
    ----------
    parameters : ColumnAParameters, optional
        Column specification.
    inputs : numpy.ndarray of shape (7,), optional
        Constant input vector. Defaults to :func:`nominal_inputs`.
    X0 : numpy.ndarray of shape (2 * NT,)
        Initial guess. A good guess (e.g., a previously computed
        steady state) is required for reliable convergence.
    residual_tol : float, optional
        Convergence threshold (infinity norm of the residual).
    max_iter : int, optional
        Maximum Newton-Krylov iterations.

    Returns
    -------
    SteadyStateResult
    """
    if inputs is None:
        inputs = nominal_inputs(parameters)

    def _residual(X: StateVector) -> StateVector:
        return column_a_rhs(0.0, X, inputs, parameters)

    try:
        X_star = newton_krylov(
            _residual,
            X0,
            f_tol=residual_tol,
            maxiter=max_iter,
        )
    except NoConvergence as exc:
        partial = np.asarray(exc.args[0], dtype=np.float64)
        residual_norm = float(np.linalg.norm(_residual(partial), ord=np.inf))
        return SteadyStateResult(
            X=partial,
            U=inputs.copy(),
            residual_norm=residual_norm,
            method="newton",
            success=False,
            message=f"newton_krylov did not converge after {max_iter} iterations",
        )

    X_star = np.asarray(X_star, dtype=np.float64)
    residual_norm = float(np.linalg.norm(_residual(X_star), ord=np.inf))
    return SteadyStateResult(
        X=X_star,
        U=inputs.copy(),
        residual_norm=residual_norm,
        method="newton",
        success=residual_norm <= residual_tol,
        message=(
            "converged"
            if residual_norm <= residual_tol
            else f"residual {residual_norm:.3e} exceeds tol {residual_tol:.1e}"
        ),
    )
