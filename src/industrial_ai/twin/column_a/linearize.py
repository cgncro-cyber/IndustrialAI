"""Numerical linearization of the Column A LV-closed-loop model.

Provides the small-signal state-space matrices ``(A, B, C)`` of the
LV-configured Column A around any operating point, plus the
steady-state gain matrix ``G(0)`` and the dominant continuous-time
constants. The linearization uses central finite differences on
:func:`column_a_rhs` after closing the level loops via
:func:`assemble_inputs_lv`, so the resulting model represents the
*regulatory*-closed plant — exactly what the Phase 2 Linear MPC
baseline (``do-mpc``) needs as its system model and what the Phase 1
mini-gate validates against the published Skogestad 1997 reference
values.

Inputs to the linearized model are the two LV-configuration
manipulated variables ``[L, V]`` followed by the two disturbance
variables ``[F, zF]`` (matching ``cola_linearize.m`` /
``cola_lv_lin.m`` in Skogestad's MATLAB suite). Outputs are the top
composition ``y_D = X[NT-1]`` and the bottom composition
``x_B = X[0]``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import numpy.typing as npt

from industrial_ai.twin.column_a.configurations.lv import (
    LVConfiguration,
    assemble_inputs_lv,
)
from industrial_ai.twin.column_a.model import column_a_rhs
from industrial_ai.twin.column_a.parameters import (
    DEFAULT_PARAMETERS,
    ColumnAParameters,
)

__all__ = [
    "LinearizedLVModel",
    "dominant_time_constants_min",
    "linearize_lv",
    "steady_state_gain",
]

LinearizeBackend = Literal["finite_difference", "casadi"]

StateVector = npt.NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class LinearizedLVModel:
    """Continuous-time linearized LV-configuration state-space model.

    Attributes
    ----------
    A : numpy.ndarray of shape (2*NT, 2*NT)
        State Jacobian ``df/dX``, evaluated at the operating point with
        the level loops already closed (level-controller gains
        contribute to ``A`` via :func:`assemble_inputs_lv`).
    B : numpy.ndarray of shape (2*NT, 4)
        Input Jacobian ``df/d[L, V, F, zF]``.
    C : numpy.ndarray of shape (2, 2*NT)
        Output selection: row 0 picks ``y_D`` (``X[NT-1]``), row 1
        picks ``x_B`` (``X[0]``).
    X_ss : numpy.ndarray of shape (2*NT,)
        Operating-point state vector.
    L_ss, V_ss, F_ss, zF_ss, qF_ss : float
        Operating-point inputs.
    """

    A: npt.NDArray[np.float64]
    B: npt.NDArray[np.float64]
    C: npt.NDArray[np.float64]
    X_ss: npt.NDArray[np.float64]
    L_ss: float
    V_ss: float
    F_ss: float
    zF_ss: float
    qF_ss: float


def _rhs_lv(
    X: StateVector,
    *,
    L: float,
    V: float,
    F: float,
    zF: float,
    qF: float,
    parameters: ColumnAParameters,
    lv_config: LVConfiguration,
) -> StateVector:
    U = assemble_inputs_lv(
        state=X, LT=L, VB=V, F=F, zF=zF, qF=qF, config=lv_config, parameters=parameters
    )
    return column_a_rhs(0.0, X, U, parameters)


def _jacobians_finite_difference(
    *,
    X_ss: StateVector,
    L_ss: float,
    V_ss: float,
    F_ss: float,
    zF_ss: float,
    qF_ss: float,
    parameters: ColumnAParameters,
    lv_config: LVConfiguration,
    fd_step_rel: float,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Central-difference Jacobians ``(A, B)`` — historical default backend."""
    n_states = 2 * parameters.NT

    def f(X: StateVector, L: float, V: float, F: float, zF: float) -> StateVector:
        return _rhs_lv(
            X, L=L, V=V, F=F, zF=zF, qF=qF_ss, parameters=parameters, lv_config=lv_config
        )

    def _fd_step(value: float) -> float:
        return max(abs(value) * fd_step_rel, 1.0e-10)

    A = np.zeros((n_states, n_states), dtype=np.float64)
    for i in range(n_states):
        h = _fd_step(X_ss[i])
        Xp = X_ss.copy()
        Xm = X_ss.copy()
        Xp[i] += h
        Xm[i] -= h
        A[:, i] = (f(Xp, L_ss, V_ss, F_ss, zF_ss) - f(Xm, L_ss, V_ss, F_ss, zF_ss)) / (2.0 * h)

    B = np.zeros((n_states, 4), dtype=np.float64)
    for col, (name, ss_value) in enumerate((("L", L_ss), ("V", V_ss), ("F", F_ss), ("zF", zF_ss))):
        h = _fd_step(ss_value)
        kwargs_plus = {"L": L_ss, "V": V_ss, "F": F_ss, "zF": zF_ss}
        kwargs_minus = {"L": L_ss, "V": V_ss, "F": F_ss, "zF": zF_ss}
        kwargs_plus[name] = ss_value + h
        kwargs_minus[name] = ss_value - h
        B[:, col] = (f(X_ss, **kwargs_plus) - f(X_ss, **kwargs_minus)) / (2.0 * h)

    return A, B


def _jacobians_casadi(
    *,
    X_ss: StateVector,
    L_ss: float,
    V_ss: float,
    F_ss: float,
    zF_ss: float,
    qF_ss: float,
    parameters: ColumnAParameters,
    lv_config: LVConfiguration,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Exact symbolic Jacobians ``(A, B)`` via the CasADi backend."""
    # Imported lazily so the finite-difference backend stays usable on
    # installations without CasADi.
    from industrial_ai.twin.column_a.casadi_model import (
        build_lv_jacobians,
        evaluate_lv_jacobians,
    )

    jacs = build_lv_jacobians(parameters=parameters, lv_config=lv_config, qF=qF_ss)
    return evaluate_lv_jacobians(jacs, X=X_ss, L=L_ss, V=V_ss, F=F_ss, zF=zF_ss)


def linearize_lv(
    *,
    X_ss: StateVector,
    L_ss: float,
    V_ss: float,
    F_ss: float,
    zF_ss: float,
    qF_ss: float = 1.0,
    parameters: ColumnAParameters = DEFAULT_PARAMETERS,
    lv_config: LVConfiguration | None = None,
    fd_step_rel: float = 1.0e-6,
    backend: LinearizeBackend = "finite_difference",
) -> LinearizedLVModel:
    """Compute the LV-configuration linearization at the given operating point.

    Two backends are available:

    - ``"finite_difference"`` (default): central differences on the
      numpy rhs, with step size ``max(|value| * fd_step_rel, 1e-10)``
      per component. Robust, dependency-free, and the implementation
      against which the original Phase 1 mini-gate was validated.
    - ``"casadi"``: exact symbolic Jacobians via CasADi algorithmic
      differentiation. Faster, machine-precision accurate, and the
      same backend the Phase 2 Linear MPC baseline (``do-mpc``)
      consumes natively. Identical numerical result to the finite-
      difference backend to ~1e-5 — see
      ``tests/twin/test_casadi_model.py``.

    Parameters
    ----------
    X_ss : numpy.ndarray of shape (2 * NT,)
        Operating-point state vector (typically the long-time
        integration result at nominal inputs).
    L_ss, V_ss : float
        Reflux and boilup at the operating point (kmol/min).
    F_ss, zF_ss : float
        Feed rate (kmol/min) and feed composition (mole fraction).
    qF_ss : float, optional
        Feed liquid fraction. Defaults to 1 (saturated liquid feed),
        matching Skogestad's canonical case.
    parameters : ColumnAParameters, optional
        Column specification.
    lv_config : LVConfiguration, optional
        LV-level-loop configuration. Defaults to Skogestad's cola_lv.m
        gains.
    fd_step_rel : float, optional
        Relative finite-difference step size. Ignored when
        ``backend="casadi"``.
    backend : {"finite_difference", "casadi"}, optional
        Differentiation backend.

    Returns
    -------
    LinearizedLVModel
    """
    cfg = lv_config if lv_config is not None else LVConfiguration()
    p = parameters
    NT = p.NT

    if backend == "casadi":
        A, B = _jacobians_casadi(
            X_ss=X_ss,
            L_ss=L_ss,
            V_ss=V_ss,
            F_ss=F_ss,
            zF_ss=zF_ss,
            qF_ss=qF_ss,
            parameters=p,
            lv_config=cfg,
        )
    elif backend == "finite_difference":
        A, B = _jacobians_finite_difference(
            X_ss=X_ss,
            L_ss=L_ss,
            V_ss=V_ss,
            F_ss=F_ss,
            zF_ss=zF_ss,
            qF_ss=qF_ss,
            parameters=p,
            lv_config=cfg,
            fd_step_rel=fd_step_rel,
        )
    else:
        raise ValueError(f"unknown backend {backend!r}; expected 'finite_difference' or 'casadi'")

    n_states = 2 * NT
    C = np.zeros((2, n_states), dtype=np.float64)
    C[0, NT - 1] = 1.0  # y_D
    C[1, 0] = 1.0  # x_B

    return LinearizedLVModel(
        A=A,
        B=B,
        C=C,
        X_ss=X_ss.copy(),
        L_ss=L_ss,
        V_ss=V_ss,
        F_ss=F_ss,
        zF_ss=zF_ss,
        qF_ss=qF_ss,
    )


def steady_state_gain(model: LinearizedLVModel) -> npt.NDArray[np.float64]:
    """Return ``G(0) = -C A^{-1} B`` of shape (2, 4).

    Columns correspond to ``[L, V, F, zF]``. The first two columns are
    the Skogestad ``G^LV(0)``; the last two are the disturbance gains
    ``G_d^LV(0)``.
    """
    return -model.C @ np.linalg.solve(model.A, model.B)


def dominant_time_constants_min(
    model: LinearizedLVModel,
    n: int = 3,
) -> npt.NDArray[np.float64]:
    """Return the ``n`` slowest continuous-time constants of the model.

    Computed as ``1 / |Re(eigenvalue)|`` for the ``n`` eigenvalues of
    ``A`` closest to zero from the stable (negative real part) side.
    Eigenvalues with positive real part are skipped — for a properly
    closed LV configuration none should be present.

    Parameters
    ----------
    model : LinearizedLVModel
    n : int, optional
        Number of slowest time constants to return. Defaults to 3 to
        match the three eigenvalues reported in Skogestad 1997
        Section 4.4.

    Returns
    -------
    numpy.ndarray of shape (n,), sorted slowest-first.
    """
    eigvals = np.linalg.eigvals(model.A)
    stable_real = eigvals.real[eigvals.real < 0.0]
    # Slowest = real part closest to zero from below.
    slowest = np.sort(stable_real)[::-1][:n]
    return 1.0 / np.abs(slowest)
