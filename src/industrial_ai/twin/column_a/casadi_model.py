"""CasADi symbolic re-implementation of the Column A nonlinear ODE.

Mirrors :func:`industrial_ai.twin.column_a.model.column_a_rhs` line by
line in :class:`casadi.SX` symbolic operations, so that exact (machine-
precision) Jacobians ``df/dx`` and ``df/du`` are available via CasADi's
algorithmic differentiation. The numpy implementation in ``model.py``
remains the integration target — the symbolic model is the
*differentiation* backend exposed to:

- the Phase 2 Linear MPC baseline (``do-mpc``), which consumes CasADi
  Functions natively;
- :func:`industrial_ai.twin.column_a.linearize.linearize_lv` when
  invoked with ``backend="casadi"``, where the symbolic Jacobian
  replaces the central-difference Jacobian and removes finite-diff
  step-size as a tuning concern.

The symbolic rhs is constructed once per ``(parameters, lv_config, qF)``
combination via :func:`build_open_loop_rhs` /
:func:`build_lv_closed_rhs`; subsequent evaluations are calls into the
compiled CasADi :class:`Function`. Construction is sub-second on the
canonical 41-stage Column A.
"""

from __future__ import annotations

from dataclasses import dataclass

import casadi as cs
import numpy as np
import numpy.typing as npt

from industrial_ai.twin.column_a.configurations.lv import LVConfiguration
from industrial_ai.twin.column_a.parameters import (
    DEFAULT_PARAMETERS,
    ColumnAParameters,
)

__all__ = [
    "LVJacobianFunctions",
    "build_lv_closed_rhs",
    "build_lv_jacobians",
    "build_open_loop_rhs",
    "evaluate_lv_jacobians",
]


def _column_a_rhs_symbolic(
    x_sym: cs.SX,
    U_sym: cs.SX,
    parameters: ColumnAParameters,
) -> cs.SX:
    """Return the symbolic ``dX/dt`` expression matching :func:`column_a_rhs`.

    Construction mirrors the numpy implementation in ``model.py``
    exactly; only the array operations are replaced by their CasADi
    SX equivalents so CasADi can build the operation graph.
    """
    p = parameters
    NT = p.NT
    NF_idx = p.feed_stage_idx

    x = x_sym[:NT]
    M = x_sym[NT : 2 * NT]

    LT = U_sym[0]
    VB = U_sym[1]
    D = U_sym[2]
    B = U_sym[3]
    F = U_sym[4]
    zF = U_sym[5]
    qF = U_sym[6]

    # VLE on stages 0 .. NT-2 (the condenser is not an equilibrium stage).
    y = (p.alpha * x[: NT - 1]) / (1.0 + (p.alpha - 1.0) * x[: NT - 1])

    # Vapor flows V[0 .. NT-2]: VB below feed, VB + (1-qF)*F above feed.
    V_entries = []
    for i in range(NT - 1):
        V_entries.append(VB if i < NF_idx else VB + (1.0 - qF) * F)
    V = cs.vertcat(*V_entries)

    # Liquid flows with linearized hydraulics + K2 effect.
    L0b = p.nominal_liquid_below_feed_kmol_per_min
    L0 = p.nominal_reflux_L0_kmol_per_min
    V0 = p.nominal_boilup_V0_kmol_per_min
    V0t = p.nominal_vapor_above_feed_kmol_per_min
    tau_L = p.liquid_dynamics_time_constant_min
    lam = p.K2_effect
    M0 = p.nominal_holdup_kmol

    L_entries: list[cs.SX] = [cs.SX(0)] * NT
    for i in range(1, NF_idx + 1):
        L_entries[i] = L0b + (M[i] - M0) / tau_L + lam * (V[i - 1] - V0)
    for i in range(NF_idx + 1, NT - 1):
        L_entries[i] = L0 + (M[i] - M0) / tau_L + lam * (V[i - 1] - V0t)
    L_entries[NT - 1] = LT

    # Material balances. Initialize inner stages, then overwrite reboiler /
    # condenser; feed enters the feed stage as an additive term.
    dMdt: list[cs.SX] = [cs.SX(0)] * NT
    dMxdt: list[cs.SX] = [cs.SX(0)] * NT
    for i in range(1, NT - 1):
        dMdt[i] = L_entries[i + 1] - L_entries[i] + V[i - 1] - V[i]
        dMxdt[i] = (
            L_entries[i + 1] * x[i + 1] - L_entries[i] * x[i] + V[i - 1] * y[i - 1] - V[i] * y[i]
        )
    dMdt[NF_idx] = dMdt[NF_idx] + F
    dMxdt[NF_idx] = dMxdt[NF_idx] + F * zF

    dMdt[0] = L_entries[1] - V[0] - B
    dMxdt[0] = L_entries[1] * x[1] - V[0] * y[0] - B * x[0]

    dMdt[NT - 1] = V[NT - 2] - LT - D
    dMxdt[NT - 1] = V[NT - 2] * y[NT - 2] - LT * x[NT - 1] - D * x[NT - 1]

    dMdt_vec = cs.vertcat(*dMdt)
    dMxdt_vec = cs.vertcat(*dMxdt)

    # dx/dt = (d(Mx)/dt - x * dM/dt) / M
    dxdt = (dMxdt_vec - x * dMdt_vec) / M
    return cs.vertcat(dxdt, dMdt_vec)


def build_open_loop_rhs(
    parameters: ColumnAParameters = DEFAULT_PARAMETERS,
) -> cs.Function:
    """Compile a CasADi Function ``f(x, u) -> dx/dt`` for the open-loop ODE.

    Parameters
    ----------
    parameters : ColumnAParameters, optional
        Column specification.

    Returns
    -------
    casadi.Function
        Two inputs: state vector ``x`` of length ``2 * NT`` and input
        vector ``u`` of length 7. One output: ``dx/dt`` of length
        ``2 * NT``. Numerical inputs (numpy arrays, lists) are
        accepted; outputs come back as :class:`casadi.DM` and convert
        to numpy with ``np.asarray(out).flatten()``.
    """
    NT = parameters.NT
    x_sym = cs.SX.sym("x", 2 * NT)
    u_sym = cs.SX.sym("u", 7)
    rhs = _column_a_rhs_symbolic(x_sym, u_sym, parameters)
    return cs.Function("column_a_rhs", [x_sym, u_sym], [rhs], ["x", "u"], ["dxdt"])


def build_lv_closed_rhs(
    *,
    parameters: ColumnAParameters = DEFAULT_PARAMETERS,
    lv_config: LVConfiguration | None = None,
    qF: float = 1.0,
) -> cs.Function:
    """Compile a CasADi Function ``f_lv(x, [L, V, F, zF]) -> dx/dt`` with the LV closure inlined.

    The two level loops (condenser-holdup → D, reboiler-holdup → B)
    are part of the operation graph, so the resulting ``dx/dt`` is
    differentiable in the supervisor-visible inputs ``[L, V, F, zF]``
    only. ``qF`` is held constant (default 1.0 = saturated liquid feed)
    matching Skogestad's canonical case; vary it by rebuilding.

    Parameters
    ----------
    parameters : ColumnAParameters, optional
    lv_config : LVConfiguration, optional
        Level-loop tuning (Kc_D, Kc_B, setpoints, biases). Defaults to
        the cola_lv.m values.
    qF : float, optional
        Feed liquid fraction. Default 1.0.

    Returns
    -------
    casadi.Function
        Two inputs: state ``x`` of length ``2 * NT`` and supervisor
        inputs ``mv = [L, V, F, zF]`` of length 4. One output: ``dx/dt``.
    """
    if lv_config is None:
        lv_config = LVConfiguration()
    NT = parameters.NT

    x_sym = cs.SX.sym("x", 2 * NT)
    mv_sym = cs.SX.sym("mv", 4)
    L, V, F, zF = mv_sym[0], mv_sym[1], mv_sym[2], mv_sym[3]
    MB = x_sym[NT]
    MD = x_sym[2 * NT - 1]
    D = lv_config.Ds + (MD - lv_config.MDs) * lv_config.Kc_D
    B = lv_config.Bs + (MB - lv_config.MBs) * lv_config.Kc_B
    U_sym = cs.vertcat(L, V, D, B, F, zF, cs.SX(qF))
    rhs = _column_a_rhs_symbolic(x_sym, U_sym, parameters)
    return cs.Function("column_a_rhs_lv", [x_sym, mv_sym], [rhs], ["x", "mv"], ["dxdt"])


@dataclass(frozen=True, slots=True)
class LVJacobianFunctions:
    """Compiled CasADi Functions for the LV-closed Jacobians.

    Attributes
    ----------
    rhs_fn : casadi.Function
        ``f_lv(x, [L, V, F, zF]) -> dx/dt``.
    A_fn : casadi.Function
        ``df_lv / dx`` of shape ``(2 * NT, 2 * NT)``.
    B_fn : casadi.Function
        ``df_lv / d[L, V, F, zF]`` of shape ``(2 * NT, 4)``.
    """

    rhs_fn: cs.Function
    A_fn: cs.Function
    B_fn: cs.Function


def build_lv_jacobians(
    *,
    parameters: ColumnAParameters = DEFAULT_PARAMETERS,
    lv_config: LVConfiguration | None = None,
    qF: float = 1.0,
) -> LVJacobianFunctions:
    """Compile the LV-closed rhs together with its exact symbolic Jacobians.

    The Jacobian Functions ``A_fn`` and ``B_fn`` accept the same
    arguments as ``rhs_fn`` and return CasADi DM matrices, which the
    caller converts to numpy via :func:`evaluate_lv_jacobians`.
    """
    NT = parameters.NT
    if lv_config is None:
        lv_config = LVConfiguration()

    x_sym = cs.SX.sym("x", 2 * NT)
    mv_sym = cs.SX.sym("mv", 4)
    L, V, F, zF = mv_sym[0], mv_sym[1], mv_sym[2], mv_sym[3]
    MB = x_sym[NT]
    MD = x_sym[2 * NT - 1]
    D = lv_config.Ds + (MD - lv_config.MDs) * lv_config.Kc_D
    B = lv_config.Bs + (MB - lv_config.MBs) * lv_config.Kc_B
    U_sym = cs.vertcat(L, V, D, B, F, zF, cs.SX(qF))
    rhs = _column_a_rhs_symbolic(x_sym, U_sym, parameters)

    A = cs.jacobian(rhs, x_sym)
    B_jac = cs.jacobian(rhs, mv_sym)

    rhs_fn = cs.Function("column_a_rhs_lv", [x_sym, mv_sym], [rhs], ["x", "mv"], ["dxdt"])
    A_fn = cs.Function("dfdx_lv", [x_sym, mv_sym], [A], ["x", "mv"], ["A"])
    B_fn = cs.Function("dfdmv_lv", [x_sym, mv_sym], [B_jac], ["x", "mv"], ["B"])
    return LVJacobianFunctions(rhs_fn=rhs_fn, A_fn=A_fn, B_fn=B_fn)


def evaluate_lv_jacobians(
    jacs: LVJacobianFunctions,
    *,
    X: npt.NDArray[np.float64],
    L: float,
    V: float,
    F: float,
    zF: float,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Evaluate the LV-closed Jacobians at ``(X, [L, V, F, zF])`` and return numpy arrays.

    Parameters
    ----------
    jacs : LVJacobianFunctions
        Pre-compiled from :func:`build_lv_jacobians`.
    X : numpy.ndarray of shape (2 * NT,)
        Operating-point state.
    L, V, F, zF : float
        Operating-point supervisor inputs.

    Returns
    -------
    tuple of (A, B)
        ``A`` of shape ``(2 * NT, 2 * NT)``, ``B`` of shape
        ``(2 * NT, 4)``. Both numpy float64.
    """
    mv = np.array([L, V, F, zF], dtype=np.float64)
    A_dm = jacs.A_fn(x=X, mv=mv)["A"]
    B_dm = jacs.B_fn(x=X, mv=mv)["B"]
    return (
        np.asarray(A_dm, dtype=np.float64),
        np.asarray(B_dm, dtype=np.float64),
    )
