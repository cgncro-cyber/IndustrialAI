"""Skogestad Column A nonlinear ODE — clean-room Python implementation.

State-vector layout (length 2 * NT = 82 for the canonical Column A):

    X[0 : NT]       light-component mole fractions on each stage,
                    ordered reboiler (X[0]) to total condenser (X[NT-1]).
    X[NT : 2 * NT]  liquid molar holdups (kmol) on each stage,
                    same ordering.

Input-vector layout (length 7):

    U[0]  reflux LT                  (kmol/min)
    U[1]  boilup VB                  (kmol/min)
    U[2]  distillate D               (kmol/min)
    U[3]  bottoms B                  (kmol/min)
    U[4]  feed F                     (kmol/min)
    U[5]  feed composition zF        (mole fraction of light component)
    U[6]  feed liquid fraction qF    (dimensionless, 1 = saturated liquid)

Modelling assumptions (Skogestad's published Column A):

- Binary mixture.
- Constant relative volatility alpha.
- Equilibrium stages from the reboiler up to (but not including) the
  total condenser; the total condenser is not an equilibrium stage.
- No vapor holdup.
- Constant molar flows on each side of the feed (the only addition is
  the vapor augmentation by ``(1 - qF) * F`` above the feed when
  ``qF < 1``).
- Linearized tray-hydraulics: liquid flow leaving a stage depends
  linearly on holdup deviation from nominal and on vapor-flow deviation
  from nominal (the latter via the K2 coefficient ``lambda``).
- Feed is mixed into the feed stage.
- The reboiler is a single equilibrium stage with bottoms drawn off.
- The condenser is total — all vapor condenses, then splits into reflux
  and distillate.

These assumptions are itemized again in
``column_a/assumptions.md`` with citations.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from industrial_ai.twin.column_a.parameters import (
    DEFAULT_PARAMETERS,
    ColumnAParameters,
)

__all__ = ["InputVector", "StateVector", "column_a_rhs"]

StateVector = npt.NDArray[np.float64]
InputVector = npt.NDArray[np.float64]


def column_a_rhs(
    t: float,
    X: StateVector,
    U: InputVector,
    parameters: ColumnAParameters = DEFAULT_PARAMETERS,
) -> StateVector:
    """Return ``dX/dt`` for the Column A nonlinear dynamic model.

    Parameters
    ----------
    t : float
        Time (min). The system is autonomous; this argument exists for
        compatibility with ``scipy.integrate.solve_ivp``.
    X : numpy.ndarray of shape (2 * NT,)
        State vector. See module docstring for layout.
    U : numpy.ndarray of shape (7,)
        Inputs and disturbances. See module docstring for layout.
    parameters : ColumnAParameters, optional
        Column specification. Defaults to ``DEFAULT_PARAMETERS``
        (Skogestad's canonical Column A).

    Returns
    -------
    numpy.ndarray of shape (2 * NT,)
        Time derivative of ``X`` in the same layout.
    """
    p = parameters
    NT = p.NT
    NF_idx = p.feed_stage_idx

    x = X[:NT]
    M = X[NT : 2 * NT]

    LT = U[0]
    VB = U[1]
    D = U[2]
    B = U[3]
    F = U[4]
    zF = U[5]
    qF = U[6]

    # Vapor-liquid equilibrium on stages 0 .. NT-2. The condenser
    # (stage NT-1) is not an equilibrium stage and is treated separately
    # below; ``y`` therefore has length NT-1.
    y = (p.alpha * x[: NT - 1]) / (1.0 + (p.alpha - 1.0) * x[: NT - 1])

    # Vapor flows assuming constant molar flows. Below the feed: V = VB.
    # Above the feed: V is augmented by (1 - qF) * F.
    V = np.empty(NT - 1, dtype=np.float64)
    V[:NF_idx] = VB
    V[NF_idx:] = VB + (1.0 - qF) * F

    # Liquid flows with linearized tray hydraulics + K2-effect.
    # L[i] for i in 1..NT-1 is the liquid leaving stage i; L[NT-1] is
    # the reflux returning to the condenser stage (set to LT). The
    # reboiler stage (i = 0) has no incoming liquid from below.
    L = np.empty(NT, dtype=np.float64)
    L0b = p.nominal_liquid_below_feed_kmol_per_min
    L0 = p.nominal_reflux_L0_kmol_per_min
    V0 = p.nominal_boilup_V0_kmol_per_min
    V0t = p.nominal_vapor_above_feed_kmol_per_min
    tau_L = p.liquid_dynamics_time_constant_min
    lam = p.K2_effect

    # Stages 1 .. NF_idx (0-indexed), inclusive: below or at the feed.
    below_or_at = np.arange(1, NF_idx + 1)
    L[below_or_at] = (
        L0b + (M[below_or_at] - p.nominal_holdup_kmol) / tau_L + lam * (V[below_or_at - 1] - V0)
    )

    # Stages NF_idx+1 .. NT-2: above the feed.
    above = np.arange(NF_idx + 1, NT - 1)
    L[above] = L0 + (M[above] - p.nominal_holdup_kmol) / tau_L + lam * (V[above - 1] - V0t)

    # The top stage receives reflux LT directly.
    L[NT - 1] = LT

    # Material balances. Initialize to zero so that the reboiler and
    # condenser cases below can overwrite their entries cleanly.
    dMdt = np.zeros(NT, dtype=np.float64)
    dMxdt = np.zeros(NT, dtype=np.float64)

    # Inner stages (1 .. NT-2, 0-indexed).
    inner = np.arange(1, NT - 1)
    dMdt[inner] = L[inner + 1] - L[inner] + V[inner - 1] - V[inner]
    dMxdt[inner] = (
        L[inner + 1] * x[inner + 1]
        - L[inner] * x[inner]
        + V[inner - 1] * y[inner - 1]
        - V[inner] * y[inner]
    )

    # Feed enters the feed stage with composition zF.
    dMdt[NF_idx] += F
    dMxdt[NF_idx] += F * zF

    # Reboiler (stage 0, equilibrium). Liquid arrives from stage 1;
    # vapor leaves to stage 1; bottoms B is drawn off.
    dMdt[0] = L[1] - V[0] - B
    dMxdt[0] = L[1] * x[1] - V[0] * y[0] - B * x[0]

    # Total condenser (stage NT-1, not an equilibrium stage). Vapor
    # from stage NT-2 condenses; the resulting liquid splits into
    # reflux LT and distillate product D.
    dMdt[NT - 1] = V[NT - 2] - LT - D
    dMxdt[NT - 1] = V[NT - 2] * y[NT - 2] - LT * x[NT - 1] - D * x[NT - 1]

    # Convert d(Mx)/dt to dx/dt via the product rule
    #   d(M x)/dt = x dM/dt + M dx/dt
    # rearranged to
    #   dx/dt    = (d(Mx)/dt - x dM/dt) / M.
    dxdt = (dMxdt - x * dMdt) / M

    out = np.empty(2 * NT, dtype=np.float64)
    out[:NT] = dxdt
    out[NT:] = dMdt
    return out
