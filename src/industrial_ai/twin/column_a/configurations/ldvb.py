"""L/D-V/B (double-ratio) configuration for Skogestad's Column A.

The L/D-V/B configuration — often abbreviated LDVB or "double-ratio" —
has the supervisor manipulate the two dimensionless ratios

    LR = L / D    (reflux ratio)
    VR = V / B    (boilup-to-bottoms ratio)

instead of absolute flows. The two level loops are closed in LV style
(condenser holdup → distillate ``D``, reboiler holdup → bottoms ``B``);
the absolute reflux and boilup are then recovered from the
supervisor-commanded ratios via ``LT = LR * D`` and ``VB = VR * B``.

LDVB is the third canonical Column A configuration considered by
Skogestad and is well known for its favorable closed-loop sensitivity
to feed-composition disturbances (Skogestad & Postlethwaite 1996,
§10.8). The underlying ODE, integrator, and steady-state machinery are
unchanged; only the supervisor-to-input mapping differs.

The implementation mirrors the published ``cola_rr.m`` (LV-style level
loop gains plus the two ratios) but is written clean-room from the
equations rather than translated line-by-line.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from industrial_ai.twin.column_a.parameters import (
    DEFAULT_PARAMETERS,
    ColumnAParameters,
)

__all__ = ["LDVBConfiguration", "assemble_inputs_ldvb", "nominal_ratios"]

StateVector = npt.NDArray[np.float64]
InputVector = npt.NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class LDVBConfiguration:
    """Tuning of the LDVB level-loop P-controllers.

    The level loops are identical to LV (the supervisor's manipulated
    variables are ratios, so the absolute D and B flows still serve as
    the level-controller outputs). Defaults reproduce the published
    ``cola_rr.m`` values: proportional gains of 10 on both holdups,
    setpoints at 0.5 kmol, nominal flow biases at 0.5 kmol/min.

    Attributes
    ----------
    Kc_D : float
        Proportional gain on the condenser-holdup-to-distillate loop.
    Kc_B : float
        Proportional gain on the reboiler-holdup-to-bottoms loop.
    MDs : float
        Condenser-holdup setpoint (kmol).
    MBs : float
        Reboiler-holdup setpoint (kmol).
    Ds : float
        Nominal distillate flow used as the controller bias term (kmol/min).
    Bs : float
        Nominal bottoms flow used as the controller bias term (kmol/min).
    """

    Kc_D: float = 10.0
    Kc_B: float = 10.0
    MDs: float = 0.5
    MBs: float = 0.5
    Ds: float = 0.5
    Bs: float = 0.5


def nominal_ratios(
    parameters: ColumnAParameters = DEFAULT_PARAMETERS,
    *,
    Ds: float = 0.5,
    Bs: float = 0.5,
) -> tuple[float, float]:
    """Return the nominal ``(LR, VR)`` ratios for the published SS.

    At Skogestad's published steady state the nominal reflux and boilup
    sit at ``L0`` and ``V0`` respectively, with both product flows at
    0.5 kmol/min. The ratios that reproduce this operating point are
    therefore ``LR = L0 / Ds`` and ``VR = V0 / Bs`` — convenient
    defaults for the supervisor's nominal command.

    Parameters
    ----------
    parameters : ColumnAParameters, optional
        Column specification (provides ``L0`` and ``V0``).
    Ds, Bs : float, optional
        Nominal distillate and bottoms flows. Defaults match
        :class:`LDVBConfiguration`.

    Returns
    -------
    tuple of (float, float)
        ``(LR_nom, VR_nom)`` — the dimensionless ratios that map to
        ``(L0, V0)`` at the published SS.
    """
    LR_nom = parameters.nominal_reflux_L0_kmol_per_min / Ds
    VR_nom = parameters.nominal_boilup_V0_kmol_per_min / Bs
    return LR_nom, VR_nom


def assemble_inputs_ldvb(
    *,
    state: StateVector,
    LR: float,
    VR: float,
    F: float,
    zF: float,
    qF: float,
    config: LDVBConfiguration | None = None,
    parameters: ColumnAParameters = DEFAULT_PARAMETERS,
) -> InputVector:
    """Build the 7-element input vector under LDVB-configuration control.

    The condenser holdup (``state[2 * NT - 1]``) and reboiler holdup
    (``state[NT]``) feed P-controllers that compute the distillate
    ``D`` and bottoms ``B`` (LV-style). The reflux ``LT`` and boilup
    ``VB`` are then recovered from the supervisor-commanded ratios via
    ``LT = LR * D`` and ``VB = VR * B``.

    Parameters
    ----------
    state : numpy.ndarray of shape (2 * NT,)
        Current state vector. The holdup portion lives at
        ``state[NT : 2 * NT]``, with the reboiler at the start of the
        slice and the condenser at the end.
    LR : float
        Reflux ratio ``L / D`` (dimensionless) commanded by the
        supervisor.
    VR : float
        Boilup-to-bottoms ratio ``V / B`` (dimensionless) commanded by
        the supervisor.
    F, zF, qF : float
        Feed rate, composition, and liquid fraction (disturbance trio).
    config : LDVBConfiguration, optional
        Level-loop tuning. Defaults to the cola_rr.m values.
    parameters : ColumnAParameters, optional
        Column specification.

    Returns
    -------
    numpy.ndarray of shape (7,)
        ``[LT, VB, D, B, F, zF, qF]`` as expected by
        :func:`industrial_ai.twin.column_a.model.column_a_rhs`.
    """
    cfg = config if config is not None else LDVBConfiguration()
    NT = parameters.NT
    MB = state[NT]
    MD = state[2 * NT - 1]
    D = cfg.Ds + (MD - cfg.MDs) * cfg.Kc_D
    B = cfg.Bs + (MB - cfg.MBs) * cfg.Kc_B
    LT = LR * D
    VB = VR * B
    return np.array([LT, VB, D, B, F, zF, qF], dtype=np.float64)
