"""DV configuration for Skogestad's Column A.

The DV configuration leaves distillate ``D`` and boilup ``VB`` as the
two free inputs for the supervisory layer, and closes the two level
loops with P-only controllers in mirror image to LV: reflux ``LT``
controls the condenser holdup (more holdup → drain it by sending more
liquid back down the column), and bottoms ``B`` controls the reboiler
holdup (same direction as LV).

DV is the second canonical Column A configuration considered by
Skogestad alongside LV and L/D-V/B (see ADR 007 and Skogestad 1997,
Trans IChemE 75:539). It changes the supervisor's input pairing but not
the underlying ODE; the integrator, model, and steady-state machinery
are reused unchanged.

The implementation mirrors the published ``cola_dv.m`` (controller
gains, holdup setpoints, nominal LT bias) but is written clean-room
from the equations rather than translated line-by-line.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from industrial_ai.twin.column_a.parameters import (
    DEFAULT_PARAMETERS,
    ColumnAParameters,
)

__all__ = ["DVConfiguration", "assemble_inputs_dv"]

StateVector = npt.NDArray[np.float64]
InputVector = npt.NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class DVConfiguration:
    """Tuning of the DV level-loop P-controllers.

    Defaults reproduce Skogestad's published ``cola_dv.m`` values:
    proportional gains of 10 on both holdups, holdup setpoints of
    0.5 kmol, the nominal LT bias matching the published steady-state
    reflux, and a 0.5 kmol/min nominal bottoms bias.

    Attributes
    ----------
    Kc_L : float
        Proportional gain on the condenser-holdup-to-reflux loop. A
        positive value means: if ``MD > MDs`` the controller increases
        ``LT`` to drain the condenser.
    Kc_B : float
        Proportional gain on the reboiler-holdup-to-bottoms loop.
    MDs : float
        Condenser-holdup setpoint (kmol).
    MBs : float
        Reboiler-holdup setpoint (kmol).
    Ls : float
        Nominal reflux flow used as the controller bias term (kmol/min).
    Bs : float
        Nominal bottoms flow used as the controller bias term (kmol/min).
    """

    Kc_L: float = 10.0
    Kc_B: float = 10.0
    MDs: float = 0.5
    MBs: float = 0.5
    Ls: float = DEFAULT_PARAMETERS.nominal_reflux_L0_kmol_per_min
    Bs: float = 0.5


def assemble_inputs_dv(
    *,
    state: StateVector,
    D: float,
    VB: float,
    F: float,
    zF: float,
    qF: float,
    config: DVConfiguration | None = None,
    parameters: ColumnAParameters = DEFAULT_PARAMETERS,
) -> InputVector:
    """Build the 7-element input vector under DV-configuration control.

    The condenser holdup (``state[2 * NT - 1]``) feeds a P-controller
    that computes the reflux ``LT``; the reboiler holdup (``state[NT]``)
    feeds a second P-controller that computes the bottoms ``B``. ``D``,
    ``VB``, ``F``, ``zF``, and ``qF`` come from the supervisor or
    operator.

    Parameters
    ----------
    state : numpy.ndarray of shape (2 * NT,)
        Current state vector. The holdup portion lives at
        ``state[NT : 2 * NT]``, with the reboiler at the start of the
        slice and the condenser at the end.
    D : float
        Distillate flow (kmol/min) commanded by the supervisor.
    VB : float
        Boilup flow (kmol/min) commanded by the supervisor.
    F, zF, qF : float
        Feed rate, composition, and liquid fraction (disturbance trio).
    config : DVConfiguration, optional
        Level-loop tuning. Defaults to the cola_dv.m values.
    parameters : ColumnAParameters, optional
        Column specification.

    Returns
    -------
    numpy.ndarray of shape (7,)
        ``[LT, VB, D, B, F, zF, qF]`` as expected by
        :func:`industrial_ai.twin.column_a.model.column_a_rhs`.
    """
    cfg = config if config is not None else DVConfiguration()
    NT = parameters.NT
    MB = state[NT]
    MD = state[2 * NT - 1]
    LT = cfg.Ls + (MD - cfg.MDs) * cfg.Kc_L
    B = cfg.Bs + (MB - cfg.MBs) * cfg.Kc_B
    return np.array([LT, VB, D, B, F, zF, qF], dtype=np.float64)
