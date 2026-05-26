"""LV configuration for Skogestad's Column A.

The LV configuration closes the condenser-holdup and reboiler-holdup
loops with P-only controllers, leaving reflux LT and boilup VB as the
two free inputs for the supervisory layer. This is the Phase 1 primary
configuration per ADR 007.

The implementation mirrors the published ``cola_lv.m`` (controller
gains, holdup setpoints, nominal product flows) but is written
clean-room from the equations rather than translated line-by-line.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from industrial_ai.twin.column_a.parameters import (
    DEFAULT_PARAMETERS,
    ColumnAParameters,
)

__all__ = ["LVConfiguration", "assemble_inputs_lv"]

StateVector = npt.NDArray[np.float64]
InputVector = npt.NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class LVConfiguration:
    """Tuning of the LV level-loop P-controllers.

    Defaults reproduce Skogestad's published ``cola_lv.m`` values:
    proportional gains of 10 on both holdups, setpoints and nominal
    flows at 0.5 kmol and 0.5 kmol/min respectively (consistent with
    the canonical Column A operating point).

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


def assemble_inputs_lv(
    *,
    state: StateVector,
    LT: float,
    VB: float,
    F: float,
    zF: float,
    qF: float,
    config: LVConfiguration | None = None,
    parameters: ColumnAParameters = DEFAULT_PARAMETERS,
) -> InputVector:
    """Build the 7-element input vector under LV-configuration control.

    The condenser holdup (``state[2 * NT - 1]``) and reboiler holdup
    (``state[NT]``) feed P-controllers that compute the distillate
    ``D`` and bottoms ``B`` flows. ``LT``, ``VB``, ``F``, ``zF``, and
    ``qF`` come from the supervisor or operator.

    Parameters
    ----------
    state : numpy.ndarray of shape (2 * NT,)
        Current state vector. The holdup portion lives at
        ``state[NT : 2 * NT]``, with the reboiler at the start of the
        slice and the condenser at the end.
    LT : float
        Reflux flow (kmol/min) commanded by the supervisor.
    VB : float
        Boilup flow (kmol/min) commanded by the supervisor.
    F, zF, qF : float
        Feed rate, composition, and liquid fraction (disturbance trio).
    config : LVConfiguration, optional
        Level-loop tuning. Defaults to the cola_lv.m values.
    parameters : ColumnAParameters, optional
        Column specification.

    Returns
    -------
    numpy.ndarray of shape (7,)
        ``[LT, VB, D, B, F, zF, qF]`` as expected by
        :func:`industrial_ai.twin.column_a.model.column_a_rhs`.
    """
    cfg = config if config is not None else LVConfiguration()
    NT = parameters.NT
    MB = state[NT]
    MD = state[2 * NT - 1]
    D = cfg.Ds + (MD - cfg.MDs) * cfg.Kc_D
    B = cfg.Bs + (MB - cfg.MBs) * cfg.Kc_B
    return np.array([LT, VB, D, B, F, zF, qF], dtype=np.float64)
