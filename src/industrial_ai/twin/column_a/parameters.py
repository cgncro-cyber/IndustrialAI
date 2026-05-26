"""Canonical parameters of Skogestad's Column A.

Values reproduce the published Column A specification (40 theoretical
stages plus a total condenser, alpha = 1.5 binary system, 99 % / 99 %
purity at the nominal operating point). Source: Skogestad &
Postlethwaite (1996), *Multivariable Feedback Control*, Wiley; and the
MATLAB code at
https://skoge.folk.ntnu.no/book/1st_edition/matlab_m/cola/colamod.m.

Units follow Skogestad's convention (kmol, min, mole fraction). No unit
conversion is applied; downstream consumers must respect the kmol/min
basis.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

__all__ = ["DEFAULT_PARAMETERS", "ColumnAParameters"]


@dataclass(frozen=True, slots=True)
class ColumnAParameters:
    """Immutable parameter container for a Column A instance.

    Defaults reproduce Skogestad's canonical Column A: 41 theoretical
    stages (reboiler at stage 1, total condenser at stage NT = 41),
    feed at stage 21, relative volatility 1.5, 99 % / 99 % light-
    component purity at steady state.

    Attributes
    ----------
    NT : int
        Total number of theoretical stages, 1-indexed in Skogestad's
        convention (stage 1 = reboiler, stage NT = total condenser).
        Downstream numpy code translates to 0-indexed arrays at the
        boundary.
    NF : int
        Feed-stage index in the same 1-indexed convention.
    alpha : float
        Constant relative volatility of the light component.
    nominal_holdup_kmol : float
        Nominal liquid holdup on each stage (kmol). Equal across
        reboiler, intermediate trays, and condenser in Column A.
    liquid_dynamics_time_constant_min : float
        Time constant tau_L of the linearized tray hydraulics (min).
        Governs how quickly tray holdup deviations relax (Skogestad
        1997, Trans IChemE 75:539, eq. relating L_i to holdup
        deviation).
    K2_effect : float
        Coefficient lambda capturing the so-called K2 effect — the
        sensitivity of liquid flow to vapor-flow deviations. Default
        zero matches the published Column A nominal case.
    nominal_feed_F_kmol_per_min : float
        Nominal feed rate (kmol/min).
    nominal_feed_liquid_fraction_qF : float
        Feed liquid fraction qF (dimensionless). qF = 1 in the
        canonical Column A (saturated liquid feed).
    nominal_reflux_L0_kmol_per_min : float
        Nominal reflux rate (kmol/min) at the published steady state.
    nominal_boilup_V0_kmol_per_min : float
        Nominal boilup rate (kmol/min) at the published steady state.
    """

    NT: int = 41
    NF: int = 21
    alpha: float = 1.5
    nominal_holdup_kmol: float = 0.5
    liquid_dynamics_time_constant_min: float = 0.063
    K2_effect: float = 0.0
    nominal_feed_F_kmol_per_min: float = 1.0
    nominal_feed_liquid_fraction_qF: float = 1.0
    nominal_reflux_L0_kmol_per_min: float = 2.70629
    nominal_boilup_V0_kmol_per_min: float = 3.20629

    @property
    def feed_stage_idx(self) -> int:
        """0-indexed location of the feed stage (i.e., ``NF - 1``)."""
        return self.NF - 1

    @property
    def n_states(self) -> int:
        """Total state-vector length: ``2 * NT`` (compositions + holdups)."""
        return 2 * self.NT

    @property
    def nominal_holdups(self) -> npt.NDArray[np.float64]:
        """Per-stage nominal holdup array of length ``NT`` (kmol)."""
        return np.full(self.NT, self.nominal_holdup_kmol, dtype=np.float64)

    @property
    def nominal_liquid_below_feed_kmol_per_min(self) -> float:
        """L0b = L0 + qF0 * F0 — nominal liquid flow on stages below the feed."""
        return self.nominal_reflux_L0_kmol_per_min + (
            self.nominal_feed_liquid_fraction_qF * self.nominal_feed_F_kmol_per_min
        )

    @property
    def nominal_vapor_above_feed_kmol_per_min(self) -> float:
        """V0t = V0 + (1 - qF0) * F0 — nominal vapor flow on stages above the feed."""
        return self.nominal_boilup_V0_kmol_per_min + (
            (1.0 - self.nominal_feed_liquid_fraction_qF) * self.nominal_feed_F_kmol_per_min
        )


DEFAULT_PARAMETERS: ColumnAParameters = ColumnAParameters()
