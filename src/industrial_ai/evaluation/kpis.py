"""Phase 2 KPI suite — six scalar KPIs scored on every simulation run.

The same set is computed for every configuration (C0 / C1 / C2 / C3),
so the metrics travel from this module into the Phase-2 baseline
benchmark, the Phase-3 agent runs, the Phase-4 safety-gate
evaluation, and the Phase-5 paper Figures 4 and 7. KPI definitions
are frozen here per the Phase 2 deliverables list in PROJECT_PLAN.md.

KPIs (lower is better unless noted):

1. **Specific energy consumption** — boilup integrated over time
   divided by distillate integrated over time. Units: kmol vapor per
   kmol distillate.
2. **Light-component yield** — light material in the distillate
   divided by light material in the feed. Range ``[0, 1]``; *higher*
   is better.
3. **Constraint-violation count** — number of regulatory ticks at
   which the top composition is below ``y_D_spec_low`` *or* the
   bottoms composition is above ``x_B_spec_high``.
4. **Settling time after disturbance** — time from disturbance onset
   until ``y_D`` and ``x_B`` both stay within the settling band of
   their post-disturbance steady values for the configured settling
   window. Returns ``+inf`` if the controlled variables never settle
   within the simulated horizon.
5. **MV activity** — total variation ``Sum |dL/dt|`` + ``Sum |dV/dt|``
   evaluated tick-by-tick. Proxy for actuator wear and reviewer-
   defends the *"but agents oscillate the valves"* objection.
6. **Integrated absolute error (IAE)** — combined IAE on both
   composition channels using the *applied* (post-slew-limiter)
   setpoints. Units: mole-fraction-minutes.

All KPIs are pure functions of :class:`SimulationResult` plus a
:class:`KPIConfig` of thresholds; nothing is read from disk and no
randomness is involved.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from industrial_ai.twin.simulate import SimulationResult

__all__ = [
    "KPIConfig",
    "KPISet",
    "compute_kpis",
]


@dataclass(frozen=True, slots=True)
class KPIConfig:
    """Thresholds and analysis windows for KPI computation.

    Attributes
    ----------
    y_D_spec_low : float
        Lowest acceptable top composition. Default 0.99 — the
        Skogestad Column A nominal purity.
    x_B_spec_high : float
        Highest acceptable bottoms composition. Default 0.01.
    settling_band : float
        Absolute mole-fraction band around the post-disturbance final
        value that defines "settled". Default 5e-4.
    settling_window_min : float
        Minimum continuous duration (min) the controlled variable
        must stay inside the settling band to be called settled.
        Default 5 min — order of one tick of the planned supervisory
        cadence (5-15 min, per ADR 006).
    disturbance_onset_min : float
        Time (min) at which the disturbance step is applied. Default
        5 min — matches :mod:`industrial_ai.control.scenarios`.
    """

    y_D_spec_low: float = 0.99
    x_B_spec_high: float = 0.01
    settling_band: float = 5.0e-4
    settling_window_min: float = 5.0
    disturbance_onset_min: float = 5.0


@dataclass(frozen=True, slots=True)
class KPISet:
    """The six Phase-2 KPIs from one closed-loop run.

    All fields are plain Python floats / ints so the set is trivially
    JSON-serializable for ``kpis.json`` artifacts.
    """

    specific_energy_kmol_per_kmol: float
    light_yield: float
    constraint_violations: int
    settling_time_min: float
    mv_activity_kmol_per_min: float
    iae_mole_fraction_min: float

    def as_dict(self) -> dict[str, float | int]:
        """Return the KPIs as a plain dict (suitable for ``kpis.json``)."""
        return asdict(self)


def compute_kpis(
    result: SimulationResult,
    *,
    config: KPIConfig | None = None,
) -> KPISet:
    """Compute the full KPI bundle for a finished simulation.

    Parameters
    ----------
    result : SimulationResult
        Trajectory returned by
        :func:`industrial_ai.twin.simulate.simulate_lv_closed_loop`.
    config : KPIConfig, optional
        Thresholds and windows. Defaults to :class:`KPIConfig`.

    Returns
    -------
    KPISet
    """
    if config is None:
        config = KPIConfig()
    if not result.success:
        # Caller-visible signal that the trajectory was truncated by a
        # solver failure: every KPI is undefined.
        return KPISet(
            specific_energy_kmol_per_kmol=float("inf"),
            light_yield=0.0,
            constraint_violations=-1,
            settling_time_min=float("inf"),
            mv_activity_kmol_per_min=float("inf"),
            iae_mole_fraction_min=float("inf"),
        )

    # --- Time grids ----------------------------------------------------------
    # State (X, y_D, x_B) is sampled at the n+1 tick times in result.t.
    # Inputs and applied setpoints are zero-order-hold values active during
    # the interval [result.t[k], result.t[k+1]] (n entries each).
    t_state = result.t
    dt_intervals = np.diff(result.t)

    y_D = result.y_D
    x_B = result.x_B
    LT = result.inputs[:, 0]
    VB = result.inputs[:, 1]
    D = result.inputs[:, 2]
    F = result.inputs[:, 4]
    zF = result.inputs[:, 5]
    y_D_sp = result.applied_setpoints[:, 0]
    x_B_sp = result.applied_setpoints[:, 1]

    # --- 1. Specific energy --------------------------------------------------
    # ZOH inputs: each VB[k] / D[k] is active for dt_intervals[k].
    total_vapor = float(np.sum(VB * dt_intervals))
    total_distillate = float(np.sum(D * dt_intervals))
    specific_energy = total_vapor / total_distillate if total_distillate > 0 else float("inf")

    # --- 2. Light-component yield -------------------------------------------
    # Yield = light out / light in. Use the *interval-end* compositions
    # (y_D at t[k+1]) for the distillate stream so it pairs with the
    # ZOH D[k] over the same interval.
    light_out = float(np.sum(D * y_D[1:] * dt_intervals))
    light_in = float(np.sum(F * zF * dt_intervals))
    light_yield = light_out / light_in if light_in > 0 else 0.0

    # --- 3. Constraint violations -------------------------------------------
    # Count ticks where either composition is off-spec.
    constraint_violations = int(np.sum((y_D < config.y_D_spec_low) | (x_B > config.x_B_spec_high)))

    # --- 4. Settling time ---------------------------------------------------
    settling_time = _settling_time(
        t=t_state,
        y_D=y_D,
        x_B=x_B,
        disturbance_onset_min=config.disturbance_onset_min,
        settling_band=config.settling_band,
        settling_window_min=config.settling_window_min,
    )

    # --- 5. MV activity ------------------------------------------------------
    mv_activity = float(np.sum(np.abs(np.diff(LT))) + np.sum(np.abs(np.diff(VB))))

    # --- 6. IAE --------------------------------------------------------------
    # Per-interval IAE with end-of-interval composition vs applied SP.
    iae_top = float(np.sum(np.abs(y_D[1:] - y_D_sp) * dt_intervals))
    iae_bottom = float(np.sum(np.abs(x_B[1:] - x_B_sp) * dt_intervals))
    iae = iae_top + iae_bottom

    return KPISet(
        specific_energy_kmol_per_kmol=specific_energy,
        light_yield=light_yield,
        constraint_violations=constraint_violations,
        settling_time_min=settling_time,
        mv_activity_kmol_per_min=mv_activity,
        iae_mole_fraction_min=iae,
    )


def _settling_time(
    *,
    t: np.ndarray,
    y_D: np.ndarray,
    x_B: np.ndarray,
    disturbance_onset_min: float,
    settling_band: float,
    settling_window_min: float,
) -> float:
    """Return the first time after onset at which both compositions are settled.

    "Settled" = each composition stays within ``settling_band`` of its
    post-disturbance final value (mean of the last 5 % of the
    trajectory) for at least ``settling_window_min`` continuous
    minutes. If no such moment exists before the trajectory ends,
    returns ``+inf``.
    """
    post = t >= disturbance_onset_min
    if not np.any(post):
        return float("inf")

    # Final value: mean of the last 5 % of the trajectory. Avoids
    # endpoint noise dominating, and decouples the metric from the
    # arbitrary horizon choice.
    tail_start = int(0.95 * len(t))
    y_D_final = float(np.mean(y_D[tail_start:]))
    x_B_final = float(np.mean(x_B[tail_start:]))

    inside_band = (np.abs(y_D - y_D_final) <= settling_band) & (
        np.abs(x_B - x_B_final) <= settling_band
    )
    # Limit the search to post-disturbance.
    inside_band[~post] = False

    # Find the earliest index from which inside_band stays True for at
    # least settling_window_min.
    for k in range(len(t)):
        if not inside_band[k]:
            continue
        # k is the first index inside the band. Check the window.
        window_end_time = t[k] + settling_window_min
        end_idx = int(np.searchsorted(t, window_end_time, side="right"))
        if end_idx >= len(t):
            # Window runs past the trajectory; not enough data to confirm.
            return float("inf")
        if np.all(inside_band[k:end_idx]):
            return float(t[k] - disturbance_onset_min)
    return float("inf")
