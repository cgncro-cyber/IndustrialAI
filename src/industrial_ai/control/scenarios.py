"""Canonical five-disturbance scenario set for Phase-2 baseline benchmarks.

Each factory in this module returns a :class:`ScenarioFn` ready to
feed into
:func:`industrial_ai.twin.simulate.simulate_lv_closed_loop`. Every
scenario starts at the Skogestad nominal operating point (``F = 1``
kmol/min, ``zF = 0.5``, ``qF = 1``, supervisor setpoints
``y_D = 0.99`` / ``x_B = 0.01``) and applies a single step
disturbance or setpoint change at ``onset_min = 5`` min. The
post-step horizon is intentionally fixed at 60 min so the same time
grid scores every controller fairly.

Five scenarios per the Phase 2 deliverables list:

1. ``F_step_+20pct`` — feed-flow disturbance, F goes from 1.0 to 1.2.
2. ``F_step_-20pct`` — feed-flow disturbance, F goes from 1.0 to 0.8.
3. ``zF_step_+10pct`` — feed-composition disturbance, zF: 0.5 to 0.55.
4. ``zF_step_-10pct`` — feed-composition disturbance, zF: 0.5 to 0.45.
5. ``yD_setpoint_+0p5pct`` — supervisory setpoint change,
   y_D: 0.99 to 0.995.

A 60-min horizon is short relative to the slow composition mode
(``tau_1`` ~ 194 min) but covers several settling windows of the
dominant fast mode (``tau_2`` ~ 12 min) and matches the disturbance
windows referenced in ``docs/figures.md`` Figure 3.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

from industrial_ai.twin.column_a import DEFAULT_PARAMETERS
from industrial_ai.twin.simulate import ScenarioFn, ScenarioStep

__all__ = [
    "DEFAULT_HORIZON_MIN",
    "DEFAULT_ONSET_MIN",
    "DEFAULT_TICK_DT_MIN",
    "SCENARIO_NAMES",
    "ScenarioSpec",
    "build_scenario",
    "build_scenarios",
]


DEFAULT_HORIZON_MIN: Final[float] = 60.0
DEFAULT_ONSET_MIN: Final[float] = 5.0
DEFAULT_TICK_DT_MIN: Final[float] = 0.05


SCENARIO_NAMES: Final[tuple[str, ...]] = (
    "F_step_+20pct",
    "F_step_-20pct",
    "zF_step_+10pct",
    "zF_step_-10pct",
    "yD_setpoint_+0p5pct",
)


@dataclass(frozen=True, slots=True)
class ScenarioSpec:
    """Static metadata describing one disturbance scenario.

    Attributes
    ----------
    name : str
        Short identifier used in run-directory paths and notebook plots.
    field : str
        Which field of :class:`ScenarioStep` is stepped — one of
        ``"F"``, ``"zF"``, or ``"y_D_setpoint"``.
    pre_step_value : float
        Value of ``field`` before the step. Always the nominal
        operating-point value.
    post_step_value : float
        Value of ``field`` after the step.
    onset_min : float
        Tick time (min) at which the step is applied.
    horizon_min : float
        Total simulation duration (min) for the scenario.
    """

    name: str
    field: str
    pre_step_value: float
    post_step_value: float
    onset_min: float = DEFAULT_ONSET_MIN
    horizon_min: float = DEFAULT_HORIZON_MIN


def _nominal_step() -> ScenarioStep:
    p = DEFAULT_PARAMETERS
    return ScenarioStep(
        y_D_setpoint=0.99,
        x_B_setpoint=0.01,
        F=p.nominal_feed_F_kmol_per_min,
        zF=0.5,
        qF=p.nominal_feed_liquid_fraction_qF,
    )


_SCENARIO_SPECS: Final[dict[str, ScenarioSpec]] = {
    "F_step_+20pct": ScenarioSpec(
        name="F_step_+20pct",
        field="F",
        pre_step_value=DEFAULT_PARAMETERS.nominal_feed_F_kmol_per_min,
        post_step_value=1.2 * DEFAULT_PARAMETERS.nominal_feed_F_kmol_per_min,
    ),
    "F_step_-20pct": ScenarioSpec(
        name="F_step_-20pct",
        field="F",
        pre_step_value=DEFAULT_PARAMETERS.nominal_feed_F_kmol_per_min,
        post_step_value=0.8 * DEFAULT_PARAMETERS.nominal_feed_F_kmol_per_min,
    ),
    "zF_step_+10pct": ScenarioSpec(
        name="zF_step_+10pct",
        field="zF",
        pre_step_value=0.5,
        post_step_value=0.55,
    ),
    "zF_step_-10pct": ScenarioSpec(
        name="zF_step_-10pct",
        field="zF",
        pre_step_value=0.5,
        post_step_value=0.45,
    ),
    "yD_setpoint_+0p5pct": ScenarioSpec(
        name="yD_setpoint_+0p5pct",
        field="y_D_setpoint",
        pre_step_value=0.99,
        post_step_value=0.995,
    ),
}


def build_scenario(name: str) -> tuple[ScenarioFn, ScenarioSpec]:
    """Return ``(scenario_fn, spec)`` for one of the canonical 5 scenarios.

    Parameters
    ----------
    name : str
        One of the entries in :data:`SCENARIO_NAMES`.

    Returns
    -------
    tuple of (ScenarioFn, ScenarioSpec)
        Closure ready for :func:`simulate_lv_closed_loop` plus the
        static metadata in :class:`ScenarioSpec`.

    Raises
    ------
    KeyError
        If ``name`` is not one of the canonical scenarios.
    """
    if name not in _SCENARIO_SPECS:
        raise KeyError(f"unknown scenario {name!r}; available: {sorted(_SCENARIO_SPECS)}")
    spec = _SCENARIO_SPECS[name]
    base = _nominal_step()

    def scenario(t: float) -> ScenarioStep:
        if t < spec.onset_min:
            return base
        return _replace_field(base, spec.field, spec.post_step_value)

    return scenario, spec


def build_scenarios() -> dict[str, tuple[ScenarioFn, ScenarioSpec]]:
    """Return the full mapping ``{name -> (scenario_fn, spec)}`` for all five."""
    return {name: build_scenario(name) for name in SCENARIO_NAMES}


def _replace_field(base: ScenarioStep, field: str, value: float) -> ScenarioStep:
    """Return a copy of ``base`` with ``field`` set to ``value``."""
    if field == "F":
        return ScenarioStep(
            y_D_setpoint=base.y_D_setpoint,
            x_B_setpoint=base.x_B_setpoint,
            F=value,
            zF=base.zF,
            qF=base.qF,
        )
    if field == "zF":
        return ScenarioStep(
            y_D_setpoint=base.y_D_setpoint,
            x_B_setpoint=base.x_B_setpoint,
            F=base.F,
            zF=value,
            qF=base.qF,
        )
    if field == "y_D_setpoint":
        return ScenarioStep(
            y_D_setpoint=value,
            x_B_setpoint=base.x_B_setpoint,
            F=base.F,
            zF=base.zF,
            qF=base.qF,
        )
    raise ValueError(f"unsupported scenario field {field!r}")
