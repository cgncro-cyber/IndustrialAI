"""Off-nominal scenario builders for the kpis.md §2.2 evaluation grid.

The canonical five disturbance scenarios in :mod:`scenarios` are
defined at the nominal operating point (F=1.0, zF=0.5). For the
Phase-3 off-nominal robustness KPI (``docs/kpis.md`` §2.2), the
same five scenarios are evaluated at 16 off-nominal OPs:

    G = { (F, zF) :  F ∈ {0.8, 0.9, 1.1, 1.2},
                     zF ∈ {0.45, 0.475, 0.525, 0.55} }

At each off-nominal OP the column starts in the LV-closed steady
state produced by :func:`lookup_lv_ss`. The scenario then applies
its disturbance *event* with the same relative magnitude as the
canonical scenario, anchored at the OP:

    F_step_+20pct  : F = F_op × 1.2 after the onset
    F_step_-20pct  : F = F_op × 0.8 after the onset
    zF_step_+10pct : zF = zF_op + 0.05 after the onset
    zF_step_-10pct : zF = zF_op - 0.05 after the onset
    yD_setpoint_+0p5pct : y_D_target steps from 0.99 to 0.995

The product specifications y_D_target=0.99 and x_B_target=0.01 are
held constant across all OPs — they are intrinsic to the column,
not the OP. At off-nominal OPs the LV-closed SS y_D/x_B differ
markedly from these targets, so the supervisory layer's task at
t=0 includes closing the gap from the off-nominal SS to the
nominal targets in addition to absorbing the scenario event.

The scenario function returned here is the same callable shape as
:func:`industrial_ai.control.scenarios.build_scenario`: a closure
``(t) -> ScenarioStep``. It can be passed to
:func:`industrial_ai.twin.simulate.simulate_lv_closed_loop` or
:func:`industrial_ai.control.c1_linear_mpc.simulate_lv_with_mpc`
without modification.

Reference. ``docs/kpis.md`` §2.2 (off-nominal grid definition).
"""

from __future__ import annotations

from industrial_ai.control.scenarios import SCENARIO_NAMES, ScenarioSpec
from industrial_ai.twin.column_a.parameters import DEFAULT_PARAMETERS
from industrial_ai.twin.simulate import ScenarioFn, ScenarioStep

__all__ = [
    "build_off_nominal_scenario",
]


_DEFAULT_ONSET_MIN = 5.0
_DEFAULT_HORIZON_MIN = 240.0


def build_off_nominal_scenario(
    name: str,
    *,
    F_op: float,
    zF_op: float,
    qF_op: float | None = None,
    onset_min: float = _DEFAULT_ONSET_MIN,
    horizon_min: float = _DEFAULT_HORIZON_MIN,
) -> tuple[ScenarioFn, ScenarioSpec]:
    """Return ``(scenario_fn, spec)`` for a canonical scenario at an off-nominal OP.

    Parameters
    ----------
    name : str
        One of the entries in
        :data:`industrial_ai.control.scenarios.SCENARIO_NAMES`.
    F_op : float
        Operating-point feed flow (kmol/min). Pre-step F equals
        ``F_op``; post-step F follows the canonical relative step.
    zF_op : float
        Operating-point feed composition (mole fraction). Pre-step
        zF equals ``zF_op``; post-step zF follows the canonical
        +/-0.05 absolute step.
    qF_op : float, optional
        Operating-point feed liquid fraction. Defaults to the
        nominal value.
    onset_min : float, optional
        Tick time at which the step is applied.
    horizon_min : float, optional
        Simulation horizon.

    Returns
    -------
    tuple of (ScenarioFn, ScenarioSpec)

    Raises
    ------
    KeyError
        If ``name`` is not one of the canonical scenarios.
    """
    if name not in SCENARIO_NAMES:
        raise KeyError(f"unknown scenario {name!r}; available: {sorted(SCENARIO_NAMES)}")
    p = DEFAULT_PARAMETERS
    qF = p.nominal_feed_liquid_fraction_qF if qF_op is None else qF_op

    base = ScenarioStep(
        y_D_setpoint=0.99,
        x_B_setpoint=0.01,
        F=F_op,
        zF=zF_op,
        qF=qF,
    )

    if name == "F_step_+20pct":
        field = "F"
        pre_value = F_op
        post_value = 1.2 * F_op
    elif name == "F_step_-20pct":
        field = "F"
        pre_value = F_op
        post_value = 0.8 * F_op
    elif name == "zF_step_+10pct":
        field = "zF"
        pre_value = zF_op
        post_value = zF_op + 0.05
    elif name == "zF_step_-10pct":
        field = "zF"
        pre_value = zF_op
        post_value = zF_op - 0.05
    elif name == "yD_setpoint_+0p5pct":
        field = "y_D_setpoint"
        pre_value = 0.99
        post_value = 0.995
    else:
        raise KeyError(f"unsupported scenario {name!r}")

    spec = ScenarioSpec(
        name=name,
        field=field,
        pre_step_value=pre_value,
        post_step_value=post_value,
        onset_min=onset_min,
        horizon_min=horizon_min,
    )

    def _stepped(field_name: str, value: float) -> ScenarioStep:
        if field_name == "F":
            return ScenarioStep(
                y_D_setpoint=base.y_D_setpoint,
                x_B_setpoint=base.x_B_setpoint,
                F=value,
                zF=base.zF,
                qF=base.qF,
            )
        if field_name == "zF":
            return ScenarioStep(
                y_D_setpoint=base.y_D_setpoint,
                x_B_setpoint=base.x_B_setpoint,
                F=base.F,
                zF=value,
                qF=base.qF,
            )
        if field_name == "y_D_setpoint":
            return ScenarioStep(
                y_D_setpoint=value,
                x_B_setpoint=base.x_B_setpoint,
                F=base.F,
                zF=base.zF,
                qF=base.qF,
            )
        raise ValueError(f"unsupported field {field_name!r}")

    def scenario(t: float) -> ScenarioStep:
        if t < onset_min:
            return base
        return _stepped(field, post_value)

    return scenario, spec
