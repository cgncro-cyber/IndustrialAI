"""Tests for the canonical 5 Phase-2 disturbance scenarios."""

from __future__ import annotations

import pytest

from industrial_ai.control.scenarios import (
    DEFAULT_ONSET_MIN,
    SCENARIO_NAMES,
    build_scenario,
    build_scenarios,
)
from industrial_ai.twin.column_a import DEFAULT_PARAMETERS


def test_all_five_scenarios_available() -> None:
    """build_scenarios returns one entry per canonical name."""
    scenarios = build_scenarios()
    assert set(scenarios) == set(SCENARIO_NAMES)
    assert len(scenarios) == 5


def test_unknown_scenario_raises() -> None:
    """build_scenario fails fast with a clear message on a bad name."""
    with pytest.raises(KeyError, match="unknown scenario"):
        build_scenario("bogus")


def test_pre_step_returns_nominal_operating_point() -> None:
    """Before onset, every scenario returns the Skogestad nominal step."""
    p = DEFAULT_PARAMETERS
    nominal_feed = p.nominal_feed_F_kmol_per_min
    nominal_q = p.nominal_feed_liquid_fraction_qF
    for name in SCENARIO_NAMES:
        scenario, _spec = build_scenario(name)
        step = scenario(0.0)
        assert step.y_D_setpoint == pytest.approx(0.99), f"{name}: pre-step y_D_setpoint"
        assert step.x_B_setpoint == pytest.approx(0.01)
        assert step.F == pytest.approx(nominal_feed)
        assert step.zF == pytest.approx(0.5)
        assert step.qF == pytest.approx(nominal_q)


@pytest.mark.parametrize(
    ("name", "field", "expected"),
    [
        ("F_step_+20pct", "F", 1.2 * DEFAULT_PARAMETERS.nominal_feed_F_kmol_per_min),
        ("F_step_-20pct", "F", 0.8 * DEFAULT_PARAMETERS.nominal_feed_F_kmol_per_min),
        ("zF_step_+10pct", "zF", 0.55),
        ("zF_step_-10pct", "zF", 0.45),
        ("yD_setpoint_+0p5pct", "y_D_setpoint", 0.995),
    ],
)
def test_post_step_value_matches_spec(name: str, field: str, expected: float) -> None:
    """After onset, the stepped field equals the spec's post-step value."""
    scenario, spec = build_scenario(name)
    post_step = scenario(spec.onset_min + 1.0)
    assert getattr(post_step, field) == pytest.approx(expected, rel=1e-12)


def test_step_is_exact_at_onset() -> None:
    """At t < onset the step has not happened; at t >= onset it has."""
    scenario, spec = build_scenario("F_step_+20pct")
    nominal_feed = DEFAULT_PARAMETERS.nominal_feed_F_kmol_per_min
    just_before = scenario(spec.onset_min - 1e-9)
    just_after = scenario(spec.onset_min)
    assert just_before.F == pytest.approx(nominal_feed)
    assert just_after.F > nominal_feed * 1.01


def test_non_stepped_fields_unchanged_after_onset() -> None:
    """An F-step must not perturb zF, qF, or the supervisor setpoints."""
    scenario, spec = build_scenario("F_step_+20pct")
    pre = scenario(0.0)
    post = scenario(spec.onset_min + 1.0)
    assert pre.zF == post.zF
    assert pre.qF == post.qF
    assert pre.y_D_setpoint == post.y_D_setpoint
    assert pre.x_B_setpoint == post.x_B_setpoint


def test_default_onset_is_five_minutes() -> None:
    """The 5-min onset is locked across all scenarios — KPI settling-time
    computation depends on it.
    """
    assert DEFAULT_ONSET_MIN == 5.0
    for name in SCENARIO_NAMES:
        _, spec = build_scenario(name)
        assert spec.onset_min == DEFAULT_ONSET_MIN


# ---------------------------------------------------------------------------
# build_scenario_at_op — off-nominal screening pass
# ---------------------------------------------------------------------------


def test_build_scenario_at_op_F_step_uses_op_as_base() -> None:
    """F_step_+20pct at OP F=0.8 should produce post = 1.2 * 0.8 = 0.96, not 1.2."""
    from industrial_ai.control.scenarios import build_scenario_at_op

    sc, spec = build_scenario_at_op("F_step_+20pct", op_F=0.8, op_zF=0.45)
    pre = sc(0.0)
    post = sc(10.0)
    assert pre.F == 0.8
    assert post.F == pytest.approx(0.96)
    # zF unchanged.
    assert pre.zF == 0.45 and post.zF == 0.45
    # Spec reflects the post value.
    assert spec.post_step_value == pytest.approx(0.96)


def test_build_scenario_at_op_zF_step_multiplicative() -> None:
    """zF_step_+10pct at OP zF=0.45 should produce post = 1.1 * 0.45 = 0.495."""
    from industrial_ai.control.scenarios import build_scenario_at_op

    sc, _ = build_scenario_at_op("zF_step_+10pct", op_F=0.8, op_zF=0.45)
    pre = sc(0.0)
    post = sc(10.0)
    assert pre.zF == 0.45
    assert post.zF == pytest.approx(0.495)


def test_build_scenario_at_op_yD_setpoint_unchanged_by_op() -> None:
    """y_D setpoint scenario is operator-spec relative, NOT OP-relative."""
    from industrial_ai.control.scenarios import build_scenario_at_op

    sc, _ = build_scenario_at_op("yD_setpoint_+0p5pct", op_F=0.8, op_zF=0.45)
    pre = sc(0.0)
    post = sc(10.0)
    assert pre.y_D_setpoint == 0.99
    assert post.y_D_setpoint == pytest.approx(0.995)


def test_build_scenario_at_op_F_negative_step() -> None:
    """F_step_-20pct at OP F=1.2 should produce post = 0.8 * 1.2 = 0.96."""
    from industrial_ai.control.scenarios import build_scenario_at_op

    sc, _ = build_scenario_at_op("F_step_-20pct", op_F=1.2, op_zF=0.55)
    assert sc(10.0).F == pytest.approx(0.96)
