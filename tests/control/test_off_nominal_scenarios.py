"""Tests for the off-nominal scenario builder."""

from __future__ import annotations

import pytest

from industrial_ai.control.off_nominal_scenarios import build_off_nominal_scenario
from industrial_ai.control.scenarios import SCENARIO_NAMES


def test_pre_step_anchored_at_op() -> None:
    """Pre-step values must equal the OP, not the nominal."""
    fn, _ = build_off_nominal_scenario("F_step_+20pct", F_op=0.8, zF_op=0.45)
    step = fn(0.0)
    assert step.F == pytest.approx(0.8)
    assert step.zF == pytest.approx(0.45)


def test_F_step_uses_relative_magnitude() -> None:
    """F_step_+20pct from F_op=0.8 lands at 0.96 (×1.2), not at 1.2."""
    fn_plus, _ = build_off_nominal_scenario("F_step_+20pct", F_op=0.8, zF_op=0.5)
    fn_minus, _ = build_off_nominal_scenario("F_step_-20pct", F_op=1.2, zF_op=0.5)
    assert fn_plus(60.0).F == pytest.approx(0.96)
    assert fn_minus(60.0).F == pytest.approx(0.96)


def test_zF_step_uses_absolute_magnitude() -> None:
    """zF_step_+10pct adds +0.05 to zF_op (so 0.45 -> 0.50, not 0.45 -> 0.55)."""
    fn_plus, _ = build_off_nominal_scenario("zF_step_+10pct", F_op=1.0, zF_op=0.45)
    fn_minus, _ = build_off_nominal_scenario("zF_step_-10pct", F_op=1.0, zF_op=0.55)
    assert fn_plus(60.0).zF == pytest.approx(0.50)
    assert fn_minus(60.0).zF == pytest.approx(0.50)


def test_yD_setpoint_is_op_invariant() -> None:
    """y_D setpoint scenario tracks the product spec, not the OP."""
    fn, _ = build_off_nominal_scenario("yD_setpoint_+0p5pct", F_op=0.8, zF_op=0.45)
    pre = fn(0.0)
    post = fn(60.0)
    assert pre.y_D_setpoint == pytest.approx(0.99)
    assert post.y_D_setpoint == pytest.approx(0.995)
    # F, zF unchanged by setpoint scenario:
    assert post.F == pytest.approx(0.8)
    assert post.zF == pytest.approx(0.45)


def test_targets_are_product_specs_not_op_dependent() -> None:
    """y_D_target=0.99 and x_B_target=0.01 hold across all OPs by design."""
    for F_op in (0.8, 0.9, 1.1, 1.2):
        for zF_op in (0.45, 0.475, 0.525, 0.55):
            fn, _ = build_off_nominal_scenario("F_step_+20pct", F_op=F_op, zF_op=zF_op)
            step = fn(0.0)
            assert step.y_D_setpoint == pytest.approx(0.99)
            assert step.x_B_setpoint == pytest.approx(0.01)


def test_unknown_scenario_raises() -> None:
    with pytest.raises(KeyError):
        build_off_nominal_scenario("not_a_real_scenario", F_op=1.0, zF_op=0.5)


def test_all_canonical_names_supported() -> None:
    """Every entry in SCENARIO_NAMES must be buildable off-nominal."""
    for name in SCENARIO_NAMES:
        fn, spec = build_off_nominal_scenario(name, F_op=0.9, zF_op=0.475)
        # Smoke: the closure is callable and produces a ScenarioStep.
        step = fn(0.0)
        assert step.F is not None
        assert spec.name == name
