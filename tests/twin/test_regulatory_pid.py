"""Tests for the regulatory PID controller."""

from __future__ import annotations

import pytest

from industrial_ai.twin.regulatory_pid import PIDController


def test_proportional_action_only() -> None:
    pid = PIDController(Kp=2.0)
    out = pid.step(measurement=0.0, setpoint=1.0, dt=0.1)
    # error = 1, Kp = 2, no integral / derivative on the first step.
    assert out == pytest.approx(2.0)


def test_integral_action_accumulates() -> None:
    pid = PIDController(Kp=0.0, Ki=1.0)
    out_a = pid.step(measurement=0.0, setpoint=1.0, dt=1.0)
    out_b = pid.step(measurement=0.0, setpoint=1.0, dt=1.0)
    # Integral builds up over time even though Kp = 0.
    assert out_b > out_a
    assert out_b == pytest.approx(1.0, abs=1e-9)


def test_saturation_clamps_output() -> None:
    pid = PIDController(Kp=10.0, output_min=-1.0, output_max=1.0)
    out_hi = pid.step(measurement=0.0, setpoint=100.0, dt=0.1)
    out_lo = pid.step(measurement=0.0, setpoint=-100.0, dt=0.1)
    assert out_hi == 1.0
    assert out_lo == -1.0


def test_anti_windup_prevents_runaway_integral() -> None:
    """Saturated PID should not accumulate runaway integral state.

    Conditional integration freezes the integrator while the output is
    saturated, so after many saturated steps the integral remains at
    its pre-saturation value (zero here).
    """
    pid = PIDController(Kp=1.0, Ki=10.0, output_min=-1.0, output_max=1.0)
    for _ in range(20):
        out = pid.step(measurement=0.0, setpoint=100.0, dt=0.1)
        assert out == 1.0, "output must stay at the upper saturation"
    # No windup: the integrator did not accumulate while saturated.
    assert pid.state.integral == 0.0


def test_anti_windup_releases_when_error_reverses() -> None:
    """After the setpoint reverses, the controller must escape saturation quickly."""
    pid = PIDController(Kp=1.0, Ki=10.0, output_min=-1.0, output_max=1.0)
    # Drive the controller into upper saturation with a positive setpoint.
    for _ in range(10):
        pid.step(measurement=0.0, setpoint=100.0, dt=0.1)
    # Now reverse the setpoint. Because no windup was accumulated, the
    # controller releases to the lower saturation on the very first step.
    out = pid.step(measurement=0.0, setpoint=-100.0, dt=0.1)
    assert out == -1.0


def test_reset_clears_state() -> None:
    pid = PIDController(Kp=1.0, Ki=1.0)
    pid.step(measurement=0.0, setpoint=1.0, dt=1.0)
    pid.reset()
    assert pid.state.integral == 0.0
    assert pid.state.previous_error == 0.0


def test_reverse_acting_sign_flip() -> None:
    """Reverse-acting controller responds with opposite sign."""
    pid_direct = PIDController(Kp=1.0, direct_acting=True)
    pid_reverse = PIDController(Kp=1.0, direct_acting=False)
    out_d = pid_direct.step(measurement=0.0, setpoint=1.0, dt=0.1)
    out_r = pid_reverse.step(measurement=0.0, setpoint=1.0, dt=0.1)
    assert out_d == -out_r


def test_nonpositive_dt_raises() -> None:
    pid = PIDController(Kp=1.0)
    with pytest.raises(ValueError, match="dt must be strictly positive"):
        pid.step(measurement=0.0, setpoint=1.0, dt=0.0)
