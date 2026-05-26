"""Tests for the setpoint interface and rate limiter."""

from __future__ import annotations

import pytest

from industrial_ai.twin.setpoint_interface import RateLimiter, SetpointInterface


def test_rate_limiter_clips_step_change() -> None:
    rl = RateLimiter(max_rate=1.0, current=0.0)
    # Request a +10 jump over 1 min; max allowed is +1.
    assert rl.update(requested=10.0, dt=1.0) == 1.0
    assert rl.current == 1.0


def test_rate_limiter_passes_small_change() -> None:
    rl = RateLimiter(max_rate=10.0, current=0.0)
    assert rl.update(requested=0.5, dt=1.0) == 0.5


def test_rate_limiter_symmetric_for_negative_steps() -> None:
    rl = RateLimiter(max_rate=1.0, current=0.0)
    assert rl.update(requested=-5.0, dt=1.0) == -1.0


def test_rate_limiter_dt_scales_allowed_step() -> None:
    rl = RateLimiter(max_rate=1.0, current=0.0)
    # Over 0.5 min the max allowed step is 0.5.
    assert rl.update(requested=10.0, dt=0.5) == 0.5


def test_rate_limiter_seed_jumps_state() -> None:
    rl = RateLimiter(max_rate=1.0, current=0.0)
    rl.seed(7.0)
    assert rl.current == 7.0


def test_rate_limiter_rejects_nonpositive_rate_and_dt() -> None:
    with pytest.raises(ValueError, match="max_rate"):
        RateLimiter(max_rate=0.0, current=0.0).update(requested=1.0, dt=1.0)
    with pytest.raises(ValueError, match="dt"):
        RateLimiter(max_rate=1.0, current=0.0).update(requested=1.0, dt=0.0)


def test_setpoint_interface_multichannel() -> None:
    iface = SetpointInterface()
    iface.register(name="yD_setpoint", max_rate=0.001, initial=0.99)
    iface.register(name="xB_setpoint", max_rate=0.001, initial=0.01)
    applied = iface.apply(
        requested={"yD_setpoint": 0.995, "xB_setpoint": 0.005},
        dt=1.0,
    )
    # Both channels rate-limited symmetrically.
    assert applied["yD_setpoint"] == pytest.approx(0.991)
    assert applied["xB_setpoint"] == pytest.approx(0.009)


def test_setpoint_interface_rejects_duplicate_registration() -> None:
    iface = SetpointInterface()
    iface.register(name="a", max_rate=1.0, initial=0.0)
    with pytest.raises(ValueError, match="already registered"):
        iface.register(name="a", max_rate=1.0, initial=0.0)


def test_setpoint_interface_rejects_unknown_channel() -> None:
    iface = SetpointInterface()
    iface.register(name="a", max_rate=1.0, initial=0.0)
    with pytest.raises(KeyError, match="unregistered"):
        iface.apply(requested={"unknown": 1.0}, dt=1.0)
