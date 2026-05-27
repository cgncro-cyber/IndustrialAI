"""Tests for the Åström-Hägglund relay-feedback auto-tuner."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import pytest

from industrial_ai.control.relay_tuning import (
    RelayResult,
    relay_test,
    tyreus_luyben,
)
from industrial_ai.twin.column_a import DEFAULT_PARAMETERS


def _y_D(state: npt.NDArray[np.float64]) -> float:
    return float(state[DEFAULT_PARAMETERS.NT - 1])


def _x_B(state: npt.NDArray[np.float64]) -> float:
    return float(state[0])


def test_top_loop_relay_produces_usable_limit_cycle(
    skogestad_reference_state: npt.NDArray[np.float64],
) -> None:
    """Top-loop relay at canonical settings yields a sensible (Ku, Pu) pair."""
    result = relay_test(
        loop="top",
        X0=skogestad_reference_state,
        setpoint=_y_D(skogestad_reference_state),
        relay_amplitude_d=0.5,
        hysteresis=5.0e-3,
        duration_min=400.0,
    )
    assert isinstance(result, RelayResult)
    assert result.loop == "top"
    assert result.Ku > 0.0
    # Pu should engage the dominant composition mode (tau_2 ~ 12 min),
    # not the fast linearized-hydraulics tail (~1-2 min) which produces
    # non-physically aggressive gains.
    assert 5.0 < result.Pu < 30.0, (
        f"top-loop Pu = {result.Pu:.2f} min outside the expected dominant-mode band"
    )
    assert result.measurement_amplitude_a > 0.0
    assert result.t.shape == result.measurement.shape
    assert result.mv.shape[0] == result.t.shape[0] - 1


def test_bottom_loop_relay_produces_usable_limit_cycle(
    skogestad_reference_state: npt.NDArray[np.float64],
) -> None:
    """Bottom-loop relay (reverse-acting) also yields a sensible (Ku, Pu)."""
    result = relay_test(
        loop="bottom",
        X0=skogestad_reference_state,
        setpoint=_x_B(skogestad_reference_state),
        relay_amplitude_d=0.5,
        hysteresis=5.0e-3,
        duration_min=400.0,
    )
    assert result.loop == "bottom"
    assert result.Ku > 0.0
    assert 1.0 < result.Pu < 20.0, f"bottom-loop Pu = {result.Pu:.2f} min outside expected band"
    assert result.measurement_amplitude_a > 0.0


def test_tyreus_luyben_formula(
    skogestad_reference_state: npt.NDArray[np.float64],
) -> None:
    """tyreus_luyben must apply Kp = Ku/3.2, Ti = 2.2 Pu exactly."""
    result = relay_test(
        loop="top",
        X0=skogestad_reference_state,
        setpoint=_y_D(skogestad_reference_state),
        relay_amplitude_d=0.5,
        hysteresis=5.0e-3,
        duration_min=400.0,
    )
    tl = tyreus_luyben(result)
    assert tl.Kp == pytest.approx(result.Ku / 3.2, rel=1e-12)
    assert tl.Ti == pytest.approx(2.2 * result.Pu, rel=1e-12)
    # Provenance fields propagate.
    assert tl.Ku == result.Ku
    assert tl.Pu == result.Pu


def test_relay_mv_alternates_around_bias(
    skogestad_reference_state: npt.NDArray[np.float64],
) -> None:
    """The relay MV trajectory must contain both +d and -d swings around the bias."""
    p = DEFAULT_PARAMETERS
    d = 0.5
    result = relay_test(
        loop="top",
        X0=skogestad_reference_state,
        setpoint=_y_D(skogestad_reference_state),
        relay_amplitude_d=d,
        hysteresis=5.0e-3,
        duration_min=200.0,
    )
    expected_high = p.nominal_reflux_L0_kmol_per_min + d
    expected_low = p.nominal_reflux_L0_kmol_per_min - d
    # MV trajectory should contain values close to both extremes.
    assert np.any(np.isclose(result.mv, expected_high, atol=1e-9))
    assert np.any(np.isclose(result.mv, expected_low, atol=1e-9))
    # No values outside the expected pair (no other MV setting in the test).
    assert np.all(
        np.isclose(result.mv, expected_high, atol=1e-9)
        | np.isclose(result.mv, expected_low, atol=1e-9)
    )


def test_relay_raises_on_too_short_duration(
    skogestad_reference_state: npt.NDArray[np.float64],
) -> None:
    """A test cut off before any limit cycle develops must fail loudly, not silently."""
    with pytest.raises(RuntimeError, match="did not reach a usable limit cycle"):
        relay_test(
            loop="top",
            X0=skogestad_reference_state,
            setpoint=_y_D(skogestad_reference_state),
            relay_amplitude_d=0.5,
            hysteresis=5.0e-3,
            duration_min=2.0,  # too short — fewer than 3 same-direction crossings
        )
