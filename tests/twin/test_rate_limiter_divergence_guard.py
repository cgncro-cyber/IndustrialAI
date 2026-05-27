"""Rate-limiter divergence-guard tests.

Phase 1 gate item: *"Setpoint rate-limiter prevents solver divergence
on +/-20 % step changes."* These tests put a +/-20 % step on the
reflux LT through a slew-rate limiter and verify the integration
stays bounded all the way to a re-converged steady state.

The continuous-time limit of :class:`industrial_ai.twin.setpoint_interface.RateLimiter`
is a piecewise-linear ramp; for the solver-divergence test it is
equivalent (and convenient) to evaluate that ramp analytically as a
function of ``t``. The discrete :class:`RateLimiter` is exercised
unit-style in ``test_setpoint_interface.py``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import pytest

from industrial_ai.twin.column_a import (
    DEFAULT_PARAMETERS,
    compute_steady_state_by_newton,
    integrate_open_loop,
)
from industrial_ai.twin.column_a.configurations.lv import assemble_inputs_lv


@dataclass(frozen=True, slots=True)
class _SlewRamp:
    """Analytic piecewise-linear ramp matching the continuous limit of RateLimiter."""

    initial: float
    target: float
    max_rate: float

    def at(self, t: float) -> float:
        if t <= 0.0:
            return self.initial
        delta = self.target - self.initial
        ramp_time = abs(delta) / self.max_rate if self.max_rate > 0 else 0.0
        if t >= ramp_time:
            return self.target
        return self.initial + float(np.sign(delta)) * self.max_rate * t


def _close_lv_inputs_fn(
    *,
    lt_ramp: _SlewRamp,
    VB: float,
    F: float,
    zF: float,
    qF: float,
) -> npt.NDArray[np.float64]:
    """Return an ``inputs_fn`` that uses the ramp for LT and the LV closure."""

    def fn(t: float, X: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        return assemble_inputs_lv(state=X, LT=lt_ramp.at(t), VB=VB, F=F, zF=zF, qF=qF)

    return fn


@pytest.mark.parametrize("step_sign", [+1.0, -1.0])
def test_lv_with_rate_limited_lt_step_does_not_diverge(
    skogestad_reference_state: npt.NDArray[np.float64],
    step_sign: float,
) -> None:
    """A +/-20 % step on LT, rate-limited at 0.1 kmol/min^2, must integrate to bounded SS."""
    p = DEFAULT_PARAMETERS
    lt_initial = p.nominal_reflux_L0_kmol_per_min
    lt_target = lt_initial * (1.0 + 0.20 * step_sign)

    ramp = _SlewRamp(initial=lt_initial, target=lt_target, max_rate=0.1)
    inputs_fn = _close_lv_inputs_fn(
        lt_ramp=ramp,
        VB=p.nominal_boilup_V0_kmol_per_min,
        F=p.nominal_feed_F_kmol_per_min,
        zF=0.5,
        qF=p.nominal_feed_liquid_fraction_qF,
    )

    # Integrate well past the ramp end (~5.4 min) and the column's slow
    # composition dynamics (tau_1 ~ 194 min per Skogestad 1997 Eq. 31).
    result = integrate_open_loop(
        X0=skogestad_reference_state,
        t_span=(0.0, 1500.0),
        inputs_fn=inputs_fn,
    )

    assert result.success, f"integrator failed: {result.message}"
    assert np.all(np.isfinite(result.X)), "state contains NaN/Inf — solver diverged"
    compositions = result.X[:, : p.NT]
    assert compositions.min() >= -1e-9
    assert compositions.max() <= 1.0 + 1e-9
    holdups = result.X[:, p.NT :]
    assert holdups.min() > 0.0


def test_rate_limited_step_reaches_newton_steady_state(
    skogestad_reference_state: npt.NDArray[np.float64],
    skogestad_reference_inputs: npt.NDArray[np.float64],
) -> None:
    """Long-time integration under a ramped +20 % LT step lands at the Newton SS at the same final inputs."""
    p = DEFAULT_PARAMETERS
    lt_initial = p.nominal_reflux_L0_kmol_per_min
    lt_target = lt_initial * 1.20

    ramp = _SlewRamp(initial=lt_initial, target=lt_target, max_rate=0.1)
    inputs_fn = _close_lv_inputs_fn(
        lt_ramp=ramp,
        VB=p.nominal_boilup_V0_kmol_per_min,
        F=p.nominal_feed_F_kmol_per_min,
        zF=0.5,
        qF=p.nominal_feed_liquid_fraction_qF,
    )
    result = integrate_open_loop(
        X0=skogestad_reference_state,
        t_span=(0.0, 3000.0),
        inputs_fn=inputs_fn,
    )
    assert result.success, result.message

    # Independently solve the SS at LT = 1.20 * L0 with the closed LV
    # loops (D and B come from the integrator's final holdups).
    final = result.X[-1]
    MB = final[p.NT]
    MD = final[2 * p.NT - 1]
    inputs_at_final = skogestad_reference_inputs.copy()
    inputs_at_final[0] = lt_target
    inputs_at_final[2] = 0.5 + 10.0 * (MD - 0.5)
    inputs_at_final[3] = 0.5 + 10.0 * (MB - 0.5)
    ss = compute_steady_state_by_newton(X0=final, inputs=inputs_at_final)
    assert ss.success, ss.message

    drift = float(np.max(np.abs(final - ss.X)))
    assert drift < 5.0e-4, (
        f"integration ended {drift:.3e} away from the Newton SS — solver may have stalled"
    )
