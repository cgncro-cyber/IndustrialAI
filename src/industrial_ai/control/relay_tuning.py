"""Åström-Hägglund relay-feedback auto-tuner for the Column A LV plant.

Replaces the controller in a single composition loop with a relay
(on/off action with optional hysteresis) and observes the steady
limit cycle that the relay-plus-plant system settles into. From the
limit-cycle peak-to-peak amplitude ``a`` of the measurement and the
relay amplitude ``d``, the ultimate gain follows the standard formula

    Ku = 4 d / (pi a)

and the ultimate period ``Pu`` is the limit-cycle period. The pair
``(Ku, Pu)`` then drives any subsequent tuning rule (Ziegler-Nichols,
Tyreus-Luyben, SIMC, ...). The level loops stay closed throughout via
:func:`assemble_inputs_lv`, so the tuner sees the *regulatory-closed*
plant — exactly what the C0 supervisory layer will drive.

Reference. Åström, K. J. and Hägglund, T. (1984). *Automatic Tuning
of Simple Regulators with Specifications on Phase and Amplitude
Margins.* Automatica 20(5), 645-651.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import numpy.typing as npt

from industrial_ai.twin.column_a.configurations.lv import (
    LVConfiguration,
    assemble_inputs_lv,
)
from industrial_ai.twin.column_a.integrator import integrate_open_loop
from industrial_ai.twin.column_a.parameters import (
    DEFAULT_PARAMETERS,
    ColumnAParameters,
)

__all__ = [
    "Loop",
    "RelayResult",
    "TyreusLuybenTuning",
    "relay_test",
    "tyreus_luyben",
]

StateVector = npt.NDArray[np.float64]
InputVector = npt.NDArray[np.float64]

Loop = Literal["top", "bottom"]


@dataclass(frozen=True, slots=True)
class RelayResult:
    """Outcome of a relay-feedback experiment on a single loop.

    Attributes
    ----------
    loop : {"top", "bottom"}
        Which composition loop the test was run on.
    Ku : float
        Ultimate gain ``4 d / (pi a)`` (kmol/min per mole fraction).
    Pu : float
        Ultimate period (min) extracted from the limit cycle.
    relay_amplitude_d : float
        The ``d`` used in the test (kmol/min half-swing around the bias).
    measurement_amplitude_a : float
        Half peak-to-peak amplitude of the controlled composition over
        the last analyzed cycles (mole fraction).
    t : numpy.ndarray
        Tick times (min).
    measurement : numpy.ndarray
        Controlled composition at each tick.
    mv : numpy.ndarray
        Relay output (manipulated variable) at each tick.
    setpoint : float
        Composition setpoint used as the relay switching point.
    """

    loop: Loop
    Ku: float
    Pu: float
    relay_amplitude_d: float
    measurement_amplitude_a: float
    t: npt.NDArray[np.float64]
    measurement: npt.NDArray[np.float64]
    mv: npt.NDArray[np.float64]
    setpoint: float


@dataclass(frozen=True, slots=True)
class TyreusLuybenTuning:
    """PI tuning derived from a relay test via Tyreus-Luyben.

    The Tyreus-Luyben rule (``Kp = Ku/3.2``, ``Ti = 2.2 Pu``) is
    deliberately more conservative than Ziegler-Nichols and is the
    standard recommendation for distillation composition loops where
    aggressive control can excite slow inverse-response modes
    (Skogestad & Postlethwaite 1996, §10).

    Attributes
    ----------
    Kp : float
    Ti : float
        Integral time (min); the integral gain ``Ki = Kp / Ti``.
    Ku, Pu : float
        Provenance — the relay-test pair this tuning was derived from.
    """

    Kp: float
    Ti: float
    Ku: float
    Pu: float


def tyreus_luyben(result: RelayResult) -> TyreusLuybenTuning:
    """Return Tyreus-Luyben PI parameters from a relay test result."""
    Kp = result.Ku / 3.2
    Ti = 2.2 * result.Pu
    return TyreusLuybenTuning(Kp=Kp, Ti=Ti, Ku=result.Ku, Pu=result.Pu)


def _bias_and_meas_index(
    *,
    loop: Loop,
    parameters: ColumnAParameters,
) -> tuple[float, float, int]:
    """Return ``(bias, sign, meas_index)`` for the given loop.

    The bias is the operating-point MV (LT or VB). The ``sign`` is
    +1 for direct-acting (top: increasing LT → increasing y_D) and -1
    for reverse-acting (bottom: increasing VB → decreasing x_B), so
    the relay logic flips the MV around the bias in the *correcting*
    direction relative to the measurement error.
    """
    NT = parameters.NT
    if loop == "top":
        return parameters.nominal_reflux_L0_kmol_per_min, +1.0, NT - 1
    return parameters.nominal_boilup_V0_kmol_per_min, -1.0, 0


def relay_test(
    *,
    loop: Loop,
    X0: StateVector,
    setpoint: float,
    relay_amplitude_d: float,
    duration_min: float = 500.0,
    tick_dt_min: float = 0.1,
    hysteresis: float = 1.0e-4,
    parameters: ColumnAParameters = DEFAULT_PARAMETERS,
    lv_config: LVConfiguration | None = None,
    F: float | None = None,
    zF: float = 0.5,
    qF: float | None = None,
    n_cycles_for_estimate: int = 3,
) -> RelayResult:
    """Run a relay-feedback experiment on the top or bottom composition loop.

    Parameters
    ----------
    loop : {"top", "bottom"}
        Which loop to test. Top manipulates ``LT`` to control ``y_D``
        (direct acting); bottom manipulates ``VB`` to control ``x_B``
        (reverse acting).
    X0 : numpy.ndarray of shape (2 * NT,)
        Initial state — typically the published steady state.
    setpoint : float
        Composition setpoint the relay switches around.
    relay_amplitude_d : float
        Half-amplitude of the MV swing (kmol/min). Typical 2-5 % of
        the operating-point flow.
    duration_min : float, optional
        Test duration. Default 500 min — enough for >~10 limit cycles
        on the canonical Column A.
    tick_dt_min : float, optional
        Sampling tick for the relay. Default 0.1 min = 6 s, fast
        enough to resolve the ultimate period without solver cost.
    hysteresis : float, optional
        Symmetric hysteresis band around the setpoint (mole fraction).
        Suppresses chatter from numerical noise; default 1e-4 is well
        below any expected limit-cycle amplitude on Column A.
    parameters : ColumnAParameters, optional
    lv_config : LVConfiguration, optional
        Level-loop closure tuning. Defaults to the cola_lv.m values.
    F, qF : float, optional
        Held-constant feed disturbances. Default to the published
        nominal values.
    zF : float, optional
        Held-constant feed composition. Default 0.5.
    n_cycles_for_estimate : int, optional
        Number of *trailing* full limit cycles to average for the
        ultimate-gain and ultimate-period estimates. Default 3.

    Returns
    -------
    RelayResult
    """
    if lv_config is None:
        lv_config = LVConfiguration()
    if F is None:
        F = parameters.nominal_feed_F_kmol_per_min
    if qF is None:
        qF = parameters.nominal_feed_liquid_fraction_qF

    bias, sign, meas_index = _bias_and_meas_index(loop=loop, parameters=parameters)
    other_bias = (
        parameters.nominal_boilup_V0_kmol_per_min
        if loop == "top"
        else parameters.nominal_reflux_L0_kmol_per_min
    )

    n_ticks = int(np.ceil(duration_min / tick_dt_min))
    t_axis = np.zeros(n_ticks + 1, dtype=np.float64)
    meas_axis = np.zeros(n_ticks + 1, dtype=np.float64)
    mv_axis = np.zeros(n_ticks, dtype=np.float64)

    X = X0.copy()
    meas_axis[0] = float(X[meas_index])
    relay_state = +1.0  # start positive; first tick will reassess

    for k in range(n_ticks):
        t_k = k * tick_dt_min
        t_next = min((k + 1) * tick_dt_min, duration_min)
        error = setpoint - float(X[meas_index])

        # Relay with hysteresis: only flip when error crosses outside
        # the hysteresis band on the correcting side.
        if relay_state > 0 and error < -hysteresis:
            relay_state = -1.0
        elif relay_state < 0 and error > hysteresis:
            relay_state = +1.0

        # Direct-acting (top): increase MV when measurement below sp.
        # Reverse-acting (bottom): decrease MV when measurement below sp.
        mv_value = bias + sign * relay_state * relay_amplitude_d

        if loop == "top":
            LT, VB = mv_value, other_bias
        else:
            LT, VB = other_bias, mv_value
        U = assemble_inputs_lv(
            state=X, LT=LT, VB=VB, F=F, zF=zF, qF=qF, config=lv_config, parameters=parameters
        )

        def _hold(t: float, X_: StateVector, _U: InputVector = U) -> InputVector:
            return _U

        result = integrate_open_loop(
            X0=X,
            t_span=(t_k, t_next),
            inputs_fn=_hold,
            parameters=parameters,
            t_eval=np.array([t_next], dtype=np.float64),
        )
        if not result.success or result.X.shape[0] == 0:
            raise RuntimeError(
                f"relay-test integration failed at tick {k} (t={t_k:.3f}): {result.message}"
            )
        X = result.X[-1]
        t_axis[k + 1] = t_next
        meas_axis[k + 1] = float(X[meas_index])
        mv_axis[k] = mv_value

    Ku, Pu, amplitude_a = _extract_ultimate_pair(
        t=t_axis,
        measurement=meas_axis,
        setpoint=setpoint,
        relay_amplitude_d=relay_amplitude_d,
        n_cycles=n_cycles_for_estimate,
    )
    return RelayResult(
        loop=loop,
        Ku=Ku,
        Pu=Pu,
        relay_amplitude_d=relay_amplitude_d,
        measurement_amplitude_a=amplitude_a,
        t=t_axis,
        measurement=meas_axis,
        mv=mv_axis,
        setpoint=setpoint,
    )


def _extract_ultimate_pair(
    *,
    t: npt.NDArray[np.float64],
    measurement: npt.NDArray[np.float64],
    setpoint: float,
    relay_amplitude_d: float,
    n_cycles: int,
) -> tuple[float, float, float]:
    """Estimate ``(Ku, Pu, a)`` from the trailing limit cycle.

    Strategy: detect zero-crossings of ``measurement - setpoint`` (i.e.
    every time the relay would switch), interpolate the crossing time
    linearly between samples. Define a *full cycle* as two consecutive
    crossings of the same direction (e.g., low -> high). Average
    period and peak-to-peak amplitude over the last ``n_cycles``
    full cycles.
    """
    deviation = measurement - setpoint
    crossings = _interpolated_zero_crossings(t, deviation)
    same_direction = crossings[::2]  # every other crossing → same direction
    if len(same_direction) < n_cycles + 1:
        raise RuntimeError(
            f"relay test did not reach a usable limit cycle: only "
            f"{len(same_direction)} same-direction crossings detected "
            f"(need {n_cycles + 1}). Increase duration_min or check the test setup."
        )

    last_cycle_times = same_direction[-(n_cycles + 1) :]
    periods = np.diff(last_cycle_times)
    Pu = float(np.mean(periods))

    # Amplitude: peak-to-peak measurement excursion over the trailing
    # window covered by the analyzed cycles.
    start_t = last_cycle_times[0]
    window = t >= start_t
    peak_to_peak = float(np.max(measurement[window]) - np.min(measurement[window]))
    amplitude_a = 0.5 * peak_to_peak
    if amplitude_a <= 0.0:
        raise RuntimeError("relay test produced zero amplitude — check setup")

    Ku = 4.0 * relay_amplitude_d / (np.pi * amplitude_a)
    return Ku, Pu, amplitude_a


def _interpolated_zero_crossings(
    t: npt.NDArray[np.float64],
    signal: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """Return the times at which ``signal`` crosses zero, linearly interpolated."""
    crossings: list[float] = []
    for i in range(len(signal) - 1):
        a, b = signal[i], signal[i + 1]
        if a == 0.0:
            crossings.append(float(t[i]))
        elif a * b < 0.0:
            frac = a / (a - b)
            crossings.append(float(t[i] + frac * (t[i + 1] - t[i])))
    return np.asarray(crossings, dtype=np.float64)
