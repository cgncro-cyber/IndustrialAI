"""LV closed-loop simulator for Column A disturbance scenarios.

This module wires together the parts that were built in isolation
during Phase 1 — the steady-state initializer, the LV configuration,
the regulatory PIDs, and the setpoint interface — into a tick-based
closed-loop simulator that executes a disturbance scenario end-to-end
and writes the full data-logging contract via :class:`RunLogger`.

Architecture (see ADR 006).

- **Supervisory cadence.** A scenario function returns the requested
  composition setpoints ``(y_D_sp, x_B_sp)`` and disturbance trio
  ``(F, zF, qF)`` at every tick of the supervisory clock. For Phase 1
  the supervisory cadence and the regulatory cadence are collapsed to
  a single timestep (typically 0.05 min / 3 s); Phase 3 will decouple
  them when the agent comes in.
- **Setpoint rate limiting.** Both composition setpoints pass through
  a :class:`SetpointInterface` slew limiter so abrupt supervisor
  jumps cannot diverge the integrator (cf. ``assumptions.md`` §4
  and the rate-limiter divergence guard test).
- **Regulatory PIDs.** Two velocity-form PIDs (top: ``y_D -> LT``,
  bottom: ``x_B -> VB`` reverse-acting) drive the LV configuration.
  Phase 1 ships intentionally conservative gains; Phase 2 retunes
  them on the baseline benchmarks.
- **LV closure.** ``D`` and ``B`` come from
  :func:`assemble_inputs_lv` (P-only level loops). The level loops
  carry the standard steady-state offset under disturbances.
- **Integration.** Between two consecutive ticks the input vector is
  held constant and :func:`integrate_open_loop` advances the ODE.
  This is the standard zero-order-hold pattern.

The output is a :class:`SimulationResult` plus, optionally, a populated
``data/runs/<config>/<scenario>/<seed>/`` directory.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

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
from industrial_ai.twin.data_logging import RunLogger
from industrial_ai.twin.regulatory_pid import PIDController
from industrial_ai.twin.setpoint_interface import SetpointInterface

__all__ = [
    "ScenarioStep",
    "SimulationResult",
    "build_default_setpoint_interface",
    "build_skogestad_phase1_pids",
    "simulate_lv_closed_loop",
]

StateVector = npt.NDArray[np.float64]
InputVector = npt.NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class ScenarioStep:
    """One tick's worth of supervisor + disturbance values.

    Attributes
    ----------
    y_D_setpoint, x_B_setpoint : float
        Requested composition setpoints. The simulator slew-limits
        these through the SetpointInterface before passing them to the
        PIDs.
    F, zF, qF : float
        Feed-side disturbance trio (kmol/min, mole fraction, dimensionless).
    """

    y_D_setpoint: float
    x_B_setpoint: float
    F: float
    zF: float
    qF: float


ScenarioFn = Callable[[float], ScenarioStep]


@dataclass(slots=True)
class SimulationResult:
    """In-memory result of an LV closed-loop simulation.

    Attributes
    ----------
    t : numpy.ndarray of shape (n_ticks + 1,)
        Tick times (min), including the initial tick at ``t=0``.
    X : numpy.ndarray of shape (n_ticks + 1, 2 * NT)
        State at every tick.
    inputs : numpy.ndarray of shape (n_ticks, 7)
        Applied input vector ``[LT, VB, D, B, F, zF, qF]`` during each
        inter-tick interval.
    applied_setpoints : numpy.ndarray of shape (n_ticks, 2)
        Slew-limited composition setpoints ``(y_D_sp, x_B_sp)`` per tick.
    requested_setpoints : numpy.ndarray of shape (n_ticks, 2)
        Raw supervisor-requested setpoints per tick (before slew limit).
    cycle_wall_clock_seconds : numpy.ndarray of shape (n_ticks,)
        Wall-clock time spent inside each tick. Populated for Figure 6.
    success : bool
    message : str
    """

    t: npt.NDArray[np.float64]
    X: npt.NDArray[np.float64]
    inputs: npt.NDArray[np.float64]
    applied_setpoints: npt.NDArray[np.float64]
    requested_setpoints: npt.NDArray[np.float64]
    cycle_wall_clock_seconds: npt.NDArray[np.float64]
    success: bool
    message: str

    @property
    def y_D(self) -> npt.NDArray[np.float64]:
        """Per-tick top-product composition (``x[NT-1]`` of each row)."""
        nt = self.X.shape[1] // 2
        return np.asarray(self.X[:, nt - 1], dtype=np.float64)

    @property
    def x_B(self) -> npt.NDArray[np.float64]:
        """Per-tick bottoms composition (``x[0]`` of each row)."""
        return np.asarray(self.X[:, 0], dtype=np.float64)


def build_skogestad_phase1_pids(
    *,
    LT_initial: float,
    VB_initial: float,
) -> tuple[PIDController, PIDController]:
    """Return ``(top_pid, bottom_pid)`` with intentionally conservative Phase-1 gains.

    Both controllers are PI (no derivative) with proportional gains in
    the *flow-per-fraction* sense. Tuning is intentionally on the
    conservative side; Phase 2 retunes both loops as part of the
    baseline benchmark — see PROJECT_PLAN Phase 2.

    Parameters
    ----------
    LT_initial, VB_initial : float
        Bias points for the controllers (the previous-output state is
        seeded here so a small initial error does not produce a large
        initial commanded MV swing).
    """
    top = PIDController(
        Kp=2.5,
        Ki=2.5 / 30.0,
        output_min=0.0,
        output_max=10.0,
        direct_acting=False,
    )
    top.state.previous_output = LT_initial
    bottom = PIDController(
        Kp=2.5,
        Ki=2.5 / 30.0,
        output_min=0.0,
        output_max=10.0,
        direct_acting=True,
    )
    bottom.state.previous_output = VB_initial
    return top, bottom


def build_default_setpoint_interface(
    *,
    y_D_initial: float,
    x_B_initial: float,
    max_rate_per_min: float = 0.01,
) -> SetpointInterface:
    """Return a SetpointInterface with composition channels at conservative slew rates.

    A 0.01 mole-fraction-per-minute default slew rate is slow enough to
    bound the regulatory transient but fast enough to track realistic
    supervisor commands over a 5-15 min cadence.
    """
    iface = SetpointInterface()
    iface.register(name="y_D", max_rate=max_rate_per_min, initial=y_D_initial)
    iface.register(name="x_B", max_rate=max_rate_per_min, initial=x_B_initial)
    return iface


def _zero_order_hold_inputs(
    input_vector: InputVector,
) -> Callable[[float, StateVector], InputVector]:
    """Return an ``inputs_fn`` that holds ``input_vector`` constant."""

    def fn(t: float, X: StateVector) -> InputVector:
        return input_vector

    return fn


def simulate_lv_closed_loop(
    *,
    X0: StateVector,
    scenario: ScenarioFn,
    duration_min: float,
    tick_dt_min: float = 0.05,
    parameters: ColumnAParameters = DEFAULT_PARAMETERS,
    pid_top: PIDController | None = None,
    pid_bottom: PIDController | None = None,
    setpoint_interface: SetpointInterface | None = None,
    lv_config: LVConfiguration | None = None,
    logger: RunLogger | None = None,
    record_tray_profile_every_n_ticks: int = 20,
    config_snapshot: dict[str, Any] | None = None,
) -> SimulationResult:
    """Run an LV closed-loop disturbance scenario end-to-end.

    Parameters
    ----------
    X0 : numpy.ndarray of shape (2 * NT,)
        Initial state. Typically a previously-computed steady state.
    scenario : Callable[[float], ScenarioStep]
        Returns the requested setpoints and disturbance trio at any
        time ``t`` (min). Called once per tick.
    duration_min : float
        Total simulation horizon (min).
    tick_dt_min : float, optional
        Regulatory tick length. Default 0.05 min = 3 s, the upper end
        of the ADR 006 regulatory-cadence band.
    parameters : ColumnAParameters, optional
        Column specification.
    pid_top, pid_bottom : PIDController, optional
        Composition-loop controllers. Defaults come from
        :func:`build_skogestad_phase1_pids`.
    setpoint_interface : SetpointInterface, optional
        Setpoint-channel slew limiter. Default from
        :func:`build_default_setpoint_interface`.
    lv_config : LVConfiguration, optional
        Level-loop tuning. Defaults to the cola_lv.m values.
    logger : RunLogger, optional
        If supplied, the simulator records the full data-logging
        contract incrementally and the caller is expected to call
        ``logger.finalize(...)`` afterwards.
    record_tray_profile_every_n_ticks : int, optional
        Subsampling factor for the tray-profile channel — every Nth
        tick is written. Default 20 (= every second at the default
        tick rate) keeps the parquet file small.
    config_snapshot : dict, optional
        Forwarded to ``logger.set_config`` for the YAML snapshot.

    Returns
    -------
    SimulationResult
    """
    NT = parameters.NT
    n_ticks = int(np.ceil(duration_min / tick_dt_min))

    # Determine initial nominal LT/VB from a single scenario probe so
    # the PIDs start with a sensible bias matching the column's SS.
    initial_step = scenario(0.0)
    LT_initial = parameters.nominal_reflux_L0_kmol_per_min
    VB_initial = parameters.nominal_boilup_V0_kmol_per_min

    if pid_top is None or pid_bottom is None:
        default_top, default_bottom = build_skogestad_phase1_pids(
            LT_initial=LT_initial, VB_initial=VB_initial
        )
        pid_top = pid_top or default_top
        pid_bottom = pid_bottom or default_bottom
    if setpoint_interface is None:
        setpoint_interface = build_default_setpoint_interface(
            y_D_initial=initial_step.y_D_setpoint,
            x_B_initial=initial_step.x_B_setpoint,
        )
    if lv_config is None:
        lv_config = LVConfiguration()

    if logger is not None and config_snapshot is not None:
        logger.set_config(config_snapshot)

    t_axis = np.zeros(n_ticks + 1, dtype=np.float64)
    X_axis = np.zeros((n_ticks + 1, 2 * NT), dtype=np.float64)
    X_axis[0] = X0
    inputs_axis = np.zeros((n_ticks, 7), dtype=np.float64)
    applied_setpoints = np.zeros((n_ticks, 2), dtype=np.float64)
    requested_setpoints = np.zeros((n_ticks, 2), dtype=np.float64)
    wall_clock = np.zeros(n_ticks, dtype=np.float64)

    X_current = X0.copy()
    success = True
    message = "ok"
    for k in range(n_ticks):
        t_k = k * tick_dt_min
        t_next = min((k + 1) * tick_dt_min, duration_min)
        wall_start = time.perf_counter()
        step = scenario(t_k)

        applied = setpoint_interface.apply(
            requested={"y_D": step.y_D_setpoint, "x_B": step.x_B_setpoint},
            dt=tick_dt_min,
        )

        y_D_meas = float(X_current[NT - 1])
        x_B_meas = float(X_current[0])
        LT = pid_top.step(measurement=y_D_meas, setpoint=applied["y_D"], dt=tick_dt_min)
        VB = pid_bottom.step(measurement=x_B_meas, setpoint=applied["x_B"], dt=tick_dt_min)

        U = assemble_inputs_lv(
            state=X_current,
            LT=LT,
            VB=VB,
            F=step.F,
            zF=step.zF,
            qF=step.qF,
            config=lv_config,
            parameters=parameters,
        )

        result = integrate_open_loop(
            X0=X_current,
            t_span=(t_k, t_next),
            inputs_fn=_zero_order_hold_inputs(U),
            parameters=parameters,
            t_eval=np.array([t_next], dtype=np.float64),
        )
        if not result.success or result.X.shape[0] == 0:
            success = False
            message = f"integration failed at tick {k} (t={t_k:.3f}): {result.message}"
            t_axis[k + 1 :] = t_next
            X_axis[k + 1 :] = X_current  # carry forward
            break
        X_current = result.X[-1]
        wall_clock[k] = time.perf_counter() - wall_start

        t_axis[k + 1] = t_next
        X_axis[k + 1] = X_current
        inputs_axis[k] = U
        applied_setpoints[k] = (applied["y_D"], applied["x_B"])
        requested_setpoints[k] = (step.y_D_setpoint, step.x_B_setpoint)

        if logger is not None:
            logger.record_timeseries(
                t=float(t_next),
                y_D=float(X_current[NT - 1]),
                x_B=float(X_current[0]),
                L=float(U[0]),
                V=float(U[1]),
                D=float(U[2]),
                B=float(U[3]),
                F=float(U[4]),
                zF=float(U[5]),
                qF=float(U[6]),
            )
            logger.record_setpoint(
                t=float(t_next),
                channel="y_D",
                requested=float(step.y_D_setpoint),
                applied=float(applied["y_D"]),
            )
            logger.record_setpoint(
                t=float(t_next),
                channel="x_B",
                requested=float(step.x_B_setpoint),
                applied=float(applied["x_B"]),
            )
            logger.record_latency(cycle_index=k, wall_clock_seconds=wall_clock[k])
            if (k % record_tray_profile_every_n_ticks) == 0:
                logger.record_tray_profile(
                    t=float(t_next),
                    compositions=X_current[:NT].copy(),
                    holdups=X_current[NT:].copy(),
                )

    if logger is not None and success:
        logger.record_tray_profile(
            t=float(t_axis[-1]),
            compositions=X_axis[-1, :NT].copy(),
            holdups=X_axis[-1, NT:].copy(),
        )

    return SimulationResult(
        t=t_axis,
        X=X_axis,
        inputs=inputs_axis,
        applied_setpoints=applied_setpoints,
        requested_setpoints=requested_setpoints,
        cycle_wall_clock_seconds=wall_clock,
        success=success,
        message=message,
    )
