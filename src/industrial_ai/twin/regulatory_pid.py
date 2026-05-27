"""Regulatory PID layer — held constant across all supervisory configurations.

Per ADR 006, the regulatory layer runs at a fast cadence (1–5 s) and is
identical across the four supervisory configurations (C0 PID-only / C1
Linear MPC / C2 Agent / C3 Agent + Safety Gate). What varies is only
the supervisor that sits above and adjusts the setpoints the regulatory
loops track.

For the Skogestad Column A LV configuration, the regulatory layer
consists of four loops:

1. **Top-composition loop** — PID on ``y_D`` manipulating reflux ``LT``.
2. **Bottom-composition loop** — PID on ``x_B`` manipulating boilup ``VB``.
3. **Condenser-level loop** — P-only on condenser holdup manipulating
   distillate ``D``. Implemented inside the LV configuration module
   itself for compactness with the cola_lv.m pattern.
4. **Reboiler-level loop** — P-only on reboiler holdup manipulating
   bottoms ``B``. Same comment.

This module implements the generic :class:`PIDController` class and
provides factory helpers for the typical Column A composition loops.
The level P-controllers stay inside
:mod:`industrial_ai.twin.column_a.configurations.lv` because they are
configuration-specific (DV and L/D-V/B would close different pairings).
"""

from __future__ import annotations

from dataclasses import dataclass, field

__all__ = ["PIDController", "PIDState"]


@dataclass(slots=True)
class PIDState:
    """Mutable internal state of the positional-form PID controller.

    Attributes
    ----------
    integral : float
        Running integral of the error. Frozen by the
        conditional-integration anti-windup whenever the unclipped
        output is saturated against the error direction.
    previous_error : float
        Last-seen error, used by the derivative term.
    previous_output : float
        Last commanded output. Exposed for downstream logging and for
        callers that seed the controller at an operating-point bias;
        not consumed by :meth:`PIDController.step` itself.
    """

    integral: float = 0.0
    previous_error: float = 0.0
    previous_output: float = 0.0


@dataclass(slots=True)
class PIDController:
    """Discrete-time PID controller with conditional-integration anti-windup.

    The controller is parameterized in textbook (parallel) form

        u(t) = Kp * e + Ki * integral(e) + Kd * d(e)/dt

    Anti-windup is implemented by **conditional integration**: the
    integrator is frozen whenever the output is saturated *and* the
    current error would push it further into saturation. As soon as
    the error reverses, integration resumes. This formulation has no
    additional tuning constant and is numerically stable for any
    positive ``dt``, including values smaller than the integral time
    constant ``Kp / Ki``.

    Parameters
    ----------
    Kp : float
        Proportional gain.
    Ki : float, optional
        Integral gain (per minute). Default 0 — yields a P-only
        controller suitable for the Column A level loops.
    Kd : float, optional
        Derivative gain (minutes). Default 0.
    output_min, output_max : float, optional
        Saturation bounds on the controller output.
    direct_acting : bool, optional
        If ``True`` (default), the controller computes
        ``u = Kp * (setpoint - measurement) + …``. If ``False``, the
        sign of the error term is inverted, useful for reverse-acting
        loops (e.g., bottoms composition increases when boilup
        decreases).

    Notes
    -----
    Time enters explicitly via ``dt`` in :meth:`step`. The caller is
    responsible for passing the elapsed simulation time between
    invocations.
    """

    Kp: float
    Ki: float = 0.0
    Kd: float = 0.0
    output_min: float = float("-inf")
    output_max: float = float("inf")
    direct_acting: bool = True
    state: PIDState = field(default_factory=PIDState)

    def reset(self) -> None:
        """Clear integral and derivative memory."""
        self.state = PIDState()

    def step(
        self,
        *,
        measurement: float,
        setpoint: float,
        dt: float,
    ) -> float:
        """Advance the controller by ``dt`` (min) and return the new output.

        Parameters
        ----------
        measurement : float
            Current process variable.
        setpoint : float
            Desired setpoint.
        dt : float
            Time elapsed since the previous :meth:`step` call (min).
            Must be strictly positive.

        Returns
        -------
        float
            Commanded output, after saturation.

        Raises
        ------
        ValueError
            If ``dt`` is not strictly positive.
        """
        if dt <= 0.0:
            raise ValueError(f"dt must be strictly positive, got {dt}")

        error = setpoint - measurement
        if not self.direct_acting:
            error = -error

        # Pre-saturation PID output (positional form: P + I + D).
        derivative = (error - self.state.previous_error) / dt
        unclipped = self.Kp * error + self.Ki * self.state.integral + self.Kd * derivative

        # Saturate.
        clipped = max(self.output_min, min(self.output_max, unclipped))

        # Conditional-integration anti-windup. Freeze the integrator
        # whenever the output is saturated and the error would push it
        # further into saturation; otherwise let the integrator update.
        saturated_high = unclipped > self.output_max
        saturated_low = unclipped < self.output_min
        pushes_further_into_saturation = (saturated_high and error > 0.0) or (
            saturated_low and error < 0.0
        )
        if not pushes_further_into_saturation:
            self.state.integral += dt * error

        self.state.previous_error = error
        self.state.previous_output = clipped
        return clipped
