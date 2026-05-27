"""SIMC (Skogestad-IMC) PI tuning rules for the LV composition loops.

Skogestad's *Simple Internal Model Control* (SIMC) recipe maps a
first-order-plus-deadtime (FOPTD) plant ``G(s) = k * e^(-theta s) / (tau s + 1)``
to a single-degree-of-freedom PI controller with closed-loop response
governed by a target time constant ``tau_c``:

    Kp = (1 / |k|) * tau / (tau_c + theta)
    Ti = min(tau, 4 (tau_c + theta))

The two-degree-of-freedom variant uses the same ``Kp`` and ``Ti`` for
*regulation* (disturbance rejection) but inserts a first-order
*setpoint filter* with time constant ``tau_c`` ahead of the PID so
that *tracking* is governed by the larger ``tau_c`` while regulation
keeps the SIMC bandwidth. This is the standard recipe for column
composition control where a sharp setpoint change would otherwise
provoke excessive overshoot.

References.

- Skogestad, S. (2003). *Simple analytic rules for model reduction
  and PID controller tuning.* Journal of Process Control 13(4),
  291-309.
- Skogestad, S. and Postlethwaite, I. (1996). *Multivariable
  Feedback Control: Analysis and Design.* Wiley, §10.

For the Column A LV plant we approximate each composition channel
as FOPTD via the dominant time constant from the linearization and
the corresponding diagonal element of ``G^LV(0)``. The negligible
deadtime in the binary-distillation idealization sets
``theta = 0``, so the SIMC formulae collapse to

    Kp = (1 / |k|) * tau / tau_c
    Ti = min(tau, 4 tau_c).
"""

from __future__ import annotations

from dataclasses import dataclass

from industrial_ai.twin.column_a.linearize import (
    LinearizedLVModel,
    dominant_time_constants_min,
    steady_state_gain,
)

__all__ = [
    "SIMCTuning",
    "simc_pi_1dof",
    "simc_pi_2dof",
    "simc_tunings_from_linearization",
]


@dataclass(frozen=True, slots=True)
class SIMCTuning:
    """One PI loop tuned by SIMC.

    Attributes
    ----------
    Kp : float
    Ti : float
        Integral time (min). Integral gain ``Ki = Kp / Ti``.
    tau_c : float
        Closed-loop time constant target (min). For the 2DoF variant
        this also doubles as the setpoint-filter time constant.
    plant_gain : float
        Linearized steady-state gain ``g`` used in the derivation
        (signed; the controller uses ``|g|`` and handles the sign via
        ``direct_acting``).
    plant_tau : float
        Linearized dominant time constant used in the derivation (min).
    method : str
        Tag identifying the recipe — ``"SIMC-1DoF"`` or ``"SIMC-2DoF"``.
    """

    Kp: float
    Ti: float
    tau_c: float
    plant_gain: float
    plant_tau: float
    method: str


def simc_pi_1dof(
    *,
    plant_gain: float,
    plant_tau: float,
    tau_c: float,
    plant_deadtime: float = 0.0,
) -> SIMCTuning:
    """Return SIMC-1DoF PI parameters for a single composition loop.

    Parameters
    ----------
    plant_gain : float
        Steady-state gain ``g_ii`` (signed) of the channel. Magnitude
        drives ``Kp``; the sign is handled at controller-construction
        time via the ``direct_acting`` flag.
    plant_tau : float
        Dominant open-loop time constant (min).
    tau_c : float
        Target closed-loop time constant (min). Smaller -> more
        aggressive; larger -> more robust.
    plant_deadtime : float, optional
        Effective deadtime (min). Default 0.

    Returns
    -------
    SIMCTuning
    """
    if plant_gain == 0.0:
        raise ValueError("plant_gain must be non-zero for SIMC tuning")
    if plant_tau <= 0.0 or tau_c <= 0.0:
        raise ValueError("plant_tau and tau_c must be strictly positive")

    Kp = (1.0 / abs(plant_gain)) * plant_tau / (tau_c + plant_deadtime)
    Ti = min(plant_tau, 4.0 * (tau_c + plant_deadtime))
    return SIMCTuning(
        Kp=Kp,
        Ti=Ti,
        tau_c=tau_c,
        plant_gain=plant_gain,
        plant_tau=plant_tau,
        method="SIMC-1DoF",
    )


def simc_pi_2dof(
    *,
    plant_gain: float,
    plant_tau: float,
    tau_c: float,
    plant_deadtime: float = 0.0,
) -> SIMCTuning:
    """Return SIMC-2DoF PI parameters for a single composition loop.

    PI gains are identical to :func:`simc_pi_1dof` — the 2DoF nature
    comes from the *setpoint filter* the caller is expected to install
    ahead of the PID with time constant ``tau_c``. The setpoint filter
    decouples tracking bandwidth from regulation bandwidth: tracking
    becomes a first-order response with time constant ``tau_c`` while
    disturbance rejection retains the full SIMC speed.

    See the module docstring for the full rationale.
    """
    base = simc_pi_1dof(
        plant_gain=plant_gain,
        plant_tau=plant_tau,
        tau_c=tau_c,
        plant_deadtime=plant_deadtime,
    )
    return SIMCTuning(
        Kp=base.Kp,
        Ti=base.Ti,
        tau_c=base.tau_c,
        plant_gain=base.plant_gain,
        plant_tau=base.plant_tau,
        method="SIMC-2DoF",
    )


def simc_tunings_from_linearization(
    model: LinearizedLVModel,
    *,
    tau_c_top_min: float = 12.0,
    tau_c_bottom_min: float = 12.0,
    variant: str = "1dof",
    effective_gain_diag: tuple[float, float] | None = None,
) -> tuple[SIMCTuning, SIMCTuning]:
    """Derive SIMC tunings for the LV top and bottom loops from a linearized model.

    Convenience that pulls ``G^LV(0)`` and the dominant time constant
    out of :class:`LinearizedLVModel`, then applies
    :func:`simc_pi_1dof` or :func:`simc_pi_2dof` to each loop.

    Parameters
    ----------
    model : LinearizedLVModel
    tau_c_top_min, tau_c_bottom_min : float, optional
        Target closed-loop time constants (min). Default 12 min ~
        tau_2 of the canonical Column A.
    variant : {"1dof", "2dof"}
        Which SIMC variant to apply.
    effective_gain_diag : tuple of (float, float), optional
        Override the per-loop plant gains with the diagonal of
        ``G(0) @ D`` after a static decoupler has been applied. The
        decoupled effective gain is ``g_ii / lambda_ii``, much smaller
        than ``g_ii`` itself; SIMC compensates by raising ``Kp`` by
        the corresponding factor. Without this argument the
        decoupled SIMC variant is silently detuned by the RGA factor
        — the exact failure mode the shootout would otherwise report.

    Returns
    -------
    tuple of (top_tuning, bottom_tuning)
    """
    if variant not in ("1dof", "2dof"):
        raise ValueError(f"variant must be '1dof' or '2dof', got {variant!r}")

    G0 = steady_state_gain(model)[:, :2]
    if effective_gain_diag is None:
        g_top = float(G0[0, 0])
        g_bottom = float(G0[1, 1])
    else:
        g_top, g_bottom = float(effective_gain_diag[0]), float(effective_gain_diag[1])
    taus = dominant_time_constants_min(model, n=1)
    tau_1 = float(taus[0])

    builder = simc_pi_1dof if variant == "1dof" else simc_pi_2dof
    top = builder(plant_gain=g_top, plant_tau=tau_1, tau_c=tau_c_top_min)
    bottom = builder(plant_gain=g_bottom, plant_tau=tau_1, tau_c=tau_c_bottom_min)
    return top, bottom
