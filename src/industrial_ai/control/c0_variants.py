"""C0 PID-only variants for the Phase-2 tuning shootout.

Phase 2 Day 2.5 promotes the C0 baseline from a single Tyreus-Luyben
tuning to the *best of a six-candidate shootout*:

    {Tyreus-Luyben, SIMC-1DoF, SIMC-2DoF}  x  {no decoupler, with decoupler}

The decoupler is the simplified static decoupler from
:func:`industrial_ai.control.decoupler.simplified_decoupler`, which is
the right answer to the LV configuration's RGA(1,1) ~ 36 (Skogestad &
Postlethwaite 1996, §10.8). The 2DoF variant adds a first-order
setpoint filter so that the SIMC tracking response uses ``tau_c``
while disturbance rejection retains the full SIMC bandwidth.

A :class:`C0Variant` carries everything needed to instantiate one
candidate: PID gains, optional decoupler, optional setpoint filter,
and a free-form reference string that the shootout JSON propagates
to ``c0_pid_tuning_shootout.json``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt

from industrial_ai.control.decoupler import (
    DecouplerSpec,
    identity_decoupler,
    simplified_decoupler,
)
from industrial_ai.control.relay_tuning import RelayResult, tyreus_luyben
from industrial_ai.control.simc import simc_tunings_from_linearization
from industrial_ai.twin.column_a.linearize import LinearizedLVModel
from industrial_ai.twin.regulatory_pid import PIDController

__all__ = [
    "C0Variant",
    "build_pids_for_variant",
    "build_six_variants",
]


@dataclass(frozen=True, slots=True)
class C0Variant:
    """One C0 candidate in the Phase-2 shootout matrix.

    Attributes
    ----------
    name : str
        Short identifier used in the shootout JSON.
    tuning_method : str
        Provenance — ``"Tyreus-Luyben"``, ``"SIMC-1DoF"``, or
        ``"SIMC-2DoF"``.
    Kp_top, Ti_top_min : float
        PI gains for the top (y_D <- LT) loop.
    Kp_bottom, Ti_bottom_min : float
        PI gains for the bottom (x_B <- VB) loop.
    decoupler : DecouplerSpec
        Static decoupler. Use :func:`identity_decoupler` to disable.
    setpoint_filter_tau_min : float, optional
        Filter time constant (min). Only meaningful for SIMC-2DoF.
    reference : str
        Free-form citation note recorded in the shootout JSON.
    """

    name: str
    tuning_method: str
    Kp_top: float
    Ti_top_min: float
    Kp_bottom: float
    Ti_bottom_min: float
    decoupler: DecouplerSpec
    setpoint_filter_tau_min: float | None
    reference: str

    def to_serializable(self) -> dict[str, Any]:
        """Return the variant in a JSON-serializable shape."""
        return {
            "name": self.name,
            "tuning_method": self.tuning_method,
            "Kp_top": self.Kp_top,
            "Ti_top_min": self.Ti_top_min,
            "Kp_bottom": self.Kp_bottom,
            "Ti_bottom_min": self.Ti_bottom_min,
            "decoupler": {
                "matrix": self.decoupler.matrix.tolist(),
                "rga_11": self.decoupler.rga_11,
                "g_effective_diag": _nan_safe_list(self.decoupler.g_effective_diag),
            },
            "setpoint_filter_tau_min": self.setpoint_filter_tau_min,
            "reference": self.reference,
        }


@dataclass(frozen=True, slots=True)
class _LoopTuning:
    """Internal helper for the {top, bottom} gain pair used in build_six_variants."""

    Kp: float
    Ti: float


def _relay_to_tunings(
    *, relay_top: RelayResult, relay_bottom: RelayResult
) -> tuple[_LoopTuning, _LoopTuning]:
    tl_top = tyreus_luyben(relay_top)
    tl_bottom = tyreus_luyben(relay_bottom)
    return (
        _LoopTuning(Kp=tl_top.Kp, Ti=tl_top.Ti),
        _LoopTuning(Kp=tl_bottom.Kp, Ti=tl_bottom.Ti),
    )


def build_six_variants(
    *,
    linearized: LinearizedLVModel,
    relay_top: RelayResult,
    relay_bottom: RelayResult,
    relay_top_decoupled: RelayResult,
    relay_bottom_decoupled: RelayResult,
    simc_tau_c_min: float = 12.0,
    simc_2dof_filter_tau_min: float | None = None,
) -> list[C0Variant]:
    """Return the six C0 candidates for the Phase-2 shootout.

    Parameters
    ----------
    linearized : LinearizedLVModel
        Linearization at the nominal operating point. Drives SIMC and
        the decoupler.
    relay_top, relay_bottom : RelayResult
        Outputs from :func:`industrial_ai.control.relay_tuning.relay_test`
        for the top and bottom composition loops on the *undecoupled*
        plant. Drives the no-decoupler Tyreus-Luyben pair.
    relay_top_decoupled, relay_bottom_decoupled : RelayResult
        Outputs from the same relay test run *with the simplified
        decoupler in the loop*. Drives the with-decoupler
        Tyreus-Luyben pair so its gains are calibrated against the
        effective plant ``G(0) D`` rather than ``G(0)`` itself — the
        previously-shipped naive variant used SISO TL gains on a
        decoupled plant and was methodologically unfair.
    simc_tau_c_min : float, optional
        Target closed-loop time constant for both SIMC variants
        (default 12 min ~ tau_2 of Column A).
    simc_2dof_filter_tau_min : float, optional
        Setpoint-filter time constant for SIMC-2DoF. Defaults to
        ``simc_tau_c_min`` per Skogestad 2003 §4.

    Returns
    -------
    list of C0Variant
        Six variants in shootout order: TL/SIMC-1DoF/SIMC-2DoF then
        each repeated with the simplified decoupler.
    """
    tl_top_loop, tl_bottom_loop = _relay_to_tunings(relay_top=relay_top, relay_bottom=relay_bottom)
    tl_top_dec_loop, tl_bottom_dec_loop = _relay_to_tunings(
        relay_top=relay_top_decoupled, relay_bottom=relay_bottom_decoupled
    )
    simc1_top, simc1_bottom = simc_tunings_from_linearization(
        linearized, tau_c_top_min=simc_tau_c_min, tau_c_bottom_min=simc_tau_c_min, variant="1dof"
    )
    simc2_top, simc2_bottom = simc_tunings_from_linearization(
        linearized, tau_c_top_min=simc_tau_c_min, tau_c_bottom_min=simc_tau_c_min, variant="2dof"
    )
    filter_tau = (
        simc_2dof_filter_tau_min if simc_2dof_filter_tau_min is not None else simc_tau_c_min
    )

    identity = identity_decoupler()
    decoupled = simplified_decoupler(linearized)

    # The simplified decoupler shrinks the effective per-loop gain to
    # g_ii / lambda_ii. Without compensating Kp, the SIMC-with-decoupler
    # variants would be silently detuned by the RGA factor and look
    # uniformly worse than the no-decoupler case — that is a tuning
    # artifact, not a real conclusion about decoupling. Re-derive SIMC
    # against the effective gain so the comparison is honest.
    g_eff = tuple(decoupled.g_effective_diag.tolist())
    simc1_dec_top, simc1_dec_bottom = simc_tunings_from_linearization(
        linearized,
        tau_c_top_min=simc_tau_c_min,
        tau_c_bottom_min=simc_tau_c_min,
        variant="1dof",
        effective_gain_diag=g_eff,
    )
    simc2_dec_top, simc2_dec_bottom = simc_tunings_from_linearization(
        linearized,
        tau_c_top_min=simc_tau_c_min,
        tau_c_bottom_min=simc_tau_c_min,
        variant="2dof",
        effective_gain_diag=g_eff,
    )

    return [
        C0Variant(
            name="TL_no_decoupler",
            tuning_method="Tyreus-Luyben",
            Kp_top=tl_top_loop.Kp,
            Ti_top_min=tl_top_loop.Ti,
            Kp_bottom=tl_bottom_loop.Kp,
            Ti_bottom_min=tl_bottom_loop.Ti,
            decoupler=identity,
            setpoint_filter_tau_min=None,
            reference="Tyreus & Luyben 1992; relay test per Astrom-Hagglund 1984",
        ),
        C0Variant(
            name="SIMC_1DoF_no_decoupler",
            tuning_method="SIMC-1DoF",
            Kp_top=simc1_top.Kp,
            Ti_top_min=simc1_top.Ti,
            Kp_bottom=simc1_bottom.Kp,
            Ti_bottom_min=simc1_bottom.Ti,
            decoupler=identity,
            setpoint_filter_tau_min=None,
            reference="Skogestad 2003 SIMC, 1-DoF PI form",
        ),
        C0Variant(
            name="SIMC_2DoF_no_decoupler",
            tuning_method="SIMC-2DoF",
            Kp_top=simc2_top.Kp,
            Ti_top_min=simc2_top.Ti,
            Kp_bottom=simc2_bottom.Kp,
            Ti_bottom_min=simc2_bottom.Ti,
            decoupler=identity,
            setpoint_filter_tau_min=filter_tau,
            reference="Skogestad 2003 SIMC, 2-DoF with setpoint filter",
        ),
        C0Variant(
            name="TL_with_decoupler_retuned",
            tuning_method="Tyreus-Luyben",
            Kp_top=tl_top_dec_loop.Kp,
            Ti_top_min=tl_top_dec_loop.Ti,
            Kp_bottom=tl_bottom_dec_loop.Kp,
            Ti_bottom_min=tl_bottom_dec_loop.Ti,
            decoupler=decoupled,
            setpoint_filter_tau_min=None,
            reference=(
                "Tyreus-Luyben gains derived from a relay test run *with* the "
                "simplified decoupler in the loop, so the (Ku, Pu) describe "
                "the effective plant G(0) D rather than G(0). Replaces the "
                "naive TL-on-decoupled-plant variant. Skogestad & Postlethwaite "
                "1996 §10.8."
            ),
        ),
        C0Variant(
            name="SIMC_1DoF_with_decoupler",
            tuning_method="SIMC-1DoF",
            Kp_top=simc1_dec_top.Kp,
            Ti_top_min=simc1_dec_top.Ti,
            Kp_bottom=simc1_dec_bottom.Kp,
            Ti_bottom_min=simc1_dec_bottom.Ti,
            decoupler=decoupled,
            setpoint_filter_tau_min=None,
            reference="SIMC-1DoF re-tuned against g_eff = g_ii / lambda_ii + simplified decoupler",
        ),
        C0Variant(
            name="SIMC_2DoF_with_decoupler",
            tuning_method="SIMC-2DoF",
            Kp_top=simc2_dec_top.Kp,
            Ti_top_min=simc2_dec_top.Ti,
            Kp_bottom=simc2_dec_bottom.Kp,
            Ti_bottom_min=simc2_dec_bottom.Ti,
            decoupler=decoupled,
            setpoint_filter_tau_min=filter_tau,
            reference="SIMC-2DoF re-tuned against g_eff + simplified decoupler + setpoint filter",
        ),
    ]


def build_pids_for_variant(
    variant: C0Variant,
    *,
    LT_initial: float,
    VB_initial: float,
    output_min: float = 0.0,
    output_max: float = 10.0,
) -> tuple[PIDController, PIDController]:
    """Construct the two PIDControllers for one variant with integrals seeded."""
    Ki_top = variant.Kp_top / variant.Ti_top_min
    Ki_bottom = variant.Kp_bottom / variant.Ti_bottom_min
    top = PIDController(
        Kp=variant.Kp_top,
        Ki=Ki_top,
        output_min=output_min,
        output_max=output_max,
        direct_acting=True,
    )
    top.state.integral = LT_initial / Ki_top
    top.state.previous_output = LT_initial
    bottom = PIDController(
        Kp=variant.Kp_bottom,
        Ki=Ki_bottom,
        output_min=output_min,
        output_max=output_max,
        direct_acting=False,
    )
    bottom.state.integral = VB_initial / Ki_bottom
    bottom.state.previous_output = VB_initial
    return top, bottom


def _nan_safe_list(arr: npt.NDArray[np.float64]) -> list[float | None]:
    return [float(v) if np.isfinite(v) else None for v in arr]
