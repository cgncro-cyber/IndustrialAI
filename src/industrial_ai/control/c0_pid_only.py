"""C0 configuration — PID-only baseline with static manual setpoints.

The C0 supervisor is the simplest reference point in the four-way
comparison defined in ADR 006. There is no setpoint adjustment by any
supervisor (no MPC, no agent, no safety gate); the two regulatory PIDs
hold the manually-chosen composition setpoints against any
disturbance, with their gains derived from the Åström-Hägglund relay
test plus the Tyreus-Luyben tuning rule (see
``src/industrial_ai/control/relay_tuning.py`` and
``data/reference/c0_pid_tuning.json``).

This module is intentionally thin: it loads the persisted tuning,
builds the two :class:`PIDController` instances, and stops there. The
actual closed-loop driver remains
:func:`industrial_ai.twin.simulate.simulate_lv_closed_loop`; C0 is
just one particular PID configuration the driver accepts.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from industrial_ai.twin.regulatory_pid import PIDController

__all__ = [
    "C0PIDTuning",
    "build_c0_pids",
    "load_c0_tuning",
]

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
_DEFAULT_TUNING_PATH = _REPO_ROOT / "data" / "reference" / "c0_pid_tuning.json"


@dataclass(frozen=True, slots=True)
class C0PIDTuning:
    """Tyreus-Luyben PI gains for the two C0 composition loops.

    Attributes
    ----------
    Kp_top, Ti_top_min : float
        Top loop: composition ``y_D`` -> reflux ``LT`` (direct-acting).
    Kp_bottom, Ti_bottom_min : float
        Bottom loop: composition ``x_B`` -> boilup ``VB``
        (reverse-acting). Sign handled by ``direct_acting=False`` in
        the constructed :class:`PIDController`.
    source : pathlib.Path
        File the tuning was loaded from — recorded so a run's
        ``manifest.json`` can hash it as an input.
    """

    Kp_top: float
    Ti_top_min: float
    Kp_bottom: float
    Ti_bottom_min: float
    source: Path


def load_c0_tuning(path: Path | None = None) -> C0PIDTuning:
    """Load C0 PI gains from a relay-tuning JSON file.

    Parameters
    ----------
    path : pathlib.Path, optional
        Tuning file path. Defaults to
        ``data/reference/c0_pid_tuning.json`` at the repo root.

    Returns
    -------
    C0PIDTuning
    """
    if path is None:
        path = _DEFAULT_TUNING_PATH
    with path.open() as fh:
        payload: dict[str, Any] = json.load(fh)
    top = payload["loops"]["top"]["tyreus_luyben"]
    bottom = payload["loops"]["bottom"]["tyreus_luyben"]
    return C0PIDTuning(
        Kp_top=float(top["Kp"]),
        Ti_top_min=float(top["Ti_min"]),
        Kp_bottom=float(bottom["Kp"]),
        Ti_bottom_min=float(bottom["Ti_min"]),
        source=path,
    )


def build_c0_pids(
    *,
    LT_initial: float,
    VB_initial: float,
    tuning: C0PIDTuning | None = None,
    detune_factor: float = 1.0,
    output_min: float = 0.0,
    output_max: float = 10.0,
) -> tuple[PIDController, PIDController]:
    """Return ``(top_pid, bottom_pid)`` configured with the C0 Tyreus-Luyben gains.

    Parameters
    ----------
    LT_initial, VB_initial : float
        Operating-point biases for the controller outputs. The PID
        ``previous_output`` state is seeded here so a small initial
        composition error does not cause a large initial MV swing.
    tuning : C0PIDTuning, optional
        Pre-loaded tuning. Defaults to
        :func:`load_c0_tuning` against the canonical JSON.
    detune_factor : float, optional
        Multiplier on ``Kp`` (and inversely on integral aggressiveness)
        applied to both loops. Default 1.0 = use the Tyreus-Luyben
        result as-is; values in ``(0, 1)`` produce a more conservative
        controller without altering ``Ti``. Useful for the C0 baseline
        if the raw TL gain proves too aggressive against the strongest
        disturbance — keeps the tuning data immutable while letting
        the baseline driver detune in code.
    output_min, output_max : float, optional
        Saturation bounds (kmol/min) applied to both MVs.

    Returns
    -------
    (PIDController, PIDController)
        Top-loop and bottom-loop PI controllers, ready to plug into
        :func:`simulate_lv_closed_loop`.
    """
    if tuning is None:
        tuning = load_c0_tuning()
    if detune_factor <= 0.0:
        raise ValueError(f"detune_factor must be > 0, got {detune_factor}")

    Kp_top = tuning.Kp_top * detune_factor
    Kp_bottom = tuning.Kp_bottom * detune_factor

    Ki_top = Kp_top / tuning.Ti_top_min
    Ki_bottom = Kp_bottom / tuning.Ti_bottom_min

    # Top loop is direct-acting: more LT -> more reflux -> y_D rises,
    # so when y_D < sp we want MORE LT. The default PIDController
    # form u = Kp*(sp - meas) + ... gives positive u for negative
    # error, i.e. direct_acting=True.
    top = PIDController(
        Kp=Kp_top,
        Ki=Ki_top,
        output_min=output_min,
        output_max=output_max,
        direct_acting=True,
    )
    # Positional-form PI: u = Kp*error + Ki*integral. Seed the integral
    # so that at zero error the controller output equals the bias MV
    # (LT_initial). Without this, the first tick at the operating point
    # outputs 0 and the controller never recovers — the LV plant has no
    # other "where am I now" reference.
    top.state.integral = LT_initial / Ki_top
    top.state.previous_output = LT_initial

    # Bottom loop is reverse-acting: more VB -> lighter component is
    # stripped up the column -> x_B drops, so when x_B > sp we want
    # MORE VB. direct_acting=False inverts the error sign to deliver
    # this.
    bottom = PIDController(
        Kp=Kp_bottom,
        Ki=Ki_bottom,
        output_min=output_min,
        output_max=output_max,
        direct_acting=False,
    )
    bottom.state.integral = VB_initial / Ki_bottom
    bottom.state.previous_output = VB_initial
    return top, bottom
