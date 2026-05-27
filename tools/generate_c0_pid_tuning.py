"""Generate ``data/reference/c0_pid_tuning.json`` from a relay-feedback test.

Runs the Åström-Hägglund relay test on the top (``y_D <- LT``) and
bottom (``x_B <- VB``) composition loops at the published Skogestad
Column A steady state, then derives Tyreus-Luyben PI parameters from
``(Ku, Pu)`` per loop.

Settings (``d = 0.5 kmol/min``, hysteresis ``5e-3`` mole fraction) are
chosen so the relay engages the column's dominant composition mode
(Pu ~ tau_2 ~ 10 min) rather than the fast linearized-hydraulics tail
(Pu ~ 2 min, which produces non-physically aggressive gains). See
``src/industrial_ai/control/relay_tuning.py`` for the test machinery
and ``docs/decisions/`` for the broader Phase 2 tuning rationale.

Invocation:

    uv run python tools/generate_c0_pid_tuning.py
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from industrial_ai.control.relay_tuning import (
    RelayResult,
    relay_test,
    tyreus_luyben,
)
from industrial_ai.twin.column_a import DEFAULT_PARAMETERS

_REPO_ROOT = Path(__file__).resolve().parent.parent
_SKOGESTAD_SS = _REPO_ROOT / "data" / "reference" / "skogestad_column_a_steady_state.json"
_DEFAULT_OUTPUT = _REPO_ROOT / "data" / "reference" / "c0_pid_tuning.json"


def _load_skogestad_ss() -> np.ndarray:
    with _SKOGESTAD_SS.open() as fh:
        ss = json.load(fh)["steady_state"]
    return np.array(ss["compositions"] + ss["holdups_kmol"], dtype=np.float64)


def _loop_payload(result: RelayResult) -> dict[str, Any]:
    tl = tyreus_luyben(result)
    return {
        "Ku": result.Ku,
        "Pu_min": result.Pu,
        "relay_amplitude_d_kmol_per_min": result.relay_amplitude_d,
        "measurement_amplitude_a": result.measurement_amplitude_a,
        "setpoint": result.setpoint,
        "tyreus_luyben": {
            "Kp": tl.Kp,
            "Ti_min": tl.Ti,
            "Ki_per_min": tl.Kp / tl.Ti,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=_DEFAULT_OUTPUT)
    parser.add_argument("--duration-min", type=float, default=500.0)
    parser.add_argument(
        "--relay-amplitude",
        type=float,
        default=0.5,
        help="Half-amplitude of the MV swing (kmol/min)",
    )
    parser.add_argument(
        "--hysteresis",
        type=float,
        default=5.0e-3,
        help="Symmetric hysteresis band around the setpoint (mole fraction)",
    )
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    X = _load_skogestad_ss()
    p = DEFAULT_PARAMETERS
    y_D = float(X[p.NT - 1])
    x_B = float(X[0])

    print(f"Skogestad SS y_D = {y_D:.5f}, x_B = {x_B:.5f}")
    print(
        f"Relay settings: d = {args.relay_amplitude:.3f} kmol/min, "
        f"hysteresis = {args.hysteresis:.4f} mole fraction"
    )

    print("Running relay test on top loop (y_D <- LT) ...")
    top = relay_test(
        loop="top",
        X0=X,
        setpoint=y_D,
        relay_amplitude_d=args.relay_amplitude,
        hysteresis=args.hysteresis,
        duration_min=args.duration_min,
    )
    print(
        f"  Ku = {top.Ku:.3f} kmol/min/fraction, Pu = {top.Pu:.2f} min, "
        f"a = {top.measurement_amplitude_a:.5f}"
    )

    print("Running relay test on bottom loop (x_B <- VB) ...")
    bottom = relay_test(
        loop="bottom",
        X0=X,
        setpoint=x_B,
        relay_amplitude_d=args.relay_amplitude,
        hysteresis=args.hysteresis,
        duration_min=args.duration_min,
    )
    print(
        f"  Ku = {bottom.Ku:.3f} kmol/min/fraction, Pu = {bottom.Pu:.2f} min, "
        f"a = {bottom.measurement_amplitude_a:.5f}"
    )

    payload = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(tz=UTC).isoformat(),
        "operating_point": {
            "case": "Skogestad Column A, nominal SS",
            "F_kmol_per_min": p.nominal_feed_F_kmol_per_min,
            "zF": 0.5,
            "qF": p.nominal_feed_liquid_fraction_qF,
            "L0_kmol_per_min": p.nominal_reflux_L0_kmol_per_min,
            "V0_kmol_per_min": p.nominal_boilup_V0_kmol_per_min,
            "y_D_at_SS": y_D,
            "x_B_at_SS": x_B,
        },
        "relay_settings": {
            "amplitude_d_kmol_per_min": args.relay_amplitude,
            "hysteresis": args.hysteresis,
            "duration_min": args.duration_min,
            "method": "Astrom-Hagglund 1984",
            "tuning_rule": "Tyreus-Luyben (Kp = Ku/3.2, Ti = 2.2 Pu)",
        },
        "loops": {
            "top": {"measurement": "y_D", "manipulated": "LT", **_loop_payload(top)},
            "bottom": {"measurement": "x_B", "manipulated": "VB", **_loop_payload(bottom)},
        },
    }
    with args.output.open("w") as fh:
        json.dump(payload, fh, indent=2)
    print(f"\nWrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
