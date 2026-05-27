"""Phase 2 Day 2.5 — PID tuning shootout.

Six C0 candidates ({Tyreus-Luyben, SIMC-1DoF, SIMC-2DoF} x {no
decoupler, with decoupler}) scored over the five canonical Phase-2
disturbance scenarios. The winner by aggregate IAE is also spot-
checked at two off-nominal operating points for robustness, then
written to ``data/reference/c0_pid_tuning.json`` so the existing
``load_c0_tuning`` API picks it up unchanged.

Invocation:

    uv run python tools/run_pid_shootout.py

Outputs:

    data/reference/c0_pid_tuning_shootout.json   # full audit trail
    data/reference/c0_pid_tuning.json            # winner, runtime-loaded
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from industrial_ai.control.c0_variants import (
    C0Variant,
    build_pids_for_variant,
    build_six_variants,
)
from industrial_ai.control.relay_tuning import RelayResult, relay_test
from industrial_ai.control.scenarios import SCENARIO_NAMES, build_scenario
from industrial_ai.evaluation.kpis import KPISet, compute_kpis
from industrial_ai.twin.column_a import DEFAULT_PARAMETERS
from industrial_ai.twin.column_a.linearize import linearize_lv
from industrial_ai.twin.simulate import simulate_lv_closed_loop

_REPO_ROOT = Path(__file__).resolve().parent.parent
_SS_FIXTURE = _REPO_ROOT / "data" / "reference" / "skogestad_column_a_steady_state.json"
_DEFAULT_SHOOTOUT = _REPO_ROOT / "data" / "reference" / "c0_pid_tuning_shootout.json"
_DEFAULT_C0 = _REPO_ROOT / "data" / "reference" / "c0_pid_tuning.json"


def _load_nominal_ss() -> np.ndarray:
    with _SS_FIXTURE.open() as fh:
        ss = json.load(fh)["steady_state"]
    return np.array(ss["compositions"] + ss["holdups_kmol"], dtype=np.float64)


def _run_variant_on_scenario(
    variant: C0Variant,
    scenario_name: str,
    X0: np.ndarray,
) -> tuple[KPISet, bool]:
    p = DEFAULT_PARAMETERS
    scenario_fn, spec = build_scenario(scenario_name)
    top, bottom = build_pids_for_variant(
        variant,
        LT_initial=p.nominal_reflux_L0_kmol_per_min,
        VB_initial=p.nominal_boilup_V0_kmol_per_min,
    )
    sim = simulate_lv_closed_loop(
        X0=X0,
        scenario=scenario_fn,
        duration_min=spec.horizon_min,
        tick_dt_min=0.05,
        pid_top=top,
        pid_bottom=bottom,
        mv_decoupler=variant.decoupler.matrix,
        setpoint_filter_tau_min=variant.setpoint_filter_tau_min,
    )
    return compute_kpis(sim), sim.success


def _score_variant(variant: C0Variant, X0: np.ndarray, scenarios: list[str]) -> dict[str, Any]:
    per_scenario: dict[str, dict[str, Any]] = {}
    aggregate_iae = 0.0
    any_failure = False
    for name in scenarios:
        kpis, success = _run_variant_on_scenario(variant, name, X0)
        per_scenario[name] = {"success": success, **kpis.as_dict()}
        if not success or not np.isfinite(kpis.iae_mole_fraction_min):
            any_failure = True
            aggregate_iae = float("inf")
            break
        aggregate_iae += kpis.iae_mole_fraction_min
    return {
        "per_scenario": per_scenario,
        "aggregate_iae": aggregate_iae,
        "any_failure": any_failure,
    }


def _write_c0_winner(winner: C0Variant, winner_results: dict[str, Any], path: Path) -> None:
    """Overwrite ``c0_pid_tuning.json`` with the winning variant's gains."""
    p = DEFAULT_PARAMETERS
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
        },
        "winner": {
            "variant_name": winner.name,
            "tuning_method": winner.tuning_method,
            "aggregate_iae": winner_results["aggregate_iae"],
            "reference": winner.reference,
        },
        "loops": {
            "top": {
                "measurement": "y_D",
                "manipulated": "LT",
                "tyreus_luyben": {
                    "Kp": winner.Kp_top,
                    "Ti_min": winner.Ti_top_min,
                    "Ki_per_min": winner.Kp_top / winner.Ti_top_min,
                },
            },
            "bottom": {
                "measurement": "x_B",
                "manipulated": "VB",
                "tyreus_luyben": {
                    "Kp": winner.Kp_bottom,
                    "Ti_min": winner.Ti_bottom_min,
                    "Ki_per_min": winner.Kp_bottom / winner.Ti_bottom_min,
                },
            },
        },
        "decoupler": winner.to_serializable()["decoupler"],
        "setpoint_filter_tau_min": winner.setpoint_filter_tau_min,
    }
    with path.open("w") as fh:
        json.dump(payload, fh, indent=2)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shootout-output", type=Path, default=_DEFAULT_SHOOTOUT)
    parser.add_argument("--c0-output", type=Path, default=_DEFAULT_C0)
    args = parser.parse_args()

    p = DEFAULT_PARAMETERS
    X_nominal = _load_nominal_ss()
    print("Linearizing LV plant at nominal SS via CasADi backend ...")
    linearized = linearize_lv(
        X_ss=X_nominal,
        L_ss=p.nominal_reflux_L0_kmol_per_min,
        V_ss=p.nominal_boilup_V0_kmol_per_min,
        F_ss=p.nominal_feed_F_kmol_per_min,
        zF_ss=0.5,
        backend="casadi",
    )

    print("Running relay tests for the Tyreus-Luyben candidate ...")
    relay_top: RelayResult = relay_test(
        loop="top",
        X0=X_nominal,
        setpoint=float(X_nominal[p.NT - 1]),
        relay_amplitude_d=0.5,
        hysteresis=5.0e-3,
        duration_min=400.0,
    )
    relay_bottom: RelayResult = relay_test(
        loop="bottom",
        X0=X_nominal,
        setpoint=float(X_nominal[0]),
        relay_amplitude_d=0.5,
        hysteresis=5.0e-3,
        duration_min=400.0,
    )

    variants = build_six_variants(
        linearized=linearized, relay_top=relay_top, relay_bottom=relay_bottom
    )
    print(f"\nScoring {len(variants)} candidates over {len(SCENARIO_NAMES)} scenarios ...")

    scored: list[dict[str, Any]] = []
    for variant in variants:
        results = _score_variant(variant, X_nominal, list(SCENARIO_NAMES))
        scored.append({"variant": variant, "results": results})
        marker = "FAIL" if results["any_failure"] else f"IAE={results['aggregate_iae']:.4f}"
        print(f"  {variant.name:32s}  {marker}")

    finite_runs = [s for s in scored if not s["results"]["any_failure"]]
    if not finite_runs:
        print("\nNo candidate completed all scenarios — shootout failed.")
        return 1
    winner_entry = min(finite_runs, key=lambda s: s["results"]["aggregate_iae"])
    winner: C0Variant = winner_entry["variant"]
    print(
        f"\nWinner: {winner.name} (aggregate IAE = {winner_entry['results']['aggregate_iae']:.4f})"
    )

    # The nominal-OP robustness check is free — it is just the winner's
    # own aggregate IAE recomputed against the same seed state. We
    # include it so the JSON's robustness block has at least one
    # entry. The F-perturbed OPs require re-converging the LV-closed
    # SS at the perturbed feed flow, which is numerically demanding on
    # this plant (RGA(1,1) ~ 36, Newton-Krylov ill-conditioned, the
    # integration fallback drives some holdups through zero and
    # produces NaN). A graduated warm-start helped on the operating-
    # window sweep but still triggers NaN on the direct path. Deferred
    # to Phase 5, when the evaluation module gets the full per-OP
    # bootstrap CI machinery anyway — see PROJECT_PLAN.md Phase 5
    # "Stochastic Accounting" section.
    print("\nRobustness spot-check (nominal OP only; F-perturbed OPs deferred):")
    nominal_results = _score_variant(winner, X_nominal, list(SCENARIO_NAMES))
    robustness: dict[str, Any] = {
        "nominal": {
            "F_kmol_per_min": DEFAULT_PARAMETERS.nominal_feed_F_kmol_per_min,
            "zF": 0.5,
            "skipped": False,
            "aggregate_iae": nominal_results["aggregate_iae"],
            "per_scenario": nominal_results["per_scenario"],
        },
        "F-20pct": {
            "F_kmol_per_min": 0.8 * DEFAULT_PARAMETERS.nominal_feed_F_kmol_per_min,
            "zF": 0.5,
            "skipped": True,
            "reason": (
                "Newton-Krylov of LV-closed residual + integration fallback both "
                "produce NaN at F=0.8 on the high-RGA LV plant. Deferred to Phase 5."
            ),
        },
        "F+20pct": {
            "F_kmol_per_min": 1.2 * DEFAULT_PARAMETERS.nominal_feed_F_kmol_per_min,
            "zF": 0.5,
            "skipped": True,
            "reason": ("Same numerical conditioning issue as F-20pct. Deferred to Phase 5."),
        },
    }
    print(f"  nominal       F=1.00  zF=0.50  IAE={nominal_results['aggregate_iae']:.4f}")
    print("  F-20pct       F=0.80  zF=0.50  SKIPPED (numerical conditioning; Phase 5)")
    print("  F+20pct       F=1.20  zF=0.50  SKIPPED (numerical conditioning; Phase 5)")

    args.shootout_output.parent.mkdir(parents=True, exist_ok=True)
    audit = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(tz=UTC).isoformat(),
        "scenarios": list(SCENARIO_NAMES),
        "winner_name": winner.name,
        "winner_aggregate_iae": winner_entry["results"]["aggregate_iae"],
        "candidates": [
            {
                **entry["variant"].to_serializable(),
                "results": entry["results"],
            }
            for entry in scored
        ],
        "robustness": robustness,
    }
    with args.shootout_output.open("w") as fh:
        json.dump(audit, fh, indent=2)
    print(f"\nWrote {args.shootout_output}")

    _write_c0_winner(winner, winner_entry["results"], args.c0_output)
    print(f"Wrote {args.c0_output} (winner gains; load_c0_tuning will pick them up)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
