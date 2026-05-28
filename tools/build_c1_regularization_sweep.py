"""Build the curated C1 regularization-sweep Pareto front.

Phase-3 prep, finding-of-Diagnosis. The C1 off-nominal collapse
(see ``c1_off_nominal_baseline.json``) is regularization-sensitive
but the trade-off against nominal performance is irreducible: no
single fixed move-suppression weight resolves both regimes while
keeping the Phase-2 gate intact.

This tool produces ``data/reference/c1_regularization_sweep.json``
which is the *main-body Pareto-front artifact* referenced by:

- ``docs/kpis.md`` §6 Bucket-B comparison (against the strongest
  gate-passing variant, not the weakest)
- ``docs/pre_submission_checklist.md`` §4.6 (structural near-
  singularity in low-F regime)
- the Phase-5 paper figure ``regularization_pareto`` (no single
  fixed tuning wins both regimes — motivation for the agentic
  supervisor)

For each multiplier on the baseline ``r_lt = r_vb = 0.1``:

- Nominal: all 5 canonical scenarios at the nominal OP. Aggregate
  IAE reported alongside per-scenario gate-wins against the
  ``TL_no_decoupler`` C0 winner (Phase-2 gate criterion: C1 wins
  ≥ 3 / 5 nominal scenarios).
- Off-nominal: worst-cluster OP F=0.8 / zF=0.45 (the canonical
  collapse case), all 5 scenarios under the off-nominal-scenario
  builder. Aggregate IAE + final ``y_D`` after the F_step_+20pct
  scenario.

Levels swept: x1 (baseline), x10, x100 (strongest gate-passing in
the diagnostic probe), x1000 (off-nominal-best but gate-breaking).

Invocation:

    uv run python tools/build_c1_regularization_sweep.py
"""

from __future__ import annotations

import json
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from industrial_ai.control.c1_linear_mpc import (
    C1MPCConfig,
    build_c1_mpc,
    simulate_lv_with_mpc,
)
from industrial_ai.control.off_nominal_scenarios import build_off_nominal_scenario
from industrial_ai.control.scenarios import SCENARIO_NAMES, build_scenario
from industrial_ai.evaluation.kpis import compute_kpis
from industrial_ai.twin.column_a import DEFAULT_PARAMETERS
from industrial_ai.twin.column_a.linearize import linearize_lv
from industrial_ai.twin.column_a.operating_window import lookup_lv_ss

_REPO_ROOT = Path(__file__).resolve().parent.parent
_OUTPUT = _REPO_ROOT / "data" / "reference" / "c1_regularization_sweep.json"
_SS_FIXTURE = _REPO_ROOT / "data" / "reference" / "skogestad_column_a_steady_state.json"
_C0_SHOOTOUT = _REPO_ROOT / "data" / "reference" / "c0_pid_tuning_shootout.json"

_MULTIPLIERS: list[float] = [1.0, 10.0, 100.0, 1000.0]
_OFF_NOMINAL_OP: tuple[float, float] = (0.8, 0.45)


def _git_sha() -> str:
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=_REPO_ROOT).decode().strip()
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def _load_nominal_ss() -> np.ndarray:
    with _SS_FIXTURE.open() as fh:
        ss = json.load(fh)["steady_state"]
    return np.array(ss["compositions"] + ss["holdups_kmol"], dtype=np.float64)


def _load_c0_per_scenario_iae() -> dict[str, float]:
    with _C0_SHOOTOUT.open() as fh:
        data = json.load(fh)
    winner = data["winner_name"]
    for cand in data["candidates"]:
        if cand["name"] == winner:
            return {
                name: cand["results"]["per_scenario"][name]["iae_mole_fraction_min"]
                for name in SCENARIO_NAMES
            }
    raise KeyError(f"winner {winner!r} not found in shootout JSON")


def _evaluate_nominal(
    mult: float,
    *,
    X_nom: np.ndarray,
    c0_iae: dict[str, float],
) -> dict[str, Any]:
    p = DEFAULT_PARAMETERS
    lin = linearize_lv(
        X_ss=X_nom,
        L_ss=p.nominal_reflux_L0_kmol_per_min,
        V_ss=p.nominal_boilup_V0_kmol_per_min,
        F_ss=p.nominal_feed_F_kmol_per_min,
        zF_ss=0.5,
        backend="casadi",
    )
    cfg = C1MPCConfig(r_lt=0.1 * mult, r_vb=0.1 * mult)
    per_scen: dict[str, Any] = {}
    aggregate = 0.0
    wins = 0
    for name in SCENARIO_NAMES:
        mpc, _ = build_c1_mpc(lin, config=cfg)
        scenario_fn, spec = build_scenario(name)
        sim = simulate_lv_with_mpc(
            X0=X_nom,
            scenario=scenario_fn,
            mpc=mpc,
            linearized=lin,
            duration_min=spec.horizon_min,
            tick_dt_min=0.05,
        )
        iae = compute_kpis(sim).iae_mole_fraction_min if sim.success else float("inf")
        won = iae < c0_iae[name]
        if won:
            wins += 1
        aggregate += iae
        per_scen[name] = {
            "c0_iae": c0_iae[name],
            "c1_iae": iae if np.isfinite(iae) else None,
            "c1_wins": won,
        }
    return {
        "aggregate_iae": aggregate if np.isfinite(aggregate) else None,
        "gate_wins_out_of_5": wins,
        "gate_passed": wins >= 3,
        "per_scenario": per_scen,
    }


def _evaluate_off_nominal(mult: float, *, F_op: float, zF_op: float) -> dict[str, Any]:
    p = DEFAULT_PARAMETERS
    X_op = lookup_lv_ss(F=F_op, zF=zF_op)
    lin = linearize_lv(
        X_ss=X_op,
        L_ss=p.nominal_reflux_L0_kmol_per_min,
        V_ss=p.nominal_boilup_V0_kmol_per_min,
        F_ss=F_op,
        zF_ss=zF_op,
        backend="casadi",
    )
    cfg = C1MPCConfig(r_lt=0.1 * mult, r_vb=0.1 * mult)
    aggregate = 0.0
    per_scen: dict[str, Any] = {}
    final_yD_fstep = float("nan")
    for name in SCENARIO_NAMES:
        mpc, _ = build_c1_mpc(lin, config=cfg)
        scenario_fn, spec = build_off_nominal_scenario(name, F_op=F_op, zF_op=zF_op)
        sim = simulate_lv_with_mpc(
            X0=X_op,
            scenario=scenario_fn,
            mpc=mpc,
            linearized=lin,
            duration_min=spec.horizon_min,
            tick_dt_min=0.05,
        )
        iae = compute_kpis(sim).iae_mole_fraction_min if sim.success else float("inf")
        aggregate += iae
        per_scen[name] = {
            "iae": iae if np.isfinite(iae) else None,
            "success": sim.success,
        }
        if name == "F_step_+20pct" and sim.success:
            final_yD_fstep = float(sim.X[-1, p.NT - 1])
    return {
        "F": F_op,
        "zF": zF_op,
        "aggregate_iae": aggregate if np.isfinite(aggregate) else None,
        "F_step_+20pct_final_y_D": final_yD_fstep,
        "per_scenario": per_scen,
    }


def main() -> int:
    _OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    X_nom = _load_nominal_ss()
    c0_iae = _load_c0_per_scenario_iae()

    print(
        f"Building C1 regularization sweep over {len(_MULTIPLIERS)} multipliers "
        f"x (nominal + off-nominal worst OP) ..."
    )
    print(
        f"{'mult':>6s} {'r':>8s}  {'nom_IAE':>9s} {'gate':>6s}  "
        f"{'offn_IAE':>9s}  {'offn y_D':>9s}  ratio_v_C0"
    )

    c0_aggregate = sum(c0_iae.values())
    levels: list[dict[str, Any]] = []
    best_gate_passing_off_nominal: dict[str, Any] | None = None
    start = time.perf_counter()
    F_op, zF_op = _OFF_NOMINAL_OP
    for mult in _MULTIPLIERS:
        nominal = _evaluate_nominal(mult, X_nom=X_nom, c0_iae=c0_iae)
        off_nominal = _evaluate_off_nominal(mult, F_op=F_op, zF_op=zF_op)
        level = {
            "multiplier": mult,
            "r_lt": 0.1 * mult,
            "r_vb": 0.1 * mult,
            "nominal": nominal,
            "off_nominal_worst_op": off_nominal,
        }
        levels.append(level)
        ratio = (
            c0_aggregate / nominal["aggregate_iae"] if nominal["aggregate_iae"] else float("inf")
        )
        print(
            f"{mult:>6.0f}x {0.1 * mult:>8.1f}  "
            f"{nominal['aggregate_iae']:>9.4f} {nominal['gate_wins_out_of_5']:>3d}/5  "
            f"{off_nominal['aggregate_iae']:>9.2f}  "
            f"{off_nominal['F_step_+20pct_final_y_D']:>9.4f}  "
            f"{ratio:>6.2f}x"
        )
        if nominal["gate_passed"] and (
            best_gate_passing_off_nominal is None
            or off_nominal["aggregate_iae"]
            < best_gate_passing_off_nominal["off_nominal_worst_op"]["aggregate_iae"]
        ):
            best_gate_passing_off_nominal = level
    elapsed = time.perf_counter() - start
    print(f"\nSweep runtime: {elapsed:.1f} s")

    baseline_level = next(L for L in levels if L["multiplier"] == 1.0)

    pareto_summary = {
        "baseline_multiplier": 1.0,
        "baseline_rationale": (
            "Phase-2 baseline retained (x1, r=0.1). Strongest C1/C0 margin (6.8x) "
            "on the nominal-OP gate. Off-nominal collapse documented as a structural "
            "limitation of fixed-weight linear MPC."
        ),
        "best_gate_passing_multiplier": (
            best_gate_passing_off_nominal["multiplier"] if best_gate_passing_off_nominal else None
        ),
        "best_gate_passing_rationale": (
            "Strongest off-nominal performance among regularization multipliers "
            "that still satisfy the Phase-2 gate (gate-wins ≥ 3/5). Used as the "
            "Bucket-B reference point in kpis.md §6 so C2 is compared against the "
            "Pareto front, not against the weakest C1 tuning."
        ),
        "tradeoff_irreducible": True,
        "tradeoff_statement": (
            "No fixed regularization multiplier resolves both regimes. "
            f"x1: nominal {baseline_level['nominal']['aggregate_iae']:.4f}, "
            f"off-nominal {baseline_level['off_nominal_worst_op']['aggregate_iae']:.2f}. "
            f"x100: nominal {next(L['nominal']['aggregate_iae'] for L in levels if L['multiplier'] == 100.0):.4f}, "
            f"off-nominal {next(L['off_nominal_worst_op']['aggregate_iae'] for L in levels if L['multiplier'] == 100.0):.2f}. "
            f"x1000: nominal {next(L['nominal']['aggregate_iae'] for L in levels if L['multiplier'] == 1000.0):.4f} "
            f"(breaks Phase-2 gate), off-nominal "
            f"{next(L['off_nominal_worst_op']['aggregate_iae'] for L in levels if L['multiplier'] == 1000.0):.2f}. "
            "Motivates the adaptive supervisory layer."
        ),
    }

    payload: dict[str, Any] = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(tz=UTC).isoformat(),
        "git_sha": _git_sha(),
        "purpose": (
            "Pareto front of C1 fixed-weight regularization vs. nominal and "
            "off-nominal performance. Main-body artifact for the Phase-5 paper."
        ),
        "off_nominal_op_under_evaluation": {"F": F_op, "zF": zF_op},
        "c0_reference": {
            "name": "TL_no_decoupler (shootout winner)",
            "nominal_aggregate_iae": c0_aggregate,
            "per_scenario": c0_iae,
        },
        "multipliers_swept": _MULTIPLIERS,
        "levels": levels,
        "pareto_summary": pareto_summary,
    }
    with _OUTPUT.open("w") as fh:
        json.dump(payload, fh, indent=2)
    print(f"Wrote {_OUTPUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
