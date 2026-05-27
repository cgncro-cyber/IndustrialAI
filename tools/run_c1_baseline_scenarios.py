"""Phase 2 Day 3 — run the C1 Linear MPC over the 5 canonical scenarios.

Writes ``data/reference/c1_baseline_kpis.json`` with per-scenario KPIs
and the aggregate IAE that the Phase-2 gate compares against the C0
shootout winner. The gate criterion from PROJECT_PLAN.md is:

    C1 must outperform C0 on at least 3 of the 5 disturbance scenarios.

Invocation:

    uv run python tools/run_c1_baseline_scenarios.py
"""

from __future__ import annotations

import argparse
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
from industrial_ai.control.scenarios import SCENARIO_NAMES, build_scenario
from industrial_ai.evaluation.kpis import compute_kpis
from industrial_ai.twin.column_a import DEFAULT_PARAMETERS
from industrial_ai.twin.column_a.linearize import linearize_lv

_REPO_ROOT = Path(__file__).resolve().parent.parent
_SS_FIXTURE = _REPO_ROOT / "data" / "reference" / "skogestad_column_a_steady_state.json"
_DEFAULT_OUT = _REPO_ROOT / "data" / "reference" / "c1_baseline_kpis.json"
_C0_SHOOTOUT = _REPO_ROOT / "data" / "reference" / "c0_pid_tuning_shootout.json"


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
    """Extract the winner's per-scenario IAE from c0_pid_tuning_shootout.json."""
    with _C0_SHOOTOUT.open() as fh:
        data = json.load(fh)
    winner_name = data["winner_name"]
    for cand in data["candidates"]:
        if cand["name"] == winner_name:
            per_scen = cand["results"]["per_scenario"]
            return {name: per_scen[name]["iae_mole_fraction_min"] for name in SCENARIO_NAMES}
    raise KeyError(f"winner {winner_name!r} not found in shootout JSON")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=_DEFAULT_OUT)
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    p = DEFAULT_PARAMETERS
    X_nominal = _load_nominal_ss()
    c0_iae = _load_c0_per_scenario_iae()

    print("Linearizing LV plant at nominal SS via CasADi backend ...")
    linearized = linearize_lv(
        X_ss=X_nominal,
        L_ss=p.nominal_reflux_L0_kmol_per_min,
        V_ss=p.nominal_boilup_V0_kmol_per_min,
        F_ss=p.nominal_feed_F_kmol_per_min,
        zF_ss=0.5,
        backend="casadi",
    )
    mpc, _ = build_c1_mpc(linearized)

    print(f"Scoring C1 (Linear MPC) over {len(SCENARIO_NAMES)} scenarios ...")
    print(f"{'scenario':25s}  {'C0 IAE':>8s}  {'C1 IAE':>8s}  ratio   winner")
    scenarios_block: dict[str, Any] = {}
    total_c1 = 0.0
    total_c0 = 0.0
    wins = 0
    start = time.perf_counter()
    for name in SCENARIO_NAMES:
        scenario_fn, spec = build_scenario(name)
        sim = simulate_lv_with_mpc(
            X0=X_nominal,
            scenario=scenario_fn,
            mpc=mpc,
            linearized=linearized,
            duration_min=spec.horizon_min,
            tick_dt_min=0.05,
        )
        kpis = compute_kpis(sim)
        iae_c1 = kpis.iae_mole_fraction_min
        iae_c0 = c0_iae[name]
        ratio = iae_c0 / iae_c1 if iae_c1 > 0 else float("inf")
        winner = "C1" if iae_c1 < iae_c0 else "C0"
        if winner == "C1":
            wins += 1
        total_c1 += iae_c1
        total_c0 += iae_c0
        scenarios_block[name] = {
            "success": sim.success,
            "c0_iae_mole_fraction_min": iae_c0,
            "c1_iae_mole_fraction_min": iae_c1,
            "ratio_c0_over_c1": ratio,
            "winner": winner,
            "max_cycle_wall_clock_seconds": float(sim.cycle_wall_clock_seconds.max()),
            **kpis.as_dict(),
        }
        print(f"  {name:25s}  {iae_c0:>8.4f}  {iae_c1:>8.4f}  {ratio:>5.2f}x  {winner}")
    duration_s = time.perf_counter() - start
    print(
        f"  {'AGGREGATE':25s}  {total_c0:>8.4f}  {total_c1:>8.4f}  "
        f"{total_c0 / total_c1:>5.2f}x  C1 wins {wins}/5  (runtime {duration_s:.1f} s)"
    )

    gate_passed = wins >= 3
    print(f"\nPhase-2 gate (C1 beats C0 on >=3/5 scenarios): {'PASS' if gate_passed else 'FAIL'}")

    payload: dict[str, Any] = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(tz=UTC).isoformat(),
        "git_sha": _git_sha(),
        "operating_point": {
            "case": "Skogestad Column A, nominal SS",
            "F_kmol_per_min": p.nominal_feed_F_kmol_per_min,
            "zF": 0.5,
            "qF": p.nominal_feed_liquid_fraction_qF,
        },
        "mpc_config": {
            "sampling_time_min": C1MPCConfig().sampling_time_min,
            "n_horizon": C1MPCConfig().n_horizon,
            "q_top": C1MPCConfig().q_top,
            "q_bottom": C1MPCConfig().q_bottom,
            "r_lt": C1MPCConfig().r_lt,
            "r_vb": C1MPCConfig().r_vb,
        },
        "aggregate": {
            "c0_iae": total_c0,
            "c1_iae": total_c1,
            "ratio_c0_over_c1": total_c0 / total_c1 if total_c1 > 0 else float("inf"),
            "scenarios_won_by_c1": wins,
            "gate_passed": gate_passed,
        },
        "scenarios": scenarios_block,
    }
    with args.output.open("w") as fh:
        json.dump(payload, fh, indent=2)
    print(f"Wrote {args.output}")
    return 0 if gate_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
