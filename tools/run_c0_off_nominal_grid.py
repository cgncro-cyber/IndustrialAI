"""Phase 3 prep — C0 PID across the kpis.md §2.2 off-nominal grid.

Symmetric companion to ``tools/run_c1_off_nominal_grid.py``. Evaluates
the fixed-gain Tyreus-Luyben C0 baseline (the shootout winner
``TL_no_decoupler``) at the same 16 off-nominal OPs. The same
controller gains are applied across the grid without re-tuning, by
design: the Phase 2 Day 2.6 finding (``docs/pre_submission_checklist.md``
§4.4) that fixed-gain TL does not extrapolate is a publishable C0
limitation, and this script quantifies it.

Outputs ``data/reference/c0_off_nominal_baseline.json`` with:

- per-OP per-scenario IAE (80 individual numbers; ``inf`` where the
  trajectory diverged to NaN)
- per-OP aggregate IAE (16 numbers)
- ``infeasibility_count``: number of (OP, scenario) cells where the
  simulator bailed out on NaN — the headline C0-locality evidence
- grid aggregate stats: mean / P95 / max over the finite per-OP
  aggregates, plus the count of finite OPs

Invocation:

    uv run python tools/run_c0_off_nominal_grid.py
"""

from __future__ import annotations

import argparse
import json
import signal
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from industrial_ai.control.c0_pid_only import build_c0_pids
from industrial_ai.control.off_nominal_scenarios import build_off_nominal_scenario
from industrial_ai.control.scenarios import SCENARIO_NAMES
from industrial_ai.evaluation.kpis import compute_kpis
from industrial_ai.twin.column_a import DEFAULT_PARAMETERS
from industrial_ai.twin.column_a.operating_window import lookup_lv_ss
from industrial_ai.twin.simulate import simulate_lv_closed_loop

_REPO_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_OUT = _REPO_ROOT / "data" / "reference" / "c0_off_nominal_baseline.json"

_F_AXIS = (0.8, 0.9, 1.1, 1.2)
_ZF_AXIS = (0.45, 0.475, 0.525, 0.55)
# Per-scenario wall-clock cap (s). The well-behaved C1 nominal sim
# is ~2.5 s; PID-saturated off-nominal sims can grind inside a single
# solve_ivp call (LSODA taking unbounded internal substeps in a stiff
# regime) for minutes without producing NaN or returning control to
# Python between ticks. We use a SIGALRM-based interrupt because the
# in-simulator per-tick cap fires only at tick boundaries and is too
# coarse for this case. Overruns count as §4.4 non-extrapolation
# events — the documented C0 locality limitation.
_PER_SCENARIO_WALL_CLOCK_S = 30.0


class _ScenarioTimeoutError(Exception):
    """Raised by the SIGALRM handler when a scenario exceeds its cap."""


def _alarm_handler(_signum: int, _frame: Any) -> None:
    raise _ScenarioTimeoutError


def _git_sha() -> str:
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=_REPO_ROOT).decode().strip()
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def _evaluate_one_op(F_op: float, zF_op: float) -> dict[str, Any]:
    p = DEFAULT_PARAMETERS
    X_op = lookup_lv_ss(F=F_op, zF=zF_op)
    L0 = p.nominal_reflux_L0_kmol_per_min
    V0 = p.nominal_boilup_V0_kmol_per_min

    scenarios_block: dict[str, Any] = {}
    op_aggregate = 0.0
    op_inf_count = 0
    max_wall = 0.0
    for name in SCENARIO_NAMES:
        scenario_fn, spec = build_off_nominal_scenario(name, F_op=F_op, zF_op=zF_op)
        # Fresh PID instances per scenario — seed integrals from the
        # current bias so a long startup transient does not skew the
        # KPI between scenarios.
        top, bottom = build_c0_pids(LT_initial=L0, VB_initial=V0)
        signal.signal(signal.SIGALRM, _alarm_handler)
        signal.setitimer(signal.ITIMER_REAL, _PER_SCENARIO_WALL_CLOCK_S)
        try:
            sim = simulate_lv_closed_loop(
                X0=X_op,
                scenario=scenario_fn,
                duration_min=spec.horizon_min,
                tick_dt_min=0.05,
                pid_top=top,
                pid_bottom=bottom,
            )
            signal.setitimer(signal.ITIMER_REAL, 0)
            timed_out = False
        except _ScenarioTimeoutError:
            signal.setitimer(signal.ITIMER_REAL, 0)
            timed_out = True
            sim = None  # type: ignore[assignment]
        if timed_out:
            iae = float("inf")
            finite = False
            sim_success = False
        elif sim.success:
            kpis = compute_kpis(sim)
            iae = kpis.iae_mole_fraction_min
            finite = bool(np.isfinite(iae))
            sim_success = True
        else:
            iae = float("inf")
            finite = False
            sim_success = False
        if not finite:
            op_inf_count += 1
        if sim is not None:
            max_wall = max(max_wall, float(sim.cycle_wall_clock_seconds.max()))
        scenarios_block[name] = {
            "success": sim_success,
            "timed_out": timed_out,
            "iae_mole_fraction_min": iae if finite else None,
            "iae_finite": finite,
        }
        op_aggregate += iae

    finite_agg = bool(np.isfinite(op_aggregate))
    return {
        "F": F_op,
        "zF": zF_op,
        "lv_closed_ss": {
            "y_D": float(X_op[DEFAULT_PARAMETERS.NT - 1]),
            "x_B": float(X_op[0]),
        },
        "aggregate_iae": op_aggregate if finite_agg else None,
        "aggregate_iae_finite": finite_agg,
        "infeasibility_count": op_inf_count,
        "max_cycle_wall_clock_seconds": max_wall,
        "scenarios": scenarios_block,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=_DEFAULT_OUT)
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)

    grid = [(F, zF) for F in _F_AXIS for zF in _ZF_AXIS]
    print(f"Evaluating C0 (TL_no_decoupler) across {len(grid)} off-nominal OPs ...")
    print(f"{'F':>5s} {'zF':>6s} {'y_D_op':>7s} {'x_B_op':>7s} {'agg_IAE':>10s} {'inf_cnt':>7s}")

    per_op_blocks: list[dict[str, Any]] = []
    finite_aggregates: list[float] = []
    total_inf = 0
    start = time.perf_counter()
    for F_op, zF_op in grid:
        block = _evaluate_one_op(F_op, zF_op)
        per_op_blocks.append(block)
        if block["aggregate_iae_finite"]:
            finite_aggregates.append(block["aggregate_iae"])
        total_inf += block["infeasibility_count"]
        print(
            f"{F_op:>5.2f} {zF_op:>6.3f} "
            f"{block['lv_closed_ss']['y_D']:>7.4f} {block['lv_closed_ss']['x_B']:>7.4f} "
            f"{(block['aggregate_iae'] if block['aggregate_iae'] is not None else float('inf')):>10.4f} "
            f"{block['infeasibility_count']:>7d}"
        )
    elapsed = time.perf_counter() - start

    if finite_aggregates:
        aggs = np.array(finite_aggregates, dtype=np.float64)
        grid_stats = {
            "mean": float(aggs.mean()),
            "p50": float(np.percentile(aggs, 50)),
            "p95": float(np.percentile(aggs, 95)),
            "max": float(aggs.max()),
            "min": float(aggs.min()),
            "ops_finite": len(finite_aggregates),
            "ops_total": len(grid),
        }
    else:
        grid_stats = {
            "mean": None,
            "p50": None,
            "p95": None,
            "max": None,
            "min": None,
            "ops_finite": 0,
            "ops_total": len(grid),
        }

    print(
        f"\nFinite OPs: {grid_stats['ops_finite']}/{len(grid)};  "
        f"total infeasibility events: {total_inf}/{len(grid) * 5}"
    )
    if grid_stats["mean"] is not None:
        print(
            f"Finite-only stats: mean={grid_stats['mean']:.4f}  "
            f"P95={grid_stats['p95']:.4f}  max={grid_stats['max']:.4f}"
        )
    print(f"Total runtime: {elapsed:.1f} s")

    payload: dict[str, Any] = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(tz=UTC).isoformat(),
        "git_sha": _git_sha(),
        "grid": {
            "F_axis": list(_F_AXIS),
            "zF_axis": list(_ZF_AXIS),
            "n_ops": len(grid),
            "source": "docs/kpis.md §2.2",
        },
        "c0_config": {
            "tuning": "TL_no_decoupler (shootout winner, c0_pid_tuning.json)",
            "retune_per_op": False,
            "per_scenario_wall_clock_cap_s": _PER_SCENARIO_WALL_CLOCK_S,
            "rationale": "Fixed-gain TL non-extrapolation is the documented C0 locality limitation (pre_submission_checklist.md §4.4); the wall-clock cap bounds pathological stiff-regime sims that the simulator cannot abort internally.",
        },
        "scenario_set": "industrial_ai.control.off_nominal_scenarios — same relative magnitudes from each OP",
        "aggregate": {
            **grid_stats,
            "total_scenario_infeasibilities": total_inf,
            "total_scenario_cells": len(grid) * len(SCENARIO_NAMES),
        },
        "per_op": per_op_blocks,
    }
    with args.output.open("w") as fh:
        json.dump(payload, fh, indent=2)
    print(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
