"""Phase 3 prep — C1 Linear MPC across the kpis.md §2.2 off-nominal grid.

Evaluates the C1 supervisor at each of 16 off-nominal operating points
(``F ∈ {0.8, 0.9, 1.1, 1.2}, zF ∈ {0.45, 0.475, 0.525, 0.55}``), with
the linearization re-computed per OP and the canonical five
disturbance scenarios applied with the same relative magnitudes from
each OP (see :mod:`industrial_ai.control.off_nominal_scenarios`).

Outputs ``data/reference/c1_off_nominal_baseline.json`` containing:

- per-OP per-scenario IAE (80 individual numbers)
- per-OP aggregate IAE (16 numbers)
- grid aggregate stats: mean, P95, max
- per-OP ``linearization_drift_g`` (kpis.md §4.1)
- ``linearization_consistency`` (kpis.md §4.1)
- ``linearization_recompute_count`` (= 16, per kpis.md §2.3)

The headline ``off_nominal_robustness_iae`` is the P95 of the
per-OP aggregate IAEs. Phase 3 will compare C2 against this number
under the Bucket B classification rule (``docs/kpis.md`` §2.4).

Invocation:

    uv run python tools/run_c1_off_nominal_grid.py
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
from industrial_ai.control.off_nominal_scenarios import build_off_nominal_scenario
from industrial_ai.control.scenarios import SCENARIO_NAMES
from industrial_ai.evaluation.kpis import compute_kpis
from industrial_ai.twin.column_a import DEFAULT_PARAMETERS
from industrial_ai.twin.column_a.linearize import linearize_lv
from industrial_ai.twin.column_a.operating_window import lookup_lv_ss

_REPO_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_OUT = _REPO_ROOT / "data" / "reference" / "c1_off_nominal_baseline.json"
_SS_FIXTURE = _REPO_ROOT / "data" / "reference" / "skogestad_column_a_steady_state.json"

_F_AXIS = (0.8, 0.9, 1.1, 1.2)
_ZF_AXIS = (0.45, 0.475, 0.525, 0.55)


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


def _evaluate_one_op(
    *,
    F_op: float,
    zF_op: float,
    A_nom: np.ndarray,
    A_nom_norm: float,
) -> dict[str, Any]:
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
    A_g = np.asarray(lin.A, dtype=np.float64)
    drift = float(np.linalg.norm(A_g - A_nom, ord=2))
    drift_rel = drift / A_nom_norm

    mpc, _ = build_c1_mpc(lin)

    scenarios_block: dict[str, Any] = {}
    op_aggregate_iae = 0.0
    max_wall = 0.0
    any_failure = False
    for name in SCENARIO_NAMES:
        scenario_fn, spec = build_off_nominal_scenario(name, F_op=F_op, zF_op=zF_op)
        sim = simulate_lv_with_mpc(
            X0=X_op,
            scenario=scenario_fn,
            mpc=mpc,
            linearized=lin,
            duration_min=spec.horizon_min,
            tick_dt_min=0.05,
        )
        if sim.success:
            kpis = compute_kpis(sim)
            iae = kpis.iae_mole_fraction_min
        else:
            iae = float("inf")
            any_failure = True
        max_wall = max(max_wall, float(sim.cycle_wall_clock_seconds.max()))
        scenarios_block[name] = {
            "success": sim.success,
            "iae_mole_fraction_min": iae if np.isfinite(iae) else None,
            "iae_finite": bool(np.isfinite(iae)),
        }
        op_aggregate_iae += iae

    return {
        "F": F_op,
        "zF": zF_op,
        "lv_closed_ss": {
            "y_D": float(X_op[DEFAULT_PARAMETERS.NT - 1]),
            "x_B": float(X_op[0]),
        },
        "linearization_drift_g": drift,
        "linearization_drift_rel": drift_rel,
        "aggregate_iae": op_aggregate_iae if np.isfinite(op_aggregate_iae) else None,
        "aggregate_iae_finite": bool(np.isfinite(op_aggregate_iae)),
        "max_cycle_wall_clock_seconds": max_wall,
        "any_scenario_failed": any_failure,
        "scenarios": scenarios_block,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=_DEFAULT_OUT)
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    p = DEFAULT_PARAMETERS

    print("Linearizing LV plant at nominal OP (reference for drift) ...")
    X_nom = _load_nominal_ss()
    lin_nom = linearize_lv(
        X_ss=X_nom,
        L_ss=p.nominal_reflux_L0_kmol_per_min,
        V_ss=p.nominal_boilup_V0_kmol_per_min,
        F_ss=p.nominal_feed_F_kmol_per_min,
        zF_ss=0.5,
        backend="casadi",
    )
    A_nom = np.asarray(lin_nom.A, dtype=np.float64)
    A_nom_norm = float(np.linalg.norm(A_nom, ord=2))

    grid = [(F, zF) for F in _F_AXIS for zF in _ZF_AXIS]
    print(f"Evaluating C1 across {len(grid)} off-nominal OPs ...")
    print(
        f"{'F':>5s} {'zF':>6s} {'y_D_op':>7s} {'x_B_op':>7s} "
        f"{'drift':>7s} {'agg_IAE':>9s} {'maxWall_s':>9s}"
    )

    per_op_blocks: list[dict[str, Any]] = []
    aggregates: list[float] = []
    finite_count = 0
    start = time.perf_counter()
    for F_op, zF_op in grid:
        block = _evaluate_one_op(F_op=F_op, zF_op=zF_op, A_nom=A_nom, A_nom_norm=A_nom_norm)
        per_op_blocks.append(block)
        agg = block["aggregate_iae"]
        if agg is not None and np.isfinite(agg):
            aggregates.append(agg)
            finite_count += 1
        print(
            f"{F_op:>5.2f} {zF_op:>6.3f} "
            f"{block['lv_closed_ss']['y_D']:>7.4f} {block['lv_closed_ss']['x_B']:>7.4f} "
            f"{block['linearization_drift_g']:>7.2f} "
            f"{(agg if agg is not None else float('inf')):>9.4f} "
            f"{block['max_cycle_wall_clock_seconds']:>9.3f}"
        )
    elapsed = time.perf_counter() - start

    if not aggregates:
        raise RuntimeError("no OP produced a finite aggregate IAE — check solver / scenarios")

    aggs = np.array(aggregates, dtype=np.float64)
    grid_stats = {
        "mean": float(aggs.mean()),
        "p50": float(np.percentile(aggs, 50)),
        "p95": float(np.percentile(aggs, 95)),
        "max": float(aggs.max()),
        "min": float(aggs.min()),
        "ops_finite": int(finite_count),
        "ops_total": len(grid),
    }
    drifts = np.array([b["linearization_drift_rel"] for b in per_op_blocks], dtype=np.float64)
    linearization_consistency = float(1.0 - np.percentile(drifts, 95))

    print(
        f"\nAggregate over {len(grid)} OPs: "
        f"mean={grid_stats['mean']:.4f}  P95={grid_stats['p95']:.4f}  "
        f"max={grid_stats['max']:.4f}  ({finite_count}/{len(grid)} finite)"
    )
    print(f"linearization_consistency (1 - P95 rel drift): {linearization_consistency:.3f}")
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
        "mpc_config": {
            "sampling_time_min": C1MPCConfig().sampling_time_min,
            "n_horizon": C1MPCConfig().n_horizon,
            "q_top": C1MPCConfig().q_top,
            "q_bottom": C1MPCConfig().q_bottom,
            "r_lt": C1MPCConfig().r_lt,
            "r_vb": C1MPCConfig().r_vb,
        },
        "scenario_set": "industrial_ai.control.off_nominal_scenarios — same relative magnitudes from each OP",
        "linearization": {
            "A_nom_spectral_norm": A_nom_norm,
            "linearization_recompute_count": len(grid),
            "linearization_consistency": linearization_consistency,
        },
        "aggregate": {
            **grid_stats,
            "off_nominal_robustness_iae": grid_stats["p95"],
        },
        "per_op": per_op_blocks,
    }
    with args.output.open("w") as fh:
        json.dump(payload, fh, indent=2)
    print(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
