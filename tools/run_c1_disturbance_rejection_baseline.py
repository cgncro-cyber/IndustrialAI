"""C1 disturbance-rejection-only off-nominal baseline (kpis.md §2.4 sub-metric).

The Bucket-B comparator that Phase-3 C2 evaluations score against on
the pure disturbance-rejection axis. The target-acquisition sub-metric
(§2.3) is the existing ``c1_off_nominal_baseline.json``; this tool
produces its disturbance-rejection-only counterpart.

Method (per kpis.md §2.4):

- For each off-nominal OP ``g = (F, zF) ∈ G``, find ``(LT*, VB*)``
  inside ``[0, 10]²`` such that the LV-closed SS produces composition
  on-target ``(y_D = 0.99, x_B = 0.01)`` within tolerance. 2D Newton
  on the composition-error map; the inner residual reuses
  ``solve_lv_closed_steady_state``.
- Cache the per-OP ``(X0_onspec, LT*, VB*)`` to
  ``data/reference/off_nominal_on_spec_pre_stages.json``. Idempotent
  on re-run.
- Build the Pareto-reference C1 (``r_lt = r_vb = 10``, the ×100
  multiplier from ``c1_regularization_sweep.json``) linearized
  *around X0_onspec*, not the natural off-nominal SS. The MPC's
  internal model and the operating point agree on where the column
  already is — the test is purely *can the supervisor reject the
  disturbance once it is composition-on-spec at this OP*.
- Apply all five canonical disturbance scenarios from
  :mod:`industrial_ai.control.off_nominal_scenarios` at each OP and
  record the per-OP aggregate IAE.

Output: ``data/reference/c1_disturbance_rejection_baseline.json``
with per-OP IAE, the pre-stage MV setpoints, the P95/max aggregate
across the grid, and the Pareto-reference MPC config used.

Infeasibility handling: OPs where ``(LT*, VB*)`` is not found inside
the bounds are excluded with an explicit ``infeasible_pre_stage_count``
alongside (none observed in the current 16-point grid; defensive
against future grid extensions).

Invocation:

    uv run python tools/run_c1_disturbance_rejection_baseline.py
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
from scipy.optimize import root

from industrial_ai.control.c1_linear_mpc import (
    C1MPCConfig,
    build_c1_mpc,
    simulate_lv_with_mpc,
)
from industrial_ai.control.off_nominal_scenarios import build_off_nominal_scenario
from industrial_ai.control.scenarios import SCENARIO_NAMES
from industrial_ai.evaluation.kpis import compute_kpis
from industrial_ai.twin.column_a import DEFAULT_PARAMETERS
from industrial_ai.twin.column_a.configurations.lv import LVConfiguration
from industrial_ai.twin.column_a.linearize import linearize_lv
from industrial_ai.twin.column_a.operating_window import (
    GridPoint,
    lookup_lv_ss,
    solve_lv_closed_steady_state,
)

_REPO_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_OUT = _REPO_ROOT / "data" / "reference" / "c1_disturbance_rejection_baseline.json"
_PRE_STAGE_CACHE = _REPO_ROOT / "data" / "reference" / "off_nominal_on_spec_pre_stages.json"

_F_AXIS = (0.8, 0.9, 1.1, 1.2)
_ZF_AXIS = (0.45, 0.475, 0.525, 0.55)
_TARGET_Y_D = 0.99
_TARGET_X_B = 0.01
_PRE_STAGE_TOL = 5.0e-4

# Pareto reference per kpis.md §6 Step 3 (x100 multiplier on the
# baseline r_lt = r_vb = 0.1).
_PARETO_MULTIPLIER = 100.0


def _git_sha() -> str:
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=_REPO_ROOT).decode().strip()
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def _find_pre_stage_mvs(
    F_op: float,
    zF_op: float,
    *,
    warm_X: np.ndarray | None = None,
) -> tuple[np.ndarray, float, float, bool, float]:
    """Find ``(LT*, VB*)`` such that LV-closed SS hits the composition target.

    Returns ``(X0_onspec, LT_star, VB_star, success, residual_norm)``.
    ``success`` is True iff a root was found inside MV bounds and the
    composition error is within ``_PRE_STAGE_TOL``.
    """
    p = DEFAULT_PARAMETERS
    NT = p.NT
    lv_config = LVConfiguration()
    # Warm-start at the natural off-nominal LV-closed SS for that OP
    # (sweep cache; this is the SS at LT=L0, VB=V0).
    if warm_X is None:
        warm_X = lookup_lv_ss(F=F_op, zF=zF_op)

    def composition_error(lt_vb: np.ndarray) -> np.ndarray:
        lt, vb = float(lt_vb[0]), float(lt_vb[1])
        # Clamp to bounds so the outer Newton doesn't probe nonsense.
        lt = max(0.0, min(10.0, lt))
        vb = max(0.0, min(10.0, vb))
        point = GridPoint(LT=lt, VB=vb, F=F_op, zF=zF_op, qF=1.0)
        X_star, _resid, ok = solve_lv_closed_steady_state(
            point=point,
            X0=warm_X,
            parameters=p,
            lv_config=lv_config,
            residual_tol=1e-7,
            max_iter=200,
        )
        if not ok:
            # Push outer Newton away from this corner by returning a
            # large penalty that's still smooth; scipy handles inf
            # poorly inside the Jacobian.
            return np.array([1.0, 1.0], dtype=np.float64)
        return np.array([X_star[NT - 1] - _TARGET_Y_D, X_star[0] - _TARGET_X_B], dtype=np.float64)

    # Initial guess: nominal LT/VB.
    lt0 = p.nominal_reflux_L0_kmol_per_min
    vb0 = p.nominal_boilup_V0_kmol_per_min
    sol = root(composition_error, x0=np.array([lt0, vb0]), method="hybr", tol=1e-6)
    lt_star = float(np.clip(sol.x[0], 0.0, 10.0))
    vb_star = float(np.clip(sol.x[1], 0.0, 10.0))

    # Solve the SS at the converged MV pair to get the actual X0_onspec.
    point = GridPoint(LT=lt_star, VB=vb_star, F=F_op, zF=zF_op, qF=1.0)
    X_onspec, _resid, ok = solve_lv_closed_steady_state(
        point=point,
        X0=warm_X,
        parameters=p,
        lv_config=lv_config,
        residual_tol=1e-7,
        max_iter=200,
    )
    err = float(
        np.linalg.norm(
            np.array([X_onspec[NT - 1] - _TARGET_Y_D, X_onspec[0] - _TARGET_X_B], dtype=np.float64),
            ord=np.inf,
        )
    )
    # The composition-error norm against the target is the source of
    # truth — scipy.optimize.root sometimes reports sol.success=False
    # when its internal step-size criterion stalls, even though the
    # converged (LT*, VB*) pair lands within tolerance on the actual
    # composition map.
    success = bool(ok and err <= _PRE_STAGE_TOL)
    return X_onspec, lt_star, vb_star, success, err


def _build_pre_stage_cache() -> dict[str, Any]:
    grid = [(F, zF) for F in _F_AXIS for zF in _ZF_AXIS]
    print(f"Pre-staging {len(grid)} off-nominal OPs to composition-on-target SS ...")
    print(f"{'F':>5s} {'zF':>6s} {'LT*':>8s} {'VB*':>8s} {'y_D':>8s} {'x_B':>8s}  err  success")
    cache: dict[str, Any] = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(tz=UTC).isoformat(),
        "git_sha": _git_sha(),
        "target": {"y_D": _TARGET_Y_D, "x_B": _TARGET_X_B},
        "pre_stage_tolerance": _PRE_STAGE_TOL,
        "ops": [],
    }
    for F_op, zF_op in grid:
        X_onspec, lt_star, vb_star, ok, err = _find_pre_stage_mvs(F_op=F_op, zF_op=zF_op)
        NT = DEFAULT_PARAMETERS.NT
        block = {
            "F": F_op,
            "zF": zF_op,
            "LT_star": lt_star,
            "VB_star": vb_star,
            "y_D_at_onspec": float(X_onspec[NT - 1]),
            "x_B_at_onspec": float(X_onspec[0]),
            "composition_error_inf_norm": err,
            "success": ok,
            "state_vector": X_onspec.tolist(),
        }
        cache["ops"].append(block)
        print(
            f"{F_op:>5.2f} {zF_op:>6.3f} {lt_star:>8.4f} {vb_star:>8.4f} "
            f"{X_onspec[NT - 1]:>8.4f} {X_onspec[0]:>8.4f}  {err:>.2e}  {ok}"
        )
    return cache


def _load_or_build_pre_stage_cache() -> dict[str, Any]:
    if _PRE_STAGE_CACHE.exists():
        with _PRE_STAGE_CACHE.open() as fh:
            return json.load(fh)
    cache = _build_pre_stage_cache()
    _PRE_STAGE_CACHE.parent.mkdir(parents=True, exist_ok=True)
    with _PRE_STAGE_CACHE.open("w") as fh:
        json.dump(cache, fh, indent=2)
    print(f"Wrote {_PRE_STAGE_CACHE}")
    return cache


def _evaluate_one_op(
    *,
    op_block: dict[str, Any],
    pareto_config: C1MPCConfig,
) -> dict[str, Any]:
    F_op = op_block["F"]
    zF_op = op_block["zF"]
    X_onspec = np.array(op_block["state_vector"], dtype=np.float64)
    lt_star = op_block["LT_star"]
    vb_star = op_block["VB_star"]
    # Linearize around the *on-spec* SS at this OP with the actual
    # bias MVs (LT*, VB*). The MPC's internal model agrees with where
    # the column is sitting before the step — the test is pure
    # disturbance-rejection.
    lin = linearize_lv(
        X_ss=X_onspec,
        L_ss=lt_star,
        V_ss=vb_star,
        F_ss=F_op,
        zF_ss=zF_op,
        backend="casadi",
    )
    mpc, _ = build_c1_mpc(lin, config=pareto_config)

    scenarios_block: dict[str, Any] = {}
    aggregate = 0.0
    max_wall = 0.0
    any_failure = False
    for name in SCENARIO_NAMES:
        scenario_fn, spec = build_off_nominal_scenario(name, F_op=F_op, zF_op=zF_op)
        sim = simulate_lv_with_mpc(
            X0=X_onspec,
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
        }
        aggregate += iae

    return {
        "F": F_op,
        "zF": zF_op,
        "pre_stage_mvs": {"LT_star": lt_star, "VB_star": vb_star},
        "y_D_at_onspec": op_block["y_D_at_onspec"],
        "x_B_at_onspec": op_block["x_B_at_onspec"],
        "aggregate_iae": aggregate if np.isfinite(aggregate) else None,
        "aggregate_iae_finite": bool(np.isfinite(aggregate)),
        "max_cycle_wall_clock_seconds": max_wall,
        "any_scenario_failed": any_failure,
        "scenarios": scenarios_block,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=_DEFAULT_OUT)
    parser.add_argument(
        "--force-pre-stage",
        action="store_true",
        help="Re-compute the pre-stage MV cache from scratch.",
    )
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    if args.force_pre_stage and _PRE_STAGE_CACHE.exists():
        _PRE_STAGE_CACHE.unlink()

    pre_stage = _load_or_build_pre_stage_cache()
    pareto = C1MPCConfig(r_lt=0.1 * _PARETO_MULTIPLIER, r_vb=0.1 * _PARETO_MULTIPLIER)

    print(
        f"\nEvaluating C1 (Pareto-reference x{_PARETO_MULTIPLIER:.0f}, "
        f"r_lt=r_vb={pareto.r_lt}) at each pre-staged OP ..."
    )
    print(f"{'F':>5s} {'zF':>6s} {'agg_IAE':>10s}  {'max_wall_s':>10s}")
    per_op: list[dict[str, Any]] = []
    finite_aggregates: list[float] = []
    pre_stage_infeas = 0
    start = time.perf_counter()
    for block in pre_stage["ops"]:
        if not block["success"]:
            pre_stage_infeas += 1
            per_op.append(
                {
                    "F": block["F"],
                    "zF": block["zF"],
                    "skipped": "pre_stage_infeasible",
                    "pre_stage_error_inf_norm": block["composition_error_inf_norm"],
                }
            )
            print(
                f"{block['F']:>5.2f} {block['zF']:>6.3f}  "
                f"PRE-STAGE INFEASIBLE (err {block['composition_error_inf_norm']:.2e})"
            )
            continue
        result = _evaluate_one_op(op_block=block, pareto_config=pareto)
        per_op.append(result)
        if result["aggregate_iae_finite"]:
            finite_aggregates.append(result["aggregate_iae"])
        print(
            f"{result['F']:>5.2f} {result['zF']:>6.3f}  "
            f"{(result['aggregate_iae'] if result['aggregate_iae'] is not None else float('inf')):>10.4f}  "
            f"{result['max_cycle_wall_clock_seconds']:>10.3f}"
        )
    elapsed = time.perf_counter() - start

    aggs = np.array(finite_aggregates, dtype=np.float64) if finite_aggregates else None
    grid_stats: dict[str, Any]
    if aggs is not None and len(aggs) > 0:
        grid_stats = {
            "mean": float(aggs.mean()),
            "p50": float(np.percentile(aggs, 50)),
            "p95": float(np.percentile(aggs, 95)),
            "max": float(aggs.max()),
            "min": float(aggs.min()),
            "ops_finite": len(finite_aggregates),
            "ops_total": len(pre_stage["ops"]),
            "off_nominal_disturbance_rejection_iae": float(np.percentile(aggs, 95)),
        }
    else:
        grid_stats = {
            "mean": None,
            "p50": None,
            "p95": None,
            "max": None,
            "min": None,
            "ops_finite": 0,
            "ops_total": len(pre_stage["ops"]),
            "off_nominal_disturbance_rejection_iae": None,
        }

    print(
        f"\nAggregate over {grid_stats['ops_finite']} finite OPs: "
        f"mean={grid_stats['mean']!r}  P95={grid_stats['p95']!r}  max={grid_stats['max']!r}"
    )
    print(f"infeasible_pre_stage_count: {pre_stage_infeas}")
    print(f"Total runtime: {elapsed:.1f} s")

    payload: dict[str, Any] = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(tz=UTC).isoformat(),
        "git_sha": _git_sha(),
        "purpose": (
            "C1 (Pareto-reference x100, r_lt=r_vb=10) disturbance-rejection-only "
            "off-nominal baseline. The Bucket-B comparator on the §2.4 sub-metric."
        ),
        "grid": {
            "F_axis": list(_F_AXIS),
            "zF_axis": list(_ZF_AXIS),
            "n_ops": len(pre_stage["ops"]),
            "source": "docs/kpis.md §2.2",
        },
        "pareto_reference": {
            "multiplier": _PARETO_MULTIPLIER,
            "r_lt": pareto.r_lt,
            "r_vb": pareto.r_vb,
            "rationale": (
                "Best gate-passing fixed-weight C1 from "
                "data/reference/c1_regularization_sweep.json (kpis.md §6 Step 3)."
            ),
        },
        "pre_stage_cache_path": "data/reference/off_nominal_on_spec_pre_stages.json",
        "infeasible_pre_stage_count": pre_stage_infeas,
        "aggregate": grid_stats,
        "per_op": per_op,
    }
    with args.output.open("w") as fh:
        json.dump(payload, fh, indent=2)
    print(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
