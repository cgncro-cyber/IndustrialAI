"""Probe — is the C1 off-nominal failure mode regularization-sensitive?

Diagnostic for the worst cluster (F=0.8 row) in the kpis.md §2.2
off-nominal grid. The current C1 baseline at F=0.8 OPs collapses
MVs to LT=VB=0 and ends in a *worse* steady state than the start
(see ``c1_off_nominal_baseline.json`` and the diagnostic plot).

Open question: is the collapse driven by under-regularized MV
moves (linear-MPC QP saturating its bounds), or by a near-singular
plant gain G_mv that no amount of move-suppression fixes?

This probe sweeps the C1 ``r_lt`` / ``r_vb`` MV-move penalty over
four levels (baseline, x10, x100, x1000) at three off-nominal OPs
and reports the per-OP aggregate IAE. Output:
``data/reference/c1_regularization_sensitivity.json``.

Three possible outcomes:
- (a) Regularization tames the collapse: settle with offset rather
  than collapse. Aggregate IAE drops from ~770 to a moderate value.
- (b) Regularization changes nothing: collapse is structural.
- (c) Partial improvement: document sensitivity, pick the most
  defensible variant as the audit baseline.

Invocation:

    uv run python tools/probe_c1_regularization_sensitivity.py
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
from industrial_ai.control.scenarios import SCENARIO_NAMES
from industrial_ai.evaluation.kpis import compute_kpis
from industrial_ai.twin.column_a import DEFAULT_PARAMETERS
from industrial_ai.twin.column_a.linearize import linearize_lv
from industrial_ai.twin.column_a.operating_window import lookup_lv_ss

_REPO_ROOT = Path(__file__).resolve().parent.parent
_OUTPUT = _REPO_ROOT / "data" / "reference" / "c1_regularization_sensitivity.json"

# Worst cluster OPs from c1_off_nominal_baseline.json.
_PROBE_OPS: list[tuple[float, float]] = [
    (0.8, 0.45),  # worst, IAE 770
    (0.8, 0.475),  # second-worst, IAE 766
    (0.9, 0.45),  # transition zone, IAE 381
]

# Multipliers on baseline r_lt = r_vb = 0.1.
_REGULARIZATION_MULTIPLIERS: list[float] = [1.0, 10.0, 100.0, 1000.0]


def _git_sha() -> str:
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=_REPO_ROOT).decode().strip()
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def _evaluate(
    *,
    F_op: float,
    zF_op: float,
    multiplier: float,
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
    cfg = C1MPCConfig(r_lt=0.1 * multiplier, r_vb=0.1 * multiplier)
    mpc, _ = build_c1_mpc(lin, config=cfg)

    per_scenario: dict[str, Any] = {}
    op_aggregate = 0.0
    final_yD = float("nan")
    final_xB = float("nan")
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
        per_scenario[name] = {
            "success": sim.success,
            "iae_mole_fraction_min": iae if np.isfinite(iae) else None,
        }
        op_aggregate += iae
        # Keep the F_step_+20pct final state for diagnostic comparison.
        if name == "F_step_+20pct" and sim.success:
            final_yD = float(sim.X[-1, p.NT - 1])
            final_xB = float(sim.X[-1, 0])

    return {
        "F": F_op,
        "zF": zF_op,
        "r_multiplier": multiplier,
        "r_lt": 0.1 * multiplier,
        "r_vb": 0.1 * multiplier,
        "aggregate_iae": op_aggregate if np.isfinite(op_aggregate) else None,
        "F_step_+20pct_final_y_D": final_yD,
        "F_step_+20pct_final_x_B": final_xB,
        "per_scenario": per_scenario,
    }


def main() -> int:
    _OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    print(
        f"Probing regularization sensitivity at {len(_PROBE_OPS)} OPs "
        f"x {len(_REGULARIZATION_MULTIPLIERS)} levels = "
        f"{len(_PROBE_OPS) * len(_REGULARIZATION_MULTIPLIERS)} combinations ..."
    )
    print(
        f"{'F':>5s} {'zF':>6s} {'mult':>6s} {'r_lt':>6s}  "
        f"{'agg_IAE':>10s}  {'final y_D':>10s}  {'final x_B':>10s}"
    )

    results: list[dict[str, Any]] = []
    start = time.perf_counter()
    for F_op, zF_op in _PROBE_OPS:
        for mult in _REGULARIZATION_MULTIPLIERS:
            block = _evaluate(F_op=F_op, zF_op=zF_op, multiplier=mult)
            results.append(block)
            agg = block["aggregate_iae"]
            print(
                f"{F_op:>5.2f} {zF_op:>6.3f} {mult:>6.0f}x {block['r_lt']:>6.1f}  "
                f"{(agg if agg is not None else float('inf')):>10.2f}  "
                f"{block['F_step_+20pct_final_y_D']:>10.4f}  "
                f"{block['F_step_+20pct_final_x_B']:>10.4f}"
            )
    elapsed = time.perf_counter() - start
    print(f"\nProbe runtime: {elapsed:.1f} s")

    payload = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(tz=UTC).isoformat(),
        "git_sha": _git_sha(),
        "probe": "C1 regularization sensitivity at worst-cluster OPs",
        "baseline_r_lt": C1MPCConfig().r_lt,
        "baseline_r_vb": C1MPCConfig().r_vb,
        "probe_ops": [{"F": F, "zF": zF} for (F, zF) in _PROBE_OPS],
        "multipliers": _REGULARIZATION_MULTIPLIERS,
        "results": results,
    }
    with _OUTPUT.open("w") as fh:
        json.dump(payload, fh, indent=2)
    print(f"Wrote {_OUTPUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
