"""Phase 3 smoke run — C2 agent vs the LV-closed plant at nominal_baseline.

One supervisory loop over a nominal (no-disturbance) horizon, single seed.
Deliberately minimal: proves the AgentRunner → MPC backend → plant
integration is functional end-to-end against the live MLX server before
the full ``tools/run_c2_agent_scenarios.py`` driver is scaffolded.

Output (single JSON, no parquet/manifest yet — that lands with the full
driver):

    data/runs/c2_smoke/nominal_baseline/seed0/smoke.json

Invocation::

    uv run python tools/run_c2_smoke.py
    uv run python tools/run_c2_smoke.py --horizon-min 30 --tick-min 5

Per-cycle output is printed live so the LLM ↔ MPC handshake is observable
in real time.
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

from industrial_ai.agents.graph import AgentRunner, GraphConfig
from industrial_ai.agents.llm_client import MLXServerLLMClient
from industrial_ai.agents.regulatory_backend import build_regulatory_backend
from industrial_ai.twin.column_a import DEFAULT_PARAMETERS

_REPO_ROOT = Path(__file__).resolve().parent.parent
_SS_FIXTURE = _REPO_ROOT / "data" / "reference" / "skogestad_column_a_steady_state.json"
_DEFAULT_OUT = (
    _REPO_ROOT / "data" / "runs" / "c2_smoke" / "nominal_baseline" / "seed0" / "smoke.json"
)
_DEFAULT_BASE_URL = "http://192.168.178.81:8080/v1"


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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--horizon-min", type=float, default=60.0)
    parser.add_argument("--tick-min", type=float, default=5.0)
    parser.add_argument("--base-url", default=_DEFAULT_BASE_URL)
    parser.add_argument("--output", type=Path, default=_DEFAULT_OUT)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    p = DEFAULT_PARAMETERS
    NT = p.NT

    print(
        f"=== C2 smoke — nominal_baseline, horizon={args.horizon_min} min, tick={args.tick_min} min ==="
    )
    print(f"endpoint: {args.base_url}")
    print(f"output:   {args.output}")
    print(f"git_sha:  {_git_sha()}")

    llm = MLXServerLLMClient(base_url=args.base_url, request_timeout_s=180.0)
    runner = AgentRunner(
        llm_client=llm,
        regulatory_backend=build_regulatory_backend("mpc"),
        config=GraphConfig(supervisor_period_min=args.tick_min),
    )

    X = _load_nominal_ss()
    n_ticks = round(args.horizon_min / args.tick_min)
    print(f"\nrunning {n_ticks} supervisor ticks...")
    print(
        f"{'cyc':>3} {'t_min':>6} "
        f"{'y_D_pre':>7} {'x_B_pre':>7} "
        f"{'y_D*':>5} {'x_B*':>5} "
        f"{'y_D_post':>8} {'x_B_post':>8} "
        f"{'wall_s':>7} {'verdict':>8}"
    )
    cycles: list[dict[str, Any]] = []
    t_total_start = time.perf_counter()
    for i in range(n_ticks):
        t_min = i * args.tick_min
        y_D_pre = float(X[NT - 1])
        x_B_pre = float(X[0])
        out = runner.step(
            cycle_index=i,
            t_min=t_min,
            X=X,
            LT_kmol_per_min=p.nominal_reflux_L0_kmol_per_min,
            VB_kmol_per_min=p.nominal_boilup_V0_kmol_per_min,
            F_kmol_per_min=p.nominal_feed_F_kmol_per_min,
            zF=0.5,
            qF=p.nominal_feed_liquid_fraction_qF,
        )
        decision = out.state.decision
        y_target = float(decision.y_D_target) if decision else float("nan")
        x_target = float(decision.x_B_target) if decision else float("nan")
        X = out.regulatory_result.X_final
        y_D_post = float(X[NT - 1])
        x_B_post = float(X[0])
        verdict = out.state.critic_verdict.decision if out.state.critic_verdict else "?"
        print(
            f"{i:>3d} {t_min:>6.1f} "
            f"{y_D_pre:>7.4f} {x_B_pre:>7.4f} "
            f"{y_target:>5.3f} {x_target:>5.3f} "
            f"{y_D_post:>8.5f} {x_B_post:>8.5f} "
            f"{out.wall_clock_seconds:>7.2f} {verdict:>8s}",
            flush=True,
        )
        cycles.append(
            {
                "cycle_index": i,
                "t_min": t_min,
                "y_D_pre": y_D_pre,
                "x_B_pre": x_B_pre,
                "y_D_target": y_target,
                "x_B_target": x_target,
                "y_D_post": y_D_post,
                "x_B_post": x_B_post,
                "wall_clock_seconds": out.wall_clock_seconds,
                "critic_verdict": verdict,
                "optimizer_rounds": out.optimizer_rounds,
                "escalated": out.escalated,
                "regulatory_simulation_success": out.regulatory_result.simulation.success,
            }
        )

    total_wall_s = time.perf_counter() - t_total_start
    aggregate_iae = float(runner._aggregate_iae)
    cycle_walls = [c["wall_clock_seconds"] for c in cycles]
    print(
        f"\nDONE: {n_ticks} cycles, aggregate IAE = {aggregate_iae:.5f}, "
        f"total wall = {total_wall_s:.1f} s, "
        f"per-cycle wall P50/P95/max = "
        f"{np.percentile(cycle_walls, 50):.2f}/{np.percentile(cycle_walls, 95):.2f}/{max(cycle_walls):.2f} s"
    )

    payload: dict[str, Any] = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(tz=UTC).isoformat(),
        "git_sha": _git_sha(),
        "config": {
            "scenario": "nominal_baseline",
            "horizon_min": args.horizon_min,
            "tick_min": args.tick_min,
            "n_ticks": n_ticks,
            "seed": args.seed,
            "base_url": args.base_url,
            "regulatory_backend": "mpc",
            "F_kmol_per_min": p.nominal_feed_F_kmol_per_min,
            "zF": 0.5,
            "qF": p.nominal_feed_liquid_fraction_qF,
        },
        "aggregate": {
            "iae_mole_fraction_min": aggregate_iae,
            "total_wall_clock_seconds": total_wall_s,
            "cycle_wall_clock_seconds_p50": float(np.percentile(cycle_walls, 50)),
            "cycle_wall_clock_seconds_p95": float(np.percentile(cycle_walls, 95)),
            "cycle_wall_clock_seconds_max": float(max(cycle_walls)),
            "completed_cycles": runner._completed_cycles,
            "all_regulatory_simulations_succeeded": all(
                c["regulatory_simulation_success"] for c in cycles
            ),
        },
        "cycles": cycles,
    }
    with args.output.open("w") as fh:
        json.dump(payload, fh, indent=2)
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
