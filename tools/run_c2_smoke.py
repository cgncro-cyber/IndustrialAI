"""Phase 3 smoke run — C2 agent vs the LV-closed plant at nominal_baseline.

One supervisory loop over a nominal (no-disturbance) horizon, single seed.
Deliberately minimal: proves the AgentRunner → MPC backend → plant
integration is functional end-to-end against the chosen LLM backend
before the full ``tools/run_c2_agent_scenarios.py`` driver is scaffolded.

Backend selection (ADR 011): ``--backend nim`` (default) uses the hosted
NVIDIA NIM Nemotron endpoint; ``--backend mac-studio`` uses the local
``mlx_lm.server`` stack retained as the Phase-5 ablation host. The
factory ``build_llm_client`` reads ``.env`` for the required vars.

Output (single JSON, no parquet/manifest yet — that lands with the full
driver):

    data/runs/c2_smoke/<scenario>/seed<seed>_<backend>/smoke.json

Invocation::

    uv run python tools/run_c2_smoke.py                       # default NIM
    uv run python tools/run_c2_smoke.py --backend mac-studio  # ablation host
    uv run python tools/run_c2_smoke.py --horizon-min 30 --tick-min 5

Per-cycle output is printed live so the LLM ↔ MPC handshake is observable
in real time.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from industrial_ai.agents.errors import (
    LLMEndpointUnreachableError,
    LLMResponseParseError,
    MissingUsageError,
)
from industrial_ai.agents.graph import AgentRunner, GraphConfig
from industrial_ai.agents.llm_client import build_llm_client
from industrial_ai.agents.regulatory_backend import build_regulatory_backend
from industrial_ai.twin.column_a import DEFAULT_PARAMETERS

_REPO_ROOT = Path(__file__).resolve().parent.parent
_SS_FIXTURE = _REPO_ROOT / "data" / "reference" / "skogestad_column_a_steady_state.json"
_SCENARIO = "nominal_baseline"


def _model_slug(model_identifier: str) -> str:
    """Last segment of the model identifier, lower-case, fs-safe.

    Examples
    --------
    >>> _model_slug("nvidia/llama-3.3-nemotron-super-49b-v1.5")
    'llama-3.3-nemotron-super-49b-v1.5'
    >>> _model_slug("deepseek-ai/deepseek-v4-flash")
    'deepseek-v4-flash'
    """
    return model_identifier.rsplit("/", 1)[-1].lower()


def _default_output_path(
    seed: int,
    backend: str,
    model_identifier: str | None,
    run_tag: str | None = None,
) -> Path:
    """Disambiguate NIM vs Mac-Studio AND per-model runs in the audit trail.

    Path shape per ADR 011 / 012::

        data/runs/c2_smoke/{scenario}/seed{seed}_{backend}_{model_slug}[_{run_tag}]/smoke.json

    ``run_tag`` carries the per-run override fingerprint for the
    Schritt-A.1 variance-diagnosis pass (e.g. ``t06``,
    ``reasoning_b2048``).
    """
    backend_slug = backend.replace("-", "_")
    leaf = f"seed{seed}_{backend_slug}"
    if model_identifier:
        leaf = f"{leaf}_{_model_slug(model_identifier)}"
    if run_tag:
        leaf = f"{leaf}_{run_tag}"
    return _REPO_ROOT / "data" / "runs" / "c2_smoke" / _SCENARIO / leaf / "smoke.json"


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
    parser.add_argument(
        "--backend",
        choices=("nim", "mac-studio"),
        default="nim",
        help="Inference backend (ADR 011). 'nim' uses the hosted "
        "NVIDIA Nemotron endpoint (default), 'mac-studio' uses the "
        "local mlx_lm.server stack retained as the Phase-5 ablation host.",
    )
    parser.add_argument(
        "--nim-model",
        default=None,
        help="Override NVIDIA_MODEL env var for this run. Useful for the "
        "ADR-011/012 multi-model comparison without editing .env. "
        "Has no effect on --backend mac-studio.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="Override the protocol's default temperature for this run. "
        "Schritt-A.1 variance-diagnosis knob; no effect when omitted.",
    )
    parser.add_argument(
        "--reasoning-mode",
        choices=("on", "off"),
        default=None,
        help="Force the first Optimizer LLM call's reasoning mode. 'on' "
        "enables chain-of-thought from cycle 0; 'off' (default behavior "
        "when omitted) uses the modal /no_think path. Critic-revision "
        "rounds always use reasoning=True regardless.",
    )
    parser.add_argument(
        "--reasoning-budget",
        type=int,
        default=None,
        help="Override the reasoning_budget threaded into "
        "NemotronExtraBodyProtocol's extra_body. Silently ignored for "
        "protocols that don't take a budget parameter.",
    )
    parser.add_argument(
        "--run-tag",
        default=None,
        help="Append this string to the default output dir name so multiple "
        "configurations of the same (seed, backend, model) tuple don't "
        "collide. Examples: 't06', 'reasoning_b2048'.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output JSON path; defaults to "
        "data/runs/c2_smoke/<scenario>/seed<seed>_<backend>_<model_slug>[_<run_tag>]/smoke.json.",
    )
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    # Per ADR 011 the --nim-model override is a per-run knob; threading
    # it through the .env layer keeps build_llm_client's contract clean.
    if args.nim_model and args.backend == "nim":
        os.environ["NVIDIA_MODEL"] = args.nim_model

    # ADR 011: factory selects the right client per backend; the
    # transport's seed-thread-through differs (NIM body `seed`, MLX
    # body `seed`) but both honor it. Schritt-A.1 overrides flow
    # through the factory's kwargs.
    llm = build_llm_client(
        backend=args.backend,
        seed=args.seed,
        temperature_override=args.temperature,
        reasoning_budget_override=args.reasoning_budget,
    )
    endpoint = getattr(llm, "base_url", None) or getattr(
        getattr(llm, "_cfg", None), "base_url", "?"
    )
    model_for_path = getattr(llm, "model", None) or getattr(
        getattr(llm, "_cfg", None), "model", None
    )
    output_path = (
        args.output
        if args.output is not None
        else _default_output_path(args.seed, args.backend, model_for_path, args.run_tag)
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    p = DEFAULT_PARAMETERS
    NT = p.NT
    model_name = getattr(llm, "model", None) or getattr(getattr(llm, "_cfg", None), "model", "?")
    # Protocol metadata for the smoke audit block. Only OpenAIChatLLMClient
    # carries a reasoning_protocol; MLXServerLLMClient stays on the legacy
    # marker-only contract.
    reasoning_protocol_obj = getattr(llm, "reasoning_protocol", None)
    reasoning_protocol_name = (
        getattr(reasoning_protocol_obj, "name", None) if reasoning_protocol_obj else "marker_only"
    )
    client_temperature = getattr(llm, "temperature", None)
    client_top_p = getattr(llm, "top_p", None)
    # First-round reasoning toggle drives both the runner's behavior
    # and the audit-block summary. Defaults to off when the CLI flag
    # is omitted, matching the modal /no_think path.
    first_round_reasoning = args.reasoning_mode == "on"
    reasoning_mode_label = "on" if first_round_reasoning else "off"
    effective_max_tokens = (
        reasoning_protocol_obj.max_tokens_for(reasoning=first_round_reasoning)
        if reasoning_protocol_obj
        else None
    )
    # If the protocol carries a reasoning_budget, record the
    # effective value for audit (overridden value or the protocol's
    # own default). For protocols without a budget, record null.
    effective_reasoning_budget = (
        getattr(reasoning_protocol_obj, "reasoning_budget", None)
        if reasoning_protocol_obj
        else None
    )

    print(
        f"=== C2 smoke — {_SCENARIO}, backend={args.backend}, "
        f"horizon={args.horizon_min} min, tick={args.tick_min} min ==="
    )
    print(f"endpoint:     {endpoint}")
    print(f"model:        {model_name}")
    print(f"protocol:     {reasoning_protocol_name}")
    print(f"temperature:  {client_temperature}")
    print(f"reasoning:    {reasoning_mode_label} (budget={effective_reasoning_budget})")
    if args.run_tag:
        print(f"run_tag:      {args.run_tag}")
    print(f"output:       {output_path}")
    print(f"git_sha:      {_git_sha()}")
    # nominal_baseline has no disturbance — canonical kpis.md §1.1
    # targets are the nominal SS values.
    runner = AgentRunner(
        llm_client=llm,
        regulatory_backend=build_regulatory_backend("mpc"),
        canonical_y_D_target=0.99,
        canonical_x_B_target=0.01,
        config=GraphConfig(
            supervisor_period_min=args.tick_min,
            first_round_reasoning=first_round_reasoning,
        ),
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
    abort_at_cycle: int | None = None
    abort_reason: str | None = None
    t_total_start = time.perf_counter()
    for i in range(n_ticks):
        t_min = i * args.tick_min
        y_D_pre = float(X[NT - 1])
        x_B_pre = float(X[0])
        # ADR 010 §2: fail-fast on LLM transport / parse errors, but
        # preserve the partial diagnostic trace — those collected
        # cycles ARE the signal the prompt iteration is asking for.
        try:
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
        except (
            LLMResponseParseError,
            MissingUsageError,
            LLMEndpointUnreachableError,
        ) as exc:
            abort_at_cycle = i
            abort_reason = f"{type(exc).__name__}: {exc!s:.500s}"
            print(f"\n!! ABORT at cycle {i}: {type(exc).__name__}", flush=True)
            print(f"   reason: {str(exc)[:200]}...", flush=True)
            break
        decision = out.state.decision
        y_target = float(decision.y_D_target) if decision else float("nan")
        x_target = float(decision.x_B_target) if decision else float("nan")
        X = out.regulatory_result.X_final
        y_D_post = float(X[NT - 1])
        x_B_post = float(X[0])
        verdict = out.state.critic_verdict.decision if out.state.critic_verdict else "?"
        proposal_in_cycle = out.state.optimizer_proposal
        rationale = proposal_in_cycle.rationale if proposal_in_cycle else ""
        print(
            f"{i:>3d} {t_min:>6.1f} "
            f"{y_D_pre:>7.4f} {x_B_pre:>7.4f} "
            f"{y_target:>5.3f} {x_target:>5.3f} "
            f"{y_D_post:>8.5f} {x_B_post:>8.5f} "
            f"{out.wall_clock_seconds:>7.2f} {verdict:>8s} "
            f"tok={out.completion_tokens:>4d}",
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
                "rationale": rationale,
                "reasoning_content": out.reasoning_content,
                "model_identifier": model_name,
                "prompt_tokens": out.prompt_tokens,
                "completion_tokens": out.completion_tokens,
                "total_tokens": out.prompt_tokens + out.completion_tokens,
            }
        )

    total_wall_s = time.perf_counter() - t_total_start
    aggregate_iae = float(runner._canonical_aggregate_iae)
    internal_tracking_iae = float(runner._internal_tracking_iae)
    if not cycles:
        # Nothing to summarise (failed before cycle 0). Surface bluntly.
        print("\n!! No cycles completed — see abort_reason in output.", flush=True)
        cycle_walls = [0.0]
        completion_tokens = [0]
        prompt_tokens = [0]
    else:
        cycle_walls = [c["wall_clock_seconds"] for c in cycles]
        completion_tokens = [c["completion_tokens"] for c in cycles]
        prompt_tokens = [c["prompt_tokens"] for c in cycles]
    status = "DONE" if abort_at_cycle is None else f"PARTIAL (aborted at cycle {abort_at_cycle})"
    print(
        f"\n{status}: {len(cycles)}/{n_ticks} cycles, "
        f"canonical IAE = {aggregate_iae:.5f}, "
        f"internal tracking IAE = {internal_tracking_iae:.5f}, "
        f"total wall = {total_wall_s:.1f} s, "
        f"per-cycle wall P50/P95/max = "
        f"{np.percentile(cycle_walls, 50):.2f}/{np.percentile(cycle_walls, 95):.2f}/{max(cycle_walls):.2f} s, "
        f"completion_tokens P50/P95/max = "
        f"{int(np.percentile(completion_tokens, 50))}/{int(np.percentile(completion_tokens, 95))}/{max(completion_tokens)}"
    )

    payload: dict[str, Any] = {
        "schema_version": 4,
        "generated_at_utc": datetime.now(tz=UTC).isoformat(),
        "git_sha": _git_sha(),
        "config": {
            "scenario": _SCENARIO,
            "horizon_min": args.horizon_min,
            "tick_min": args.tick_min,
            "n_ticks": n_ticks,
            "seed": args.seed,
            "backend": args.backend,
            "base_url": endpoint,
            "model": model_name,
            "model_identifier": model_name,
            "reasoning_protocol": reasoning_protocol_name,
            "reasoning_mode": reasoning_mode_label,
            "reasoning_budget": effective_reasoning_budget,
            "run_tag": args.run_tag,
            "temperature": client_temperature,
            "top_p": client_top_p,
            "max_tokens": effective_max_tokens,
            "regulatory_backend": "mpc",
            "F_kmol_per_min": p.nominal_feed_F_kmol_per_min,
            "zF": 0.5,
            "qF": p.nominal_feed_liquid_fraction_qF,
        },
        "aggregate": {
            "iae_mole_fraction_min": aggregate_iae,
            "internal_tracking_iae_mole_fraction_min": internal_tracking_iae,
            "total_wall_clock_seconds": total_wall_s,
            "cycle_wall_clock_seconds_p50": float(np.percentile(cycle_walls, 50)),
            "cycle_wall_clock_seconds_p95": float(np.percentile(cycle_walls, 95)),
            "cycle_wall_clock_seconds_max": float(max(cycle_walls)),
            "prompt_tokens_total": sum(prompt_tokens),
            "completion_tokens_total": sum(completion_tokens),
            "completion_tokens_p50": int(np.percentile(completion_tokens, 50)),
            "completion_tokens_p95": int(np.percentile(completion_tokens, 95)),
            "completion_tokens_max": int(max(completion_tokens)),
            "completed_cycles": runner._completed_cycles,
            "aborted_at_cycle": abort_at_cycle,
            "abort_reason": abort_reason,
            "all_regulatory_simulations_succeeded": all(
                c["regulatory_simulation_success"] for c in cycles
            )
            if cycles
            else False,
        },
        "cycles": cycles,
    }
    with output_path.open("w") as fh:
        json.dump(payload, fh, indent=2)
    print(f"wrote {output_path}")
    # ADR 010: a partial run is non-zero exit so CI surfaces it.
    return 0 if abort_at_cycle is None else 2


if __name__ == "__main__":
    raise SystemExit(main())
