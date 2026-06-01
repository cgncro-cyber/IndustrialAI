"""Phase-3 DoE confirmation driver — N=10 validation at the optimum cell.

Consumes ``confirmation_spec.json`` produced by
``tools/analyze_doe_sweep.py`` and runs ``N=5`` additional smokes at
seeds 5–9 against the optimum cell's factor levels. Combined with the
screening ``N=5`` at seeds 0–4, this lands the kpis.md §1.1 ``N=10``
metrics at the cell that will be pinned for downstream evaluation.

Resilience mirrors the main sweep driver: atomic-write manifest,
SIGINT handler, idempotent restart.

If any of the confirmation seeds breaches the kpis.md §1.1 threshold
(0.01 mole-fraction·min), the driver does **not** abort — it
completes all five seeds, computes the ``N=10`` bootstrap CI, and
writes ``kpis_md_section_1_1_pass: false`` in the result. The flag
is data, not a gate; the operator decides whether to re-pin to the
second-ranked cell or document the observed variance.

Invocation::

    uv run python tools/run_doe_confirmation.py \\
        --analysis-root data/runs/c2_doe_sampling/nemotron-3-super-120b-a12b
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import random
import signal
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from industrial_ai.agents.errors import MissingConfirmationSpecError
from industrial_ai.io import atomic_write_json

_REPO_ROOT = Path(__file__).resolve().parent.parent

_KPIS_THRESHOLD_IAE = 0.01
_BOOTSTRAP_REPS = 1000
_BOOTSTRAP_SEED = 20260601


def _utc_now_iso() -> str:
    return _dt.datetime.now(tz=_dt.UTC).isoformat()


def _bootstrap_ci_mean(
    values: list[float], *, reps: int = _BOOTSTRAP_REPS, seed: int = _BOOTSTRAP_SEED
) -> tuple[float, float]:
    if not values:
        return float("nan"), float("nan")
    if len(values) == 1:
        return float(values[0]), float(values[0])
    rng = random.Random(seed)
    means: list[float] = []
    n = len(values)
    for _ in range(reps):
        sample = [values[rng.randrange(n)] for _ in range(n)]
        means.append(statistics.fmean(sample))
    means.sort()
    lo = means[int(0.025 * reps)]
    hi = means[min(int(0.975 * reps), reps - 1)]
    return float(lo), float(hi)


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return float("nan")
    if len(values) == 1:
        return float(values[0])
    sorted_values = sorted(values)
    k = (len(sorted_values) - 1) * (pct / 100.0)
    f = int(k)
    c = min(f + 1, len(sorted_values) - 1)
    if f == c:
        return float(sorted_values[f])
    return float(sorted_values[f] + (sorted_values[c] - sorted_values[f]) * (k - f))


def _build_manifest(spec: dict[str, Any]) -> dict[str, Any]:
    cell = spec["optimum_cell"]
    cells: list[dict[str, Any]] = []
    for seed in spec["confirmation_seeds"]:
        cells.append(
            {
                "cell_id": f"{cell['cell_id']}_S={seed}",
                "temperature": cell["temperature"],
                "top_p": cell["top_p"],
                "reasoning_config": cell["reasoning_config"],
                "reasoning_budget": cell.get("reasoning_budget"),
                "seed": seed,
                "status": "pending",
                "started_at": None,
                "completed_at": None,
                "smoke_path": None,
                "error_class": None,
                "error_message_excerpt": None,
                "canonical_iae": None,
                "completion_tokens_p95": None,
                "wall_clock_p95": None,
                "completed_cycles": None,
            }
        )
    return {
        "model_identifier": spec["model_identifier"],
        "optimum_cell": cell,
        "screening_metrics_n5": spec["screening_metrics_n5"],
        "started_at": _utc_now_iso(),
        "cells": cells,
    }


def _smoke_command(
    cell: dict[str, Any],
    model: str,
    cell_dir: Path,
) -> list[str]:
    cmd = [
        "uv",
        "run",
        "python",
        "tools/run_c2_smoke.py",
        "--backend",
        "nim",
        "--nim-model",
        model,
        "--seed",
        str(cell["seed"]),
        "--temperature",
        str(cell["temperature"]),
        "--top-p",
        str(cell["top_p"]),
        "--output-dir",
        str(cell_dir),
    ]
    if cell["reasoning_config"] != "off":
        cmd += ["--reasoning-mode", "on"]
        budget = cell.get("reasoning_budget")
        if budget is not None:
            cmd += ["--reasoning-budget", str(budget)]
    else:
        cmd += ["--reasoning-mode", "off"]
    return cmd


def _classify_smoke_failure(
    returncode: int,
    stdout: str,
    stderr: str,
) -> tuple[str, str]:
    combined = (stdout + "\n" + stderr).lower()
    if "rate limit" in combined or "429" in combined:
        return "rate_limit", combined[-400:]
    if "5xx" in combined or "503" in combined or "504" in combined:
        return "server_5xx", combined[-400:]
    if "timeout" in combined or returncode in (124, 137):
        return "timeout", combined[-400:]
    if "llmresponseparseerror" in combined or "llmresponseformat" in combined:
        return "smoke_parse", combined[-400:]
    return f"exit_{returncode}", combined[-400:]


def _extract_summary_metrics(smoke_path: Path) -> dict[str, Any]:
    with smoke_path.open() as fh:
        data = json.load(fh)
    aggregate = data.get("aggregate", {})
    return {
        "canonical_iae": aggregate.get("iae_mole_fraction_min"),
        "completion_tokens_p95": aggregate.get("completion_tokens_p95"),
        "wall_clock_p95": aggregate.get("cycle_wall_clock_seconds_p95"),
        "completed_cycles": aggregate.get("completed_cycles", 0),
    }


def _run_one_cell(
    cell: dict[str, Any],
    *,
    model: str,
    output_root: Path,
    manifest_path: Path,
    manifest: dict[str, Any],
    timeout_s: float,
) -> None:
    cell["status"] = "running"
    cell["started_at"] = _utc_now_iso()
    cell["error_class"] = None
    cell["error_message_excerpt"] = None
    atomic_write_json(manifest_path, manifest)
    cell_dir = output_root / cell["cell_id"]
    cell_dir.mkdir(parents=True, exist_ok=True)
    cmd = _smoke_command(cell, model, cell_dir)
    try:
        completed = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout_s,
            cwd=_REPO_ROOT,
            check=False,
        )
        rc = completed.returncode
        out = completed.stdout
        err = completed.stderr
    except subprocess.TimeoutExpired as exc:
        rc = 124
        out = exc.stdout or ""
        err = (exc.stderr or "") + f"\nTimeoutExpired after {timeout_s}s"
    smoke_path = cell_dir / "smoke.json"
    if rc == 0 and smoke_path.exists():
        cell["status"] = "done"
        cell["smoke_path"] = str(smoke_path)
        cell.update(_extract_summary_metrics(smoke_path))
    else:
        cell["status"] = "failed"
        error_class, excerpt = _classify_smoke_failure(rc, out, err)
        cell["error_class"] = error_class
        cell["error_message_excerpt"] = excerpt[-400:]
    cell["completed_at"] = _utc_now_iso()
    atomic_write_json(manifest_path, manifest)


def _write_confirmation_result(
    analysis_root: Path,
    spec: dict[str, Any],
    manifest: dict[str, Any],
    confirmation_path: Path,
) -> dict[str, Any]:
    screening_seeds = spec["screening_metrics_n5"]["seeds_used"]
    confirmation_seeds = [c["seed"] for c in manifest["cells"]]
    # Reconstruct individual canonical IAE per seed by merging the
    # screening's per-cell smokes (already on disk) with the
    # confirmation cells (just produced).
    screening_iae_by_seed = _load_screening_iae_per_seed(analysis_root, spec, screening_seeds)
    confirmation_iae_by_seed = {
        c["seed"]: c.get("canonical_iae")
        for c in manifest["cells"]
        if c.get("canonical_iae") is not None
    }
    individual: dict[str, float] = {}
    for seed in screening_seeds:
        v = screening_iae_by_seed.get(seed)
        if v is not None:
            individual[str(seed)] = v
    for seed, v in confirmation_iae_by_seed.items():
        if v is not None:
            individual[str(seed)] = v
    iae_values = [v for v in individual.values()]
    tokens_p95 = [
        c["completion_tokens_p95"]
        for c in manifest["cells"]
        if c.get("completion_tokens_p95") is not None
    ]
    wall_p95 = [
        c["wall_clock_p95"] for c in manifest["cells"] if c.get("wall_clock_p95") is not None
    ]
    ci_lo, ci_hi = _bootstrap_ci_mean(iae_values)
    mean_iae = statistics.fmean(iae_values) if iae_values else float("nan")
    n10_passes = ci_hi < _KPIS_THRESHOLD_IAE if iae_values else False
    result = {
        "model_identifier": spec["model_identifier"],
        "optimum_cell": spec["optimum_cell"],
        "screening_seeds": screening_seeds,
        "confirmation_seeds": confirmation_seeds,
        "n_seeds_total": len(individual),
        "individual_canonical_iae_by_seed": individual,
        "n10_metrics": {
            "mean_canonical_iae": mean_iae,
            "ci_95_canonical_iae_bootstrap_b1000": [ci_lo, ci_hi],
            "completion_tokens_p95": max(tokens_p95) if tokens_p95 else None,
            "wall_clock_p95": max(wall_p95) if wall_p95 else None,
        },
        "kpis_md_section_1_1_pass": n10_passes,
        "kpis_md_section_1_1_threshold": _KPIS_THRESHOLD_IAE,
        "completed_at": _utc_now_iso(),
    }
    atomic_write_json(confirmation_path, result)
    return result


def _load_screening_iae_per_seed(
    analysis_root: Path, spec: dict[str, Any], screening_seeds: list[int]
) -> dict[int, float]:
    """Reload the screening-phase per-seed canonical IAE for the optimum cell.

    The sweep's per-cell smoke.json files contain the canonical IAE
    for that (factor-cell, seed) pair; we look up those files for the
    optimum cell's factor levels at the screening seeds.
    """
    out: dict[int, float] = {}
    optimum = spec["optimum_cell"]
    base_cell_id = optimum["cell_id"]
    for seed in screening_seeds:
        smoke_path = analysis_root / f"{base_cell_id}_S={seed}" / "smoke.json"
        if not smoke_path.exists():
            continue
        try:
            with smoke_path.open() as fh:
                data = json.load(fh)
        except (OSError, json.JSONDecodeError):
            continue
        val = data.get("aggregate", {}).get("iae_mole_fraction_min")
        if val is not None:
            out[seed] = float(val)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--analysis-root",
        type=Path,
        required=True,
        help="Sweep root containing confirmation_spec.json (from analyze_doe_sweep.py).",
    )
    parser.add_argument("--cell-timeout", type=float, default=600.0)
    parser.add_argument("--cell-sleep", type=float, default=2.0)
    args = parser.parse_args()

    spec_path = args.analysis_root / "confirmation_spec.json"
    if not spec_path.exists():
        raise MissingConfirmationSpecError(
            f"no confirmation_spec.json at {spec_path}. The analysis "
            "step must produce it before the confirmation driver runs. "
            "Per ADR 010 §2 this is a fail-fast, not a recovery scenario."
        )
    with spec_path.open() as fh:
        spec = json.load(fh)

    output_root = Path(spec["confirmation_output_root"])
    output_root.mkdir(parents=True, exist_ok=True)
    manifest_path = output_root / "confirmation_manifest.json"
    if manifest_path.exists():
        with manifest_path.open() as fh:
            manifest = json.load(fh)
    else:
        manifest = _build_manifest(spec)
        atomic_write_json(manifest_path, manifest)

    pending = [c for c in manifest["cells"] if c["status"] != "done"]
    if not pending:
        print(
            f"[{spec['model_identifier']}] confirmation already complete; "
            "writing confirmation_result.json from cached cells.",
            flush=True,
        )
    else:
        print(
            f"[{spec['model_identifier']}] confirmation start: "
            f"{len(manifest['cells']) - len(pending)}/{len(manifest['cells'])} "
            "already done.",
            flush=True,
        )

    # SIGINT handler — same pattern as the sweep driver.
    current_cell: dict[str, Any] | None = None

    def _handler(signum: int, frame: Any) -> None:
        del signum, frame
        if current_cell is not None and current_cell.get("status") == "running":
            current_cell["status"] = "interrupted"
            current_cell["completed_at"] = _utc_now_iso()
        atomic_write_json(manifest_path, manifest)
        sys.exit(130)

    signal.signal(signal.SIGINT, _handler)

    for i, cell in enumerate(pending, start=1):
        current_cell = cell
        _run_one_cell(
            cell,
            model=spec["model_identifier"],
            output_root=output_root,
            manifest_path=manifest_path,
            manifest=manifest,
            timeout_s=args.cell_timeout,
        )
        if i < len(pending):
            time.sleep(args.cell_sleep)

    result_path = args.analysis_root / "confirmation_result.json"
    result = _write_confirmation_result(args.analysis_root, spec, manifest, result_path)
    pass_str = "PASS" if result["kpis_md_section_1_1_pass"] else "FAIL"
    n10 = result["n10_metrics"]
    print(
        f"[{spec['model_identifier']}] Confirmation complete. "
        f"N={result['n_seeds_total']} canonical IAE mean="
        f"{n10['mean_canonical_iae']:.6f}, "
        f"CI95=[{n10['ci_95_canonical_iae_bootstrap_b1000'][0]:.6f}, "
        f"{n10['ci_95_canonical_iae_bootstrap_b1000'][1]:.6f}]. "
        f"kpis.md §1.1 threshold {_KPIS_THRESHOLD_IAE}: {pass_str}.",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
