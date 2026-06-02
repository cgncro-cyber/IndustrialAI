"""Phase-3 Schritt B — off-nominal screening grid driver.

Full Cartesian product over the kpis.md §2.5 amended 3-corner screening grid:

  OP               in {(1.2, 0.45), (1.2, 0.55), (0.8, 0.55)}                  3
  scenario         in kpis.md §1.2 5-scenario set                              5
  submetric        in {target_acquisition, disturbance_rejection}              2
  seed             in {0, 1, ..., 9}                                          10

  → 3 × 5 × 2 × 10 = 300 cells per model.

The fourth corner ``(F=0.8, zF=0.45)`` is **excluded** from the
Schritt-B screening grid per ``docs/kpis.md`` §2.5 Changelog 2026-06-02
and ``docs/pre_submission_checklist.md`` §4.6 Empirical confirmation
2026-06-02. The LV configuration is near-singular at that operating
point (``cond(G_mv) ≈ 6800`` vs 150 nominal); the nominal LT/VB
applied during target_acquisition push the plant into a catastrophic
regime that both C1 (P95 IAE ≈ 161) and C2 cannot escape. Audit
trail at ``docs/analyses/2026-06-02_schritt_b_failure_diagnosis.md``.
The corner remains in the kpis.md §2.2 16-point HEADLINE grid for
Schritt C, where the joint-architectural limit becomes part of the
paper finding.

Each cell shells out to ``tools/run_c2_smoke.py`` with --scenario,
--op-F, --op-zF, --submetric, --seed plus --output-dir. Sampling
defaults (T=0.3, top_p=0.95, reasoning=off) come from the
DoE-pinned ``NemotronExtraBodyProtocol`` (ADR 011 Sub-Amendment
2026-06-02); the driver does not override them.

Resilience mirrors ``tools/run_doe_sweep.py``: atomic-write
manifest, state machine pending → running → done / failed /
interrupted, SIGINT handler, idempotent restart. Per ADR 010 §2:
HTTP 429 / 5xx / 4xx all mark the cell failed (no silent retry);
``--cell-timeout`` defaults to 900 s (vs DoE's 600) because off-
nominal scenarios with longer dynamics may legitimately consume
more wall-clock per cycle than ``nominal_baseline``.

Invocation::

    uv run python tools/run_offnominal_screening.py \\
        --model nvidia/nemotron-3-super-120b-a12b \\
        --output-root data/runs/c2_offnominal_screening/nemotron-3-super-120b-a12b

    # Resume after a crash — same command. Already-done cells are skipped.
    # Re-attempt only previously-failed cells:
    uv run python tools/run_offnominal_screening.py ... --retry-failed
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import signal
import subprocess
import sys
import time
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from industrial_ai.io import atomic_write_json

_REPO_ROOT = Path(__file__).resolve().parent.parent

#: kpis.md §2.5 amended 3-OP corner-grid (subset of the 16-point §2.2
#: headline grid; the F=0.8/zF=0.475 anomaly and the LV-singular
#: F=0.8/zF=0.45 corner are NOT in this set). See module docstring.
_OPS: tuple[tuple[float, float], ...] = (
    (1.2, 0.45),
    (1.2, 0.55),
    (0.8, 0.55),
)

#: kpis.md §1.2 canonical 5-scenario set.
_SCENARIOS: tuple[str, ...] = (
    "F_step_+20pct",
    "F_step_-20pct",
    "zF_step_+10pct",
    "zF_step_-10pct",
    "yD_setpoint_+0p5pct",
)

#: Two sub-metric arms per kpis.md §2.3 / §2.4.
_SUBMETRICS: tuple[str, ...] = ("target_acquisition", "disturbance_rejection")

#: N=10 per cell per the Schritt-B specification (no screen-then-confirm;
#: direct N=10 because the screening surface IS the analysis surface).
_SEEDS: tuple[int, ...] = tuple(range(10))

_RATE_LIMIT_SLEEP_S = 60.0
_HTTP_5XX_SLEEP_S = 30.0


def _utc_now_iso() -> str:
    return _dt.datetime.now(tz=_dt.UTC).isoformat()


def _to_str(maybe: bytes | str | None) -> str:
    """Coerce subprocess.run stdout/stderr to str (bugfix from 2e0e50c)."""
    if maybe is None:
        return ""
    if isinstance(maybe, bytes):
        return maybe.decode("utf-8", errors="replace")
    return maybe


def _cell_id(op_F: float, op_zF: float, scenario: str, submetric: str, seed: int) -> str:
    """Stable lexicographic cell id encoding all four factors + seed."""
    return f"F={op_F:g}_zF={op_zF:g}_S={scenario}_M={submetric}_seed{seed}"


def enumerate_cells() -> list[dict[str, Any]]:
    """Return the full 400-cell list in stable order."""
    cells: list[dict[str, Any]] = []
    for op_F, op_zF in _OPS:
        for scenario in _SCENARIOS:
            for submetric in _SUBMETRICS:
                for seed in _SEEDS:
                    cells.append(
                        {
                            "cell_id": _cell_id(op_F, op_zF, scenario, submetric, seed),
                            "op_F": op_F,
                            "op_zF": op_zF,
                            "scenario": scenario,
                            "submetric": submetric,
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
                            "canonical_iae_warning": None,
                        }
                    )
    return cells


def _build_manifest(model: str) -> dict[str, Any]:
    return {
        "model_identifier": model,
        "factor_space": {
            "ops": [list(op) for op in _OPS],
            "scenarios": list(_SCENARIOS),
            "submetrics": list(_SUBMETRICS),
            "seeds": list(_SEEDS),
        },
        "total_cells": len(_OPS) * len(_SCENARIOS) * len(_SUBMETRICS) * len(_SEEDS),
        "started_at": _utc_now_iso(),
        "cells": enumerate_cells(),
    }


def _load_or_create_manifest(manifest_path: Path, model: str) -> dict[str, Any]:
    if manifest_path.exists():
        try:
            with manifest_path.open() as fh:
                data: dict[str, Any] = json.load(fh)
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError(
                f"manifest at {manifest_path} is corrupted ({exc}). "
                "Operator decides recovery: investigate, rename, or remove "
                "before re-running the screening."
            ) from exc
        if data.get("model_identifier") != model:
            raise RuntimeError(
                f"manifest at {manifest_path} belongs to model "
                f"{data.get('model_identifier')!r} but the driver was "
                f"invoked with --model {model!r}."
            )
        return data
    return _build_manifest(model)


def _smoke_command(cell: dict[str, Any], model: str, cell_dir: Path) -> list[str]:
    return [
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
        "--scenario",
        cell["scenario"],
        "--op-F",
        str(cell["op_F"]),
        "--op-zF",
        str(cell["op_zF"]),
        "--submetric",
        cell["submetric"],
        "--output-dir",
        str(cell_dir),
    ]


def _classify_smoke_failure(returncode: int, stdout: str, stderr: str) -> tuple[str, str]:
    combined = (stdout + "\n" + stderr).lower()
    if "infeasiblesubmetric" in combined:
        return "pre_stage_infeasible", combined[-400:]
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
    completed_cycles = aggregate.get("completed_cycles", 0)
    return {
        "canonical_iae": aggregate.get("iae_mole_fraction_min"),
        "completion_tokens_p95": aggregate.get("completion_tokens_p95"),
        "wall_clock_p95": aggregate.get("cycle_wall_clock_seconds_p95"),
        "completed_cycles": completed_cycles,
        "canonical_iae_warning": "partial_run"
        if completed_cycles is not None and completed_cycles < 12
        else None,
    }


def _run_one_cell(
    cell: dict[str, Any],
    *,
    model: str,
    output_root: Path,
    manifest_path: Path,
    manifest: dict[str, Any],
    timeout_s: float,
    dry_run: bool,
) -> None:
    cell["status"] = "running"
    cell["started_at"] = _utc_now_iso()
    cell["error_class"] = None
    cell["error_message_excerpt"] = None
    atomic_write_json(manifest_path, manifest)
    cell_dir = output_root / cell["cell_id"]
    cell_dir.mkdir(parents=True, exist_ok=True)
    if dry_run:
        cell["status"] = "done"
        cell["completed_at"] = _utc_now_iso()
        cell["smoke_path"] = str(cell_dir / "smoke.json")
        cell["canonical_iae"] = 0.0
        cell["completion_tokens_p95"] = 0
        cell["wall_clock_p95"] = 0.0
        cell["completed_cycles"] = 12
        atomic_write_json(manifest_path, manifest)
        return
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
        out = _to_str(exc.stdout)
        err = _to_str(exc.stderr) + f"\nTimeoutExpired after {timeout_s}s"
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
        if error_class == "rate_limit":
            time.sleep(_RATE_LIMIT_SLEEP_S)
        elif error_class == "server_5xx":
            time.sleep(_HTTP_5XX_SLEEP_S)
    cell["completed_at"] = _utc_now_iso()
    atomic_write_json(manifest_path, manifest)


def _select_pending(
    cells: list[dict[str, Any]],
    *,
    retry_failed: bool,
    max_cells: int | None,
) -> Iterable[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for cell in cells:
        status = cell.get("status", "pending")
        if status == "done":
            continue
        if status == "failed" and not retry_failed:
            continue
        selected.append(cell)
    if max_cells is not None:
        selected = selected[:max_cells]
    return selected


def _install_sigint_handler(
    manifest_path: Path, manifest: dict[str, Any], current_cell: dict[str, Any] | None
) -> None:
    def _handler(signum: int, frame: Any) -> None:
        del signum, frame
        if current_cell is not None and current_cell.get("status") == "running":
            current_cell["status"] = "interrupted"
            current_cell["completed_at"] = _utc_now_iso()
        try:
            atomic_write_json(manifest_path, manifest)
        finally:
            sys.exit(130)

    signal.signal(signal.SIGINT, _handler)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="NIM model identifier")
    parser.add_argument(
        "--output-root",
        type=Path,
        required=True,
        help="Root directory for the screening manifest + per-cell smoke outputs",
    )
    parser.add_argument(
        "--cell-timeout",
        type=float,
        default=900.0,
        help="Per-cell subprocess timeout in seconds (default 900; off-nominal scenarios may legitimately need more than nominal_baseline's 600).",
    )
    parser.add_argument(
        "--cell-sleep",
        type=float,
        default=2.0,
        help="Seconds to sleep between cells (rate-limit defensiveness).",
    )
    parser.add_argument(
        "--retry-failed",
        action="store_true",
        help="Re-attempt cells with status='failed' from a previous run.",
    )
    parser.add_argument(
        "--max-cells",
        type=int,
        default=None,
        help="Stop after this many cells (for driver-shape verification).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Do not invoke the per-cell smoke; mark each touched cell done with zeroed metrics.",
    )
    args = parser.parse_args()

    args.output_root.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_root / "sweep_manifest.json"
    manifest = _load_or_create_manifest(manifest_path, args.model)
    atomic_write_json(manifest_path, manifest)

    cells = manifest["cells"]
    pending = list(_select_pending(cells, retry_failed=args.retry_failed, max_cells=args.max_cells))
    if not pending:
        print(f"[{args.model}] No pending cells. Screening already complete.", flush=True)
        return 0

    total = manifest["total_cells"]
    done_before = sum(1 for c in cells if c["status"] == "done")
    failed_before = sum(1 for c in cells if c["status"] == "failed")
    print(
        f"[{args.model} off-nominal] screening start: {done_before}/{total} done, "
        f"{failed_before} failed previously, {len(pending)} to run this invocation",
        flush=True,
    )

    current_cell: dict[str, Any] | None = None
    _install_sigint_handler(manifest_path, manifest, current_cell)

    sweep_t0 = time.perf_counter()
    for i, cell in enumerate(pending, start=1):
        current_cell = cell
        _install_sigint_handler(manifest_path, manifest, current_cell)
        _run_one_cell(
            cell,
            model=args.model,
            output_root=args.output_root,
            manifest_path=manifest_path,
            manifest=manifest,
            timeout_s=args.cell_timeout,
            dry_run=args.dry_run,
        )
        if i % 25 == 0 or i == len(pending):
            done = sum(1 for c in cells if c["status"] == "done")
            failed = sum(1 for c in cells if c["status"] == "failed")
            elapsed_s = time.perf_counter() - sweep_t0
            eta_s = (elapsed_s / i) * (len(pending) - i) if i > 0 else 0.0
            print(
                f"[{args.model} off-nominal] {done}/{total} cells done, {failed} failed, "
                f"this invocation: {i}/{len(pending)}, elapsed {elapsed_s / 3600:.1f}h, "
                f"ETA {eta_s / 3600:.1f}h",
                flush=True,
            )
        if i < len(pending):
            time.sleep(args.cell_sleep)

    done = sum(1 for c in cells if c["status"] == "done")
    failed = sum(1 for c in cells if c["status"] == "failed")
    print(
        f"[{args.model} off-nominal] screening finished: {done}/{total} done, {failed} failed",
        flush=True,
    )
    return 0 if done == total else 1


if __name__ == "__main__":
    raise SystemExit(main())
