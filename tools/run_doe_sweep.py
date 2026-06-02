"""Phase-3 DoE sweep driver — hyperparameter screening on nominal_baseline.

Full-factorial DoE per ``docs/prompts/2026-06-01_phase3_doe_sampling.md``:

  temperature      ∈ {0.0, 0.3, 0.6, 0.8, 1.0}     (5 levels)
  top_p            ∈ {0.8, 0.95, 1.0}              (3 levels)
  reasoning_config ∈ {off, on_budget_1024, on_budget_4096}  (3 levels)
  seed             ∈ {0, 1, 2, 3, 4}               (N=5)

  → 5 × 3 × 3 × 5 = 225 cells per model.

Each cell shells out to ``tools/run_c2_smoke.py`` with the matching
override flags and an explicit ``--output-dir`` so the cell's
smoke.json lands in its own dedicated directory. Resilience is
priority #1 — the manifest is atomically rewritten after every cell
transition so a mid-sweep crash or laptop sleep loses at most one
cell's progress, and re-running with the same arguments resumes
where it left off.

Invocation::

    uv run python tools/run_doe_sweep.py \\
        --model nvidia/nemotron-3-super-120b-a12b \\
        --output-root data/runs/c2_doe_sampling/nemotron-3-super-120b-a12b

    # Resume after a crash — same command. Already-done cells are
    # skipped.

    # Re-attempt only previously-failed cells:
    uv run python tools/run_doe_sweep.py ... --retry-failed

    # Tests / quick dry-runs:
    uv run python tools/run_doe_sweep.py ... --dry-run --max-cells 2
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import re
import signal
import subprocess
import sys
import time
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from industrial_ai.io import atomic_write_json

_REPO_ROOT = Path(__file__).resolve().parent.parent

#: Hardcoded factor levels per the DoE prompt. Edit only via an
#: ADR-amendment commit that documents the change.
_FACTORS_TEMPERATURE: tuple[float, ...] = (0.0, 0.3, 0.6, 0.8, 1.0)
_FACTORS_TOP_P: tuple[float, ...] = (0.8, 0.95, 1.0)
_FACTORS_REASONING: tuple[str, ...] = ("off", "on_budget_1024", "on_budget_4096")
_FACTORS_SEED: tuple[int, ...] = (0, 1, 2, 3, 4)

_RATE_LIMIT_SLEEP_S = 60.0
_HTTP_5XX_SLEEP_S = 30.0

#: How V4-Flash maps the budget factor onto its reasoning_effort dial
#: (DeepSeek doesn't accept an integer budget). The mapping is
#: surfaced in the manifest cell record so reviewers can confirm.
_DEEPSEEK_BUDGET_TO_EFFORT: dict[str, str] = {
    "on_budget_1024": "medium",
    "on_budget_4096": "high",
}


def _utc_now_iso() -> str:
    return _dt.datetime.now(tz=_dt.UTC).isoformat()


def _parse_budget(reasoning_config: str) -> int | None:
    """Map ``on_budget_NNNN`` → NNNN; ``off`` → None."""
    if reasoning_config == "off":
        return None
    match = re.match(r"on_budget_(\d+)$", reasoning_config)
    if match is None:
        raise ValueError(f"unrecognised reasoning_config: {reasoning_config!r}")
    return int(match.group(1))


def enumerate_cells() -> list[dict[str, Any]]:
    """Return the full-factorial cell list in lexicographic-stable order."""
    cells: list[dict[str, Any]] = []
    for temperature in _FACTORS_TEMPERATURE:
        for top_p in _FACTORS_TOP_P:
            for reasoning in _FACTORS_REASONING:
                for seed in _FACTORS_SEED:
                    cell_id = f"T={temperature:g}_p={top_p:g}_R={reasoning}_S={seed}"
                    cells.append(
                        {
                            "cell_id": cell_id,
                            "temperature": temperature,
                            "top_p": top_p,
                            "reasoning_config": reasoning,
                            "reasoning_budget": _parse_budget(reasoning),
                            "seed": seed,
                            "status": "pending",
                            "started_at": None,
                            "completed_at": None,
                            "smoke_path": None,
                            "error_class": None,
                            "error_message_excerpt": None,
                            # Summary metrics filled in when status="done".
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
            "temperature": list(_FACTORS_TEMPERATURE),
            "top_p": list(_FACTORS_TOP_P),
            "reasoning_config": list(_FACTORS_REASONING),
            "seed": list(_FACTORS_SEED),
        },
        "total_cells": len(_FACTORS_TEMPERATURE)
        * len(_FACTORS_TOP_P)
        * len(_FACTORS_REASONING)
        * len(_FACTORS_SEED),
        "deepseek_budget_to_effort": _DEEPSEEK_BUDGET_TO_EFFORT,
        "started_at": _utc_now_iso(),
        "cells": enumerate_cells(),
    }


def _load_or_create_manifest(manifest_path: Path, model: str) -> dict[str, Any]:
    if manifest_path.exists():
        try:
            with manifest_path.open() as fh:
                data: dict[str, Any] = json.load(fh)
        except (OSError, json.JSONDecodeError) as exc:
            # ADR 010 §2: a corrupt existing manifest is operator-visible,
            # not silently overwritten — rename or delete it before
            # restarting.
            raise RuntimeError(
                f"manifest at {manifest_path} is corrupted ({exc}). "
                "Operator decides recovery: investigate, rename, or remove "
                "before re-running the sweep."
            ) from exc
        if data.get("model_identifier") != model:
            raise RuntimeError(
                f"manifest at {manifest_path} belongs to model "
                f"{data.get('model_identifier')!r} but the driver was "
                f"invoked with --model {model!r}. Use the matching "
                "--output-root or rename the manifest."
            )
        return data
    return _build_manifest(model)


def _cell_smoke_dir(output_root: Path, cell_id: str) -> Path:
    return output_root / cell_id


def _smoke_command(
    cell: dict[str, Any],
    model: str,
    cell_dir: Path,
) -> list[str]:
    """Build the per-cell ``run_c2_smoke.py`` invocation."""
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
        # NemotronExtraBodyProtocol takes a numeric budget;
        # DeepSeekExtraBodyProtocol silently ignores it (it uses
        # reasoning_effort instead, mapped server-side via the budget
        # bucket; the protocol's own default_reasoning_effort='high'
        # is preserved because the driver doesn't have a CLI for it
        # yet — V4-Flash budget→effort mapping is recorded in the
        # manifest's `deepseek_budget_to_effort` for reviewer audit).
        budget = cell["reasoning_budget"]
        if budget is not None:
            cmd += ["--reasoning-budget", str(budget)]
    else:
        cmd += ["--reasoning-mode", "off"]
    return cmd


def _to_str(maybe: bytes | str | None) -> str:
    """Coerce ``subprocess.run`` stdout/stderr to ``str``, decoding bytes if needed.

    With ``text=True`` on ``subprocess.run`` the success-path streams come
    back as ``str``, but on the ``TimeoutExpired`` branch the kernel-side
    streams haven't been decoded yet — Python's stdlib exposes them as
    bytes there. The DoE driver consumed those values raw and crashed
    with a ``TypeError: can't concat str to bytes`` on the first
    timeout (cell ``T=0.8_p=1_R=on_budget_1024_S=3`` at 06:10 UTC
    2026-06-02). This helper is the single fix-point so both drivers
    behave identically.
    """
    if maybe is None:
        return ""
    if isinstance(maybe, bytes):
        return maybe.decode("utf-8", errors="replace")
    return maybe


def _classify_smoke_failure(
    returncode: int,
    stdout: str,
    stderr: str,
) -> tuple[str, str]:
    """Heuristic classification of failures into (error_class, excerpt).

    Used only for the manifest summary — the operator can re-read
    the full per-cell log in stdout/stderr if needed.
    """
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
    """Read a finished cell's smoke.json and pull the headline numbers."""
    with smoke_path.open() as fh:
        data = json.load(fh)
    aggregate = data.get("aggregate", {})
    completed_cycles = aggregate.get("completed_cycles", 0)
    canonical_iae = aggregate.get("iae_mole_fraction_min")
    warning = None
    if completed_cycles is not None and completed_cycles < 12:
        warning = "partial_run"
    return {
        "canonical_iae": canonical_iae,
        "completion_tokens_p95": aggregate.get("completion_tokens_p95"),
        "wall_clock_p95": aggregate.get("cycle_wall_clock_seconds_p95"),
        "completed_cycles": completed_cycles,
        "canonical_iae_warning": warning,
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
    """Execute one cell end-to-end, updating the manifest after each transition."""
    cell["status"] = "running"
    cell["started_at"] = _utc_now_iso()
    cell["error_class"] = None
    cell["error_message_excerpt"] = None
    atomic_write_json(manifest_path, manifest)
    cell_dir = _cell_smoke_dir(output_root, cell["cell_id"])
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
        returncode = completed.returncode
        stdout = completed.stdout
        stderr = completed.stderr
    except subprocess.TimeoutExpired as exc:
        returncode = 124
        stdout = _to_str(exc.stdout)
        stderr = _to_str(exc.stderr) + f"\nTimeoutExpired after {timeout_s}s"
    smoke_path = cell_dir / "smoke.json"
    if returncode == 0 and smoke_path.exists():
        cell["status"] = "done"
        cell["smoke_path"] = str(smoke_path)
        metrics = _extract_summary_metrics(smoke_path)
        cell.update(metrics)
    else:
        cell["status"] = "failed"
        error_class, excerpt = _classify_smoke_failure(returncode, stdout, stderr)
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
    """Pick cells that still need work, respecting --retry-failed and --max-cells."""
    selected: list[dict[str, Any]] = []
    for cell in cells:
        status = cell.get("status", "pending")
        if status == "done":
            continue
        if status == "failed" and not retry_failed:
            continue
        if status == "running":
            # Resume from crash: the cell may have a partial smoke.json
            # on disk that we can inspect; the per-cell runner will
            # decide whether to keep or re-run.
            pass
        selected.append(cell)
    if max_cells is not None:
        selected = selected[:max_cells]
    return selected


def _install_sigint_handler(
    manifest_path: Path, manifest: dict[str, Any], current_cell: dict[str, Any] | None
) -> None:
    """Mark the currently-running cell as interrupted on Ctrl-C, save, exit clean."""

    def _handler(signum: int, frame: Any) -> None:
        del signum, frame
        if current_cell is not None and current_cell.get("status") == "running":
            current_cell["status"] = "interrupted"
            current_cell["completed_at"] = _utc_now_iso()
        try:
            atomic_write_json(manifest_path, manifest)
        finally:
            sys.exit(130)  # standard SIGINT exit code

    signal.signal(signal.SIGINT, _handler)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="NIM model identifier")
    parser.add_argument(
        "--output-root",
        type=Path,
        required=True,
        help="Root directory for the sweep's manifest + per-cell smoke outputs",
    )
    parser.add_argument(
        "--cell-timeout",
        type=float,
        default=600.0,
        help="Per-cell subprocess timeout in seconds (default 600).",
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
        help="Stop after this many cells (for verification / smoke testing).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Do not invoke the per-cell smoke; mark each cell done with "
        "zeroed metrics. For driver-shape verification only.",
    )
    args = parser.parse_args()

    args.output_root.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_root / "sweep_manifest.json"
    manifest = _load_or_create_manifest(manifest_path, args.model)
    atomic_write_json(manifest_path, manifest)

    cells = manifest["cells"]
    pending = list(_select_pending(cells, retry_failed=args.retry_failed, max_cells=args.max_cells))
    if not pending:
        print(f"[{args.model}] No pending cells. Sweep already complete.", flush=True)
        return 0

    total = manifest["total_cells"]
    done_before = sum(1 for c in cells if c["status"] == "done")
    failed_before = sum(1 for c in cells if c["status"] == "failed")
    print(
        f"[{args.model}] sweep start: {done_before}/{total} done, "
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
        if i % 10 == 0 or i == len(pending):
            done = sum(1 for c in cells if c["status"] == "done")
            failed = sum(1 for c in cells if c["status"] == "failed")
            elapsed_s = time.perf_counter() - sweep_t0
            remaining = total - done - failed
            eta_s = (elapsed_s / i) * (len(pending) - i) if i > 0 else 0.0
            print(
                f"[{args.model}] {done}/{total} done, {failed} failed, "
                f"{remaining} remaining (this invocation: {i}/{len(pending)}, "
                f"elapsed {elapsed_s / 60:.1f} min, ETA {eta_s / 60:.1f} min)",
                flush=True,
            )
        if i < len(pending):
            time.sleep(args.cell_sleep)

    done = sum(1 for c in cells if c["status"] == "done")
    failed = sum(1 for c in cells if c["status"] == "failed")
    print(
        f"[{args.model}] sweep finished: {done}/{total} done, {failed} failed",
        flush=True,
    )
    # Non-zero exit if any cell is still not done; the chained launch
    # uses `&&` so this stops the chain only when something's wrong.
    return 0 if done == total else 1


if __name__ == "__main__":
    raise SystemExit(main())
