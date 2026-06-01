"""Phase-3 DoE response-surface analysis + confirmation_spec output.

Consumes the sweep manifest + per-cell ``smoke.json`` files produced
by ``tools/run_doe_sweep.py``, aggregates the per-cell N=5 metrics,
runs a B=1000 bootstrap CI on canonical IAE per cell, and writes:

  analysis.json          — machine-readable cell aggregates + ranking.
  analysis_summary.md    — human-readable response surface tables.
  confirmation_spec.json — machine-readable spec naming the optimum
                           cell for downstream confirmation runs.

Selection rule for the optimum cell:

  1. Compute per-cell N=5 statistics: mean canonical IAE, 95 %
     bootstrap CI (B=1000) on the mean, completion_tokens_p95
     (max over the 5 seeds — the worst-case headroom), and
     wall_clock_p95 (same).
  2. Filter to cells where the 95 % CI upper bound is below the
     kpis.md §1.1 threshold (0.01 mole-fraction·min).
  3. Apply secondary constraints: wall_clock_p95 < 60 s and
     completion_tokens_p95 < 1000.
  4. Argmin canonical IAE mean over the filtered set; ties broken
     by ascending wall_clock_p95, then by lexicographic cell_id.
  5. If the filtered set is empty (no cell unambiguously clears
     the threshold even at N=5), fall back to argmin over the
     full surface — with the selection_rationale field calling
     out the fall-back explicitly so reviewers see it.

Invocation::

    uv run python tools/analyze_doe_sweep.py \\
        --output-root data/runs/c2_doe_sampling/nemotron-3-super-120b-a12b

Re-invocations overwrite the analysis artifacts; the sweep manifest
itself is read-only here.
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from industrial_ai.io import atomic_write_json

#: kpis.md §1.1 acceptance threshold for canonical IAE on
#: nominal_baseline (mole-fraction·min).
_KPIS_THRESHOLD_IAE = 0.01

_SECONDARY_WALL_CLOCK_CEILING_S = 60.0
_SECONDARY_COMPLETION_TOKENS_CEILING = 1000

_BOOTSTRAP_REPS = 1000
_BOOTSTRAP_SEED = 20260601  # reproducible bootstrap across re-runs

_CONFIRMATION_SEEDS: tuple[int, ...] = (5, 6, 7, 8, 9)


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


def _factor_key(cell: dict[str, Any]) -> tuple[float, float, str]:
    return (cell["temperature"], cell["top_p"], cell["reasoning_config"])


def _factor_cell_id(cell: dict[str, Any]) -> str:
    return f"T={cell['temperature']:g}_p={cell['top_p']:g}_R={cell['reasoning_config']}"


def _aggregate_by_factor_cell(
    cells: Iterable[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Aggregate per-seed cells into per-(T, top_p, reasoning_config) records."""
    by_key: dict[tuple[float, float, str], list[dict[str, Any]]] = {}
    for cell in cells:
        # Only aggregate done cells with finite canonical IAE.
        if cell.get("status") != "done":
            continue
        iae = cell.get("canonical_iae")
        if iae is None:
            continue
        by_key.setdefault(_factor_key(cell), []).append(cell)
    aggregates: list[dict[str, Any]] = []
    for _key, seed_cells in by_key.items():
        iae_values = [float(c["canonical_iae"]) for c in seed_cells]
        tokens_p95 = [
            float(c["completion_tokens_p95"])
            for c in seed_cells
            if c.get("completion_tokens_p95") is not None
        ]
        walls_p95 = [
            float(c["wall_clock_p95"]) for c in seed_cells if c.get("wall_clock_p95") is not None
        ]
        ci_lo, ci_hi = _bootstrap_ci_mean(iae_values)
        sample_cell = seed_cells[0]
        aggregates.append(
            {
                "cell_id": _factor_cell_id(sample_cell),
                "temperature": sample_cell["temperature"],
                "top_p": sample_cell["top_p"],
                "reasoning_config": sample_cell["reasoning_config"],
                "reasoning_budget": sample_cell.get("reasoning_budget"),
                "n_seeds_used": len(seed_cells),
                "seeds_used": sorted(c["seed"] for c in seed_cells),
                "mean_canonical_iae": statistics.fmean(iae_values),
                "stdev_canonical_iae": statistics.stdev(iae_values) if len(iae_values) > 1 else 0.0,
                "ci_95_canonical_iae": [ci_lo, ci_hi],
                "max_completion_tokens_p95": max(tokens_p95) if tokens_p95 else None,
                "max_wall_clock_p95": max(walls_p95) if walls_p95 else None,
                "individual_canonical_iae": iae_values,
            }
        )
    return aggregates


def _select_optimum(
    aggregates: list[dict[str, Any]],
) -> tuple[dict[str, Any], str]:
    """Return (best_cell, rationale_string)."""
    if not aggregates:
        raise RuntimeError(
            "no cell aggregates available — analysis cannot recommend an "
            "optimum. Inspect the sweep manifest for done cells."
        )

    def _passes_secondary(c: dict[str, Any]) -> bool:
        wall = c["max_wall_clock_p95"]
        tokens = c["max_completion_tokens_p95"]
        if wall is None or tokens is None:
            return False
        return (
            wall < _SECONDARY_WALL_CLOCK_CEILING_S and tokens < _SECONDARY_COMPLETION_TOKENS_CEILING
        )

    ci_filtered = [
        c
        for c in aggregates
        if c["ci_95_canonical_iae"][1] < _KPIS_THRESHOLD_IAE and _passes_secondary(c)
    ]
    if ci_filtered:
        ranked = sorted(
            ci_filtered,
            key=lambda c: (
                c["mean_canonical_iae"],
                c["max_wall_clock_p95"] if c["max_wall_clock_p95"] is not None else 1e9,
                c["cell_id"],
            ),
        )
        rationale = (
            f"argmin mean canonical IAE among cells with 95 % CI upper "
            f"bound < {_KPIS_THRESHOLD_IAE}, wall_clock_p95 < "
            f"{_SECONDARY_WALL_CLOCK_CEILING_S} s, completion_tokens_p95 < "
            f"{_SECONDARY_COMPLETION_TOKENS_CEILING}; tiebreak on "
            "wall_clock_p95 then cell_id"
        )
        return ranked[0], rationale
    # Fall back: argmin over full surface, no CI filter.
    ranked = sorted(
        aggregates,
        key=lambda c: (
            c["mean_canonical_iae"],
            c["max_wall_clock_p95"] if c["max_wall_clock_p95"] is not None else 1e9,
            c["cell_id"],
        ),
    )
    rationale = (
        f"CI-filtered set was empty (no cell with 95 % CI upper bound < "
        f"{_KPIS_THRESHOLD_IAE} that also satisfies secondary constraints); "
        "argmin selected from the full surface — operator decides whether "
        "to confirm at the chosen cell or accept the variance with "
        "documentation"
    )
    return ranked[0], rationale


def _render_summary_md(
    manifest: dict[str, Any],
    aggregates: list[dict[str, Any]],
    optimum: dict[str, Any],
    rationale: str,
    total_cells: int,
    done_cells: int,
    failed_cells: int,
) -> str:
    lines: list[str] = []
    model = manifest["model_identifier"]
    lines.append(f"# DoE Response Surface — {model}")
    lines.append("")
    lines.append(
        f"Total factor-cells: **{len(aggregates)}**"
        f" (sweep cells done: {done_cells}/{total_cells}, failed: {failed_cells})"
    )
    lines.append("")
    lines.append("## Optimum cell")
    lines.append("")
    lines.append(f"- **cell_id:** `{optimum['cell_id']}`")
    lines.append(f"- temperature: **{optimum['temperature']}**")
    lines.append(f"- top_p: **{optimum['top_p']}**")
    lines.append(f"- reasoning_config: **{optimum['reasoning_config']}**")
    lines.append(
        f"- mean canonical IAE: **{optimum['mean_canonical_iae']:.6f}** "
        f"(95 % CI [{optimum['ci_95_canonical_iae'][0]:.6f}, "
        f"{optimum['ci_95_canonical_iae'][1]:.6f}])"
    )
    lines.append(
        f"- max wall_clock_p95: {optimum['max_wall_clock_p95']} s, "
        f"max completion_tokens_p95: {optimum['max_completion_tokens_p95']}"
    )
    lines.append(f"- _selection rationale:_ {rationale}")
    lines.append("")
    lines.append("## Per-reasoning-config response tables (mean canonical IAE)")
    lines.append("")
    grouped: dict[str, list[dict[str, Any]]] = {}
    for c in aggregates:
        grouped.setdefault(c["reasoning_config"], []).append(c)
    for cfg in sorted(grouped.keys()):
        lines.append(f"### {cfg}")
        lines.append("")
        # Header: temperatures across columns
        temps = sorted({c["temperature"] for c in grouped[cfg]})
        tops = sorted({c["top_p"] for c in grouped[cfg]})
        header = "| top_p \\ T | " + " | ".join(f"{t:g}" for t in temps) + " |"
        sep = "|" + "---|" * (len(temps) + 1)
        lines.append(header)
        lines.append(sep)
        for top in tops:
            cells_for_top = {c["temperature"]: c for c in grouped[cfg] if c["top_p"] == top}
            row = (
                f"| {top:g} | "
                + " | ".join(
                    f"{cells_for_top[t]['mean_canonical_iae']:.4f}" if t in cells_for_top else "—"
                    for t in temps
                )
                + " |"
            )
            lines.append(row)
        lines.append("")
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()

    manifest_path = args.output_root / "sweep_manifest.json"
    if not manifest_path.exists():
        print(f"!! No sweep_manifest.json at {manifest_path}", flush=True)
        return 2
    with manifest_path.open() as fh:
        manifest = json.load(fh)
    total_cells = manifest["total_cells"]
    cells = manifest["cells"]
    done_cells = sum(1 for c in cells if c["status"] == "done")
    failed_cells = sum(1 for c in cells if c["status"] == "failed")
    aggregates = _aggregate_by_factor_cell(cells)
    if not aggregates:
        print(
            "!! No done cells in the manifest — cannot analyse. "
            f"({done_cells}/{total_cells} done, {failed_cells} failed)",
            flush=True,
        )
        return 3
    optimum, rationale = _select_optimum(aggregates)

    # Re-ranked list for the analysis output: all factor-cells ranked
    # by mean canonical IAE.
    ranked = sorted(
        aggregates,
        key=lambda c: (
            c["mean_canonical_iae"],
            c["max_wall_clock_p95"] if c["max_wall_clock_p95"] is not None else 1e9,
            c["cell_id"],
        ),
    )

    analysis = {
        "model_identifier": manifest["model_identifier"],
        "kpis_md_threshold_canonical_iae": _KPIS_THRESHOLD_IAE,
        "secondary_constraints": {
            "wall_clock_p95_ceiling_s": _SECONDARY_WALL_CLOCK_CEILING_S,
            "completion_tokens_p95_ceiling": _SECONDARY_COMPLETION_TOKENS_CEILING,
        },
        "bootstrap": {"reps": _BOOTSTRAP_REPS, "seed": _BOOTSTRAP_SEED},
        "done_cells": done_cells,
        "failed_cells": failed_cells,
        "total_cells": total_cells,
        "factor_cells_aggregated": len(aggregates),
        "optimum_cell_id": optimum["cell_id"],
        "selection_rationale": rationale,
        "ranked_cells": ranked,
    }
    atomic_write_json(args.output_root / "analysis.json", analysis)

    summary = _render_summary_md(
        manifest, aggregates, optimum, rationale, total_cells, done_cells, failed_cells
    )
    (args.output_root / "analysis_summary.md").write_text(summary, encoding="utf-8")

    confirmation_spec = {
        "model_identifier": manifest["model_identifier"],
        "optimum_cell": {
            "temperature": optimum["temperature"],
            "top_p": optimum["top_p"],
            "reasoning_config": optimum["reasoning_config"],
            "reasoning_budget": optimum.get("reasoning_budget"),
            "cell_id": optimum["cell_id"],
        },
        "screening_metrics_n5": {
            "mean_canonical_iae": optimum["mean_canonical_iae"],
            "ci_95_canonical_iae": optimum["ci_95_canonical_iae"],
            "completion_tokens_p95": optimum["max_completion_tokens_p95"],
            "wall_clock_p95": optimum["max_wall_clock_p95"],
            "seeds_used": optimum["seeds_used"],
        },
        "confirmation_seeds": list(_CONFIRMATION_SEEDS),
        "confirmation_output_root": str(args.output_root) + "_confirmation",
        "selection_rationale": rationale,
        "kpis_md_threshold_canonical_iae": _KPIS_THRESHOLD_IAE,
    }
    atomic_write_json(args.output_root / "confirmation_spec.json", confirmation_spec)

    print(
        f"[{manifest['model_identifier']}] analysis complete. "
        f"Optimum cell: {optimum['cell_id']} "
        f"(mean IAE {optimum['mean_canonical_iae']:.6f}, "
        f"95 % CI [{optimum['ci_95_canonical_iae'][0]:.6f}, "
        f"{optimum['ci_95_canonical_iae'][1]:.6f}]).",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
