"""Phase-3 Schritt B — off-nominal screening response surface + Bucket-B verdict.

Reads the screening manifest + per-cell smokes produced by
``tools/run_offnominal_screening.py``, aggregates the per-(OP, scenario,
submetric) statistics over N=10 seeds, then walks the kpis.md §6 Step 3
decision tree against the Pareto-reference C1 (×100 regularization)
baselines at the same 4 corner OPs.

Outputs:

  analysis.json                 — machine-readable, full per-cell aggregates.
  analysis_summary.md           — human-readable tables grouped by sub-metric.
  bucket_b_classification.json  — kpis.md §6 Step 3 verdict per sub-metric.

Selection / threshold rules per the prompt:

- Per-(OP, scenario, submetric) cell: mean canonical IAE over 10 seeds,
  B=1000 bootstrap 95 % CI.
- Per-(OP, submetric) headline: P95 of scenario means across the 5
  scenarios. Matches kpis.md §2.3 (target_acquisition) and §2.4
  (disturbance_rejection) headline KPI definition.
- Per-submetric grid aggregate: P95 across all 4 OPs of the per-OP P95s.
- Bucket-B verdict: ratio of C2's grid-aggregate (computed at the upper
  CI bound for ADR-010 conservatism) over the C1 Pareto-reference's
  grid-aggregate. Bands per kpis.md §6 Step 3:
    < 0.50  → "strong"      (>= 2.0x improvement)
    < 0.67  → "moderate"    (>= 1.5x improvement, "clears" Bucket-B)
    >= 0.67 and CI brackets 0.67 → "ambiguous"
    >= 0.67 and CI clears 0.67 → "fails"
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

_REPO_ROOT = Path(__file__).resolve().parent.parent

_C1_OFF_NOMINAL = _REPO_ROOT / "data" / "reference" / "c1_off_nominal_baseline.json"
_C1_DISTURBANCE = _REPO_ROOT / "data" / "reference" / "c1_disturbance_rejection_baseline.json"

_BOOTSTRAP_REPS = 1000
_BOOTSTRAP_SEED = 20260602  # reproducible bootstrap across re-runs

_SCENARIOS: tuple[str, ...] = (
    "F_step_+20pct",
    "F_step_-20pct",
    "zF_step_+10pct",
    "zF_step_-10pct",
    "yD_setpoint_+0p5pct",
)
#: kpis.md §2.5 amended 3-OP corner-grid (Changelog 2026-06-02).
#: The (0.8, 0.45) LV-singular corner is excluded; see
#: tools/run_offnominal_screening.py module docstring + docs/analyses/
#: 2026-06-02_schritt_b_failure_diagnosis.md for evidence.
_OPS: tuple[tuple[float, float], ...] = (
    (1.2, 0.45),
    (1.2, 0.55),
    (0.8, 0.55),
)
_SUBMETRICS: tuple[str, ...] = ("target_acquisition", "disturbance_rejection")

_BUCKET_B_THRESHOLD = 0.67
_BUCKET_B_STRONG_THRESHOLD = 0.50


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


def _bootstrap_ci_p95_of_means(
    per_scenario_means_per_seed: list[list[float]],
    *,
    reps: int = _BOOTSTRAP_REPS,
    seed: int = _BOOTSTRAP_SEED,
) -> tuple[float, float]:
    """Bootstrap CI on the P95 of per-scenario means at the per-(OP, submetric) level."""
    if not per_scenario_means_per_seed or not per_scenario_means_per_seed[0]:
        return float("nan"), float("nan")
    n_seeds = len(per_scenario_means_per_seed[0])
    rng = random.Random(seed)
    bootstrap_p95s: list[float] = []
    for _ in range(reps):
        # Resample seeds with replacement, recompute per-scenario means
        # for the bootstrap sample, then P95 over scenarios.
        seed_indices = [rng.randrange(n_seeds) for _ in range(n_seeds)]
        scenario_means = [
            statistics.fmean(scenario_seeds[idx] for idx in seed_indices)
            for scenario_seeds in per_scenario_means_per_seed
        ]
        bootstrap_p95s.append(_percentile(scenario_means, 95.0))
    bootstrap_p95s.sort()
    lo = bootstrap_p95s[int(0.025 * reps)]
    hi = bootstrap_p95s[min(int(0.975 * reps), reps - 1)]
    return float(lo), float(hi)


def _classify_evidence_band(ratio_at_upper_ci: float) -> str:
    if ratio_at_upper_ci < _BUCKET_B_STRONG_THRESHOLD:
        return "strong"
    if ratio_at_upper_ci < _BUCKET_B_THRESHOLD:
        return "moderate"
    return "fails"


def _aggregate_cells(
    cells: Iterable[dict[str, Any]],
) -> dict[tuple[float, float, str, str], dict[str, Any]]:
    """Aggregate per-seed cells into per-(OP, scenario, submetric) records."""
    by_key: dict[tuple[float, float, str, str], list[dict[str, Any]]] = {}
    for cell in cells:
        if cell.get("status") != "done":
            continue
        iae = cell.get("canonical_iae")
        if iae is None:
            continue
        key = (cell["op_F"], cell["op_zF"], cell["scenario"], cell["submetric"])
        by_key.setdefault(key, []).append(cell)
    out: dict[tuple[float, float, str, str], dict[str, Any]] = {}
    for key, seed_cells in by_key.items():
        iae_values = [float(c["canonical_iae"]) for c in seed_cells]
        ci_lo, ci_hi = _bootstrap_ci_mean(iae_values)
        out[key] = {
            "op_F": key[0],
            "op_zF": key[1],
            "scenario": key[2],
            "submetric": key[3],
            "n_seeds_used": len(seed_cells),
            "seeds_used": sorted(c["seed"] for c in seed_cells),
            "mean_canonical_iae": statistics.fmean(iae_values),
            "stdev_canonical_iae": statistics.stdev(iae_values) if len(iae_values) > 1 else 0.0,
            "ci_95_canonical_iae": [ci_lo, ci_hi],
            "individual_canonical_iae": iae_values,
        }
    return out


def _per_op_p95(
    aggregates: dict[tuple[float, float, str, str], dict[str, Any]],
    submetric: str,
) -> dict[tuple[float, float], dict[str, Any]]:
    """Per-OP P95 over the 5 scenarios (mean canonical IAE per scenario)."""
    out: dict[tuple[float, float], dict[str, Any]] = {}
    for op_F, op_zF in _OPS:
        scenario_means: list[float] = []
        scenario_records: list[dict[str, Any]] = []
        seed_grid: list[list[float]] = []
        for scenario in _SCENARIOS:
            key = (op_F, op_zF, scenario, submetric)
            if key not in aggregates:
                continue
            agg = aggregates[key]
            scenario_means.append(agg["mean_canonical_iae"])
            scenario_records.append(agg)
            seed_grid.append(agg["individual_canonical_iae"])
        if not scenario_means:
            continue
        p95 = _percentile(scenario_means, 95.0)
        ci_lo, ci_hi = _bootstrap_ci_p95_of_means(seed_grid)
        out[(op_F, op_zF)] = {
            "op_F": op_F,
            "op_zF": op_zF,
            "n_scenarios": len(scenario_means),
            "p95_canonical_iae": p95,
            "ci_95_p95": [ci_lo, ci_hi],
            "scenario_means": dict(
                zip(
                    [r["scenario"] for r in scenario_records],
                    scenario_means,
                    strict=True,
                )
            ),
        }
    return out


def _grid_p95(per_op: dict[tuple[float, float], dict[str, Any]]) -> dict[str, Any]:
    if not per_op:
        return {"p95_of_p95s": float("nan"), "n_ops": 0, "per_op_values": []}
    values = [v["p95_canonical_iae"] for v in per_op.values()]
    return {
        "p95_of_p95s": _percentile(values, 95.0),
        "max_of_p95s": max(values),
        "mean_of_p95s": statistics.fmean(values),
        "n_ops": len(values),
        "per_op_values": values,
    }


def _c1_per_op_p95(c1: dict[str, Any], ops: tuple[tuple[float, float], ...]) -> dict[str, Any]:
    """Compute C1's per-OP P95 over the 5 scenarios for the requested OPs."""
    per_op: list[float] = []
    per_op_detail: list[dict[str, Any]] = []
    for op_F, op_zF in ops:
        match = next((o for o in c1["per_op"] if o["F"] == op_F and o["zF"] == op_zF), None)
        if match is None:
            continue
        scenarios = match.get("scenarios")
        if not isinstance(scenarios, dict) or not scenarios:
            continue
        scenario_iaes = [
            float(v["iae_mole_fraction_min"])
            for v in scenarios.values()
            if v.get("iae_finite", True) and v.get("iae_mole_fraction_min") is not None
        ]
        if not scenario_iaes:
            continue
        p95 = _percentile(scenario_iaes, 95.0)
        per_op.append(p95)
        per_op_detail.append({"op_F": op_F, "op_zF": op_zF, "p95_canonical_iae": p95})
    return {
        "per_op_p95": per_op_detail,
        "grid_p95_of_p95s": _percentile(per_op, 95.0) if per_op else float("nan"),
        "grid_max_of_p95s": max(per_op) if per_op else float("nan"),
        "grid_mean_of_p95s": statistics.fmean(per_op) if per_op else float("nan"),
        "n_ops": len(per_op),
    }


def _classify_submetric(
    submetric: str,
    c2_grid: dict[str, Any],
    c1_grid: dict[str, Any],
    per_op_c2: dict[tuple[float, float], dict[str, Any]],
) -> dict[str, Any]:
    c1_point = c1_grid["grid_p95_of_p95s"]
    if c1_point is None or c1_point != c1_point:  # NaN check
        return {
            "submetric": submetric,
            "evidence_band": "no_c1_baseline",
            "note": "C1 baseline P95 not computable at the 4 corner OPs",
        }
    c2_p95_values = c2_grid["per_op_values"]
    # CI on the grid P95 using the per-OP CI upper bounds (worst-case):
    # ADR-010 conservatism — compare C2's worst-case grid summary
    # against C1's point estimate.
    upper_ci_values = [v["ci_95_p95"][1] for v in per_op_c2.values()]
    c2_upper_p95 = _percentile(upper_ci_values, 95.0) if upper_ci_values else float("nan")
    c2_lower_p95 = (
        _percentile([v["ci_95_p95"][0] for v in per_op_c2.values()], 95.0)
        if per_op_c2
        else float("nan")
    )
    if c1_point == 0:
        ratio_at_upper = float("inf") if c2_upper_p95 > 0 else 1.0
        ratio_at_point = float("inf") if c2_grid["p95_of_p95s"] > 0 else 1.0
    else:
        ratio_at_upper = c2_upper_p95 / c1_point
        ratio_at_point = c2_grid["p95_of_p95s"] / c1_point
    threshold_clear = ratio_at_upper < _BUCKET_B_THRESHOLD
    band = _classify_evidence_band(ratio_at_upper)
    # ambiguity check: if point estimate clears but upper CI doesn't.
    if not threshold_clear and ratio_at_point < _BUCKET_B_THRESHOLD:
        band = "ambiguous"
    return {
        "submetric": submetric,
        "c1_grid_p95_of_p95s": c1_point,
        "c2_grid_p95_of_p95s_point": c2_grid["p95_of_p95s"],
        "c2_grid_p95_of_p95s_ci95_lower_at_per_op_lower": c2_lower_p95,
        "c2_grid_p95_of_p95s_ci95_upper_at_per_op_upper": c2_upper_p95,
        "ratio_c2_over_c1_at_point": ratio_at_point,
        "ratio_c2_over_c1_at_upper_ci": ratio_at_upper,
        "bucket_b_threshold": _BUCKET_B_THRESHOLD,
        "bucket_b_strong_threshold": _BUCKET_B_STRONG_THRESHOLD,
        "bucket_b_threshold_clear": threshold_clear,
        "evidence_band": band,
        "c2_per_op_values": c2_p95_values,
        "c1_per_op_values": [r["p95_canonical_iae"] for r in c1_grid["per_op_p95"]],
    }


def _render_summary_md(
    manifest: dict[str, Any],
    aggregates: dict[tuple[float, float, str, str], dict[str, Any]],
    per_op_per_submetric: dict[str, dict[tuple[float, float], dict[str, Any]]],
    bucket_b: dict[str, Any],
    done_cells: int,
    failed_cells: int,
    total_cells: int,
) -> str:
    lines: list[str] = []
    lines.append(f"# Off-Nominal Screening Response Surface — {manifest['model_identifier']}")
    lines.append("")
    lines.append(
        f"Cells aggregated: **{len(aggregates)}** factor-cells "
        f"(sweep cells done: {done_cells}/{total_cells}, failed: {failed_cells})"
    )
    lines.append("")
    lines.append("## Bucket-B verdict per sub-metric")
    lines.append("")
    for submetric_key in ("submetric_a_target_acquisition", "submetric_b_disturbance_rejection"):
        v = bucket_b.get(submetric_key, {})
        lines.append(f"### {submetric_key}")
        lines.append("")
        lines.append(f"- evidence_band: **{v.get('evidence_band')}**")
        lines.append(f"- C1 grid-P95-of-P95s: {v.get('c1_grid_p95_of_p95s')}")
        lines.append(f"- C2 grid-P95-of-P95s (point): {v.get('c2_grid_p95_of_p95s_point')}")
        lines.append(
            f"- C2 grid-P95-of-P95s (upper-CI per-OP): {v.get('c2_grid_p95_of_p95s_ci95_upper_at_per_op_upper')}"
        )
        lines.append(f"- ratio C2 / C1 at point: {v.get('ratio_c2_over_c1_at_point')}")
        lines.append(
            f"- ratio C2 / C1 at upper-CI: {v.get('ratio_c2_over_c1_at_upper_ci')} "
            f"(threshold {v.get('bucket_b_threshold')} for Bucket-B clear)"
        )
        lines.append("")
    overall = bucket_b.get("bucket_b_overall_classification", "(not computed)")
    lines.append(f"## Overall: **{overall}**")
    lines.append("")
    lines.append("## Per-OP P95 tables (mean canonical IAE per scenario; P95 across scenarios)")
    lines.append("")
    for submetric in _SUBMETRICS:
        lines.append(f"### {submetric}")
        lines.append("")
        per_op = per_op_per_submetric.get(submetric, {})
        if not per_op:
            lines.append("_(no cells)_")
            lines.append("")
            continue
        lines.append("| OP (F, zF) | P95 (mean IAE) | CI95 lower | CI95 upper |")
        lines.append("|---|---|---|---|")
        for op_key, v in sorted(per_op.items()):
            lines.append(
                f"| ({op_key[0]:g}, {op_key[1]:g}) | {v['p95_canonical_iae']:.4f} | "
                f"{v['ci_95_p95'][0]:.4f} | {v['ci_95_p95'][1]:.4f} |"
            )
        lines.append("")
    return "\n".join(lines) + "\n"


def _decide_overall(
    bucket_b_submetric_a: dict[str, Any], bucket_b_submetric_b: dict[str, Any]
) -> str:
    a_clear = bucket_b_submetric_a.get("bucket_b_threshold_clear", False)
    b_clear = bucket_b_submetric_b.get("bucket_b_threshold_clear", False)
    a_amb = bucket_b_submetric_a.get("evidence_band") == "ambiguous"
    b_amb = bucket_b_submetric_b.get("evidence_band") == "ambiguous"
    if a_clear and b_clear:
        return "Bucket B (both)"
    if a_clear:
        return "Bucket B (target_acquisition)"
    if b_clear:
        return "Bucket B (disturbance_rejection)"
    if a_amb or b_amb:
        return "ambiguous"
    return "does_not_clear"


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
    cells = manifest["cells"]
    total_cells = manifest["total_cells"]
    done_cells = sum(1 for c in cells if c["status"] == "done")
    failed_cells = sum(1 for c in cells if c["status"] == "failed")

    aggregates = _aggregate_cells(cells)
    if not aggregates:
        print(
            f"!! No done cells in the manifest — cannot analyse. "
            f"({done_cells}/{total_cells} done, {failed_cells} failed)",
            flush=True,
        )
        return 3

    per_op_per_submetric: dict[str, dict[tuple[float, float], dict[str, Any]]] = {}
    grid_per_submetric: dict[str, dict[str, Any]] = {}
    for submetric in _SUBMETRICS:
        per_op = _per_op_p95(aggregates, submetric)
        per_op_per_submetric[submetric] = per_op
        grid_per_submetric[submetric] = _grid_p95(per_op)

    # C1 baselines.
    with _C1_OFF_NOMINAL.open() as fh:
        c1_off_nominal = json.load(fh)
    with _C1_DISTURBANCE.open() as fh:
        c1_disturbance = json.load(fh)
    c1_target_grid = _c1_per_op_p95(c1_off_nominal, _OPS)
    c1_disturbance_grid = _c1_per_op_p95(c1_disturbance, _OPS)

    bucket_b_a = _classify_submetric(
        "target_acquisition",
        grid_per_submetric["target_acquisition"],
        c1_target_grid,
        per_op_per_submetric["target_acquisition"],
    )
    bucket_b_b = _classify_submetric(
        "disturbance_rejection",
        grid_per_submetric["disturbance_rejection"],
        c1_disturbance_grid,
        per_op_per_submetric["disturbance_rejection"],
    )

    bucket_b_overall = _decide_overall(bucket_b_a, bucket_b_b)

    # Stringify tuple keys for JSON serialization.
    aggregates_json = [
        {**v, "key": list(k)}  # type: ignore[dict-item]
        for k, v in aggregates.items()
    ]
    per_op_json = {
        submetric: [{**v, "op_key": list(op_key)} for op_key, v in per_op.items()]
        for submetric, per_op in per_op_per_submetric.items()
    }

    analysis = {
        "model_identifier": manifest["model_identifier"],
        "done_cells": done_cells,
        "failed_cells": failed_cells,
        "total_cells": total_cells,
        "bootstrap": {"reps": _BOOTSTRAP_REPS, "seed": _BOOTSTRAP_SEED},
        "factor_cells_aggregated": len(aggregates),
        "aggregates": aggregates_json,
        "per_op_per_submetric": per_op_json,
        "grid_per_submetric": grid_per_submetric,
        "c1_baselines": {
            "target_acquisition_path": str(_C1_OFF_NOMINAL),
            "disturbance_rejection_path": str(_C1_DISTURBANCE),
            "target_acquisition_grid": c1_target_grid,
            "disturbance_rejection_grid": c1_disturbance_grid,
        },
    }
    atomic_write_json(args.output_root / "analysis.json", analysis)

    bucket_b = {
        "primary_model": manifest["model_identifier"],
        "pareto_reference_c1": {
            "target_acquisition_source": str(_C1_OFF_NOMINAL),
            "disturbance_rejection_source": str(_C1_DISTURBANCE),
            "regularization": "r_lt=r_vb=10 (x100, Pareto-reference per kpis.md §6 Step 3)",
        },
        "kpis_md_thresholds": {
            "bucket_b_threshold": _BUCKET_B_THRESHOLD,
            "bucket_b_strong_threshold": _BUCKET_B_STRONG_THRESHOLD,
        },
        "submetric_a_target_acquisition": bucket_b_a,
        "submetric_b_disturbance_rejection": bucket_b_b,
        "bucket_b_overall_classification": bucket_b_overall,
        "linearization_drift_correlation": {
            "note": (
                "kpis.md §6 Step 3 validation: improvement should concentrate at OPs "
                "with highest linearization_drift_g. Per-OP C2 and C1 values reported "
                "in analysis.json grid_per_submetric for manual cross-reference "
                "against data/reference/c1_regularization_sweep.json."
            ),
        },
    }
    atomic_write_json(args.output_root / "bucket_b_classification.json", bucket_b)

    summary = _render_summary_md(
        manifest, aggregates, per_op_per_submetric, bucket_b, done_cells, failed_cells, total_cells
    )
    (args.output_root / "analysis_summary.md").write_text(summary, encoding="utf-8")

    print(
        f"[{manifest['model_identifier']} off-nominal] analysis complete. "
        f"Bucket-B overall: {bucket_b_overall}. "
        f"target_acquisition band: {bucket_b_a.get('evidence_band')}, "
        f"disturbance_rejection band: {bucket_b_b.get('evidence_band')}.",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
