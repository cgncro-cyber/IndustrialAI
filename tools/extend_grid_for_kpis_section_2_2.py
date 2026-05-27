"""Extend ``operating_window_states.parquet`` to cover the kpis.md §2.2 grid.

The Phase-1 sweep grid (``default_lv_grid_spec()``, 1125 points) uses
a 0.05-step zF axis: ``zF ∈ {0.30, 0.35, ..., 0.70}``. The Phase-3
off-nominal evaluation grid pre-registered in ``docs/kpis.md`` §2.2
requires the finer-grained ``zF ∈ {0.45, 0.475, 0.525, 0.55}``. Two of
those (0.475 and 0.525) are absent from the existing cache; per §2.3
the off-nominal tools must use cached SS lookups (no runtime
Newton-Krylov), so the cache itself has to be extended.

This tool runs Newton-Krylov SS solves only for the missing
``(F, zF, LT, VB)`` combinations, concatenates the new rows onto the
existing parquet, and writes back. Existing rows are untouched —
re-runs of this tool are idempotent (duplicates filtered out).

Invocation:

    uv run python tools/extend_grid_for_kpis_section_2_2.py
"""

from __future__ import annotations

import json
import time
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

from industrial_ai.twin.column_a.configurations.lv import LVConfiguration
from industrial_ai.twin.column_a.operating_window import (
    GridPoint,
    solve_lv_closed_steady_state,
)
from industrial_ai.twin.column_a.parameters import DEFAULT_PARAMETERS

_REPO_ROOT = Path(__file__).resolve().parent.parent
_STATES_PATH = _REPO_ROOT / "data" / "reference" / "operating_window_states.parquet"
_CSV_PATH = _REPO_ROOT / "data" / "baseline_operating_window.csv"
_SKOGESTAD_SS = _REPO_ROOT / "data" / "reference" / "skogestad_column_a_steady_state.json"


def _load_skogestad_ss() -> np.ndarray:
    with _SKOGESTAD_SS.open() as fh:
        ss = json.load(fh)["steady_state"]
    return np.array(ss["compositions"] + ss["holdups_kmol"], dtype=np.float64)


def _extension_grid() -> list[GridPoint]:
    p = DEFAULT_PARAMETERS
    L0 = p.nominal_reflux_L0_kmol_per_min
    V0 = p.nominal_boilup_V0_kmol_per_min
    F_axis = [0.8, 0.9, 1.0, 1.1, 1.2]
    zF_axis = [0.475, 0.525]
    LT_ratios = [0.9, 0.95, 1.0, 1.05, 1.1]
    VB_ratios = [0.9, 0.95, 1.0, 1.05, 1.1]
    return [
        GridPoint(LT=LT_r * L0, VB=VB_r * V0, F=F, zF=zF, qF=1.0)
        for F, zF, LT_r, VB_r in product(F_axis, zF_axis, LT_ratios, VB_ratios)
    ]


def _already_present(df: pd.DataFrame, point: GridPoint, tol: float = 1e-8) -> bool:
    mask = (
        np.isclose(df["F"], point.F, atol=tol)
        & np.isclose(df["zF"], point.zF, atol=tol)
        & np.isclose(df["LT"], point.LT, atol=tol)
        & np.isclose(df["VB"], point.VB, atol=tol)
        & np.isclose(df["qF"], point.qF, atol=tol)
    )
    return bool(mask.any())


def main() -> int:
    if not _STATES_PATH.exists():
        raise FileNotFoundError(
            f"states cache missing: {_STATES_PATH}. Run tools/run_operating_window_sweep.py first."
        )
    parameters = DEFAULT_PARAMETERS
    NT = parameters.NT
    lv_config = LVConfiguration()

    df_existing = pd.read_parquet(_STATES_PATH)
    print(f"existing parquet rows: {len(df_existing)}")

    grid = _extension_grid()
    print(f"extension grid candidates: {len(grid)} (F=5, zF=2, LT=5, VB=5)")
    missing = [pt for pt in grid if not _already_present(df_existing, pt)]
    print(f"missing rows to solve: {len(missing)}")
    if not missing:
        print("nothing to do; cache already covers the kpis.md §2.2 grid")
        return 0

    X_guess = _load_skogestad_ss()
    new_rows: list[dict[str, float]] = []
    csv_rows: list[dict[str, float | bool]] = []
    start = time.perf_counter()
    for i, point in enumerate(missing):
        X_star, residual_norm, success = solve_lv_closed_steady_state(
            point=point,
            X0=X_guess,
            parameters=parameters,
            lv_config=lv_config,
            residual_tol=1e-7,
            max_iter=200,
        )
        csv_rows.append(
            {
                "F": point.F,
                "zF": point.zF,
                "LT": point.LT,
                "VB": point.VB,
                "qF": point.qF,
                "y_D": float(X_star[NT - 1]),
                "x_B": float(X_star[0]),
                "residual_norm": residual_norm,
                "success": success,
            }
        )
        if success:
            X_guess = X_star
            new_rows.append(
                {
                    "F": point.F,
                    "zF": point.zF,
                    "LT": point.LT,
                    "VB": point.VB,
                    "qF": point.qF,
                    **{f"state_{j:03d}": float(X_star[j]) for j in range(2 * NT)},
                }
            )
        if (i + 1) % 20 == 0:
            print(f"  solved {i + 1}/{len(missing)} ...")
    elapsed = time.perf_counter() - start
    n_success = sum(1 for r in csv_rows if r["success"])
    print(
        f"solved {len(missing)} points in {elapsed:.1f} s; "
        f"convergence {n_success}/{len(missing)} ({n_success / len(missing):.1%})"
    )

    if n_success < len(missing):
        print("WARNING: not all extension points converged — see CSV for diagnostics")

    df_new = pd.DataFrame(new_rows)
    df_merged = pd.concat([df_existing, df_new], ignore_index=True)
    df_merged.to_parquet(_STATES_PATH, index=False)
    print(f"wrote {_STATES_PATH} ({len(df_existing)} -> {len(df_merged)} rows)")

    if _CSV_PATH.exists():
        df_csv = pd.read_csv(_CSV_PATH)
        df_csv_new = pd.DataFrame(csv_rows)
        df_csv_merged = pd.concat([df_csv, df_csv_new], ignore_index=True)
        df_csv_merged.to_csv(_CSV_PATH, index=False)
        print(f"wrote {_CSV_PATH} ({len(df_csv)} -> {len(df_csv_merged)} rows)")

    return 0 if n_success == len(missing) else 1


if __name__ == "__main__":
    raise SystemExit(main())
