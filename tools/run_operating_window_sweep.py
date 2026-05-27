"""Materialize the >=1000-point baseline operating-window sweep.

Invocation:

    uv run python tools/run_operating_window_sweep.py

Writes ``data/baseline_operating_window.csv`` per the Phase 1 gate item
*">=1000 logged steady-state points across nominal operating window."*
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from industrial_ai.twin.column_a.operating_window import (
    default_lv_grid_spec,
    sweep_operating_window,
)

_REPO_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_OUTPUT = _REPO_ROOT / "data" / "baseline_operating_window.csv"
_DEFAULT_STATES = _REPO_ROOT / "data" / "reference" / "operating_window_states.parquet"
_SKOGESTAD_SS = _REPO_ROOT / "data" / "reference" / "skogestad_column_a_steady_state.json"


def _load_skogestad_ss() -> np.ndarray:
    with _SKOGESTAD_SS.open() as fh:
        ss = json.load(fh)["steady_state"]
    return np.array(ss["compositions"] + ss["holdups_kmol"], dtype=np.float64)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=_DEFAULT_OUTPUT)
    parser.add_argument(
        "--states-output",
        type=Path,
        default=_DEFAULT_STATES,
        help=(
            "Companion parquet file with the full converged state vector per "
            "row. Consumed by industrial_ai.twin.column_a.operating_window."
            "lookup_lv_ss() for downstream off-nominal SS lookups (Phase 2 "
            "robustness, Phase 3 MPC linearizations at perturbed OPs)."
        ),
    )
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    spec = default_lv_grid_spec()
    print(f"sweeping {spec.n_points()} operating points ...")
    start = time.perf_counter()
    df = sweep_operating_window(spec, X_init=_load_skogestad_ss(), states_path=args.states_output)
    duration = time.perf_counter() - start
    success_rate = float(df["success"].mean())

    df.to_csv(args.output, index=False)
    print(f"wrote {len(df)} rows to {args.output}")
    print(f"wrote {df['success'].sum()} state vectors to {args.states_output}")
    print(f"convergence: {success_rate:.1%} (target >=99 %); runtime {duration:.1f} s")
    return 0 if success_rate >= 0.99 else 1


if __name__ == "__main__":
    raise SystemExit(main())
