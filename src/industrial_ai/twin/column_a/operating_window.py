"""Operating-window sweep across the LV configuration.

Generates a deterministic grid of ``(F, zF, LT, VB, qF)`` operating
points and re-converges the Column A steady state at each one,
warm-started from the previous successful solve. This is the Phase 1
gate item *"Twin converges across the full intended LV operating
window without manual intervention"* — non-convergence at any grid
point signals either a model defect or an unrealistic input region.

Output is a :class:`pandas.DataFrame` with one row per grid point and
columns ``(F, zF, LT, VB, qF, y_D, x_B, residual_norm, success)``.
The accompanying CLI in ``tools/run_operating_window_sweep.py``
materializes the full >=1000-point sweep into
``data/baseline_operating_window.csv`` per ``docs/figures.md``.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from itertools import product
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pandas as pd
from scipy.optimize import NoConvergence, newton_krylov

from industrial_ai.twin.column_a.configurations.lv import (
    LVConfiguration,
    assemble_inputs_lv,
)
from industrial_ai.twin.column_a.integrator import integrate_open_loop
from industrial_ai.twin.column_a.model import column_a_rhs
from industrial_ai.twin.column_a.parameters import (
    DEFAULT_PARAMETERS,
    ColumnAParameters,
)
from industrial_ai.twin.column_a.steady_state import flat_initial_state

__all__ = [
    "GridPoint",
    "GridSpec",
    "build_grid",
    "default_lv_grid_spec",
    "lookup_lv_ss",
    "solve_lv_closed_steady_state",
    "sweep_operating_window",
]

StateVector = npt.NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class GridSpec:
    """Specification of an LV operating-window grid.

    Attributes
    ----------
    F : sequence of float
        Feed flows to sweep (kmol/min).
    zF : sequence of float
        Feed compositions to sweep (mole fraction).
    LT_ratios : sequence of float
        Reflux flows as ratios of the nominal ``L0``.
    VB_ratios : sequence of float
        Boilups as ratios of the nominal ``V0``.
    qF : sequence of float, optional
        Feed liquid fractions to sweep. Default is the canonical
        ``[1.0]`` (saturated liquid feed only).
    """

    F: Sequence[float]
    zF: Sequence[float]
    LT_ratios: Sequence[float]
    VB_ratios: Sequence[float]
    qF: Sequence[float] = field(default_factory=lambda: [1.0])

    def n_points(self) -> int:
        return len(self.F) * len(self.zF) * len(self.LT_ratios) * len(self.VB_ratios) * len(self.qF)


def default_lv_grid_spec() -> GridSpec:
    """Return a grid spec that yields >=1000 LV operating points.

    Spans the realistic Skogestad LV operating window. The reflux and
    boilup ratios are kept at +/-10 % around their nominals (rather
    than +/-20 %) because the LV configuration enforces only the level
    loops, not the overall mass-balance pair (L - V) <-> (D - B), so
    independently extreme LT and VB combinations push the column into
    physically unrealizable regimes. Feed flow and composition retain
    a wider +/-20 % range because they enter as disturbances and the
    closed loops absorb them naturally.

    - F:  +/-20 % around 1.0 kmol/min
    - zF: 0.30 .. 0.70 (around the symmetric nominal 0.5)
    - LT: +/-10 % around the published L0 = 2.70629
    - VB: +/-10 % around the published V0 = 3.20629
    - qF: 1.0 only (saturated liquid feed)

    Default sizes: 5 x 9 x 5 x 5 x 1 = 1125 grid points. The 5-point
    F / LT_ratio / VB_ratio axes intentionally include the nominal
    values (F=1.0, LT=L0, VB=V0) so downstream code can look up the
    canonical operating point without a separate sweep — for example
    :func:`lookup_lv_ss` for off-nominal robustness checks.
    """
    return GridSpec(
        F=tuple(np.linspace(0.8, 1.2, 5).tolist()),
        zF=tuple(np.linspace(0.3, 0.7, 9).tolist()),
        LT_ratios=tuple(np.linspace(0.9, 1.1, 5).tolist()),
        VB_ratios=tuple(np.linspace(0.9, 1.1, 5).tolist()),
        qF=(1.0,),
    )


@dataclass(frozen=True, slots=True)
class GridPoint:
    """One LV operating point in the sweep (free parameters only)."""

    LT: float
    VB: float
    F: float
    zF: float
    qF: float


def build_grid(
    spec: GridSpec,
    parameters: ColumnAParameters = DEFAULT_PARAMETERS,
) -> Iterable[GridPoint]:
    """Yield :class:`GridPoint` instances for every point in the grid.

    Only the five LV free parameters (LT, VB, F, zF, qF) are emitted.
    The LV-closure provides ``D`` and ``B`` from the converged state
    inside :func:`sweep_operating_window` — fixing them in the input
    vector would over-determine the system and prevent Newton-Krylov
    from finding a self-consistent steady state.
    """
    L0 = parameters.nominal_reflux_L0_kmol_per_min
    V0 = parameters.nominal_boilup_V0_kmol_per_min
    for F, zF, LT_ratio, VB_ratio, qF in product(
        spec.F, spec.zF, spec.LT_ratios, spec.VB_ratios, spec.qF
    ):
        yield GridPoint(LT=LT_ratio * L0, VB=VB_ratio * V0, F=F, zF=zF, qF=qF)


def solve_lv_closed_steady_state(
    *,
    point: GridPoint,
    X0: StateVector,
    parameters: ColumnAParameters,
    lv_config: LVConfiguration,
    residual_tol: float,
    max_iter: int,
    integration_fallback_min: float = 5_000.0,
) -> tuple[StateVector, float, bool]:
    """Solve ``f(X*, U_LV(X*)) = 0`` with the LV closure inlined into the residual.

    Strategy is Newton-Krylov first (cheap when it works), with a
    long-time-integration fallback for points where Newton stalls. The
    fallback uses ``integrate_open_loop`` with the LV closure live —
    the same machinery exercised by ``test_lv_disturbance_scenarios``
    and known to be robust across the operating window. The fallback
    is invoked only when Newton fails, so the warm-start fast path is
    preserved for the bulk of the sweep.
    """

    def residual(X: StateVector) -> StateVector:
        U = assemble_inputs_lv(
            state=X,
            LT=point.LT,
            VB=point.VB,
            F=point.F,
            zF=point.zF,
            qF=point.qF,
            config=lv_config,
            parameters=parameters,
        )
        return column_a_rhs(0.0, X, U, parameters)

    try:
        X_star = newton_krylov(residual, X0, f_tol=residual_tol, maxiter=max_iter)
        X_star = np.asarray(X_star, dtype=np.float64)
        residual_norm = float(np.linalg.norm(residual(X_star), ord=np.inf))
        if residual_norm <= residual_tol:
            return X_star, residual_norm, True
    except NoConvergence:
        pass

    # Newton stalled — run a long-time LV-closed integration.
    def closed_inputs(t: float, X: StateVector) -> npt.NDArray[np.float64]:
        return assemble_inputs_lv(
            state=X,
            LT=point.LT,
            VB=point.VB,
            F=point.F,
            zF=point.zF,
            qF=point.qF,
            config=lv_config,
            parameters=parameters,
        )

    integ = integrate_open_loop(
        X0=X0,
        t_span=(0.0, integration_fallback_min),
        inputs_fn=closed_inputs,
        parameters=parameters,
        rtol=1e-9,
        atol=1e-11,
        t_eval=np.array([integration_fallback_min], dtype=np.float64),
    )
    if not integ.success or integ.X.shape[0] == 0:
        return X0, float("inf"), False

    X_final = np.asarray(integ.X[-1], dtype=np.float64)
    residual_norm = float(np.linalg.norm(residual(X_final), ord=np.inf))
    # Integration fallback accepts a slightly looser tolerance because
    # 5000 min is finite, not infinite.
    return X_final, residual_norm, residual_norm <= max(residual_tol, 1e-4)


def sweep_operating_window(
    spec: GridSpec,
    *,
    parameters: ColumnAParameters = DEFAULT_PARAMETERS,
    X_init: StateVector | None = None,
    lv_config: LVConfiguration | None = None,
    residual_tol: float = 1.0e-7,
    max_iter: int = 200,
    warm_start: bool = True,
    states_path: Path | None = None,
) -> pd.DataFrame:
    """Sweep the operating window and return one DataFrame row per grid point.

    Parameters
    ----------
    spec : GridSpec
        Grid to evaluate.
    parameters : ColumnAParameters, optional
        Column specification.
    X_init : numpy.ndarray of shape (2 * NT,), optional
        Initial guess for the first point. Defaults to a flat
        composition profile; for highest convergence rates supply the
        published Skogestad SS.
    lv_config : LVConfiguration, optional
        Level-loop tuning used inside the LV-closed residual.
    residual_tol : float, optional
        Convergence threshold passed to Newton-Krylov.
    max_iter : int, optional
        Iteration cap per grid point.
    warm_start : bool, optional
        If ``True`` (default), the previous successful solve seeds the
        next point. If ``False``, every point starts from ``X_init``.
    states_path : pathlib.Path, optional
        If given, the full state vector ``X_star`` for every converged
        row is written to a companion parquet file with one column per
        state index (``state_000``..``state_{2*NT-1}``) plus the input
        keys ``F, zF, LT, VB, qF``. Use :func:`lookup_lv_ss` to query
        the cache later. Failed rows are omitted from the parquet.

    Returns
    -------
    pandas.DataFrame
        Columns: ``F, zF, LT, VB, qF, y_D, x_B, residual_norm, success``.
    """
    if X_init is None:
        X_init = flat_initial_state(parameters)
    if lv_config is None:
        lv_config = LVConfiguration()

    NT = parameters.NT
    rows: list[dict[str, float | bool]] = []
    states_rows: list[dict[str, float]] = []
    X_guess = X_init.copy()
    for point in build_grid(spec, parameters):
        X_star, residual_norm, success = solve_lv_closed_steady_state(
            point=point,
            X0=X_guess,
            parameters=parameters,
            lv_config=lv_config,
            residual_tol=residual_tol,
            max_iter=max_iter,
        )
        rows.append(
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
        if warm_start and success:
            X_guess = X_star
        if states_path is not None and success:
            states_rows.append(
                {
                    "F": point.F,
                    "zF": point.zF,
                    "LT": point.LT,
                    "VB": point.VB,
                    "qF": point.qF,
                    **{f"state_{i:03d}": float(X_star[i]) for i in range(2 * NT)},
                }
            )
    df = pd.DataFrame(rows)
    if states_path is not None:
        states_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(states_rows).to_parquet(states_path, index=False)
    return df


def lookup_lv_ss(
    *,
    F: float,
    zF: float,
    qF: float = 1.0,
    LT: float | None = None,
    VB: float | None = None,
    states_path: Path | None = None,
    parameters: ColumnAParameters = DEFAULT_PARAMETERS,
    tol: float = 1.0e-6,
) -> npt.NDArray[np.float64]:
    """Return the cached LV-closed steady-state vector for an operating point.

    Looks up the row in the sweep states parquet whose ``(F, zF, qF,
    LT, VB)`` match the request within ``tol`` and returns the full
    state vector. ``LT`` and ``VB`` default to the column's nominal
    operating-point values.

    Parameters
    ----------
    F, zF : float
        Feed flow (kmol/min) and composition (mole fraction).
    qF : float, optional
        Feed liquid fraction. Default 1.0.
    LT, VB : float, optional
        Reflux and boilup. Default to the parameters' nominal
        operating-point values.
    states_path : pathlib.Path, optional
        Override for the sweep states parquet. Defaults to
        ``data/reference/operating_window_states.parquet`` at the
        repo root.
    parameters : ColumnAParameters, optional
    tol : float, optional
        Per-column absolute matching tolerance. Default 1e-6 — tight
        enough to require exact (rather than nearest) lookups while
        absorbing floating-point round-trip noise.

    Returns
    -------
    numpy.ndarray of shape (2 * NT,)
        Converged state vector at the requested operating point.

    Raises
    ------
    FileNotFoundError
        If the states parquet does not exist (run
        ``tools/run_operating_window_sweep.py`` first).
    KeyError
        If no row matches the requested operating point within ``tol``.
    """
    if states_path is None:
        states_path = (
            Path(__file__).resolve().parents[4]
            / "data"
            / "reference"
            / "operating_window_states.parquet"
        )
    if not states_path.exists():
        raise FileNotFoundError(
            f"sweep states cache missing: {states_path}. "
            "Run `uv run python tools/run_operating_window_sweep.py` first."
        )
    LT_q = parameters.nominal_reflux_L0_kmol_per_min if LT is None else LT
    VB_q = parameters.nominal_boilup_V0_kmol_per_min if VB is None else VB
    df = pd.read_parquet(states_path)
    mask = (
        np.isclose(df["F"], F, atol=tol)
        & np.isclose(df["zF"], zF, atol=tol)
        & np.isclose(df["qF"], qF, atol=tol)
        & np.isclose(df["LT"], LT_q, atol=tol)
        & np.isclose(df["VB"], VB_q, atol=tol)
    )
    hits = df[mask]
    if len(hits) == 0:
        raise KeyError(
            f"no cached SS for (F={F}, zF={zF}, qF={qF}, LT={LT_q}, VB={VB_q}); "
            f"extend the sweep grid or recompute on demand"
        )
    row = hits.iloc[0]
    NT = parameters.NT
    return np.array([row[f"state_{i:03d}"] for i in range(2 * NT)], dtype=np.float64)
