"""Phase 1 mini-gate: Skogestad Column A port vs. published references.

Two-layer validation strategy per ADR 007 and the mini-gate plan:

**Option C — paper-defensive validation.** Tests against published
scalar quantities from Skogestad (1997), *Trans IChemE* 75(A),
539-562:

- Steady-state gain matrix G^LV(0), Equation (31) on page 9,
  tolerance ±5 % per entry.
- Dominant continuous-time constant tau_1, Section 4.4 on page 15
  ("the slowest mode, with time constant 194 min"), tolerance ±2 %.
- The second and third slowest time constants from the same passage,
  same tolerance.

**Option A — engineering cross-check.** Tests three open-loop
trajectories produced by running Skogestad's own MATLAB code in Octave
(via ``tools/generate_skogestad_reference_trajectories.m``,
relative-tolerance 1e-10):

- +1 % step in reflux L_T at t = 0.
- -10 % step in feed composition z_F at t = 0.
- +10 % step in feed rate F at t = 0.

Together these cover the three canonical disturbance axes of a
distillation model (reflux, composition, mass flow) and tolerate
relative differences below 1e-6 — the integrator's own numerical
floor.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pytest

from industrial_ai.twin.column_a import DEFAULT_PARAMETERS, integrate_open_loop
from industrial_ai.twin.column_a.configurations.lv import assemble_inputs_lv
from industrial_ai.twin.column_a.linearize import (
    dominant_time_constants_min,
    linearize_lv,
    steady_state_gain,
)

# ---------------------------------------------------------------------------
# Fixtures.
# ---------------------------------------------------------------------------

TRAJECTORY_REFERENCE_PATH = (
    Path(__file__).parent.parent / "data" / "reference" / "skogestad_column_a_trajectories.json"
)


@pytest.fixture(scope="module")
def trajectory_reference() -> dict[str, Any]:
    """Load the Octave-generated reference trajectories."""
    with TRAJECTORY_REFERENCE_PATH.open() as fh:
        data: dict[str, Any] = json.load(fh)
    return data


@pytest.fixture(scope="module")
def linearized_lv_at_nominal(
    skogestad_reference_state: npt.NDArray[np.float64],
) -> Any:
    """Linearize the LV-closed plant at the published nominal operating point."""
    p = DEFAULT_PARAMETERS
    return linearize_lv(
        X_ss=skogestad_reference_state,
        L_ss=p.nominal_reflux_L0_kmol_per_min,
        V_ss=p.nominal_boilup_V0_kmol_per_min,
        F_ss=p.nominal_feed_F_kmol_per_min,
        zF_ss=0.5,
    )


# ---------------------------------------------------------------------------
# Option C — paper-defensive scalar checks.
# ---------------------------------------------------------------------------

# Skogestad (1997) Trans IChemE 75:539, Eq. (31) on page 9.
# Matrix elements are mole-fraction gains (rows = [y_D, x_B], cols = [L, V]).
G_LV_PUBLISHED = np.array(
    [
        [0.8754, -0.8618],
        [1.0846, -1.0982],
    ],
    dtype=np.float64,
)

# Skogestad (1997) Section 4.4 on page 15.
TAU_1_PUBLISHED_MIN = 193.9
TAU_2_PUBLISHED_MIN = 12.0
TAU_3_PUBLISHED_MIN = 3.5


def test_steady_state_gain_matrix_matches_skogestad_1997_eq_31(
    linearized_lv_at_nominal: Any,
) -> None:
    """G^LV(0) per Skogestad 1997 Eq. (31). Tolerance ±5 % per entry."""
    G_full = steady_state_gain(linearized_lv_at_nominal)
    G_LV = G_full[:, :2]
    rel_err = np.abs(G_LV - G_LV_PUBLISHED) / np.abs(G_LV_PUBLISHED)
    assert np.max(rel_err) < 0.05, (
        f"G^LV(0) deviates from Skogestad 1997 Eq. (31) by up to "
        f"{np.max(rel_err) * 100:.2f} % (>5 %). Computed:\n{G_LV}\n"
        f"Published:\n{G_LV_PUBLISHED}"
    )


def test_dominant_time_constant_matches_skogestad_1997_section_4_4(
    linearized_lv_at_nominal: Any,
) -> None:
    """tau_1 per Skogestad 1997 Section 4.4. Tolerance ±2 %."""
    taus = dominant_time_constants_min(linearized_lv_at_nominal, n=1)
    rel_err = abs(taus[0] - TAU_1_PUBLISHED_MIN) / TAU_1_PUBLISHED_MIN
    assert rel_err < 0.02, (
        f"tau_1 = {taus[0]:.2f} min deviates from Skogestad 1997 ({TAU_1_PUBLISHED_MIN} min) "
        f"by {rel_err * 100:.2f} % (>2 %)"
    )


def test_secondary_time_constants_match_skogestad_1997_section_4_4(
    linearized_lv_at_nominal: Any,
) -> None:
    """tau_2 and tau_3 per Skogestad 1997 Section 4.4. Tolerance ±2 %."""
    taus = dominant_time_constants_min(linearized_lv_at_nominal, n=3)
    rel_err_2 = abs(taus[1] - TAU_2_PUBLISHED_MIN) / TAU_2_PUBLISHED_MIN
    rel_err_3 = abs(taus[2] - TAU_3_PUBLISHED_MIN) / TAU_3_PUBLISHED_MIN
    assert rel_err_2 < 0.02, (
        f"tau_2 = {taus[1]:.2f} min vs published {TAU_2_PUBLISHED_MIN} min: "
        f"{rel_err_2 * 100:.2f} % deviation (>2 %)"
    )
    assert rel_err_3 < 0.02, (
        f"tau_3 = {taus[2]:.2f} min vs published {TAU_3_PUBLISHED_MIN} min: "
        f"{rel_err_3 * 100:.2f} % deviation (>2 %)"
    )


# ---------------------------------------------------------------------------
# Option A — engineering cross-check against Octave-generated trajectories.
# ---------------------------------------------------------------------------

TRAJECTORY_TOL = 1.0e-6  # integrator-limit, per the mini-gate plan


def _simulate_step(
    *,
    X0: npt.NDArray[np.float64],
    t_eval: npt.NDArray[np.float64],
    LT: float,
    VB: float,
    F: float,
    zF: float,
    qF: float = 1.0,
) -> Any:
    """Run the Python port with constant inputs and return the full result."""

    def inputs_fn(t: float, X: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        return assemble_inputs_lv(
            state=X,
            LT=LT,
            VB=VB,
            F=F,
            zF=zF,
            qF=qF,
        )

    return integrate_open_loop(
        X0=X0,
        t_span=(float(t_eval[0]), float(t_eval[-1])),
        inputs_fn=inputs_fn,
        rtol=1.0e-10,
        atol=1.0e-12,
        t_eval=t_eval,
    )


def _max_relative_error(ours: np.ndarray, ref: np.ndarray) -> float:
    """Max |ours - ref| / max(|ref|, 1e-3) — scaled to avoid divide-by-zero near 0."""
    scale = np.maximum(np.abs(ref), 1.0e-3)
    return float(np.max(np.abs(ours - ref) / scale))


@pytest.mark.parametrize(
    "scenario_name,LT_mul,zF_mul,F_mul",
    [
        ("L_plus_1pct", 1.01, 1.0, 1.0),
        ("zF_minus_10pct", 1.0, 0.90, 1.0),
        ("F_plus_10pct", 1.0, 1.0, 1.10),
    ],
)
def test_trajectory_matches_octave_reference(
    skogestad_reference_state: npt.NDArray[np.float64],
    trajectory_reference: dict[str, Any],
    scenario_name: str,
    LT_mul: float,
    zF_mul: float,
    F_mul: float,
) -> None:
    """Open-loop step response matches Skogestad's Octave-MATLAB trajectory."""
    p = DEFAULT_PARAMETERS
    ref = trajectory_reference["scenarios"][scenario_name]
    t_eval = np.array(ref["t_min"], dtype=np.float64)

    LT = p.nominal_reflux_L0_kmol_per_min * LT_mul
    VB = p.nominal_boilup_V0_kmol_per_min
    F = p.nominal_feed_F_kmol_per_min * F_mul
    zF = 0.5 * zF_mul

    result = _simulate_step(
        X0=skogestad_reference_state,
        t_eval=t_eval,
        LT=LT,
        VB=VB,
        F=F,
        zF=zF,
    )
    assert result.success, f"integration failed: {result.message}"

    yD_ours = result.X[:, p.NT - 1]
    xB_ours = result.X[:, 0]
    yD_ref = np.array(ref["y_D"], dtype=np.float64)
    xB_ref = np.array(ref["x_B"], dtype=np.float64)

    err_yD = _max_relative_error(yD_ours, yD_ref)
    err_xB = _max_relative_error(xB_ours, xB_ref)

    assert err_yD < TRAJECTORY_TOL, (
        f"y_D trajectory deviates from Octave reference by {err_yD:.3e} (> {TRAJECTORY_TOL:.0e})"
    )
    assert err_xB < TRAJECTORY_TOL, (
        f"x_B trajectory deviates from Octave reference by {err_xB:.3e} (> {TRAJECTORY_TOL:.0e})"
    )
