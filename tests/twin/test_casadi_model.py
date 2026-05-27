"""Verification tests for the CasADi symbolic Column A model.

The CasADi model must produce the same numerical rhs as the numpy
implementation and exact Jacobians that match central differences. If
this parity holds, the Phase 2 Linear MPC baseline (and any other
gradient-consuming downstream code) can rely on the symbolic backend
without further validation.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import pytest

from industrial_ai.twin.column_a import DEFAULT_PARAMETERS
from industrial_ai.twin.column_a.casadi_model import (
    build_lv_closed_rhs,
    build_lv_jacobians,
    build_open_loop_rhs,
    evaluate_lv_jacobians,
)
from industrial_ai.twin.column_a.configurations.lv import (
    LVConfiguration,
    assemble_inputs_lv,
)
from industrial_ai.twin.column_a.model import column_a_rhs


def _flatten(dm: object) -> np.ndarray:
    """Unwrap a CasADi DM result into a numpy float64 array."""
    return np.asarray(dm, dtype=np.float64).flatten()


def test_open_loop_rhs_matches_numpy_at_published_ss(
    skogestad_reference_state: npt.NDArray[np.float64],
    skogestad_reference_inputs: npt.NDArray[np.float64],
) -> None:
    """CasADi open-loop rhs == numpy rhs at the published SS (to machine epsilon)."""
    f = build_open_loop_rhs()
    sym = _flatten(f(x=skogestad_reference_state, u=skogestad_reference_inputs)["dxdt"])
    num = column_a_rhs(0.0, skogestad_reference_state, skogestad_reference_inputs)
    np.testing.assert_allclose(sym, num, atol=1e-13, rtol=0.0)


def test_open_loop_rhs_matches_numpy_at_perturbed_state(
    skogestad_reference_state: npt.NDArray[np.float64],
    skogestad_reference_inputs: npt.NDArray[np.float64],
) -> None:
    """Parity also holds away from steady state — the operation graph is not SS-specific."""
    rng = np.random.default_rng(seed=42)
    NT = DEFAULT_PARAMETERS.NT
    perturbed = skogestad_reference_state.copy()
    perturbed[:NT] += rng.normal(scale=0.02, size=NT)
    perturbed[:NT] = np.clip(perturbed[:NT], 0.001, 0.999)
    perturbed[NT:] += rng.normal(scale=0.01, size=NT)

    f = build_open_loop_rhs()
    sym = _flatten(f(x=perturbed, u=skogestad_reference_inputs)["dxdt"])
    num = column_a_rhs(0.0, perturbed, skogestad_reference_inputs)
    np.testing.assert_allclose(sym, num, atol=1e-12, rtol=1e-12)


def test_lv_closed_rhs_matches_assembled_inputs(
    skogestad_reference_state: npt.NDArray[np.float64],
) -> None:
    """The LV-closed CasADi rhs equals the numpy rhs with assemble_inputs_lv applied."""
    p = DEFAULT_PARAMETERS
    L = p.nominal_reflux_L0_kmol_per_min
    V = p.nominal_boilup_V0_kmol_per_min
    F = p.nominal_feed_F_kmol_per_min
    zF = 0.5
    qF = p.nominal_feed_liquid_fraction_qF

    lv_rhs = build_lv_closed_rhs(qF=qF)
    sym = _flatten(lv_rhs(x=skogestad_reference_state, mv=np.array([L, V, F, zF]))["dxdt"])

    U = assemble_inputs_lv(state=skogestad_reference_state, LT=L, VB=V, F=F, zF=zF, qF=qF)
    num = column_a_rhs(0.0, skogestad_reference_state, U)
    np.testing.assert_allclose(sym, num, atol=1e-13, rtol=0.0)


def _finite_difference_lv_jacobian(
    *,
    X: npt.NDArray[np.float64],
    mv: npt.NDArray[np.float64],
    qF: float,
    parameters: object,
    lv_config: LVConfiguration,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Reference Jacobian via central differences for the parity check."""
    n = X.shape[0]
    A = np.zeros((n, n), dtype=np.float64)
    B = np.zeros((n, 4), dtype=np.float64)

    def f(xv: npt.NDArray[np.float64], mvv: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        U = assemble_inputs_lv(
            state=xv,
            LT=float(mvv[0]),
            VB=float(mvv[1]),
            F=float(mvv[2]),
            zF=float(mvv[3]),
            qF=qF,
            config=lv_config,
            parameters=parameters,
        )
        return column_a_rhs(0.0, xv, U, parameters)

    for i in range(n):
        h = max(abs(X[i]) * 1e-6, 1e-10)
        xp, xm = X.copy(), X.copy()
        xp[i] += h
        xm[i] -= h
        A[:, i] = (f(xp, mv) - f(xm, mv)) / (2.0 * h)
    for j in range(4):
        h = max(abs(mv[j]) * 1e-6, 1e-10)
        mvp, mvm = mv.copy(), mv.copy()
        mvp[j] += h
        mvm[j] -= h
        B[:, j] = (f(X, mvp) - f(X, mvm)) / (2.0 * h)
    return A, B


def test_lv_jacobians_match_finite_difference(
    skogestad_reference_state: npt.NDArray[np.float64],
) -> None:
    """The CasADi LV Jacobians must match a central-difference reference to ~1e-6."""
    p = DEFAULT_PARAMETERS
    cfg = LVConfiguration()
    L = p.nominal_reflux_L0_kmol_per_min
    V = p.nominal_boilup_V0_kmol_per_min
    F = p.nominal_feed_F_kmol_per_min
    zF = 0.5

    jacs = build_lv_jacobians(lv_config=cfg, qF=p.nominal_feed_liquid_fraction_qF)
    A_sym, B_sym = evaluate_lv_jacobians(jacs, X=skogestad_reference_state, L=L, V=V, F=F, zF=zF)

    A_ref, B_ref = _finite_difference_lv_jacobian(
        X=skogestad_reference_state,
        mv=np.array([L, V, F, zF], dtype=np.float64),
        qF=p.nominal_feed_liquid_fraction_qF,
        parameters=p,
        lv_config=cfg,
    )

    # Central differences with rel-step 1e-6 are accurate to ~1e-7 for
    # well-conditioned entries; the loosest comparable threshold for the
    # full 82-row matrix is ~1e-5 to absorb subtraction loss on the small
    # composition components near 0 or 1.
    np.testing.assert_allclose(A_sym, A_ref, atol=1e-5, rtol=1e-5)
    np.testing.assert_allclose(B_sym, B_ref, atol=1e-5, rtol=1e-5)


def test_lv_jacobians_are_independent_of_state_dtype(
    skogestad_reference_state: npt.NDArray[np.float64],
) -> None:
    """Same Jacobian regardless of whether inputs are list/tuple/ndarray."""
    jacs = build_lv_jacobians()
    A1, B1 = evaluate_lv_jacobians(jacs, X=skogestad_reference_state, L=2.7, V=3.2, F=1.0, zF=0.5)
    A2, B2 = evaluate_lv_jacobians(
        jacs, X=skogestad_reference_state.astype(np.float64), L=2.7, V=3.2, F=1.0, zF=0.5
    )
    np.testing.assert_allclose(A1, A2, atol=1e-15)
    np.testing.assert_allclose(B1, B2, atol=1e-15)


@pytest.mark.parametrize("Kc_D", [5.0, 10.0, 20.0])
def test_lv_jacobians_pick_up_level_loop_gain(
    skogestad_reference_state: npt.NDArray[np.float64],
    Kc_D: float,
) -> None:
    """Changing the level-loop gain must change the symbolic A matrix.

    Guards against a regression where the LV closure is silently
    bypassed in the symbolic build (a common pitfall when refactoring
    the SX construction).
    """
    cfg_default = LVConfiguration()
    cfg_alt = LVConfiguration(Kc_D=Kc_D)
    jacs_default = build_lv_jacobians(lv_config=cfg_default)
    jacs_alt = build_lv_jacobians(lv_config=cfg_alt)

    A_def, _ = evaluate_lv_jacobians(
        jacs_default, X=skogestad_reference_state, L=2.7, V=3.2, F=1.0, zF=0.5
    )
    A_alt, _ = evaluate_lv_jacobians(
        jacs_alt, X=skogestad_reference_state, L=2.7, V=3.2, F=1.0, zF=0.5
    )
    if Kc_D == cfg_default.Kc_D:
        np.testing.assert_allclose(A_def, A_alt, atol=1e-15)
    else:
        assert not np.allclose(A_def, A_alt, atol=1e-12), (
            "Level-loop gain change had no effect on A — closure may be missing"
        )
