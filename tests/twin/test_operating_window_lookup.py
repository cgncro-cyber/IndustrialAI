"""Tests for the sweep-cache state lookup helper.

The Phase-1 operating-window sweep persists per-OP converged states
to a parquet so downstream code (Phase-2 robustness spot-check,
Phase-3 MPC off-nominal linearizations) can warm-start without
re-running Newton-Krylov. The lookups must round-trip and the loaded
states must close the algebraic balances.
"""

from __future__ import annotations

import numpy as np
import pytest

from industrial_ai.twin.column_a import DEFAULT_PARAMETERS
from industrial_ai.twin.column_a.balances import check_balances
from industrial_ai.twin.column_a.configurations.lv import LVConfiguration
from industrial_ai.twin.column_a.operating_window import lookup_lv_ss


@pytest.mark.parametrize("F", [0.8, 0.9, 1.0, 1.1, 1.2])
def test_lookup_yields_state_closing_mass_balance(F: float) -> None:
    """Every cached OP must close mass balance with the LV-implied (D, B)."""
    p = DEFAULT_PARAMETERS
    cfg = LVConfiguration()
    X = lookup_lv_ss(F=F, zF=0.5)
    assert X.shape == (2 * p.NT,)
    D = cfg.Ds + (X[2 * p.NT - 1] - cfg.MDs) * cfg.Kc_D
    B = cfg.Bs + (X[p.NT] - cfg.MBs) * cfg.Kc_B
    U = np.array(
        [
            p.nominal_reflux_L0_kmol_per_min,
            p.nominal_boilup_V0_kmol_per_min,
            D,
            B,
            F,
            0.5,
            1.0,
        ],
        dtype=np.float64,
    )
    res = check_balances(state=X, inputs=U)
    assert res.max_abs() < 1.0e-6, f"F={F}: cached SS fails algebraic balance: residuals={res}"


def test_lookup_raises_keyerror_for_unsampled_point() -> None:
    """Requesting an OP that is not on the grid must fail loudly."""
    with pytest.raises(KeyError, match="no cached SS"):
        lookup_lv_ss(F=1.05, zF=0.51)


def test_lookup_returns_82_element_vector_for_default_column() -> None:
    """State shape matches the canonical Column A (2 * 41 = 82)."""
    X = lookup_lv_ss(F=1.0, zF=0.5)
    assert X.shape == (82,)
    # Compositions must lie in [0, 1].
    NT = DEFAULT_PARAMETERS.NT
    compositions = X[:NT]
    holdups = X[NT:]
    assert (compositions >= 0.0).all() and (compositions <= 1.0).all()
    assert (holdups > 0.0).all()
