"""Unit tests for the Phase-2 KPI suite.

KPIs are pure functions of a :class:`SimulationResult`. The tests
construct synthetic results so each KPI can be verified against a
hand-computed expected value without running the full Column A
integration.
"""

from __future__ import annotations

import numpy as np
import pytest

from industrial_ai.evaluation.kpis import KPIConfig, KPISet, compute_kpis
from industrial_ai.twin.simulate import SimulationResult


def _make_result(
    *,
    t: np.ndarray,
    y_D: np.ndarray,
    x_B: np.ndarray,
    L: np.ndarray,
    V: np.ndarray,
    D: np.ndarray,
    B: np.ndarray,
    F: np.ndarray,
    zF: np.ndarray,
    y_D_sp: np.ndarray,
    x_B_sp: np.ndarray,
    success: bool = True,
) -> SimulationResult:
    """Assemble a SimulationResult from individual channels.

    Length conventions follow simulate_lv_closed_loop: ``t``, ``y_D``,
    ``x_B`` have ``n_ticks + 1`` entries; input and setpoint channels
    have ``n_ticks`` entries.
    """
    NT = 41
    n = len(t)
    X = np.zeros((n, 2 * NT))
    X[:, 0] = x_B
    X[:, NT - 1] = y_D
    inputs = np.column_stack([L, V, D, B, F, zF, np.ones_like(F)])
    setpoints = np.column_stack([y_D_sp, x_B_sp])
    return SimulationResult(
        t=t,
        X=X,
        inputs=inputs,
        applied_setpoints=setpoints,
        requested_setpoints=setpoints,
        cycle_wall_clock_seconds=np.zeros(n - 1),
        success=success,
        message="synthetic",
    )


def test_specific_energy_constant_inputs() -> None:
    """With constant V and D, specific energy equals V/D exactly."""
    n = 100
    t = np.linspace(0.0, 1.0, n + 1)
    result = _make_result(
        t=t,
        y_D=np.full(n + 1, 0.99),
        x_B=np.full(n + 1, 0.01),
        L=np.full(n, 2.7),
        V=np.full(n, 3.2),
        D=np.full(n, 0.5),
        B=np.full(n, 0.5),
        F=np.full(n, 1.0),
        zF=np.full(n, 0.5),
        y_D_sp=np.full(n, 0.99),
        x_B_sp=np.full(n, 0.01),
    )
    kpis = compute_kpis(result)
    assert kpis.specific_energy_kmol_per_kmol == pytest.approx(3.2 / 0.5, rel=1e-12)


def test_light_yield_at_nominal_skogestad_operating_point() -> None:
    """At 99 % top purity, balanced D and B, yield should be 0.99 * D / (F * zF) = 0.99."""
    n = 50
    t = np.linspace(0.0, 1.0, n + 1)
    result = _make_result(
        t=t,
        y_D=np.full(n + 1, 0.99),
        x_B=np.full(n + 1, 0.01),
        L=np.full(n, 2.7),
        V=np.full(n, 3.2),
        D=np.full(n, 0.5),
        B=np.full(n, 0.5),
        F=np.full(n, 1.0),
        zF=np.full(n, 0.5),
        y_D_sp=np.full(n, 0.99),
        x_B_sp=np.full(n, 0.01),
    )
    kpis = compute_kpis(result)
    expected = 0.5 * 0.99 / (1.0 * 0.5)
    assert kpis.light_yield == pytest.approx(expected, rel=1e-12)


def test_constraint_violations_count_below_spec_only() -> None:
    """Half the ticks below spec on y_D → exactly that many violations."""
    n = 100
    t = np.linspace(0.0, 1.0, n + 1)
    y_D = np.where(np.arange(n + 1) < 50, 0.98, 0.995)  # 50 violations
    result = _make_result(
        t=t,
        y_D=y_D,
        x_B=np.full(n + 1, 0.005),  # always in spec
        L=np.full(n, 2.7),
        V=np.full(n, 3.2),
        D=np.full(n, 0.5),
        B=np.full(n, 0.5),
        F=np.full(n, 1.0),
        zF=np.full(n, 0.5),
        y_D_sp=np.full(n, 0.99),
        x_B_sp=np.full(n, 0.01),
    )
    kpis = compute_kpis(result)
    assert kpis.constraint_violations == 50


def test_settling_time_immediately_after_onset_when_already_settled() -> None:
    """If the trajectory never moves, settling time = 0 (settles at onset)."""
    n = 1200  # 60 min at 0.05 min ticks
    t = np.arange(n + 1) * 0.05
    y_D = np.full(n + 1, 0.99)
    x_B = np.full(n + 1, 0.01)
    result = _make_result(
        t=t,
        y_D=y_D,
        x_B=x_B,
        L=np.full(n, 2.7),
        V=np.full(n, 3.2),
        D=np.full(n, 0.5),
        B=np.full(n, 0.5),
        F=np.full(n, 1.0),
        zF=np.full(n, 0.5),
        y_D_sp=np.full(n, 0.99),
        x_B_sp=np.full(n, 0.01),
    )
    kpis = compute_kpis(result)
    # Onset at 5 min; the system was already at the final value, so
    # settling time relative to onset is 0.
    assert kpis.settling_time_min == pytest.approx(0.0, abs=1e-12)


def test_settling_time_inf_when_oscillating_forever() -> None:
    """An oscillation larger than the settling band returns +inf."""
    n = 1200
    t = np.arange(n + 1) * 0.05
    # y_D oscillates by +/- 0.005 around 0.99 — much larger than the
    # default 5e-4 settling band.
    y_D = 0.99 + 0.005 * np.sin(2.0 * np.pi * t / 5.0)
    x_B = np.full(n + 1, 0.01)
    result = _make_result(
        t=t,
        y_D=y_D,
        x_B=x_B,
        L=np.full(n, 2.7),
        V=np.full(n, 3.2),
        D=np.full(n, 0.5),
        B=np.full(n, 0.5),
        F=np.full(n, 1.0),
        zF=np.full(n, 0.5),
        y_D_sp=np.full(n, 0.99),
        x_B_sp=np.full(n, 0.01),
    )
    kpis = compute_kpis(result)
    assert np.isinf(kpis.settling_time_min)


def test_mv_activity_is_total_variation() -> None:
    """MV activity = sum of |dLT| + |dVB| over the trajectory."""
    n = 5
    t = np.linspace(0.0, 1.0, n + 1)
    L = np.array([2.7, 2.9, 2.5, 3.0, 2.8])  # |diffs| = 0.2 + 0.4 + 0.5 + 0.2 = 1.3
    V = np.array([3.2, 3.4, 3.4, 3.6, 3.5])  # |diffs| = 0.2 + 0.0 + 0.2 + 0.1 = 0.5
    result = _make_result(
        t=t,
        y_D=np.full(n + 1, 0.99),
        x_B=np.full(n + 1, 0.01),
        L=L,
        V=V,
        D=np.full(n, 0.5),
        B=np.full(n, 0.5),
        F=np.full(n, 1.0),
        zF=np.full(n, 0.5),
        y_D_sp=np.full(n, 0.99),
        x_B_sp=np.full(n, 0.01),
    )
    kpis = compute_kpis(result)
    assert kpis.mv_activity_kmol_per_min == pytest.approx(1.3 + 0.5, abs=1e-12)


def test_iae_on_constant_offset() -> None:
    """IAE with constant 0.005 offset on y_D over 1 min interval = 0.005."""
    t = np.array([0.0, 1.0])
    y_D = np.array([0.985, 0.985])
    x_B = np.array([0.01, 0.01])
    y_D_sp = np.array([0.99])
    x_B_sp = np.array([0.01])
    result = _make_result(
        t=t,
        y_D=y_D,
        x_B=x_B,
        L=np.array([2.7]),
        V=np.array([3.2]),
        D=np.array([0.5]),
        B=np.array([0.5]),
        F=np.array([1.0]),
        zF=np.array([0.5]),
        y_D_sp=y_D_sp,
        x_B_sp=x_B_sp,
    )
    kpis = compute_kpis(result)
    assert kpis.iae_mole_fraction_min == pytest.approx(0.005, rel=1e-12)


def test_failed_simulation_returns_inf_kpis() -> None:
    """If the simulation reports failure, KPIs return their failure sentinels."""
    n = 10
    t = np.linspace(0.0, 0.5, n + 1)
    result = _make_result(
        t=t,
        y_D=np.full(n + 1, 0.99),
        x_B=np.full(n + 1, 0.01),
        L=np.full(n, 2.7),
        V=np.full(n, 3.2),
        D=np.full(n, 0.5),
        B=np.full(n, 0.5),
        F=np.full(n, 1.0),
        zF=np.full(n, 0.5),
        y_D_sp=np.full(n, 0.99),
        x_B_sp=np.full(n, 0.01),
        success=False,
    )
    kpis = compute_kpis(result)
    assert np.isinf(kpis.specific_energy_kmol_per_kmol)
    assert kpis.light_yield == 0.0
    assert kpis.constraint_violations == -1
    assert np.isinf(kpis.settling_time_min)
    assert np.isinf(kpis.mv_activity_kmol_per_min)
    assert np.isinf(kpis.iae_mole_fraction_min)


def test_as_dict_is_json_safe() -> None:
    """KPISet.as_dict() must produce a JSON-serializable dict."""
    import json

    s = KPISet(
        specific_energy_kmol_per_kmol=6.4,
        light_yield=0.99,
        constraint_violations=3,
        settling_time_min=15.5,
        mv_activity_kmol_per_min=2.1,
        iae_mole_fraction_min=0.45,
    )
    d = s.as_dict()
    blob = json.dumps(d)
    assert json.loads(blob) == d


def test_custom_kpi_config_thresholds_applied() -> None:
    """A tighter spec band catches violations that the default tolerates."""
    n = 20
    t = np.linspace(0.0, 1.0, n + 1)
    # y_D = 0.991 — passes the default 0.99 spec, fails a stricter 0.992.
    y_D = np.full(n + 1, 0.991)
    result = _make_result(
        t=t,
        y_D=y_D,
        x_B=np.full(n + 1, 0.005),
        L=np.full(n, 2.7),
        V=np.full(n, 3.2),
        D=np.full(n, 0.5),
        B=np.full(n, 0.5),
        F=np.full(n, 1.0),
        zF=np.full(n, 0.5),
        y_D_sp=np.full(n, 0.99),
        x_B_sp=np.full(n, 0.01),
    )
    loose = compute_kpis(result, config=KPIConfig())
    tight = compute_kpis(result, config=KPIConfig(y_D_spec_low=0.992))
    assert loose.constraint_violations == 0
    assert tight.constraint_violations == n + 1
