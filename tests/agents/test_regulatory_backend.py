"""Tests for the regulatory-backend protocol and the two adapters."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from industrial_ai.agents.errors import RegulatoryBackendError
from industrial_ai.agents.regulatory_backend import (
    MPCBackend,
    PIDBackend,
    RegulatoryBackend,
    build_regulatory_backend,
)
from industrial_ai.control.c1_linear_mpc import C1MPCConfig
from industrial_ai.twin.column_a import DEFAULT_PARAMETERS


@pytest.fixture(scope="module")
def nominal_X() -> np.ndarray:
    ss_path = (
        Path(__file__).resolve().parent.parent.parent
        / "data"
        / "reference"
        / "skogestad_column_a_steady_state.json"
    )
    with ss_path.open() as fh:
        ss = json.load(fh)["steady_state"]
    return np.array(ss["compositions"] + ss["holdups_kmol"], dtype=np.float64)


def test_build_mpc_backend_default_linearization(nominal_X: np.ndarray) -> None:
    backend = build_regulatory_backend("mpc")
    assert isinstance(backend, MPCBackend)
    assert backend.name == "mpc"


def test_build_pid_backend() -> None:
    backend = build_regulatory_backend("pid")
    assert isinstance(backend, PIDBackend)
    assert backend.name == "pid"


def test_unknown_backend_kind_raises_named_error() -> None:
    """ADR 010 §1: typed exception, not generic ValueError."""
    with pytest.raises(RegulatoryBackendError) as exc_info:
        build_regulatory_backend("unknown")  # type: ignore[arg-type]
    assert "unknown" in str(exc_info.value)


def test_backends_satisfy_protocol() -> None:
    assert isinstance(build_regulatory_backend("mpc"), RegulatoryBackend)
    assert isinstance(build_regulatory_backend("pid"), RegulatoryBackend)


def test_mpc_backend_step_holds_nominal_to_target(nominal_X: np.ndarray) -> None:
    backend = build_regulatory_backend("mpc")
    p = DEFAULT_PARAMETERS
    result = backend.step(
        X0=nominal_X,
        t_start_min=0.0,
        cycle_duration_min=5.0,
        y_D_target=0.99,
        x_B_target=0.01,
        F=p.nominal_feed_F_kmol_per_min,
        zF=0.5,
        qF=1.0,
    )
    assert result.simulation.success
    NT = p.NT
    # At nominal SS with nominal targets, the column must stay near spec.
    assert result.X_final[NT - 1] == pytest.approx(0.99, abs=5e-3)
    assert result.X_final[0] == pytest.approx(0.01, abs=5e-3)
    assert result.backend_name == "mpc"


def test_pid_backend_step_holds_nominal_to_target(nominal_X: np.ndarray) -> None:
    backend = build_regulatory_backend("pid")
    p = DEFAULT_PARAMETERS
    result = backend.step(
        X0=nominal_X,
        t_start_min=0.0,
        cycle_duration_min=5.0,
        y_D_target=0.99,
        x_B_target=0.01,
        F=p.nominal_feed_F_kmol_per_min,
        zF=0.5,
        qF=1.0,
    )
    assert result.simulation.success
    NT = p.NT
    assert result.X_final[NT - 1] == pytest.approx(0.99, abs=2e-2)
    assert result.X_final[0] == pytest.approx(0.01, abs=2e-2)
    assert result.backend_name == "pid"


def test_mpc_backend_respects_explicit_config(nominal_X: np.ndarray) -> None:
    """``C1MPCConfig`` overrides flow through to the backend."""
    aggressive = C1MPCConfig(r_lt=0.01, r_vb=0.01)
    backend = build_regulatory_backend("mpc", mpc_config=aggressive)
    assert isinstance(backend, MPCBackend)
    assert backend.mpc_config.r_lt == 0.01
