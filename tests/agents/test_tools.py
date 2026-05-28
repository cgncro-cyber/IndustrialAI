"""Schema + functional tests for the agent tool surface."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from pydantic import ValidationError

from industrial_ai.agents.tools import (
    SetpointProposalInput,
    propose_setpoint,
    query_kpi,
    read_recent_disturbance,
    read_twin_state,
)
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


def test_read_twin_state_extracts_compositions(nominal_X: np.ndarray) -> None:
    snap = read_twin_state(
        cycle_index=0,
        t_min=0.0,
        X=nominal_X,
        LT_kmol_per_min=2.706,
        VB_kmol_per_min=3.206,
        F_kmol_per_min=1.0,
        zF=0.5,
        qF=1.0,
    )
    NT = DEFAULT_PARAMETERS.NT
    assert snap.y_D == pytest.approx(float(nominal_X[NT - 1]))
    assert snap.x_B == pytest.approx(float(nominal_X[0]))
    assert snap.cycle_index == 0


def test_read_recent_disturbance_relative_F() -> None:
    win = read_recent_disturbance(
        window_min=30.0,
        F_history=[1.0, 1.0, 1.2],
        zF_history=[0.5, 0.5, 0.5],
    )
    assert win.F_start == 1.0
    assert win.F_end == 1.2
    assert win.F_delta_relative == pytest.approx(0.2)
    assert win.zF_delta_absolute == pytest.approx(0.0)


def test_read_recent_disturbance_rejects_empty_history() -> None:
    with pytest.raises(ValueError):
        read_recent_disturbance(window_min=5.0, F_history=[], zF_history=[0.5])


def test_read_recent_disturbance_rejects_non_positive_window() -> None:
    with pytest.raises(ValueError):
        read_recent_disturbance(window_min=0.0, F_history=[1.0], zF_history=[0.5])


def test_setpoint_proposal_rejects_inverted_column() -> None:
    with pytest.raises(ValidationError) as exc_info:
        SetpointProposalInput(y_D_target=0.4, x_B_target=0.5, rationale="invert")
    assert "inverted" in str(exc_info.value)


def test_setpoint_proposal_rejects_out_of_bounds() -> None:
    with pytest.raises(ValidationError):
        SetpointProposalInput(y_D_target=1.5, x_B_target=0.01, rationale="x")


def test_propose_setpoint_passthrough() -> None:
    inp = SetpointProposalInput(y_D_target=0.99, x_B_target=0.01, rationale="hold")
    assert propose_setpoint(inp) is inp


def test_query_kpi_round_trip() -> None:
    snap = query_kpi(cycle_index=4, aggregate_iae_so_far=0.123, completed_cycles=4)
    assert snap.aggregate_iae_so_far == pytest.approx(0.123)
    assert snap.cycle_index == 4
    assert snap.completed_cycles == 4
