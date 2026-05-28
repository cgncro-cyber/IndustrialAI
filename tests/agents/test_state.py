"""Schema validation for the agent state objects."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from industrial_ai.agents.state import (
    MV_BOUNDS,
    SETPOINT_BOUNDS,
    AgentState,
    CriticVerdict,
    ObserverReport,
    OptimizerProposal,
)


def test_observer_report_is_frozen() -> None:
    rep = ObserverReport(
        cycle_index=0,
        t_min=0.0,
        y_D=0.99,
        x_B=0.01,
        LT_kmol_per_min=2.7,
        VB_kmol_per_min=3.2,
        F_kmol_per_min=1.0,
        zF=0.5,
        qF=1.0,
        recent_aggregate_iae=0.0,
    )
    with pytest.raises(ValidationError):
        rep.y_D = 0.5  # type: ignore[misc]


def test_observer_report_rejects_negative_cycle_index() -> None:
    with pytest.raises(ValidationError):
        ObserverReport(
            cycle_index=-1,
            t_min=0.0,
            y_D=0.99,
            x_B=0.01,
            LT_kmol_per_min=2.7,
            VB_kmol_per_min=3.2,
            F_kmol_per_min=1.0,
            zF=0.5,
            qF=1.0,
            recent_aggregate_iae=0.0,
        )


def test_optimizer_proposal_in_bounds_helper() -> None:
    good = OptimizerProposal(y_D_target=0.99, x_B_target=0.01, rationale="test")
    assert good.in_bounds()
    # Pydantic frozen, but we can still construct an out-of-bounds proposal
    # via the bypass route — assert the helper catches it.
    bad = OptimizerProposal.model_construct(y_D_target=1.5, x_B_target=0.01, rationale="x")
    assert not bad.in_bounds()


def test_optimizer_proposal_requires_rationale() -> None:
    with pytest.raises(ValidationError):
        OptimizerProposal(y_D_target=0.99, x_B_target=0.01, rationale="")


def test_critic_verdict_decision_enum() -> None:
    with pytest.raises(ValidationError):
        CriticVerdict(decision="unknown", reason="bogus")  # type: ignore[arg-type]
    for value in ("accept", "revise", "escalate"):
        v = CriticVerdict(decision=value, reason="ok")  # type: ignore[arg-type]
        assert v.decision == value


def test_agent_state_terminal_detection() -> None:
    state = AgentState()
    assert not state.is_terminal()
    state.critic_verdict = CriticVerdict(decision="revise", reason="..")
    assert not state.is_terminal()
    state.critic_verdict = CriticVerdict(decision="accept", reason="..")
    assert state.is_terminal()
    state.critic_verdict = CriticVerdict(decision="escalate", reason="..")
    assert state.is_terminal()


def test_setpoint_bounds_are_canonical() -> None:
    assert SETPOINT_BOUNDS["y_D_target"] == (0.0, 1.0)
    assert SETPOINT_BOUNDS["x_B_target"] == (0.0, 1.0)


def test_mv_bounds_match_assumptions() -> None:
    # Same bounds the C1 MPC enforces.
    assert MV_BOUNDS["LT"] == (0.0, 10.0)
    assert MV_BOUNDS["VB"] == (0.0, 10.0)
