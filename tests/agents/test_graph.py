"""End-to-end graph tests: observer → optimizer → critic → regulatory backend."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from industrial_ai.agents.errors import CriticLoopLimitError
from industrial_ai.agents.graph import (
    AgentRunner,
    GraphConfig,
    run_one_cycle,
)
from industrial_ai.agents.llm_client import LLMClient, LLMResponse, MockLLMClient
from industrial_ai.agents.regulatory_backend import build_regulatory_backend
from industrial_ai.agents.tools import SetpointProposalInput
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


def _nominal_run(
    nominal_X: np.ndarray,
    llm: LLMClient,
    backend_kind: str = "mpc",
    config: GraphConfig | None = None,
):
    p = DEFAULT_PARAMETERS
    backend = build_regulatory_backend(backend_kind)
    return run_one_cycle(
        cycle_index=0,
        t_min=0.0,
        X=nominal_X,
        LT_kmol_per_min=p.nominal_reflux_L0_kmol_per_min,
        VB_kmol_per_min=p.nominal_boilup_V0_kmol_per_min,
        F_kmol_per_min=p.nominal_feed_F_kmol_per_min,
        zF=0.5,
        qF=1.0,
        recent_aggregate_iae=0.0,
        llm_client=llm,
        regulatory_backend=backend,
        config=config,
    )


def test_single_cycle_nominal_mpc_accepts_and_holds_spec(nominal_X: np.ndarray) -> None:
    out = _nominal_run(nominal_X, MockLLMClient(policy="nominal"))
    assert out.optimizer_rounds == 1
    assert not out.escalated
    assert out.state.critic_verdict.decision == "accept"
    assert out.state.decision.y_D_target == pytest.approx(0.99)
    assert out.regulatory_result.simulation.success
    NT = DEFAULT_PARAMETERS.NT
    assert out.regulatory_result.X_final[NT - 1] == pytest.approx(0.99, abs=5e-3)


def test_single_cycle_nominal_pid_runs_end_to_end(nominal_X: np.ndarray) -> None:
    out = _nominal_run(nominal_X, MockLLMClient(policy="nominal"), backend_kind="pid")
    assert out.regulatory_result.backend_name == "pid"
    assert out.regulatory_result.simulation.success
    assert out.state.critic_verdict.decision == "accept"


def test_wall_clock_is_recorded(nominal_X: np.ndarray) -> None:
    out = _nominal_run(nominal_X, MockLLMClient(policy="nominal"))
    assert out.wall_clock_seconds > 0
    # Loose ceiling — the LLM is a mock and MPC is fast.
    assert out.wall_clock_seconds < 30.0


class _InvertingMockLLM(LLMClient):
    """Pathological mock that proposes an inverted column on every call.

    Bypasses pydantic validation via ``model_construct`` — the real
    LLM has no compile-time guarantee its JSON output is valid, so
    the graph must be robust to invalid proposals reaching the Critic.
    Used to verify the revise-then-escalate flow and the hard
    recursion budget.
    """

    name = "inverting_mock"

    def __init__(self) -> None:
        self.call_count = 0

    def complete(self, **kwargs: object) -> LLMResponse:
        self.call_count += 1
        # model_construct skips validators on purpose.
        proposal = SetpointProposalInput.model_construct(
            y_D_target=0.05,
            x_B_target=0.95,
            rationale="run column inverted (this is a bug bait)",
        )
        return LLMResponse(proposal=proposal, raw_text="bait")


def test_inverted_proposal_first_cycle_raises_critic_loop_limit(
    nominal_X: np.ndarray,
) -> None:
    """ADR 010 §5: first-cycle exhaustion with no previous_accepted must crash.

    The graph's escalate verdict re-uses the previous accepted target as
    a designed safe-state transition. On the very first cycle of a run
    there is nothing to fall back to, so substituting a default would
    be a silent fallback. The runner raises CriticLoopLimitError
    instead.
    """
    bad = _InvertingMockLLM()
    cfg = GraphConfig(max_critic_optimizer_rounds=3)
    with pytest.raises(CriticLoopLimitError) as exc_info:
        _nominal_run(nominal_X, bad, config=cfg)
    # Optimizer was actually invoked the budget number of times — i.e.
    # the hard-limit branch fires, not some earlier guard.
    assert "optimizer_rounds=" in str(exc_info.value)
    assert bad.call_count >= cfg.max_critic_optimizer_rounds


def test_escalation_falls_back_to_previous_accepted(nominal_X: np.ndarray) -> None:
    """When the Critic escalates, the runner must use the previous accepted target."""
    cfg = GraphConfig(max_critic_optimizer_rounds=2)
    runner = AgentRunner(
        llm_client=MockLLMClient(policy="nominal"),
        regulatory_backend=build_regulatory_backend("mpc"),
        config=cfg,
    )
    p = DEFAULT_PARAMETERS
    # First cycle: nominal mock accepts (0.99, 0.01).
    out1 = runner.step(
        cycle_index=0,
        t_min=0.0,
        X=nominal_X,
        LT_kmol_per_min=p.nominal_reflux_L0_kmol_per_min,
        VB_kmol_per_min=p.nominal_boilup_V0_kmol_per_min,
        F_kmol_per_min=1.0,
        zF=0.5,
        qF=1.0,
    )
    assert out1.state.critic_verdict.decision == "accept"
    # Swap to pathological client and step again.
    runner.llm_client = _InvertingMockLLM()
    out2 = runner.step(
        cycle_index=1,
        t_min=5.0,
        X=out1.regulatory_result.X_final,
        LT_kmol_per_min=p.nominal_reflux_L0_kmol_per_min,
        VB_kmol_per_min=p.nominal_boilup_V0_kmol_per_min,
        F_kmol_per_min=1.0,
        zF=0.5,
        qF=1.0,
    )
    assert out2.escalated
    # The regulatory backend ran with the previous accepted target (0.99, 0.01),
    # not the inverted (0.05, 0.95) — so the column stays near spec.
    NT = DEFAULT_PARAMETERS.NT
    assert out2.regulatory_result.X_final[NT - 1] == pytest.approx(0.99, abs=5e-3)


def test_runner_accumulates_iae_over_cycles(nominal_X: np.ndarray) -> None:
    runner = AgentRunner(
        llm_client=MockLLMClient(policy="nominal"),
        regulatory_backend=build_regulatory_backend("mpc"),
    )
    p = DEFAULT_PARAMETERS
    iae_before = runner._aggregate_iae
    runner.step(
        cycle_index=0,
        t_min=0.0,
        X=nominal_X,
        LT_kmol_per_min=p.nominal_reflux_L0_kmol_per_min,
        VB_kmol_per_min=p.nominal_boilup_V0_kmol_per_min,
        F_kmol_per_min=1.0,
        zF=0.5,
        qF=1.0,
    )
    assert runner._aggregate_iae >= iae_before
    assert runner._completed_cycles == 1


class _OutOfBoundsYDMock(LLMClient):
    """Always proposes y_D above 1.0 to exercise the bounds-violation escalate path."""

    name = "out_of_bounds_yd"

    def complete(self, **kwargs: object) -> LLMResponse:
        return LLMResponse(
            proposal=SetpointProposalInput.model_construct(
                y_D_target=1.5,
                x_B_target=0.01,
                rationale="exceed y_D bound (bug bait)",
            ),
            raw_text="bait",
        )


class _OutOfBoundsXBMock(LLMClient):
    """Always proposes x_B above 1.0 (with y_D below) to exercise the x_B escalate path."""

    name = "out_of_bounds_xb"

    def complete(self, **kwargs: object) -> LLMResponse:
        return LLMResponse(
            proposal=SetpointProposalInput.model_construct(
                y_D_target=0.5,
                x_B_target=1.5,
                rationale="exceed x_B bound (bug bait)",
            ),
            raw_text="bait",
        )


def test_y_D_out_of_bounds_escalates_after_budget(nominal_X: np.ndarray) -> None:
    """ADR-010-compliant: budget-exhausted bounds violation triggers CriticLoopLimitError."""
    cfg = GraphConfig(max_critic_optimizer_rounds=2)
    with pytest.raises(CriticLoopLimitError):
        _nominal_run(nominal_X, _OutOfBoundsYDMock(), config=cfg)


def test_x_B_out_of_bounds_escalates_after_budget(nominal_X: np.ndarray) -> None:
    """Same for x_B; both Critic branches must escalate at budget."""
    cfg = GraphConfig(max_critic_optimizer_rounds=2)
    with pytest.raises(CriticLoopLimitError):
        _nominal_run(nominal_X, _OutOfBoundsXBMock(), config=cfg)


class _YDDropMock(LLMClient):
    """Proposes a y_D well below the observed value to trigger the soft-drop revise check."""

    name = "y_d_drop"

    def __init__(self) -> None:
        self.call_count = 0

    def complete(self, **kwargs: object) -> LLMResponse:
        self.call_count += 1
        # First proposal: drop > 0.3 (triggers Critic revise heuristic).
        # Second proposal: stay within drop tolerance (Critic accepts).
        y_D = 0.5 if self.call_count == 1 else 0.98
        return LLMResponse(
            proposal=SetpointProposalInput(y_D_target=y_D, x_B_target=0.01, rationale="t"),
            raw_text="x",
        )


def test_large_y_D_drop_triggers_revise_then_recovery(nominal_X: np.ndarray) -> None:
    """The soft 'drop > 0.3' heuristic should cause a revise, then accept on next round."""
    mock = _YDDropMock()
    cfg = GraphConfig(max_critic_optimizer_rounds=3)
    out = _nominal_run(nominal_X, mock, config=cfg)
    assert out.optimizer_rounds == 2
    assert out.state.critic_verdict.decision == "accept"
    assert mock.call_count == 2


def test_adaptive_mock_proposes_interim_at_offnominal(nominal_X: np.ndarray) -> None:
    """The adaptive mock policy realises the off-nominal regime and proposes interim targets.

    This exercises the Bucket-B target-sequencing path through the
    graph without an LLM in the loop.
    """
    from industrial_ai.twin.column_a.operating_window import lookup_lv_ss

    X_off = lookup_lv_ss(F=0.8, zF=0.45)
    backend = build_regulatory_backend("mpc")
    out = run_one_cycle(
        cycle_index=0,
        t_min=0.0,
        X=X_off,
        LT_kmol_per_min=DEFAULT_PARAMETERS.nominal_reflux_L0_kmol_per_min,
        VB_kmol_per_min=DEFAULT_PARAMETERS.nominal_boilup_V0_kmol_per_min,
        F_kmol_per_min=0.8,
        zF=0.45,
        qF=1.0,
        recent_aggregate_iae=0.0,
        llm_client=MockLLMClient(policy="adaptive"),
        regulatory_backend=backend,
    )
    assert out.state.decision.y_D_target == pytest.approx(0.97)
    assert out.state.decision.x_B_target == pytest.approx(0.02)
    assert out.state.critic_verdict.decision == "accept"
