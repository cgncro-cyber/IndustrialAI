"""End-to-end graph tests: observer → optimizer → critic → regulatory backend."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

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


def test_inverted_proposal_triggers_revise_then_escalate(nominal_X: np.ndarray) -> None:
    bad = _InvertingMockLLM()
    cfg = GraphConfig(max_critic_optimizer_rounds=3)
    out = _nominal_run(nominal_X, bad, config=cfg)
    # The Critic must have asked for revise until the budget exhausted,
    # then escalated. Optimizer calls ≤ max_critic_optimizer_rounds + 1
    # because the post-budget verdict is escalate.
    assert out.escalated
    assert out.state.critic_verdict.decision == "escalate"
    assert out.optimizer_rounds <= cfg.max_critic_optimizer_rounds + 1
    assert bad.call_count == out.optimizer_rounds


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
