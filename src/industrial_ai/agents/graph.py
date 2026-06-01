"""LangGraph orchestration for one supervisor cycle of the C2 / C3 agent.

The graph implements the Observer → Optimizer → Critic decomposition
named in PROJECT_PLAN Phase 3. The Optimizer-Critic edge is bounded
by a hard recursion limit (``max_critic_optimizer_rounds``) so a
mis-tuned LLM cannot trap the supervisor in an infinite revise loop.

The graph is built on top of LangGraph's ``StateGraph`` machinery
but does not depend on LangGraph for the test surface — the runner
exposes :func:`run_one_cycle` and :func:`AgentRunner.run_loop` as
the integration points the agentic-simulator wrappers use.

Wall-clock per cycle is measured between entering the graph at the
Observer and emitting the final ``AgentState`` decision; this is the
``supervisory_cycle_wallclock`` KPI from ``docs/kpis.md`` §5.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt

from industrial_ai.agents.errors import CriticLoopLimitError
from industrial_ai.agents.llm_client import LLMClient
from industrial_ai.agents.regulatory_backend import RegulatoryBackend, RegulatoryStepResult
from industrial_ai.agents.state import (
    SETPOINT_BOUNDS,
    AgentState,
    CriticVerdict,
    ObserverReport,
    OptimizerProposal,
)
from industrial_ai.agents.tools import (
    read_twin_state,
)
from industrial_ai.twin.column_a import DEFAULT_PARAMETERS

__all__ = [
    "AgentRunner",
    "CycleOutcome",
    "GraphConfig",
    "run_one_cycle",
]

StateVector = npt.NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class GraphConfig:
    """Hard limits and prompt knobs for the agent graph.

    Attributes
    ----------
    max_critic_optimizer_rounds : int
        Maximum number of Optimizer invocations per supervisor cycle.
        Default 3: one initial proposal + up to two revisions before
        the Critic escalates. Prevents pathological LLM loops.
    supervisor_period_min : float
        Wall-clock cadence of the supervisor (ADR 006: 5–15 min).
    system_prompt : str
        Stable system prompt shared across Observer / Optimizer
        calls. The Critic uses a separate prompt below.
    """

    max_critic_optimizer_rounds: int = 3
    supervisor_period_min: float = 5.0
    system_prompt: str = (
        "You are the supervisory layer of a binary distillation column. "
        "Your job is to choose composition setpoints (y_D_target, x_B_target) "
        "that minimize IAE under disturbance. Respect physical bounds: "
        "y_D_target ∈ [0, 1], x_B_target ∈ [0, 1], and y_D_target must exceed "
        "x_B_target. Reply in a single JSON object with keys y_D_target, "
        "x_B_target, rationale."
    )


@dataclass(slots=True)
class CycleOutcome:
    """Full result of one supervisor cycle: decision + regulatory tick + wallclock.

    ``prompt_tokens`` and ``completion_tokens`` are summed over **all**
    Optimizer LLM calls in this cycle (including revise rounds), so a
    cycle with two Optimizer rounds reports the combined token cost.
    The ``MockLLMClient`` emits a rough character-count // 4 estimate
    on every call so unit tests can observe propagation; the live MLX
    client extracts the real ``usage`` block from
    ``/v1/completions`` and raises
    :class:`LLMResponseMissingUsageError` at the source if the server
    omits it (ADR 010 §2). A client that legitimately has no usage
    information (e.g. a future fixture) sets ``prompt_tokens`` /
    ``completion_tokens`` on the ``LLMResponse`` to ``None`` and the
    accumulator treats that as zero for the cycle.
    """

    state: AgentState
    regulatory_result: RegulatoryStepResult
    wall_clock_seconds: float
    optimizer_rounds: int
    escalated: bool
    prompt_tokens: int
    completion_tokens: int


# ---------------------------------------------------------------------------
# Node implementations.
# ---------------------------------------------------------------------------


def _observer_node(
    *,
    cycle_index: int,
    t_min: float,
    X: StateVector,
    LT_kmol_per_min: float,
    VB_kmol_per_min: float,
    F_kmol_per_min: float,
    zF: float,
    qF: float,
    recent_aggregate_iae: float,
) -> ObserverReport:
    """Read the twin state and pack into an :class:`ObserverReport`."""
    snap = read_twin_state(
        cycle_index=cycle_index,
        t_min=t_min,
        X=X,
        LT_kmol_per_min=LT_kmol_per_min,
        VB_kmol_per_min=VB_kmol_per_min,
        F_kmol_per_min=F_kmol_per_min,
        zF=zF,
        qF=qF,
    )
    return ObserverReport(
        cycle_index=snap.cycle_index,
        t_min=snap.t_min,
        y_D=snap.y_D,
        x_B=snap.x_B,
        LT_kmol_per_min=snap.LT_kmol_per_min,
        VB_kmol_per_min=snap.VB_kmol_per_min,
        F_kmol_per_min=snap.F_kmol_per_min,
        zF=snap.zF,
        qF=snap.qF,
        recent_aggregate_iae=recent_aggregate_iae,
    )


def _optimizer_node(
    *,
    observer_report: ObserverReport,
    llm_client: LLMClient,
    system_prompt: str,
    critic_feedback: str | None = None,
) -> tuple[OptimizerProposal, int, int]:
    """Call the LLM to propose a ``(y_D_target, x_B_target)`` pair.

    Builds a structured user prompt from the Observer report; if a
    Critic-revision feedback is supplied, it is appended verbatim so
    the LLM can adapt the next proposal.
    """
    body = (
        f"Cycle {observer_report.cycle_index} at t={observer_report.t_min:.1f} min. "
        f"Plant: y_D={observer_report.y_D:.4f}, x_B={observer_report.x_B:.4f}, "
        f"LT={observer_report.LT_kmol_per_min:.3f}, VB={observer_report.VB_kmol_per_min:.3f}, "
        f"F={observer_report.F_kmol_per_min:.3f}, zF={observer_report.zF:.3f}, "
        f"qF={observer_report.qF:.3f}. "
        f"Run IAE so far: {observer_report.recent_aggregate_iae:.4f} mole-fraction·min."
    )
    if critic_feedback:
        body += f"\n\nCritic feedback on previous proposal: {critic_feedback}"
    # Modal reasoning toggle per ADR 005 amendment 2026-05-28:
    # Round 1 (no critic feedback) runs in fast /no_think mode for
    # the typical accept path; revisions enable chain-of-thought so
    # the Optimizer has room to reconsider against the Critic's
    # objection. The LLMClient implementation (MLXServerLLMClient)
    # consumes ``reasoning`` to inject the marker and pick the
    # max-tokens budget.
    reasoning = critic_feedback is not None
    reply = llm_client.complete(
        system_prompt=system_prompt,
        user_prompt=body,
        reasoning=reasoning,
    )
    proposal = OptimizerProposal(
        y_D_target=reply.proposal.y_D_target,
        x_B_target=reply.proposal.x_B_target,
        rationale=reply.proposal.rationale,
    )
    # `or 0` handles a client that legitimately has no usage info
    # (LLMResponse fields default to None). MockLLMClient supplies a
    # char-count // 4 estimate so unit tests see real propagation; the
    # live MLX client raises LLMResponseMissingUsageError if the
    # server itself fails to return a `usage` block (ADR 010 §2).
    prompt_tokens = reply.prompt_tokens or 0
    completion_tokens = reply.completion_tokens or 0
    return proposal, prompt_tokens, completion_tokens


def _critic_node(
    *,
    proposal: OptimizerProposal,
    observer_report: ObserverReport,
    optimizer_rounds: int,
    config: GraphConfig,
) -> CriticVerdict:
    """Rule-based critic — no LLM call.

    Checks:

    - In-bounds (Pydantic already enforces this; defensive double-check).
    - Monotonic ordering (``y_D_target > x_B_target``).
    - Round-budget: if the Optimizer has already been invoked
      ``max_critic_optimizer_rounds`` times, the next verdict
      automatically escalates instead of asking for another revise
      pass.
    - Physical plausibility heuristic: ``y_D_target`` should not be
      below the observed ``y_D`` minus a generous slack — if the
      Optimizer is proposing to lower y_D way below the current
      state, the Critic asks for a revise.
    """
    yD_lo, yD_hi = SETPOINT_BOUNDS["y_D_target"]
    xB_lo, xB_hi = SETPOINT_BOUNDS["x_B_target"]
    if not (yD_lo <= proposal.y_D_target <= yD_hi):
        if optimizer_rounds >= config.max_critic_optimizer_rounds:
            return CriticVerdict(
                decision="escalate",
                reason=f"y_D_target out of bounds and budget exhausted; round={optimizer_rounds}",
            )
        return CriticVerdict(decision="revise", reason="y_D_target out of bounds")
    if not (xB_lo <= proposal.x_B_target <= xB_hi):
        if optimizer_rounds >= config.max_critic_optimizer_rounds:
            return CriticVerdict(
                decision="escalate",
                reason=f"x_B_target out of bounds and budget exhausted; round={optimizer_rounds}",
            )
        return CriticVerdict(decision="revise", reason="x_B_target out of bounds")
    if proposal.y_D_target <= proposal.x_B_target:
        if optimizer_rounds >= config.max_critic_optimizer_rounds:
            return CriticVerdict(
                decision="escalate",
                reason="y_D_target ≤ x_B_target (inverted) and budget exhausted",
            )
        return CriticVerdict(
            decision="revise",
            reason=(
                f"y_D_target ({proposal.y_D_target:.4f}) must exceed x_B_target "
                f"({proposal.x_B_target:.4f}); column cannot run inverted."
            ),
        )
    # Sanity: if the proposal demands a y_D drop > 0.3 from observed,
    # the agent is probably confused — ask once for revision.
    if (
        observer_report.y_D - proposal.y_D_target > 0.3
        and optimizer_rounds < config.max_critic_optimizer_rounds
    ):
        return CriticVerdict(
            decision="revise",
            reason=(
                f"Proposed y_D_target {proposal.y_D_target:.3f} drops more than 0.3 "
                f"below observed {observer_report.y_D:.3f}; reconsider."
            ),
        )
    return CriticVerdict(decision="accept", reason="proposal passes all checks")


# ---------------------------------------------------------------------------
# Single-cycle runner.
# ---------------------------------------------------------------------------


def run_one_cycle(
    *,
    cycle_index: int,
    t_min: float,
    X: StateVector,
    LT_kmol_per_min: float,
    VB_kmol_per_min: float,
    F_kmol_per_min: float,
    zF: float,
    qF: float,
    recent_aggregate_iae: float,
    llm_client: LLMClient,
    regulatory_backend: RegulatoryBackend,
    previous_accepted: OptimizerProposal | None = None,
    config: GraphConfig | None = None,
) -> CycleOutcome:
    """Execute Observer → Optimizer → Critic → regulatory backend for one cycle.

    Parameters mirror the per-cycle plant state. ``previous_accepted``
    is the most recent accepted proposal in the run; the runner uses
    it when the Critic escalates so the regulatory backend has a
    sensible fallback target rather than a NaN.

    Returns a :class:`CycleOutcome` containing the final AgentState,
    the regulatory simulation result for the cycle, wall-clock time
    spent inside the graph, the number of optimizer rounds executed,
    and whether the Critic escalated.
    """
    cfg = config or GraphConfig()
    state = AgentState()

    wall_start = time.perf_counter()
    state.observer_report = _observer_node(
        cycle_index=cycle_index,
        t_min=t_min,
        X=X,
        LT_kmol_per_min=LT_kmol_per_min,
        VB_kmol_per_min=VB_kmol_per_min,
        F_kmol_per_min=F_kmol_per_min,
        zF=zF,
        qF=qF,
        recent_aggregate_iae=recent_aggregate_iae,
    )

    critic_feedback: str | None = None
    prompt_tokens_total = 0
    completion_tokens_total = 0
    while True:
        # Hard ceiling — defensive belt-and-suspenders branch. The
        # Critic's own budget check (see _critic_node) escalates one
        # round earlier, so this loop guard should be unreachable in
        # any path the Critic actually traverses. Kept as a final
        # safety net against a future refactor that drops the Critic
        # budget check.
        if (
            state.optimizer_rounds >= cfg.max_critic_optimizer_rounds + 1
        ):  # pragma: no cover - unreachable while the Critic budget check is in place
            state.critic_verdict = CriticVerdict(
                decision="escalate",
                reason=f"optimizer round budget {cfg.max_critic_optimizer_rounds} exceeded",
            )
            break
        proposal, call_prompt_tokens, call_completion_tokens = _optimizer_node(
            observer_report=state.observer_report,
            llm_client=llm_client,
            system_prompt=cfg.system_prompt,
            critic_feedback=critic_feedback,
        )
        prompt_tokens_total += call_prompt_tokens
        completion_tokens_total += call_completion_tokens
        state.optimizer_proposal = proposal
        state.optimizer_rounds += 1
        verdict = _critic_node(
            proposal=proposal,
            observer_report=state.observer_report,
            optimizer_rounds=state.optimizer_rounds,
            config=cfg,
        )
        state.critic_verdict = verdict
        if verdict.decision == "accept":
            state.decision = proposal
            break
        if verdict.decision == "escalate":
            state.decision = previous_accepted
            break
        # decision == revise → loop with feedback
        critic_feedback = verdict.reason

    escalated = state.critic_verdict is not None and state.critic_verdict.decision == "escalate"
    if state.decision is None:
        # ADR 010 §5: the documented escalate verdict re-uses the
        # previous accepted target as a logged safe-state transition.
        # When NO previous accepted target exists (e.g., first-ever
        # cycle of a run with a misbehaving LLM), there is nothing
        # safe to fall back to — substituting a default would be a
        # silent fallback. Abort the run loudly instead.
        raise CriticLoopLimitError(
            "Critic loop limit reached on the first supervisor cycle with no "
            "previous accepted proposal available; the agent has not produced "
            "any valid setpoint yet. Inspect the LLM prompt, the policy, "
            "and the max_critic_optimizer_rounds budget. "
            f"optimizer_rounds={state.optimizer_rounds}, "
            f"last_verdict={state.critic_verdict.decision if state.critic_verdict else 'none'}."
        )
    target = state.decision

    reg_result = regulatory_backend.step(
        X0=X,
        t_start_min=t_min,
        cycle_duration_min=cfg.supervisor_period_min,
        y_D_target=target.y_D_target,
        x_B_target=target.x_B_target,
        F=F_kmol_per_min,
        zF=zF,
        qF=qF,
    )
    wall_end = time.perf_counter()

    return CycleOutcome(
        state=state,
        regulatory_result=reg_result,
        wall_clock_seconds=wall_end - wall_start,
        optimizer_rounds=state.optimizer_rounds,
        escalated=escalated,
        prompt_tokens=prompt_tokens_total,
        completion_tokens=completion_tokens_total,
    )


# ---------------------------------------------------------------------------
# Multi-cycle runner (driver for the agent + regulatory backend loop).
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class AgentRunner:
    """Drive the agent + regulatory backend over an N-cycle supervisor loop.

    A thin orchestration class that tracks plant state, the previous
    accepted proposal (for escalation fallback), and per-cycle KPI
    accumulation. The bulk of the per-cycle logic lives in
    :func:`run_one_cycle`.

    ``canonical_y_D_target`` and ``canonical_x_B_target`` are the
    reference values against which the ``docs/kpis.md`` §1.1 IAE is
    accumulated. They are scenario-defined (not agent-chosen) and must
    match the scenario's canonical targets per the §1.1 contract — for
    disturbance-rejection scenarios, the nominal SS values
    ``(0.99, 0.01)``; for ``yD_setpoint_+0p5pct``, the stepped
    setpoint. Required at construction (no defaults) per ADR 010 §2;
    omitting them raises ``TypeError`` rather than silently defaulting
    to the nominal SS.
    """

    llm_client: LLMClient
    regulatory_backend: RegulatoryBackend
    canonical_y_D_target: float
    canonical_x_B_target: float
    config: GraphConfig = field(default_factory=GraphConfig)
    _previous_accepted: OptimizerProposal | None = None
    #: Canonical IAE accumulated against the scenario-defined targets
    #: (``canonical_y_D_target``, ``canonical_x_B_target``). This is
    #: the ``kpis.md`` §1.1 headline KPI, comparable across C0/C1/C2/C3.
    _canonical_aggregate_iae: float = 0.0
    #: Internal MPC tracking gap to the agent's OWN chosen targets.
    #: Diagnostic only — not the ``kpis.md`` §1.1 KPI. Useful for
    #: spotting when the agent picks targets the regulatory layer
    #: cannot actually reach.
    _internal_tracking_iae: float = 0.0
    _completed_cycles: int = 0

    def step(
        self,
        *,
        cycle_index: int,
        t_min: float,
        X: StateVector,
        LT_kmol_per_min: float,
        VB_kmol_per_min: float,
        F_kmol_per_min: float,
        zF: float,
        qF: float,
    ) -> CycleOutcome:
        """Run one cycle and update internal accumulators."""
        outcome = run_one_cycle(
            cycle_index=cycle_index,
            t_min=t_min,
            X=X,
            LT_kmol_per_min=LT_kmol_per_min,
            VB_kmol_per_min=VB_kmol_per_min,
            F_kmol_per_min=F_kmol_per_min,
            zF=zF,
            qF=qF,
            recent_aggregate_iae=self._canonical_aggregate_iae,
            llm_client=self.llm_client,
            regulatory_backend=self.regulatory_backend,
            previous_accepted=self._previous_accepted,
            config=self.config,
        )
        # Update agent memory only on accept; escalate keeps previous.
        if (
            outcome.state.critic_verdict is not None
            and outcome.state.critic_verdict.decision == "accept"
            and outcome.state.decision is not None
        ):
            self._previous_accepted = outcome.state.decision
        sim = outcome.regulatory_result.simulation
        if sim.success:
            decision = outcome.state.decision
            # Invariant: run_one_cycle raises CriticLoopLimitError when
            # no decision can be reached, so a successful regulatory
            # simulation implies a non-None decision. ADR 010 §2:
            # asserted, not silently defaulted.
            assert decision is not None
            NT = DEFAULT_PARAMETERS.NT
            dt_intervals = np.diff(sim.t)
            yD_trajectory = sim.X[1:, NT - 1]
            xB_trajectory = sim.X[1:, 0]
            canonical_iae = float(
                np.sum(np.abs(yD_trajectory - self.canonical_y_D_target) * dt_intervals)
                + np.sum(np.abs(xB_trajectory - self.canonical_x_B_target) * dt_intervals)
            )
            self._canonical_aggregate_iae += canonical_iae
            internal_iae = float(
                np.sum(np.abs(yD_trajectory - decision.y_D_target) * dt_intervals)
                + np.sum(np.abs(xB_trajectory - decision.x_B_target) * dt_intervals)
            )
            self._internal_tracking_iae += internal_iae
        self._completed_cycles += 1
        return outcome
