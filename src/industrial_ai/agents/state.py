"""LangGraph state schema for the C2 / C3 agentic supervisor (Phase 3).

The supervisor decomposes one supervisory cycle into three nodes
(per PROJECT_PLAN Phase 3): an Observer that reads the plant + KPI
state, an Optimizer that proposes a setpoint update, and a Critic
that reviews the proposal. The state graph passes a single
:class:`AgentState` between nodes and accumulates the per-cycle
decision trail.

The schema is intentionally minimal — only fields the agent
actually reads or writes flow through ``AgentState``. Backend
identity, regulatory result, and plant integration live in the
runner (:mod:`industrial_ai.agents.graph`), not in the state.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import numpy.typing as npt
from pydantic import BaseModel, ConfigDict, Field

__all__ = [
    "MV_BOUNDS",
    "SETPOINT_BOUNDS",
    "AgentState",
    "CriticVerdict",
    "ObserverReport",
    "OptimizerProposal",
]

StateVector = npt.NDArray[np.float64]

# Hard plant-physical bounds for composition setpoints.
# y_D / x_B are mole fractions; the safety-gate list in kpis.md §3.3
# defines the tighter constraints, but the agent must never *propose*
# unphysical values.
SETPOINT_BOUNDS: dict[str, tuple[float, float]] = {
    "y_D_target": (0.0, 1.0),
    "x_B_target": (0.0, 1.0),
}

# MV operating envelope per column_a/assumptions.md.
MV_BOUNDS: dict[str, tuple[float, float]] = {
    "LT": (0.0, 10.0),
    "VB": (0.0, 10.0),
}


class ObserverReport(BaseModel):
    """Output of the Observer node — what the plant looks like right now.

    Composition signals are top (``y_D``) and bottom (``x_B``); MV
    signals are the most recent ``(LT, VB)`` actually applied; the
    operating-point disturbance trio ``(F, zF, qF)`` is carried
    through so the Optimizer can spot regime changes.
    """

    model_config = ConfigDict(frozen=True)

    cycle_index: int = Field(ge=0, description="Zero-based supervisor cycle counter.")
    t_min: float = Field(description="Wall-clock time of this observation, in plant minutes.")
    y_D: float = Field(description="Top-tray composition (mole fraction).")
    x_B: float = Field(description="Bottom-tray composition (mole fraction).")
    LT_kmol_per_min: float = Field(description="Most recent applied reflux.")
    VB_kmol_per_min: float = Field(description="Most recent applied boilup.")
    F_kmol_per_min: float = Field(description="Current feed flow.")
    zF: float = Field(description="Current feed composition.")
    qF: float = Field(description="Current feed liquid fraction.")
    recent_aggregate_iae: float = Field(
        ge=0.0,
        description=(
            "IAE accumulated so far in this run (mole-fraction·min). Lets "
            "the Optimizer judge whether the current strategy is working."
        ),
    )


class OptimizerProposal(BaseModel):
    """Output of the Optimizer node — the next composition-target pair.

    The Optimizer reads the Observer report (and optionally previous
    Critic feedback in the same cycle) and emits a new
    ``(y_D_target, x_B_target)`` pair plus a free-text rationale.
    """

    model_config = ConfigDict(frozen=True)

    y_D_target: float = Field(description="Distillate composition target (mole fraction).")
    x_B_target: float = Field(description="Bottoms composition target (mole fraction).")
    rationale: str = Field(
        min_length=1,
        description=(
            "Short natural-language justification for the proposal. Used "
            "in the per-cycle decision log and for paper-grade audit "
            "trails of the agent's reasoning."
        ),
    )

    def in_bounds(self) -> bool:
        """Return ``True`` if both targets are within their physical bounds."""
        return (
            SETPOINT_BOUNDS["y_D_target"][0] <= self.y_D_target <= SETPOINT_BOUNDS["y_D_target"][1]
            and SETPOINT_BOUNDS["x_B_target"][0]
            <= self.x_B_target
            <= SETPOINT_BOUNDS["x_B_target"][1]
        )


class CriticVerdict(BaseModel):
    """Output of the Critic node — accept, revise, or escalate.

    The Critic reviews the Optimizer's proposal against simple
    plausibility checks (in-bounds, agent-internal sanity) and emits
    one of three verdicts. ``"revise"`` triggers another Optimizer
    round inside the same supervisor cycle, bounded by the hard
    recursion limit in the graph runner.
    """

    model_config = ConfigDict(frozen=True)

    decision: Literal["accept", "revise", "escalate"] = Field(
        description=(
            "accept: proposal is forwarded to the regulatory backend. "
            "revise: Optimizer is asked to refine, bounded by the graph's "
            "max-Critic-Optimizer-rounds limit. "
            "escalate: fall back to the previous accepted setpoint, mark "
            "the cycle as agent-escalated for downstream analysis."
        )
    )
    reason: str = Field(min_length=1, description="Short justification for the verdict.")


class AgentState(BaseModel):
    """LangGraph state that flows through Observer → Optimizer → Critic.

    Mutated in place across node visits; the graph runner inspects
    ``critic_verdict`` to decide whether to loop the Critic-Optimizer
    sub-graph or terminate the cycle. ``optimizer_rounds`` is
    incremented every time the Optimizer is re-entered and is the
    counter the hard recursion limit guards.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    observer_report: ObserverReport | None = None
    optimizer_proposal: OptimizerProposal | None = None
    critic_verdict: CriticVerdict | None = None
    optimizer_rounds: int = Field(
        default=0, ge=0, description="Number of Optimizer invocations in the current cycle."
    )
    decision: OptimizerProposal | None = Field(
        default=None,
        description=(
            "Final accepted proposal for this cycle, or ``None`` if the "
            "Critic escalated and the previous accepted proposal must be "
            "reused by the runner."
        ),
    )

    def is_terminal(self) -> bool:
        """Return ``True`` if the graph has produced a final decision."""
        if self.critic_verdict is None:
            return False
        return self.critic_verdict.decision in ("accept", "escalate")
