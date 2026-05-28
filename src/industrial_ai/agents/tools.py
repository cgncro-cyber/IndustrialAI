"""Tool surface that the LLM-driven agent uses to query the twin.

Pydantic schemas + thin functional adapters. The LLM never sees the
plant state directly — it sees structured tool results. Keeping the
tool surface narrow makes the agent's reasoning auditable and
constrains the prompt-injection surface to a known shape.

The four tools defined here are the ones named in the Phase-3 spec:

- :func:`read_twin_state` — current ``(y_D, x_B, LT, VB, F, zF, qF)``.
- :func:`read_recent_disturbance` — what changed in the last
  ``window_min`` minutes.
- :func:`propose_setpoint` — the agent's commit point. The schema
  enforces the SETPOINT_BOUNDS contract from
  :mod:`industrial_ai.agents.state`.
- :func:`query_kpi` — IAE accumulated so far in the current run.

Each tool returns a frozen pydantic model; the LLM client serializes
those to JSON before they enter the prompt context.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
from pydantic import BaseModel, ConfigDict, Field, model_validator

from industrial_ai.agents.state import SETPOINT_BOUNDS
from industrial_ai.twin.column_a import DEFAULT_PARAMETERS

__all__ = [
    "DisturbanceWindow",
    "KPISnapshot",
    "SetpointProposalInput",
    "TwinStateSnapshot",
    "propose_setpoint",
    "query_kpi",
    "read_recent_disturbance",
    "read_twin_state",
]

StateVector = npt.NDArray[np.float64]


class TwinStateSnapshot(BaseModel):
    """Compact view of the plant state at one supervisor cycle."""

    model_config = ConfigDict(frozen=True)

    cycle_index: int = Field(ge=0)
    t_min: float
    y_D: float
    x_B: float
    LT_kmol_per_min: float
    VB_kmol_per_min: float
    F_kmol_per_min: float
    zF: float
    qF: float


class DisturbanceWindow(BaseModel):
    """Summary of feed-disturbance changes over a recent window."""

    model_config = ConfigDict(frozen=True)

    window_min: float = Field(gt=0.0)
    F_start: float
    F_end: float
    F_delta_relative: float = Field(
        description="``(F_end - F_start) / F_start`` — relative change in feed flow."
    )
    zF_start: float
    zF_end: float
    zF_delta_absolute: float = Field(
        description="``zF_end - zF_start`` — absolute change in feed composition."
    )


class SetpointProposalInput(BaseModel):
    """Input schema for the :func:`propose_setpoint` tool.

    The agent emits an instance of this; the schema enforces the
    same bounds the runner uses to gate the regulatory backend.
    """

    model_config = ConfigDict(frozen=True)

    y_D_target: float = Field(
        ge=SETPOINT_BOUNDS["y_D_target"][0],
        le=SETPOINT_BOUNDS["y_D_target"][1],
        description="Distillate composition target (mole fraction).",
    )
    x_B_target: float = Field(
        ge=SETPOINT_BOUNDS["x_B_target"][0],
        le=SETPOINT_BOUNDS["x_B_target"][1],
        description="Bottoms composition target (mole fraction).",
    )
    rationale: str = Field(min_length=1)

    @model_validator(mode="after")
    def _check_ordering(self) -> SetpointProposalInput:
        # Top must be at least as light-rich as bottom — otherwise the
        # column is being asked to invert.
        if self.y_D_target <= self.x_B_target:
            raise ValueError(
                f"y_D_target ({self.y_D_target}) must exceed x_B_target "
                f"({self.x_B_target}); column cannot run inverted."
            )
        return self


class KPISnapshot(BaseModel):
    """Run-level KPI accumulator visible to the agent."""

    model_config = ConfigDict(frozen=True)

    cycle_index: int = Field(ge=0)
    aggregate_iae_so_far: float = Field(ge=0.0)
    completed_cycles: int = Field(ge=0)


def read_twin_state(
    *,
    cycle_index: int,
    t_min: float,
    X: StateVector,
    LT_kmol_per_min: float,
    VB_kmol_per_min: float,
    F_kmol_per_min: float,
    zF: float,
    qF: float,
) -> TwinStateSnapshot:
    """Build a :class:`TwinStateSnapshot` from raw twin signals.

    Composition extraction is the same convention used elsewhere in
    the codebase: ``y_D = X[NT - 1]``, ``x_B = X[0]``.
    """
    NT = DEFAULT_PARAMETERS.NT
    return TwinStateSnapshot(
        cycle_index=cycle_index,
        t_min=t_min,
        y_D=float(X[NT - 1]),
        x_B=float(X[0]),
        LT_kmol_per_min=LT_kmol_per_min,
        VB_kmol_per_min=VB_kmol_per_min,
        F_kmol_per_min=F_kmol_per_min,
        zF=zF,
        qF=qF,
    )


def read_recent_disturbance(
    *,
    window_min: float,
    F_history: list[float],
    zF_history: list[float],
) -> DisturbanceWindow:
    """Compare the start and end of a recent disturbance window.

    The histories are expected to be supervisor-cycle samples (one
    entry per cycle), most-recent last. A window shorter than two
    samples reports zero delta.
    """
    if window_min <= 0.0:
        raise ValueError(f"window_min must be > 0, got {window_min}")
    if not F_history or not zF_history:
        raise ValueError("F_history and zF_history must each contain at least one sample")
    F_start = F_history[0]
    F_end = F_history[-1]
    zF_start = zF_history[0]
    zF_end = zF_history[-1]
    F_delta_relative = (F_end - F_start) / F_start if F_start != 0.0 else 0.0
    return DisturbanceWindow(
        window_min=window_min,
        F_start=F_start,
        F_end=F_end,
        F_delta_relative=F_delta_relative,
        zF_start=zF_start,
        zF_end=zF_end,
        zF_delta_absolute=zF_end - zF_start,
    )


def propose_setpoint(proposal: SetpointProposalInput) -> SetpointProposalInput:
    """Pass-through validator that records the agent's commit point.

    The pydantic schema does the work; this function exists so the
    LLM tool-calling layer has a typed callable surface to dispatch
    against.
    """
    return proposal


def query_kpi(
    *,
    cycle_index: int,
    aggregate_iae_so_far: float,
    completed_cycles: int,
) -> KPISnapshot:
    """Return the current run-level KPI snapshot."""
    return KPISnapshot(
        cycle_index=cycle_index,
        aggregate_iae_so_far=aggregate_iae_so_far,
        completed_cycles=completed_cycles,
    )
