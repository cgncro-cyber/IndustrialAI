"""Pluggable regulatory backend for the agentic supervisor (ADR 009).

The C2 / C3 agent emits ``(y_D_target, x_B_target)`` setpoints at the
supervisory cadence (5 min default, ADR 006). Those targets need to
be translated into ``(LT, VB)`` plant inputs by a regulatory layer
that runs at ~0.05-min ticks underneath the supervisor.

ADR 009 names two regulatory backends:

- **MPC** (primary, the C2 / C3 four-step ladder): the Linear MPC
  from :mod:`industrial_ai.control.c1_linear_mpc` takes the agent's
  targets, solves the QP each supervisory tick, and emits ``(LT,
  VB)`` that the regulatory clock holds between solves.
- **PID** (deployment-economics branch): the C0 multi-loop PID from
  :mod:`industrial_ai.control.c0_pid_only` tracks the agent's
  targets directly without an MPC layer in between.

Both backends share the :class:`RegulatoryBackend` protocol so the
agent graph (:mod:`industrial_ai.agents.graph`) does not branch on
backend identity. The Phase-5 evaluation pass switches backends via
a single config flag.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Protocol, runtime_checkable

import numpy as np
import numpy.typing as npt

from industrial_ai.agents.errors import RegulatoryBackendError
from industrial_ai.control.c0_pid_only import build_c0_pids
from industrial_ai.control.c1_linear_mpc import (
    C1MPCConfig,
    build_c1_mpc,
    simulate_lv_with_mpc,
)
from industrial_ai.twin.column_a import DEFAULT_PARAMETERS
from industrial_ai.twin.column_a.linearize import LinearizedLVModel, linearize_lv
from industrial_ai.twin.column_a.parameters import ColumnAParameters
from industrial_ai.twin.simulate import (
    ScenarioFn,
    ScenarioStep,
    SimulationResult,
    simulate_lv_closed_loop,
)

__all__ = [
    "MPCBackend",
    "PIDBackend",
    "RegulatoryBackend",
    "RegulatoryStepResult",
    "build_regulatory_backend",
]

StateVector = npt.NDArray[np.float64]


@dataclass(slots=True)
class RegulatoryStepResult:
    """Outcome of one supervisor cycle worth of regulatory tracking.

    Attributes
    ----------
    X_final : numpy.ndarray of shape (2 * NT,)
        Plant state at the end of the cycle.
    simulation : SimulationResult
        Full per-tick trajectory inside the cycle (for KPI accounting
        and figure regeneration).
    backend_name : str
        ``"mpc"`` or ``"pid"`` — recorded in the per-cycle manifest.
    """

    X_final: StateVector
    simulation: SimulationResult
    backend_name: str


@runtime_checkable
class RegulatoryBackend(Protocol):
    """Interface the agent uses to drive the plant for one supervisor cycle.

    Implementations track the agent's ``(y_D_target, x_B_target)``
    setpoints over a fixed wall-clock window (``cycle_duration_min``)
    starting at ``t_start_min`` from plant state ``X0``, with the
    feed disturbance ``(F, zF, qF)`` held constant during the window.

    The contract is the same for MPC and PID backends — the agent
    graph does not branch on backend identity.
    """

    name: str

    def step(
        self,
        *,
        X0: StateVector,
        t_start_min: float,
        cycle_duration_min: float,
        y_D_target: float,
        x_B_target: float,
        F: float,
        zF: float,
        qF: float,
    ) -> RegulatoryStepResult: ...


def _constant_setpoint_scenario(
    *,
    y_D_target: float,
    x_B_target: float,
    F: float,
    zF: float,
    qF: float,
) -> ScenarioFn:
    """Build a scenario closure that holds the agent's targets constant for one cycle."""
    step = ScenarioStep(
        y_D_setpoint=y_D_target,
        x_B_setpoint=x_B_target,
        F=F,
        zF=zF,
        qF=qF,
    )

    def scenario(_t: float) -> ScenarioStep:
        return step

    return scenario


@dataclass(slots=True)
class MPCBackend:
    """C1 Linear MPC adapter for the supervisor.

    Holds a *linearized model* and a pre-built ``do-mpc`` controller.
    The linearization is computed once per ``MPCBackend`` instance —
    the agent re-instantiates the backend when it decides to refresh
    the linearization point (the ``linearization_drift_g`` signal
    from :mod:`industrial_ai.twin.column_a.linearize`).
    """

    linearized: LinearizedLVModel
    mpc_config: C1MPCConfig
    name: str = "mpc"

    def step(
        self,
        *,
        X0: StateVector,
        t_start_min: float,
        cycle_duration_min: float,
        y_D_target: float,
        x_B_target: float,
        F: float,
        zF: float,
        qF: float,
    ) -> RegulatoryStepResult:
        del t_start_min  # MPC backend is time-translation-invariant within a cycle
        mpc, _ = build_c1_mpc(self.linearized, config=self.mpc_config)
        scenario = _constant_setpoint_scenario(
            y_D_target=y_D_target,
            x_B_target=x_B_target,
            F=F,
            zF=zF,
            qF=qF,
        )
        sim = simulate_lv_with_mpc(
            X0=X0,
            scenario=scenario,
            mpc=mpc,
            linearized=self.linearized,
            duration_min=cycle_duration_min,
            tick_dt_min=0.05,
            supervisor_period_min=self.mpc_config.sampling_time_min,
        )
        return RegulatoryStepResult(
            X_final=sim.X[-1] if sim.success else X0,
            simulation=sim,
            backend_name=self.name,
        )


@dataclass(slots=True)
class PIDBackend:
    """C0 PID adapter for the deployment-economics branch (ADR 009 Option B).

    Reuses the C0 ``TL_no_decoupler`` shootout-winner tuning without
    retuning across operating points — that is the *point* of the
    "MPC-free deployment" framing.
    """

    name: str = "pid"
    detune_factor: float = 1.0
    parameters: ColumnAParameters = DEFAULT_PARAMETERS

    def step(
        self,
        *,
        X0: StateVector,
        t_start_min: float,
        cycle_duration_min: float,
        y_D_target: float,
        x_B_target: float,
        F: float,
        zF: float,
        qF: float,
    ) -> RegulatoryStepResult:
        del t_start_min
        L0 = self.parameters.nominal_reflux_L0_kmol_per_min
        V0 = self.parameters.nominal_boilup_V0_kmol_per_min
        top, bottom = build_c0_pids(
            LT_initial=L0,
            VB_initial=V0,
            detune_factor=self.detune_factor,
        )
        scenario = _constant_setpoint_scenario(
            y_D_target=y_D_target,
            x_B_target=x_B_target,
            F=F,
            zF=zF,
            qF=qF,
        )
        sim = simulate_lv_closed_loop(
            X0=X0,
            scenario=scenario,
            duration_min=cycle_duration_min,
            tick_dt_min=0.05,
            pid_top=top,
            pid_bottom=bottom,
        )
        return RegulatoryStepResult(
            X_final=sim.X[-1] if sim.success else X0,
            simulation=sim,
            backend_name=self.name,
        )


def build_regulatory_backend(
    kind: Literal["mpc", "pid"],
    *,
    X_linearization: StateVector | None = None,
    F_linearization: float | None = None,
    zF_linearization: float | None = None,
    mpc_config: C1MPCConfig | None = None,
) -> RegulatoryBackend:
    """Instantiate either the MPC or PID regulatory backend.

    Parameters
    ----------
    kind : {"mpc", "pid"}
        Backend selector. ``"mpc"`` matches the primary ADR-009 four-
        step ladder; ``"pid"`` matches the deployment-economics
        branch.
    X_linearization, F_linearization, zF_linearization : optional
        Linearization point. Required for ``kind="mpc"``. Defaults to
        the Phase-2 nominal SS at F=1, zF=0.5 when not supplied.
    mpc_config : C1MPCConfig, optional
        MPC tuning. Defaults to :class:`C1MPCConfig`.

    Returns
    -------
    RegulatoryBackend
    """
    if kind == "pid":
        return PIDBackend()
    if kind != "mpc":
        raise RegulatoryBackendError(
            f"unknown regulatory backend kind: {kind!r} (expected 'mpc' or 'pid')"
        )

    p = DEFAULT_PARAMETERS
    if X_linearization is None:
        import json
        from pathlib import Path

        ss_path = (
            Path(__file__).resolve().parents[3]
            / "data"
            / "reference"
            / "skogestad_column_a_steady_state.json"
        )
        with ss_path.open() as fh:
            ss = json.load(fh)["steady_state"]
        X_linearization = np.array(ss["compositions"] + ss["holdups_kmol"], dtype=np.float64)
    if F_linearization is None:
        F_linearization = p.nominal_feed_F_kmol_per_min
    if zF_linearization is None:
        zF_linearization = 0.5
    lin = linearize_lv(
        X_ss=X_linearization,
        L_ss=p.nominal_reflux_L0_kmol_per_min,
        V_ss=p.nominal_boilup_V0_kmol_per_min,
        F_ss=F_linearization,
        zF_ss=zF_linearization,
        backend="casadi",
    )
    return MPCBackend(linearized=lin, mpc_config=mpc_config or C1MPCConfig())
