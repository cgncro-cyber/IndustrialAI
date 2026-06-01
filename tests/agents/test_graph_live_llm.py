"""Live-LLM integration tests for the AgentRunner / one-cycle graph.

These tests run the real ``MLXServerLLMClient`` against the Mac Studio
endpoint specified in ADR 005 amendment 2026-05-28
(``http://192.168.178.81:8080/v1`` by default). They are skipped unless
``--run-live-llm`` is passed (see ``tests/conftest.py``); the unit suite
must remain independent of the Mac Studio runtime.

ADR 010 (fail-fast) discipline: if ``--run-live-llm`` is passed but the
server is unreachable, the fixture calls ``pytest.fail`` rather than
auto-skipping — the user explicitly opted into the live path.

Endpoint override: set ``MLX_SERVER_BASE_URL`` to point at a different
host (e.g. localhost when running on the Mac Studio itself).
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np
import pytest

from industrial_ai.agents.graph import AgentRunner, run_one_cycle
from industrial_ai.agents.llm_client import MLXServerLLMClient
from industrial_ai.agents.regulatory_backend import build_regulatory_backend
from industrial_ai.twin.column_a import DEFAULT_PARAMETERS

_DEFAULT_BASE_URL = "http://192.168.178.81:8080/v1"


pytestmark = pytest.mark.live_llm


@pytest.fixture(scope="module")
def live_mlx_client() -> MLXServerLLMClient:
    """Build a live MLX client and confirm the endpoint answers a probe."""
    base_url = os.environ.get("MLX_SERVER_BASE_URL", _DEFAULT_BASE_URL)
    client = MLXServerLLMClient(base_url=base_url, request_timeout_s=120.0)
    # Pinned smoke prompt — identical structure to phase3_llm_smoke_check
    # Test B so a green probe here means the gated transport is healthy.
    sys_prompt = (
        "You are the supervisory layer of a binary distillation column. "
        "Choose a (y_D_target, x_B_target) pair that minimizes IAE under the "
        "observed plant state. Physical bounds: y_D_target in [0,1], "
        "x_B_target in [0,1], y_D_target must strictly exceed x_B_target. "
        "Reply with a SINGLE JSON object on its own line: "
        '{"y_D_target": <float>, "x_B_target": <float>, "rationale": "<short reason>"}. '
        "No surrounding markdown, no trailing prose, no JSON arrays."
    )
    user_prompt = (
        "Cycle 0 at t=0.0 min. Plant: y_D=0.99, x_B=0.01, LT=2.706, VB=3.206, "
        "F=1.000, zF=0.500, qF=1.000. Run IAE so far: 0.0000."
    )
    try:
        response = client.complete(
            system_prompt=sys_prompt,
            user_prompt=user_prompt,
            reasoning=False,
        )
    except Exception as exc:
        pytest.fail(
            f"MLX server probe against {base_url} failed: "
            f"{type(exc).__name__}: {exc}. ADR 010: no fallback, the live "
            "endpoint must be reachable when --run-live-llm is requested."
        )
    assert response.proposal is not None
    return client


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


def test_live_single_cycle_nominal_mpc_accepts_and_holds_spec(
    live_mlx_client: MLXServerLLMClient, nominal_X: np.ndarray
) -> None:
    p = DEFAULT_PARAMETERS
    backend = build_regulatory_backend("mpc")
    t0 = time.perf_counter()
    out = run_one_cycle(
        cycle_index=0,
        t_min=0.0,
        X=nominal_X,
        LT_kmol_per_min=p.nominal_reflux_L0_kmol_per_min,
        VB_kmol_per_min=p.nominal_boilup_V0_kmol_per_min,
        F_kmol_per_min=p.nominal_feed_F_kmol_per_min,
        zF=0.5,
        qF=1.0,
        recent_aggregate_iae=0.0,
        llm_client=live_mlx_client,
        regulatory_backend=backend,
    )
    wall = time.perf_counter() - t0
    # Live LLM modal default reasoning=False: ADR 005 amendment empirical
    # P95 = 6.1 s; allow generous margin for revise rounds + MPC solve.
    assert wall < 60.0, f"single cycle wall-clock {wall:.1f} s exceeds 60 s budget"
    assert out.state.critic_verdict.decision == "accept"
    assert out.regulatory_result.simulation.success
    NT = DEFAULT_PARAMETERS.NT
    assert out.regulatory_result.X_final[NT - 1] == pytest.approx(0.99, abs=1e-2)


def test_live_single_cycle_off_nominal_proposes_in_bounds(
    live_mlx_client: MLXServerLLMClient,
) -> None:
    """At F=0.8, zF=0.45 (low-F regime, §4.6) the LLM must propose a valid
    in-bounds setpoint pair; whether the Critic accepts or escalates is
    informational — both are ADR-010-compliant outcomes."""
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
        llm_client=live_mlx_client,
        regulatory_backend=backend,
    )
    decision = out.state.decision
    assert 0.0 <= decision.x_B_target < decision.y_D_target <= 1.0
    assert out.state.critic_verdict.decision in {"accept", "escalate", "revise"}


def test_live_two_cycle_runner_accumulates_iae(
    live_mlx_client: MLXServerLLMClient, nominal_X: np.ndarray
) -> None:
    runner = AgentRunner(
        llm_client=live_mlx_client,
        regulatory_backend=build_regulatory_backend("mpc"),
    )
    p = DEFAULT_PARAMETERS
    out1 = runner.step(
        cycle_index=0,
        t_min=0.0,
        X=nominal_X,
        LT_kmol_per_min=p.nominal_reflux_L0_kmol_per_min,
        VB_kmol_per_min=p.nominal_boilup_V0_kmol_per_min,
        F_kmol_per_min=p.nominal_feed_F_kmol_per_min,
        zF=0.5,
        qF=1.0,
    )
    iae_after_one = runner._aggregate_iae
    assert runner._completed_cycles == 1
    assert iae_after_one >= 0.0

    runner.step(
        cycle_index=1,
        t_min=5.0,
        X=out1.regulatory_result.X_final,
        LT_kmol_per_min=p.nominal_reflux_L0_kmol_per_min,
        VB_kmol_per_min=p.nominal_boilup_V0_kmol_per_min,
        F_kmol_per_min=p.nominal_feed_F_kmol_per_min,
        zF=0.5,
        qF=1.0,
    )
    assert runner._completed_cycles == 2
    assert runner._aggregate_iae >= iae_after_one
