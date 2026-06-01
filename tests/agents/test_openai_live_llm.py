"""Live-LLM integration test for the OpenAIChatLLMClient against NIM.

Runs the real ``OpenAIChatLLMClient`` against the NVIDIA NIM endpoint
configured in ``.env`` per ADR 011. Skipped unless ``--run-live-llm``
is passed (see ``tests/conftest.py``); the unit suite must remain
independent of the NIM availability and from spending NIM quota.

ADR 010 (fail-fast) discipline: if ``--run-live-llm`` is passed but
``NVIDIA_API_KEY`` / ``NVIDIA_BASE_URL`` / ``NVIDIA_MODEL`` are
missing, the fixture calls ``pytest.fail`` rather than auto-skipping
— the operator explicitly opted into the live path.
"""

from __future__ import annotations

import pytest

from industrial_ai.agents.llm_client import OpenAIChatLLMClient, build_llm_client

pytestmark = pytest.mark.live_llm


@pytest.fixture(scope="module")
def live_nim_client() -> OpenAIChatLLMClient:
    """Build the NIM-backed client per ADR 011 and confirm it answers a probe."""
    try:
        client = build_llm_client(backend="nim")
    except Exception as exc:
        pytest.fail(
            f"build_llm_client(backend='nim') failed: "
            f"{type(exc).__name__}: {exc}. Populate .env per ADR 011."
        )
    assert isinstance(client, OpenAIChatLLMClient)
    return client


def test_live_nim_returns_setpoint_with_usage(live_nim_client: OpenAIChatLLMClient) -> None:
    """One short call against the live NIM endpoint must return a parseable
    setpoint proposal with a real ``usage`` block (ADR 010 §2).
    """
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
    resp = live_nim_client.complete(
        system_prompt=sys_prompt,
        user_prompt=user_prompt,
        reasoning=False,
    )
    # Bounds + ordering invariant from the system prompt.
    assert 0.0 <= resp.proposal.x_B_target < resp.proposal.y_D_target <= 1.0
    assert resp.proposal.rationale  # non-empty
    # ADR 010 §2: usage is required, no zero-defaulting.
    assert resp.prompt_tokens is not None and resp.prompt_tokens > 0
    assert resp.completion_tokens is not None and resp.completion_tokens > 0
