"""Tests for the LLM client abstraction.

LMStudioLLMClient is not exercised against a live endpoint here —
the Mac Studio stack is not always reachable from CI. The class is
import-and-instantiate tested only; the live smoke check happens in
``tools/phase3_llm_smoke_check.py`` once Schritt 4 runs.
"""

from __future__ import annotations

import pytest

from industrial_ai.agents.llm_client import (
    LLMResponse,
    LMStudioLLMClient,
    MockLLMClient,
    _parse_setpoint_json,
)
from industrial_ai.agents.tools import SetpointProposalInput


def test_mock_nominal_policy_is_deterministic() -> None:
    mock = MockLLMClient(policy="nominal")
    r1 = mock.complete(system_prompt="sys", user_prompt="anything")
    r2 = mock.complete(system_prompt="sys", user_prompt="something else")
    assert r1.proposal.y_D_target == r2.proposal.y_D_target == 0.99
    assert r1.proposal.x_B_target == r2.proposal.x_B_target == 0.01


def test_mock_adaptive_extracts_off_nominal_y_D() -> None:
    mock = MockLLMClient(policy="adaptive")
    r_off = mock.complete(
        system_prompt="sys",
        user_prompt="Plant: y_D=0.720, x_B=0.000, ...",
    )
    assert r_off.proposal.y_D_target == pytest.approx(0.97)
    assert r_off.proposal.x_B_target == pytest.approx(0.02)
    # Adaptive at on-spec falls back to nominal targets.
    r_on = mock.complete(
        system_prompt="sys",
        user_prompt="Plant: y_D=0.990, x_B=0.010, ...",
    )
    assert r_on.proposal.y_D_target == pytest.approx(0.99)
    assert r_on.proposal.x_B_target == pytest.approx(0.01)


def test_mock_rejects_unknown_policy() -> None:
    with pytest.raises(ValueError):
        MockLLMClient(policy="random")


def test_mock_returns_LLMResponse_with_token_counts() -> None:
    mock = MockLLMClient(policy="nominal")
    response = mock.complete(system_prompt="sys", user_prompt="user prompt text")
    assert isinstance(response, LLMResponse)
    assert response.prompt_tokens is not None
    assert response.completion_tokens is not None
    assert response.raw_text  # non-empty


def test_lm_studio_client_lazy_construction() -> None:
    """Instantiating the LM Studio client must not require a running server."""
    client = LMStudioLLMClient(base_url="http://nowhere.invalid:9999/v1")
    assert client.name == "lm_studio"
    assert client._client is None  # not built until first complete() call


def test_parse_setpoint_json_finds_object_in_chatty_text() -> None:
    text = (
        "Thinking... the column is on-spec, so I propose:\n"
        '{"y_D_target": 0.99, "x_B_target": 0.01, "rationale": "hold"}\n'
        "Done."
    )
    parsed = _parse_setpoint_json(text)
    assert isinstance(parsed, SetpointProposalInput)
    assert parsed.y_D_target == pytest.approx(0.99)


def test_parse_setpoint_json_rejects_no_object() -> None:
    with pytest.raises(ValueError):
        _parse_setpoint_json("no JSON in here at all")
