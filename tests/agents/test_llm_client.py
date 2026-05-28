"""Tests for the LLM client abstraction.

LMStudioLLMClient is not exercised against a live endpoint here —
the Mac Studio stack is not always reachable from CI. The class is
import-and-instantiate tested only; the live smoke check happens in
``tools/phase3_llm_smoke_check.py`` once Schritt 4 runs.
"""

from __future__ import annotations

import pytest

from industrial_ai.agents.errors import (
    LLMEndpointUnreachableError,
    LLMResponseParseError,
    MockLLMClientMisuseError,
)
from industrial_ai.agents.llm_client import (
    LLMResponse,
    LMStudioLLMClient,
    MLXServerLLMClient,
    MockLLMClient,
    _load_jinja_template,
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
    with pytest.raises(LLMResponseParseError):
        _parse_setpoint_json("no JSON in here at all")


def test_parse_setpoint_json_rejects_malformed_json() -> None:
    with pytest.raises(LLMResponseParseError):
        _parse_setpoint_json("{this is not valid JSON}")


def test_mock_construction_in_pytest_is_permitted() -> None:
    """Inside pytest, PYTEST_CURRENT_TEST is set so the mock guard passes."""
    # Default allow_mock=False: sanctioned because pytest is active.
    mock = MockLLMClient(policy="nominal")
    assert mock.name == "mock"


def test_mock_construction_outside_pytest_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    """ADR 010 §3: outside pytest, allow_mock=False rejects construction."""
    import sys

    monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)
    # Hide the pytest module too so the second guard arm fails as well.
    monkeypatch.setitem(sys.modules, "pytest", None)
    with pytest.raises(MockLLMClientMisuseError):
        MockLLMClient(policy="nominal")


def test_mock_construction_outside_pytest_with_allow_mock_passes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """ADR 010 §3: explicit allow_mock=True is permitted even outside pytest."""
    import sys

    monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)
    monkeypatch.setitem(sys.modules, "pytest", None)
    mock = MockLLMClient(policy="nominal", allow_mock=True)
    assert mock.name == "mock"


def test_mock_guard_pytest_module_arm(monkeypatch: pytest.MonkeyPatch) -> None:
    """Guard's second arm: env var unset but pytest module loaded → permitted."""
    monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)
    mock = MockLLMClient(policy="nominal")
    assert mock.name == "mock"


def test_extract_y_D_returns_none_on_unmatched_prompt() -> None:
    """The adaptive policy falls through to nominal when the prompt has no y_D."""
    from industrial_ai.agents.llm_client import _extract_y_D_from_prompt

    assert _extract_y_D_from_prompt("no signal in this prompt") is None


def test_mlx_server_client_chat_template_renders_to_expected_shape() -> None:
    """Client-side jinja2 render must produce the same Llama-3 shape as the model tokenizer.

    Byte-identical comparison against the format empirically confirmed
    during Schritt-4 diagnosis (Test 2 vs server-rendered prompt).
    """
    template = _load_jinja_template("data/reference/nemotron_super_v1_5_chat_template.jinja")
    rendered = template.render(
        messages=[
            {"role": "system", "content": "You are a process control engineer. Answer briefly."},
            {"role": "user", "content": "What does RGA measure?"},
        ],
        tools=None,
        add_generation_prompt=True,
    )
    expected = (
        "<|begin_of_text|>"
        "<|start_header_id|>system<|end_header_id|>\n\n"
        "You are a process control engineer. Answer briefly.\n\n"
        "<|eot_id|>"
        "<|start_header_id|>user<|end_header_id|>\n\n"
        "What does RGA measure?"
        "<|eot_id|>"
        "<|start_header_id|>assistant<|end_header_id|>\n\n"
    )
    assert rendered == expected


def test_mlx_server_no_think_marker_injects_empty_think_stub() -> None:
    """``/no_think`` in the system content must trigger the template's empty think stub.

    The chat_template.jinja appends ``<think>\\n\\n</think>\\n\\n``
    after the assistant header when ``enable_thinking`` is false.
    That stub is what gates the model into skip-reasoning mode and is
    the mechanism behind the modal reasoning toggle in the
    MLXServerLLMClient (ADR 005 amendment 2026-05-28).
    """
    template = _load_jinja_template("data/reference/nemotron_super_v1_5_chat_template.jinja")
    rendered_no_think = template.render(
        messages=[
            {"role": "system", "content": "/no_think You are a controller. Be brief."},
            {"role": "user", "content": "Status?"},
        ],
        tools=None,
        add_generation_prompt=True,
    )
    assert rendered_no_think.endswith(
        "<|start_header_id|>assistant<|end_header_id|>\n\n<think>\n\n</think>\n\n"
    )
    # And without the marker, no stub.
    rendered_with_think = template.render(
        messages=[
            {"role": "system", "content": "You are a controller. Be brief."},
            {"role": "user", "content": "Status?"},
        ],
        tools=None,
        add_generation_prompt=True,
    )
    assert "<think>" not in rendered_with_think


def test_mlx_server_client_lazy_construction() -> None:
    """Instantiating the MLX server client must not require a running server."""
    client = MLXServerLLMClient(base_url="http://nowhere.invalid:9999/v1")
    assert client.name == "mlx_server"
    assert client._client is None  # lazy until first call


def test_mlx_server_client_against_dead_endpoint_raises_named_error() -> None:
    """ADR 010 §2: a single attempt, then LLMEndpointUnreachableError with the URL."""
    client = MLXServerLLMClient(
        base_url="http://192.0.2.1:65535/v1",
        request_timeout_s=2.0,
    )
    with pytest.raises(LLMEndpointUnreachableError) as exc_info:
        client.complete(system_prompt="sys", user_prompt="user")
    assert "192.0.2.1:65535" in str(exc_info.value)


def test_mlx_server_client_unknown_chat_template_path_raises() -> None:
    """Missing template fixture must fail loudly at construction time."""
    with pytest.raises(FileNotFoundError):
        MLXServerLLMClient(
            base_url="http://localhost:9999/v1",
            chat_template_path="/nonexistent/path/chat_template.jinja",
        )


def test_lm_studio_client_against_dead_endpoint_raises_named_error() -> None:
    """ADR 010 §2: a single connection attempt, then a named exception.

    Uses a port that has no listener (RFC-5737 documentation address is
    routed-but-no-service in most environments). The client must raise
    LLMEndpointUnreachableError, not hang and not silently degrade.
    """
    client = LMStudioLLMClient(
        base_url="http://192.0.2.1:65535/v1",
        request_timeout_s=2.0,
    )
    with pytest.raises(LLMEndpointUnreachableError) as exc_info:
        client.complete(system_prompt="sys", user_prompt="user")
    assert "192.0.2.1:65535" in str(exc_info.value)
