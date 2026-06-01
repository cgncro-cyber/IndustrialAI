"""Tests for the LLM client abstraction.

LMStudioLLMClient is not exercised against a live endpoint here —
the Mac Studio stack is not always reachable from CI. The class is
import-and-instantiate tested only; the live smoke check happens in
``tools/phase3_llm_smoke_check.py`` once Schritt 4 runs.
"""

from __future__ import annotations

from typing import ClassVar

import pytest

from industrial_ai.agents.errors import (
    LLMEndpointUnreachableError,
    LLMResponseParseError,
    MissingUsageError,
    MockLLMClientMisuseError,
)
from industrial_ai.agents.llm_client import (
    LLMResponse,
    LMStudioLLMClient,
    MLXServerLLMClient,
    MockLLMClient,
    ReasoningProtocol,
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


def test_parse_setpoint_json_accepts_control_chars_in_rationale() -> None:
    """Nemotron embeds literal tabs/newlines in markdown-bullet rationales.

    Surfaced empirically on the 2026-06-01 C2 smoke re-run on
    nominal_baseline (Cycle 1 response contained an embedded
    multi-line bulleted rationale). The numeric fields are
    unaffected by the relaxed parser.
    """
    text = (
        '{"y_D_target": 0.995, "x_B_target": 0.005, '
        '"rationale": "Targets balance:\n'
        "\t- **y_D_target**: slightly above current to drive purification.\n"
        '\t- **x_B_target**: below current to reduce bottoms light fraction."}'
    )
    parsed = _parse_setpoint_json(text)
    assert parsed.y_D_target == pytest.approx(0.995)
    assert parsed.x_B_target == pytest.approx(0.005)
    assert "y_D_target" in parsed.rationale


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


# ---------------------------------------------------------------------------
# MLXServerLLMClient usage-block surfacing (Item 2 of the 2026-06-01 pre-
# prompt-hardening pass). Tests inject a fake OpenAI client so the code
# path that extracts `usage` from /v1/completions is exercised without
# needing a live mlx_lm.server endpoint.
# ---------------------------------------------------------------------------


class _FakeUsageOK:
    prompt_tokens = 137
    completion_tokens = 42


class _FakeChoiceOK:
    text = '{"y_D_target": 0.99, "x_B_target": 0.01, "rationale": "ok"}'


class _FakeReplyOK:
    choices: ClassVar = [_FakeChoiceOK()]
    usage: ClassVar = _FakeUsageOK()


class _FakeCompletionsOK:
    def create(self, **_: object) -> _FakeReplyOK:
        return _FakeReplyOK()


class _FakeOpenAIOK:
    completions = _FakeCompletionsOK()


def _build_mlx_client_with_fake(fake_openai: object) -> MLXServerLLMClient:
    client = MLXServerLLMClient(base_url="http://fake/v1", request_timeout_s=10.0)
    client._client = fake_openai  # type: ignore[assignment]
    return client


def test_mlx_client_surfaces_usage_in_llm_response() -> None:
    """Per Item 2: prompt iteration needs per-call token counts."""
    client = _build_mlx_client_with_fake(_FakeOpenAIOK())
    resp = client.complete(system_prompt="sys", user_prompt="usr", reasoning=False)
    assert resp.prompt_tokens == 137
    assert resp.completion_tokens == 42
    assert resp.proposal.y_D_target == pytest.approx(0.99)


class _FakeReplyNoUsage:
    choices: ClassVar = [_FakeChoiceOK()]
    usage: ClassVar = None


class _FakeCompletionsNoUsage:
    def create(self, **_: object) -> _FakeReplyNoUsage:
        return _FakeReplyNoUsage()


class _FakeOpenAINoUsage:
    completions = _FakeCompletionsNoUsage()


def test_mlx_client_raises_when_usage_block_missing() -> None:
    """ADR 010 §2: missing `usage` is a transport regression, not a default-to-zero."""
    client = _build_mlx_client_with_fake(_FakeOpenAINoUsage())
    with pytest.raises(MissingUsageError) as exc_info:
        client.complete(system_prompt="sys", user_prompt="usr", reasoning=False)
    assert "fake" in str(exc_info.value)


class _FakeUsagePartial:
    prompt_tokens = 100
    completion_tokens = None  # the regression we want to catch


class _FakeReplyPartialUsage:
    choices: ClassVar = [_FakeChoiceOK()]
    usage: ClassVar = _FakeUsagePartial()


class _FakeCompletionsPartial:
    def create(self, **_: object) -> _FakeReplyPartialUsage:
        return _FakeReplyPartialUsage()


class _FakeOpenAIPartial:
    completions = _FakeCompletionsPartial()


def test_mlx_client_raises_when_usage_field_partial() -> None:
    """ADR 010 §2: a `usage` block with a None field is also a fail-fast."""
    client = _build_mlx_client_with_fake(_FakeOpenAIPartial())
    with pytest.raises(MissingUsageError) as exc_info:
        client.complete(system_prompt="sys", user_prompt="usr", reasoning=False)
    assert "completion_tokens" in str(exc_info.value)


class _FakeCompletionsCapturingKwargs:
    last_kwargs: ClassVar[dict[str, object]] = {}

    def create(self, **kwargs: object) -> _FakeReplyOK:
        _FakeCompletionsCapturingKwargs.last_kwargs = dict(kwargs)
        return _FakeReplyOK()


class _FakeOpenAICapturing:
    completions = _FakeCompletionsCapturingKwargs()


def test_mlx_client_threads_seed_into_completions_request() -> None:
    """Item 3: --seed is wired through to mlx_lm.server 0.31.3 /v1/completions."""
    client = MLXServerLLMClient(base_url="http://fake/v1", request_timeout_s=10.0, seed=42)
    client._client = _FakeOpenAICapturing()  # type: ignore[assignment]
    client.complete(system_prompt="sys", user_prompt="usr", reasoning=False)
    assert _FakeCompletionsCapturingKwargs.last_kwargs.get("seed") == 42


def test_mlx_client_omits_seed_when_unset() -> None:
    """``seed=None`` must NOT be sent — let the server pick its own."""
    client = MLXServerLLMClient(base_url="http://fake/v1", request_timeout_s=10.0)
    client._client = _FakeOpenAICapturing()  # type: ignore[assignment]
    _FakeCompletionsCapturingKwargs.last_kwargs = {}
    client.complete(system_prompt="sys", user_prompt="usr", reasoning=False)
    assert "seed" not in _FakeCompletionsCapturingKwargs.last_kwargs


# ---------------------------------------------------------------------------
# OpenAIChatLLMClient (ADR 011) — mocked httpx transport so the chat-
# completions code path is exercised without a live NIM endpoint.
# ---------------------------------------------------------------------------


from industrial_ai.agents.errors import (  # noqa: E402 — keep grouped with usage
    LLMResponseFormatError,
    LLMServerError,
    MissingAPIKeyError,
    MissingBackendConfigError,
)
from industrial_ai.agents.llm_client import (  # noqa: E402
    OpenAIChatLLMClient,
    build_llm_client,
)


class _StubResponseOK:
    status_code = 200
    text = ""

    def json(self) -> dict[str, object]:
        return {
            "choices": [
                {
                    "message": {
                        "content": '{"y_D_target": 0.99, "x_B_target": 0.01, "rationale": "ok"}'
                    }
                }
            ],
            "usage": {"prompt_tokens": 211, "completion_tokens": 47, "total_tokens": 258},
        }


def _build_openai_client_with_stub(
    stub_response: object,
    *,
    reasoning_protocol: ReasoningProtocol | None = None,
    model: str = "nvidia/llama-3.3-nemotron-super-49b-v1.5",
) -> OpenAIChatLLMClient:
    from industrial_ai.agents.llm_client import NemotronMarkerProtocol

    protocol: ReasoningProtocol = reasoning_protocol or NemotronMarkerProtocol()
    client = OpenAIChatLLMClient(
        base_url="https://fake.invalid/v1",
        api_key="nvapi-test",
        model=model,
        reasoning_protocol=protocol,
        temperature=protocol.default_temperature,
    )
    captured: dict[str, object] = {}

    def _stub_post(url: str, payload: dict[str, object], headers: dict[str, str]) -> object:
        captured["url"] = url
        captured["payload"] = payload
        captured["headers"] = headers
        return stub_response

    client._post = _stub_post  # type: ignore[assignment]
    client._last_captured = captured  # type: ignore[attr-defined]
    return client


def test_openai_client_requires_api_key() -> None:
    """ADR 010 §2: empty api_key is fail-fast at construction."""
    from industrial_ai.agents.llm_client import NemotronMarkerProtocol

    with pytest.raises(MissingAPIKeyError):
        OpenAIChatLLMClient(
            base_url="https://x/v1",
            api_key="",
            model="m",
            reasoning_protocol=NemotronMarkerProtocol(),
        )


def test_openai_client_happy_path_parses_setpoint_and_usage() -> None:
    client = _build_openai_client_with_stub(_StubResponseOK())
    resp = client.complete(system_prompt="sys", user_prompt="usr", reasoning=False)
    assert resp.proposal.y_D_target == pytest.approx(0.99)
    assert resp.proposal.x_B_target == pytest.approx(0.01)
    assert resp.prompt_tokens == 211
    assert resp.completion_tokens == 47


def test_openai_client_sends_bearer_auth_and_chat_completions_path() -> None:
    client = _build_openai_client_with_stub(_StubResponseOK())
    client.complete(system_prompt="sys", user_prompt="usr", reasoning=False)
    captured = client._last_captured  # type: ignore[attr-defined]
    assert captured["url"].endswith("/chat/completions")
    assert captured["headers"]["Authorization"].startswith("Bearer ")


def test_openai_client_injects_no_think_marker_on_reasoning_false() -> None:
    client = _build_openai_client_with_stub(_StubResponseOK())
    client.complete(system_prompt="SYS", user_prompt="usr", reasoning=False)
    captured = client._last_captured  # type: ignore[attr-defined]
    sys_msg = captured["payload"]["messages"][0]
    assert sys_msg["role"] == "system"
    assert sys_msg["content"].startswith("/no_think ")
    assert captured["payload"]["max_tokens"] == 512


def test_openai_client_omits_no_think_marker_on_reasoning_true() -> None:
    client = _build_openai_client_with_stub(_StubResponseOK())
    client.complete(system_prompt="SYS", user_prompt="usr", reasoning=True)
    captured = client._last_captured  # type: ignore[attr-defined]
    sys_msg = captured["payload"]["messages"][0]
    assert not sys_msg["content"].startswith("/no_think ")
    assert captured["payload"]["max_tokens"] == 4096


def test_openai_client_threads_seed_into_payload() -> None:
    from industrial_ai.agents.llm_client import NemotronMarkerProtocol

    client = OpenAIChatLLMClient(
        base_url="https://x/v1",
        api_key="k",
        model="m",
        reasoning_protocol=NemotronMarkerProtocol(),
        seed=42,
    )
    captured: dict[str, object] = {}
    client._post = lambda url, payload, headers: (  # type: ignore[assignment]
        captured.update({"payload": payload}) or _StubResponseOK()
    )
    client.complete(system_prompt="sys", user_prompt="usr", reasoning=False)
    assert captured["payload"]["seed"] == 42


def test_openai_client_omits_seed_when_unset() -> None:
    client = _build_openai_client_with_stub(_StubResponseOK())
    client.complete(system_prompt="sys", user_prompt="usr", reasoning=False)
    captured = client._last_captured  # type: ignore[attr-defined]
    assert "seed" not in captured["payload"]


class _StubResponseNon2xx:
    status_code = 503
    text = "Service Unavailable"

    def json(self) -> dict[str, object]:
        return {}


def test_openai_client_raises_llm_server_error_on_non_2xx() -> None:
    client = _build_openai_client_with_stub(_StubResponseNon2xx())
    with pytest.raises(LLMServerError) as exc_info:
        client.complete(system_prompt="sys", user_prompt="usr", reasoning=False)
    assert "503" in str(exc_info.value)


class _StubResponseNonJSON:
    status_code = 200
    text = "<html>captive portal</html>"

    def json(self) -> dict[str, object]:
        raise ValueError("not JSON")


def test_openai_client_raises_format_error_on_non_json_body() -> None:
    client = _build_openai_client_with_stub(_StubResponseNonJSON())
    with pytest.raises(LLMResponseFormatError):
        client.complete(system_prompt="sys", user_prompt="usr", reasoning=False)


class _StubResponseNoUsage:
    status_code = 200
    text = ""

    def json(self) -> dict[str, object]:
        return {
            "choices": [
                {
                    "message": {
                        "content": '{"y_D_target": 0.99, "x_B_target": 0.01, "rationale": "x"}'
                    }
                }
            ],
        }


def test_openai_client_raises_missing_usage_when_usage_block_absent() -> None:
    client = _build_openai_client_with_stub(_StubResponseNoUsage())
    with pytest.raises(MissingUsageError):
        client.complete(system_prompt="sys", user_prompt="usr", reasoning=False)


class _StubResponseMissingChoices:
    status_code = 200
    text = ""

    def json(self) -> dict[str, object]:
        return {"choices": [], "usage": {"prompt_tokens": 1, "completion_tokens": 1}}


def test_openai_client_raises_format_error_when_choices_missing() -> None:
    client = _build_openai_client_with_stub(_StubResponseMissingChoices())
    with pytest.raises(LLMResponseFormatError):
        client.complete(system_prompt="sys", user_prompt="usr", reasoning=False)


# ---------------------------------------------------------------------------
# build_llm_client factory (ADR 011)
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# ReasoningProtocol strategies (ADR 011/012)
# ---------------------------------------------------------------------------


def test_nemotron_marker_protocol_injects_no_think_only_when_reasoning_off() -> None:
    from industrial_ai.agents.llm_client import NemotronMarkerProtocol

    p = NemotronMarkerProtocol()
    assert p.apply_to_system_prompt("SYS", reasoning=False).startswith("/no_think ")
    assert p.apply_to_system_prompt("SYS", reasoning=True) == "SYS"
    assert p.apply_to_extra_body(reasoning=False) is None
    assert p.apply_to_extra_body(reasoning=True) is None
    assert p.extract_reasoning_content({"content": "x"}) is None
    assert p.extract_reasoning_content({"content": "x", "reasoning_content": "y"}) is None
    assert p.max_tokens_for(reasoning=False) == 512
    assert p.max_tokens_for(reasoning=True) == 4096
    assert p.default_temperature == pytest.approx(0.6)


def test_nemotron_extra_body_protocol_emits_chat_template_kwargs() -> None:
    from industrial_ai.agents.llm_client import NemotronExtraBodyProtocol

    p = NemotronExtraBodyProtocol(reasoning_budget=4096)
    # System prompt is API-modal, not marker-modal.
    assert p.apply_to_system_prompt("SYS", reasoning=False) == "SYS"
    assert p.apply_to_system_prompt("SYS", reasoning=True) == "SYS"
    # extra_body always present, reasoning_budget is 0 when off.
    off = p.apply_to_extra_body(reasoning=False)
    assert off == {
        "chat_template_kwargs": {"enable_thinking": False},
        "reasoning_budget": 0,
    }
    on = p.apply_to_extra_body(reasoning=True)
    assert on == {
        "chat_template_kwargs": {"enable_thinking": True},
        "reasoning_budget": 4096,
    }
    # Reasoning trace comes back as reasoning_content.
    assert p.extract_reasoning_content({"reasoning_content": "trace"}) == "trace"
    assert p.extract_reasoning_content({"content": "x"}) is None
    assert p.max_tokens_for(reasoning=False) == 8192
    assert p.max_tokens_for(reasoning=True) == 8192
    assert p.default_temperature == pytest.approx(1.0)


def test_nemotron_extra_body_protocol_respects_reasoning_budget_override() -> None:
    from industrial_ai.agents.llm_client import NemotronExtraBodyProtocol

    p = NemotronExtraBodyProtocol(reasoning_budget=16384)
    on = p.apply_to_extra_body(reasoning=True)
    assert on["reasoning_budget"] == 16384


def test_deepseek_extra_body_protocol_uses_thinking_and_reasoning_effort() -> None:
    from industrial_ai.agents.llm_client import DeepSeekExtraBodyProtocol

    p = DeepSeekExtraBodyProtocol(reasoning_effort="high")
    assert p.apply_to_system_prompt("SYS", reasoning=False) == "SYS"
    assert p.apply_to_system_prompt("SYS", reasoning=True) == "SYS"
    off = p.apply_to_extra_body(reasoning=False)
    assert off == {"chat_template_kwargs": {"thinking": False}}
    on = p.apply_to_extra_body(reasoning=True)
    assert on == {"chat_template_kwargs": {"thinking": True, "reasoning_effort": "high"}}
    # Trace may be on `reasoning` or `reasoning_content`; check both.
    assert p.extract_reasoning_content({"reasoning": "trace1"}) == "trace1"
    assert p.extract_reasoning_content({"reasoning_content": "trace2"}) == "trace2"
    # `reasoning` takes precedence when both present (DeepSeek snippet order).
    assert (
        p.extract_reasoning_content({"reasoning": "primary", "reasoning_content": "secondary"})
        == "primary"
    )
    assert p.extract_reasoning_content({}) is None
    assert p.max_tokens_for(reasoning=False) == 8192
    assert p.max_tokens_for(reasoning=True) == 8192
    assert p.default_temperature == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# OpenAIChatLLMClient with each protocol — mock-based assertions on the
# wire-shape of the request.
# ---------------------------------------------------------------------------


class _StubResponseWithReasoningContent:
    status_code = 200
    text = ""

    def json(self) -> dict[str, object]:
        return {
            "choices": [
                {
                    "message": {
                        "content": '{"y_D_target": 0.99, "x_B_target": 0.01, "rationale": "ok"}',
                        "reasoning_content": "step 1: hold. step 2: target = current.",
                    }
                }
            ],
            "usage": {"prompt_tokens": 211, "completion_tokens": 47, "total_tokens": 258},
        }


def test_openai_client_with_nemotron_extra_body_merges_chat_template_kwargs() -> None:
    from industrial_ai.agents.llm_client import NemotronExtraBodyProtocol

    client = _build_openai_client_with_stub(
        _StubResponseWithReasoningContent(),
        reasoning_protocol=NemotronExtraBodyProtocol(reasoning_budget=4096),
        model="nvidia/nemotron-3-super-120b-a12b",
    )
    resp = client.complete(system_prompt="SYS", user_prompt="usr", reasoning=True)
    captured = client._last_captured  # type: ignore[attr-defined]
    payload = captured["payload"]
    # System content not marker-modified.
    assert payload["messages"][0]["content"] == "SYS"
    # extra_body fields merged at top level.
    assert payload["chat_template_kwargs"] == {"enable_thinking": True}
    assert payload["reasoning_budget"] == 4096
    # max_tokens from protocol default 8192.
    assert payload["max_tokens"] == 8192
    # reasoning_content surfaced on the LLMResponse.
    assert resp.reasoning_content is not None
    assert "step 1" in resp.reasoning_content


def test_openai_client_with_deepseek_protocol_emits_thinking_off_when_reasoning_false() -> None:
    from industrial_ai.agents.llm_client import DeepSeekExtraBodyProtocol

    client = _build_openai_client_with_stub(
        _StubResponseOK(),
        reasoning_protocol=DeepSeekExtraBodyProtocol(),
        model="deepseek-ai/deepseek-v4-flash",
    )
    client.complete(system_prompt="SYS", user_prompt="usr", reasoning=False)
    captured = client._last_captured  # type: ignore[attr-defined]
    payload = captured["payload"]
    assert payload["chat_template_kwargs"] == {"thinking": False}
    assert "reasoning_effort" not in payload["chat_template_kwargs"]


def test_openai_client_reasoning_content_none_for_marker_protocol() -> None:
    """NemotronMarkerProtocol always returns None for reasoning_content."""
    client = _build_openai_client_with_stub(_StubResponseWithReasoningContent())
    resp = client.complete(system_prompt="sys", user_prompt="usr", reasoning=False)
    assert resp.reasoning_content is None


def test_openai_client_emits_per_protocol_temperature() -> None:
    from industrial_ai.agents.llm_client import NemotronExtraBodyProtocol

    client = _build_openai_client_with_stub(
        _StubResponseOK(),
        reasoning_protocol=NemotronExtraBodyProtocol(),
        model="nvidia/nemotron-3-super-120b-a12b",
    )
    client.complete(system_prompt="sys", user_prompt="usr", reasoning=True)
    captured = client._last_captured  # type: ignore[attr-defined]
    # The 120B-default temperature is 1.0, threaded through __init__ via
    # the helper. Sanity-check the wire value.
    assert captured["payload"]["temperature"] == pytest.approx(1.0)


def test_build_llm_client_rejects_unknown_backend() -> None:
    with pytest.raises(ValueError):
        build_llm_client(backend="rocm")


@pytest.fixture
def _no_op_dotenv(monkeypatch: pytest.MonkeyPatch) -> None:
    """Prevent build_llm_client's load_dotenv() from re-reading the project .env.

    Otherwise monkeypatch.delenv is undone by load_dotenv reading the real
    .env file back in.
    """
    import dotenv

    import industrial_ai.agents.llm_client as llm_client_module

    def _noop(*_: object, **__: object) -> bool:
        return False

    monkeypatch.setattr(dotenv, "load_dotenv", _noop)
    monkeypatch.setattr(llm_client_module, "load_dotenv", _noop, raising=False)


def test_build_llm_client_nim_returns_openai_chat_client(
    monkeypatch: pytest.MonkeyPatch, _no_op_dotenv: None
) -> None:
    monkeypatch.setenv("NVIDIA_API_KEY", "nvapi-test")
    monkeypatch.setenv("NVIDIA_BASE_URL", "https://nim.test/v1")
    monkeypatch.setenv("NVIDIA_MODEL", "nvidia/llama-3.3-nemotron-super-49b-v1.5")
    client = build_llm_client(backend="nim")
    assert isinstance(client, OpenAIChatLLMClient)
    assert client.base_url == "https://nim.test/v1"
    assert client.model == "nvidia/llama-3.3-nemotron-super-49b-v1.5"


def test_build_llm_client_nim_dispatches_protocol_by_model_prefix(
    monkeypatch: pytest.MonkeyPatch, _no_op_dotenv: None
) -> None:
    """build_llm_client picks the right ReasoningProtocol per ADR-011 registry."""
    from industrial_ai.agents.llm_client import (
        DeepSeekExtraBodyProtocol,
        NemotronExtraBodyProtocol,
        NemotronMarkerProtocol,
    )

    monkeypatch.setenv("NVIDIA_API_KEY", "k")
    monkeypatch.setenv("NVIDIA_BASE_URL", "https://x")

    monkeypatch.setenv("NVIDIA_MODEL", "nvidia/llama-3.3-nemotron-super-49b-v1.5")
    c1 = build_llm_client(backend="nim")
    assert isinstance(c1.reasoning_protocol, NemotronMarkerProtocol)
    assert c1.temperature == pytest.approx(0.6)

    monkeypatch.setenv("NVIDIA_MODEL", "nvidia/nemotron-3-super-120b-a12b")
    c2 = build_llm_client(backend="nim")
    assert isinstance(c2.reasoning_protocol, NemotronExtraBodyProtocol)
    assert c2.temperature == pytest.approx(1.0)

    monkeypatch.setenv("NVIDIA_MODEL", "deepseek-ai/deepseek-v4-flash")
    c3 = build_llm_client(backend="nim")
    assert isinstance(c3.reasoning_protocol, DeepSeekExtraBodyProtocol)
    assert c3.temperature == pytest.approx(1.0)


def test_build_llm_client_nim_unknown_model_raises_named_error(
    monkeypatch: pytest.MonkeyPatch, _no_op_dotenv: None
) -> None:
    """ADR-010 §2: unknown model identifier raises, no silent default."""
    from industrial_ai.agents.errors import UnknownReasoningProtocolError

    monkeypatch.setenv("NVIDIA_API_KEY", "k")
    monkeypatch.setenv("NVIDIA_BASE_URL", "https://x")
    monkeypatch.setenv("NVIDIA_MODEL", "unknown-vendor/mystery-model-7b")
    with pytest.raises(UnknownReasoningProtocolError) as exc_info:
        build_llm_client(backend="nim")
    assert "mystery-model-7b" in str(exc_info.value)


def test_build_llm_client_mac_studio_returns_mlx_client_with_defaults(
    monkeypatch: pytest.MonkeyPatch, _no_op_dotenv: None
) -> None:
    monkeypatch.delenv("MAC_STUDIO_BASE_URL", raising=False)
    monkeypatch.delenv("MAC_STUDIO_MODEL", raising=False)
    client = build_llm_client(backend="mac-studio")
    assert isinstance(client, MLXServerLLMClient)


def test_build_llm_client_nim_raises_with_all_missing_vars_listed(
    monkeypatch: pytest.MonkeyPatch, _no_op_dotenv: None
) -> None:
    monkeypatch.delenv("NVIDIA_API_KEY", raising=False)
    monkeypatch.delenv("NVIDIA_BASE_URL", raising=False)
    monkeypatch.delenv("NVIDIA_MODEL", raising=False)
    with pytest.raises(MissingBackendConfigError) as exc_info:
        build_llm_client(backend="nim")
    msg = str(exc_info.value)
    assert "NVIDIA_API_KEY" in msg
    assert "NVIDIA_BASE_URL" in msg
    assert "NVIDIA_MODEL" in msg


def test_build_llm_client_nim_lists_only_missing_vars(
    monkeypatch: pytest.MonkeyPatch, _no_op_dotenv: None
) -> None:
    """Only the missing vars are listed, present ones are not."""
    monkeypatch.setenv("NVIDIA_API_KEY", "k")
    monkeypatch.setenv("NVIDIA_BASE_URL", "https://x")
    monkeypatch.delenv("NVIDIA_MODEL", raising=False)
    with pytest.raises(MissingBackendConfigError) as exc_info:
        build_llm_client(backend="nim")
    msg = str(exc_info.value)
    assert "NVIDIA_MODEL" in msg
    # Be tolerant of phrasing — but at minimum the message
    # must call out the missing list distinctly.
    assert "missing" in msg.lower()
