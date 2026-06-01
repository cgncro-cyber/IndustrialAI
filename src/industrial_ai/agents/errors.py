"""Typed exception hierarchy for the agent skeleton (ADR 010).

ADR 010 forbids silent fallbacks: errors propagate as named typed
exceptions so an operator sees the root cause in the stack trace
and downstream KPI computation never receives contaminated data.

This module is the authoritative location for the exception names
referenced in ADR 010.
"""

from __future__ import annotations

__all__ = [
    "AgentError",
    "CriticLoopLimitError",
    "LLMEndpointUnreachableError",
    "LLMResponseFormatError",
    "LLMResponseParseError",
    "LLMServerError",
    "MissingAPIKeyError",
    "MissingBackendConfigError",
    "MissingConfirmationSpecError",
    "MissingUsageError",
    "MockLLMClientMisuseError",
    "RegulatoryBackendError",
    "UnknownReasoningProtocolError",
]


class AgentError(Exception):
    """Base class for all agent-pipeline errors.

    All named exceptions in this module derive from
    :class:`AgentError` so an operator can ``except AgentError`` at
    the run boundary to log + re-raise + exit, without masking the
    specific subtype.
    """


class LLMEndpointUnreachableError(AgentError):
    """The configured LLM endpoint refused the connection or timed out.

    Per ADR 010 §2, a single connection attempt is made and this
    exception is raised on failure. No retry-until-default loop, no
    silent provider switch. The exception message must include the
    endpoint URL so the operator can fix the configuration.
    """


class LLMResponseParseError(AgentError):
    """The LLM returned text that did not parse to the expected schema.

    Per ADR 010 §2, this aborts the run. The message must include
    the offending text so the operator can iterate on the prompt or
    the schema contract.
    """


class MissingUsageError(AgentError):
    """The LLM response is missing the documented ``usage`` block.

    The OpenAI-compatible ``/v1/completions`` and ``/v1/chat/completions``
    contracts require a ``usage`` block with ``prompt_tokens`` and
    ``completion_tokens``. Per ADR 010 §2, a missing or partial
    ``usage`` is a transport regression worth surfacing rather than
    silently emitting zero counts — Phase-3 prompt iteration relies
    on these numbers to debug latency / output-length drift.

    Surfaced by both :class:`MLXServerLLMClient` (against
    ``/v1/completions``) and :class:`OpenAIChatLLMClient` (against
    ``/v1/chat/completions`` on NIM and similar hosted backends per
    ADR 011).
    """


class LLMServerError(AgentError):
    """The LLM endpoint returned a non-2xx HTTP response.

    Per ADR 010 §2, a single attempt is made; on any non-2xx the
    client raises this error including the status code and a short
    response-body excerpt so the operator can diagnose the upstream
    issue (rate-limit / auth / model-not-available / 5xx). No retries,
    no silent fall-through to a different backend.
    """


class LLMResponseFormatError(AgentError):
    """The LLM endpoint returned a body that did not parse as JSON.

    Distinct from :class:`LLMResponseParseError` (which is about
    failing to find a JSON setpoint object inside the assistant
    ``content`` field): this fires when the *outer* HTTP response
    body itself is not parseable. Typically signals upstream proxy
    misbehavior or a captive-portal interception rather than a
    model-side issue.
    """


class MissingAPIKeyError(AgentError):
    """A hosted LLM backend was constructed without an API key.

    Per ADR 010 §2, an empty-string or ``None`` API key passed to
    e.g. :class:`OpenAIChatLLMClient` raises this error at
    construction time rather than letting the first network call
    fail with a generic 401 — the operator sees the configuration
    bug at startup.
    """


class UnknownReasoningProtocolError(AgentError):
    """No :class:`ReasoningProtocol` is registered for the requested model identifier.

    Raised by :func:`build_llm_client` when the model prefix does not
    match any entry in ``_PROTOCOL_REGISTRY``. Per ADR 010 §2, an
    unknown identifier is a configuration error rather than a silent
    default to a marker-style protocol — different reasoning
    families have different on/off conventions and a silent default
    would produce wrong-shaped requests against a wrong-shaped API.
    """


class MissingConfirmationSpecError(AgentError):
    """``run_doe_confirmation.py`` was invoked but no ``confirmation_spec.json`` exists.

    Per ADR 010 §2, the missing artifact indicates an upstream
    analysis failure rather than a recovery scenario — the DoE
    analysis step is responsible for producing the spec. The
    confirmation driver fails fast so the operator can fix the
    upstream issue rather than re-run with a wrong / stale spec.
    """


class MissingBackendConfigError(AgentError):
    """One or more required env vars for the selected LLM backend are missing.

    Raised by :func:`build_llm_client` when, for example,
    ``backend="nim"`` is requested but ``NVIDIA_API_KEY`` is unset.
    The error message lists **all** missing env vars in one shot so
    the operator can fix the ``.env`` in a single pass rather than
    discover them one error at a time.
    """


class CriticLoopLimitError(AgentError):
    """The agent graph exhausted ``max_critic_optimizer_rounds`` without an accept verdict.

    Raised only when no previous accepted proposal exists yet (e.g.,
    the very first supervisor cycle). When a previous accepted
    proposal exists the graph's documented ``escalate`` verdict path
    applies instead (per ADR 010 §5 — designed safe-state transition,
    logged and counted, not a silent fallback).
    """


class RegulatoryBackendError(AgentError):
    """The regulatory backend (MPC or PID) could not be constructed or stepped.

    Per ADR 010 §2, this aborts the run. The message must name which
    backend (``"mpc"`` / ``"pid"``) and which operation
    (``"construct"`` / ``"step"``) failed.
    """


class MockLLMClientMisuseError(AgentError):
    """Constructing ``MockLLMClient`` outside a sanctioned test context.

    Per ADR 010 §3, the mock is exclusively a test double. The only
    sanctioned constructions are inside a pytest run or with an
    explicit ``allow_mock=True`` flag that must be set in test code
    only.
    """
