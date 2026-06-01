"""LLM client abstraction with deterministic Mock + LM Studio backends.

The C2 / C3 agent never imports a specific LLM provider directly.
It calls :class:`LLMClient.complete` with a system + user prompt and
gets back a structured :class:`SetpointProposalInput`. The two
implementations here cover the Phase-3 workflow:

- :class:`MockLLMClient` — deterministic canned responses, no
  network. Lets the LangGraph orchestration be tested end-to-end
  without a live model; supports seed-controlled reproducibility for
  per-cycle decision logs.
- :class:`LMStudioLLMClient` — thin wrapper over ``langchain-openai``
  pointed at the LM Studio endpoint (``http://localhost:1234/v1`` by
  default per ADR 005). One-line swap to a remote provider via the
  ``base_url`` constructor argument.

Both implementations return :class:`SetpointProposalInput`. The
schema-validation is the contract that lets the rest of the graph
treat both clients identically.
"""

from __future__ import annotations

import os
import random
import sys
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

from industrial_ai.agents.errors import (
    LLMEndpointUnreachableError,
    LLMResponseFormatError,
    LLMResponseParseError,
    LLMServerError,
    MissingAPIKeyError,
    MissingBackendConfigError,
    MissingUsageError,
    MockLLMClientMisuseError,
)
from industrial_ai.agents.state import SETPOINT_BOUNDS
from industrial_ai.agents.tools import SetpointProposalInput

__all__ = [
    "LLMClient",
    "LLMResponse",
    "LMStudioLLMClient",
    "MLXServerLLMClient",
    "MockLLMClient",
    "OpenAIChatLLMClient",
    "build_llm_client",
]

_DEFAULT_NOMINAL_TARGETS = {"y_D_target": 0.99, "x_B_target": 0.01}


@dataclass(slots=True)
class LLMResponse:
    """One LLM call's full result: parsed proposal + raw assistant text + token metrics."""

    proposal: SetpointProposalInput
    raw_text: str
    prompt_tokens: int | None = None
    completion_tokens: int | None = None


class LLMClient(ABC):
    """Abstract LLM client interface used by the agent graph.

    The ``reasoning`` flag selects between Nemotron-Super v1.5's two
    inference modes (per ADR 005 amendment 2026-05-28):

    - ``reasoning=False`` (default for tool-call cycles): ``/no_think``
      marker injected into the system prompt, ``<think></think>``
      stubbed by the model; produces JSON-only output in ~10-20 s.
    - ``reasoning=True`` (used for Critic-revision rounds): chain-of-
      thought reasoning enabled, larger ``max_tokens`` budget so the
      JSON reaches the response after a long deliberation. Costs
      ~80-150 s per call.

    Implementations are free to interpret ``max_tokens=None`` as
    "use the reasoning-state-appropriate default". Mocks ignore the
    flag entirely.
    """

    name: str

    @abstractmethod
    def complete(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int | None = None,
        temperature: float = 0.6,
        top_p: float = 0.95,
        reasoning: bool = False,
    ) -> LLMResponse: ...


class MockLLMClient(LLMClient):
    """Deterministic mock that emits canned proposals.

    The mock implements two response policies:

    - ``"nominal"`` (default): always proposes ``y_D=0.99, x_B=0.01``
      — the nominal product spec from the canonical scenario set.
      Tests the orchestration and the graph's hard limits.
    - ``"adaptive"``: if the user prompt mentions ``y_D=0.XX`` with
      ``XX < 0.95``, the mock recognises an off-nominal state and
      proposes an interim target ``y_D = 0.97, x_B = 0.02`` (closer
      to reachable). Lets the graph be exercised on a Bucket-B-style
      target-sequencing path without an LLM in the loop.

    A seed is accepted for completeness but only influences the
    ``rationale`` string (small textual jitter); the numeric proposal
    is always deterministic for a given policy + prompt pair.
    """

    name: str = "mock"

    def __init__(
        self,
        *,
        policy: str = "nominal",
        seed: int = 0,
        allow_mock: bool = False,
    ) -> None:
        if not _is_sanctioned_mock_context(allow_mock):
            raise MockLLMClientMisuseError(
                "MockLLMClient is a test double (ADR 010 §3). "
                "Construction outside pytest requires allow_mock=True, which is "
                "permitted only in test code. In production / notebook / "
                "evaluation runs, configure a real LLMClient (e.g., "
                "LMStudioLLMClient) instead."
            )
        if policy not in ("nominal", "adaptive"):
            raise ValueError(f"unknown mock policy: {policy!r}")
        self.policy = policy
        self._rng = random.Random(seed)

    def complete(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int | None = None,
        temperature: float = 0.6,
        top_p: float = 0.95,
        reasoning: bool = False,
    ) -> LLMResponse:
        del system_prompt, max_tokens, temperature, top_p, reasoning

        if self.policy == "nominal":
            y_D, x_B = (
                _DEFAULT_NOMINAL_TARGETS["y_D_target"],
                _DEFAULT_NOMINAL_TARGETS["x_B_target"],
            )
            rationale = "Nominal product spec, hold at (0.99, 0.01)."
        else:  # adaptive
            y_D_observed = _extract_y_D_from_prompt(user_prompt)
            if y_D_observed is not None and y_D_observed < 0.95:
                y_D, x_B = 0.97, 0.02
                rationale = (
                    f"Observed y_D={y_D_observed:.3f} suggests off-nominal regime; "
                    "propose interim (0.97, 0.02) before pushing to spec."
                )
            else:
                y_D, x_B = (
                    _DEFAULT_NOMINAL_TARGETS["y_D_target"],
                    _DEFAULT_NOMINAL_TARGETS["x_B_target"],
                )
                rationale = "On-spec or unknown regime; hold at (0.99, 0.01)."

        # Clamp defensively to bounds — never violate the contract.
        y_D = min(max(y_D, SETPOINT_BOUNDS["y_D_target"][0]), SETPOINT_BOUNDS["y_D_target"][1])
        x_B = min(max(x_B, SETPOINT_BOUNDS["x_B_target"][0]), SETPOINT_BOUNDS["x_B_target"][1])

        # Reserved for future rationale-jitter via self._rng; the
        # numeric proposal must remain deterministic per the policy
        # contract above.
        _ = self._rng

        proposal = SetpointProposalInput(
            y_D_target=y_D,
            x_B_target=x_B,
            rationale=rationale,
        )
        raw = f'{{"y_D_target": {y_D}, "x_B_target": {x_B}, "rationale": "{rationale}"}}'
        return LLMResponse(
            proposal=proposal,
            raw_text=raw,
            prompt_tokens=len(user_prompt) // 4,
            completion_tokens=len(raw) // 4,
        )


def _is_sanctioned_mock_context(allow_mock: bool) -> bool:
    """Return ``True`` if constructing :class:`MockLLMClient` is permitted here.

    Sanctioned contexts (per ADR 010 §3):

    - pytest is currently active (``PYTEST_CURRENT_TEST`` env var set,
      or the ``pytest`` module is loaded).
    - the caller explicitly passed ``allow_mock=True``.

    Anything else is treated as a production / notebook / evaluation
    run and the mock is rejected at construction time.
    """
    if allow_mock:
        return True
    if "PYTEST_CURRENT_TEST" in os.environ:
        return True
    return sys.modules.get("pytest") is not None


def _extract_y_D_from_prompt(prompt: str) -> float | None:
    """Find ``y_D=0.XX`` in the prompt body, return the value or ``None``.

    Tolerant against trailing punctuation and whitespace; first match
    wins. Used by the adaptive mock policy.
    """
    import re

    m = re.search(r"y_D\s*=\s*([0-9]+\.[0-9]+)", prompt)
    if m is None:
        return None
    try:
        return float(m.group(1))
    except ValueError:  # pragma: no cover - regex match guarantees a parseable float
        return None


@dataclass
class _LMStudioConfig:
    base_url: str = "http://localhost:1234/v1"
    model: str = "nvidia/Llama-3.3-Nemotron-Super-49B-v1.5"
    api_key: str = "lm-studio"  # LM Studio ignores the value, library wants a string
    request_timeout_s: float = 60.0


class LMStudioLLMClient(LLMClient):
    """Wrapper over ``langchain-openai`` against the LM Studio endpoint.

    Construction is deferred to first ``complete`` call so importing
    this module does not require a running LM Studio. The model and
    endpoint default per ADR 005; pass overrides if running against
    a remote endpoint instead.
    """

    name: str = "lm_studio"

    def __init__(
        self,
        *,
        base_url: str | None = None,
        model: str | None = None,
        api_key: str | None = None,
        request_timeout_s: float | None = None,
    ) -> None:
        self._cfg = _LMStudioConfig(
            base_url=base_url or _LMStudioConfig.base_url,
            model=model or _LMStudioConfig.model,
            api_key=api_key or _LMStudioConfig.api_key,
            request_timeout_s=request_timeout_s or _LMStudioConfig.request_timeout_s,
        )
        self._client: Any | None = None

    def _ensure_client(self) -> Any:
        if (
            self._client is not None
        ):  # pragma: no cover - cached path exercised only on the second call against a live endpoint (Schritt-4 smoke check)
            return self._client
        from langchain_openai import ChatOpenAI
        from pydantic import SecretStr

        self._client = ChatOpenAI(
            base_url=self._cfg.base_url,
            api_key=SecretStr(self._cfg.api_key),
            model=self._cfg.model,
            timeout=self._cfg.request_timeout_s,
        )
        return self._client

    def complete(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int | None = None,
        temperature: float = 0.6,
        top_p: float = 0.95,
        reasoning: bool = False,
    ) -> LLMResponse:
        del reasoning  # LM Studio endpoint expects the model to manage modes
        client = self._ensure_client()
        messages = [
            ("system", system_prompt),
            ("user", user_prompt),
        ]
        # ADR 010 §2: single attempt, named exception on network failure.
        # No retry-until-default, no silent provider switch.
        try:
            reply = client.invoke(
                messages,
                max_tokens=max_tokens if max_tokens is not None else 256,
                temperature=temperature,
                top_p=top_p,
            )
        except Exception as exc:
            raise LLMEndpointUnreachableError(
                f"LM Studio endpoint {self._cfg.base_url!r} unreachable: "
                f"{type(exc).__name__}: {exc}"
            ) from exc
        # pragma: no cover — requires a live LM Studio endpoint response,
        # exercised by tools/phase3_llm_smoke_check.py once Schritt 4 runs.
        raw = reply.content if hasattr(reply, "content") else str(reply)  # pragma: no cover
        proposal = _parse_setpoint_json(raw)  # pragma: no cover
        return LLMResponse(  # pragma: no cover
            proposal=proposal,
            raw_text=raw,
        )


def _parse_setpoint_json(text: str) -> SetpointProposalInput:
    """Extract the first JSON object from ``text`` and parse to ``SetpointProposalInput``.

    The LM Studio prompt asks the model to wrap its answer in a JSON
    object with keys ``y_D_target``, ``x_B_target``, ``rationale``.
    This helper enforces the contract; schema validation rejects
    out-of-bounds values.

    ``strict=False`` is passed to :func:`json.loads` so literal tab /
    newline characters inside the ``rationale`` string value do not
    abort parsing. Nemotron-Super 49B v1.5 frequently formats
    rationales with markdown-style bullet indentation that embeds
    these control characters; strict JSON would reject them, but
    they are semantically harmless inside the rationale's free-text
    field. The numeric fields are untouched by this relaxation.
    """
    import json
    import re

    match = re.search(r"\{[^{}]*\}", text, flags=re.DOTALL)
    if match is None:
        raise LLMResponseParseError(f"could not find a JSON object in LLM response: {text!r}")
    try:
        payload = json.loads(match.group(0), strict=False)
    except json.JSONDecodeError as exc:
        raise LLMResponseParseError(
            f"invalid JSON in LLM response: {match.group(0)!r} ({exc})"
        ) from exc
    return SetpointProposalInput(**payload)


# ---------------------------------------------------------------------------
# Native mlx_lm.server backend (ADR 005 amendment, post-Schritt-4 diagnosis).
# ---------------------------------------------------------------------------


@dataclass
class _MLXServerConfig:
    """Configuration for the native ``mlx_lm.server`` transport.

    Used when the LM Studio bundled mlx-engine cannot load nemotron-nas
    (LM Studio bug #704) and we fall back to a freshly-installed
    ``mlx-lm`` server with explicit chat-template rendering on the
    client side. See ADR 005 amendment 2026-05-28 for the empirical
    chain that motivates this transport.
    """

    base_url: str = "http://192.168.178.81:8080/v1"
    model: str = "default_model"
    api_key: str = (
        "mlx-server"  # mlx_lm.server ignores it; openai client requires a non-empty string
    )
    request_timeout_s: float = 600.0
    #: Best-effort determinism seed forwarded to ``/v1/completions``.
    #: mlx_lm.server 0.31.3 reads it from the request body and seeds
    #: ``mx.random`` before generation (verified against
    #: ``mlx_lm/server.py`` body validation + ``mx.random.seed`` call).
    #: ``None`` lets the server pick a fresh random seed per request.
    seed: int | None = None


class MLXServerLLMClient(LLMClient):
    """Native ``mlx_lm.server`` transport with client-side chat-template rendering.

    The server hits ``/v1/completions`` (NOT ``/v1/chat/completions``)
    so we bypass the transformers 5.x
    ``apply_chat_template(tokenize=True)`` regression for
    nemotron-nas tokenizers (empirically isolated post-Schritt-4: the
    chat-completions tokenization path returns a batched
    ``Encoding`` list that the mlx server cannot consume, producing
    a degenerate ``a a a ...`` repetition loop).

    The chat template is rendered on the client side via ``jinja2``
    from the model's bundled ``chat_template.jinja`` file. The
    pinned template fixture lives at
    ``data/reference/nemotron_super_v1_5_chat_template.jinja`` with
    its SHA-256 recorded in the ADR 005 amendment for
    reproducibility — any model swap requires re-pinning the
    template.

    Network errors raise :class:`LLMEndpointUnreachableError`; JSON
    extraction failures raise :class:`LLMResponseParseError`. ADR
    010 contract preserved.
    """

    name: str = "mlx_server"

    def __init__(
        self,
        *,
        base_url: str | None = None,
        model: str | None = None,
        api_key: str | None = None,
        request_timeout_s: float | None = None,
        chat_template_path: str | None = None,
        seed: int | None = None,
    ) -> None:
        self._cfg = _MLXServerConfig(
            base_url=base_url or _MLXServerConfig.base_url,
            model=model or _MLXServerConfig.model,
            api_key=api_key or _MLXServerConfig.api_key,
            request_timeout_s=request_timeout_s or _MLXServerConfig.request_timeout_s,
            seed=seed,
        )
        if chat_template_path is None:
            from pathlib import Path

            chat_template_path = str(
                Path(__file__).resolve().parents[3]
                / "data"
                / "reference"
                / "nemotron_super_v1_5_chat_template.jinja"
            )
        self._template = _load_jinja_template(chat_template_path)
        self._client: Any | None = None

    def _ensure_client(self) -> Any:
        if (
            self._client is not None
        ):  # pragma: no cover - cached path exercised only on the second call against a live endpoint
            return self._client
        from openai import OpenAI

        self._client = OpenAI(
            base_url=self._cfg.base_url,
            api_key=self._cfg.api_key,
            timeout=self._cfg.request_timeout_s,
        )
        return self._client

    def complete(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int | None = None,
        temperature: float = 0.6,
        top_p: float = 0.95,
        reasoning: bool = False,
    ) -> LLMResponse:
        client = self._ensure_client()
        # Modal reasoning toggle per ADR 005 amendment 2026-05-28:
        #   reasoning=False  → /no_think marker, smaller budget,
        #                      typical tool-call cycle (10-20 s).
        #   reasoning=True   → no marker, large budget,
        #                      Critic-revision deliberation (~80-150 s).
        # The /no_think marker is detected and stripped by the
        # chat_template.jinja itself (see template's system_content
        # handling at the top of the file).
        if reasoning:
            effective_system_prompt = system_prompt
            effective_max_tokens = max_tokens if max_tokens is not None else 4096
        else:
            effective_system_prompt = "/no_think " + system_prompt
            effective_max_tokens = max_tokens if max_tokens is not None else 512
        # Client-side template rendering mirrors what
        # tokenizer.apply_chat_template(tokenize=False) would emit.
        rendered = self._template.render(
            messages=[
                {"role": "system", "content": effective_system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            tools=None,
            add_generation_prompt=True,
        )
        # ADR 010 §2: single attempt, named exception on network failure.
        # mlx_lm.server 0.31.3 accepts a per-request `seed` in the body
        # and threads it into ``mx.random.seed`` before generation;
        # ``None`` lets the server pick a fresh random seed per request.
        completion_kwargs: dict[str, Any] = {
            "model": self._cfg.model,
            "prompt": rendered,
            "max_tokens": effective_max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "stop": ["<|eot_id|>"],
        }
        if self._cfg.seed is not None:
            completion_kwargs["seed"] = self._cfg.seed
        try:
            reply = client.completions.create(**completion_kwargs)
        except Exception as exc:
            raise LLMEndpointUnreachableError(
                f"mlx_lm.server endpoint {self._cfg.base_url!r} unreachable: "
                f"{type(exc).__name__}: {exc}"
            ) from exc
        raw = reply.choices[0].text
        # ADR 010 §2: the /v1/completions contract is documented to
        # return a `usage` block; missing or partial usage is a
        # transport regression worth surfacing rather than silently
        # emitting zeros that would mask latency / output-length drift
        # in Phase-3 prompt iteration.
        usage = getattr(reply, "usage", None)
        if usage is None:
            raise MissingUsageError(
                f"mlx_lm.server response from {self._cfg.base_url!r} "
                "missing the documented `usage` block."
            )
        prompt_tokens = getattr(usage, "prompt_tokens", None)
        completion_tokens = getattr(usage, "completion_tokens", None)
        if prompt_tokens is None or completion_tokens is None:
            raise MissingUsageError(
                f"mlx_lm.server response from {self._cfg.base_url!r} "
                "`usage` block is missing required fields: "
                f"prompt_tokens={prompt_tokens!r}, "
                f"completion_tokens={completion_tokens!r}."
            )
        proposal = _parse_setpoint_json(raw)
        return LLMResponse(
            proposal=proposal,
            raw_text=raw,
            prompt_tokens=int(prompt_tokens),
            completion_tokens=int(completion_tokens),
        )


# ---------------------------------------------------------------------------
# OpenAI-Chat-API backend (ADR 011, NIM primary).
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class OpenAIChatLLMClient(LLMClient):
    """Standard OpenAI ``/v1/chat/completions`` transport, used against NIM.

    Per ADR 011, this is the Phase-3 primary inference path.
    Server-side chat-template rendering (no client-side jinja2
    render, unlike :class:`MLXServerLLMClient`) — the vLLM backend
    on NIM correctly handles Nemotron's chat template, so we just
    POST a ``messages`` array and receive parsed assistant content.

    The ``reasoning`` modal convention follows ADR 005 amendment:
    ``reasoning=False`` prefixes the system content with
    ``/no_think`` (Nemotron's chat template detects it and stubs
    the ``<think></think>`` block) and uses a ~512-token budget;
    ``reasoning=True`` omits the marker and raises the budget to
    4096.

    ADR 010 §2: single attempt, no retry. Named errors:

    - :class:`MissingAPIKeyError` at construction time when
      ``api_key`` is empty.
    - :class:`LLMServerError` on non-2xx HTTP responses
      (includes status code and body excerpt).
    - :class:`LLMResponseFormatError` when the response body is
      not parseable as JSON.
    - :class:`LLMEndpointUnreachableError` on connection / timeout
      failures (httpx-level transport errors).
    - :class:`MissingUsageError` if the response is missing the
      OpenAI-spec ``usage`` block or any required field of it.
    - :class:`LLMResponseParseError` if the assistant ``content``
      field does not contain a parseable setpoint JSON object.
    """

    base_url: str
    api_key: str
    model: str
    request_timeout_s: float = 180.0
    seed: int | None = None
    name: str = "openai_chat"

    def __post_init__(self) -> None:
        if not self.api_key:
            raise MissingAPIKeyError(
                f"OpenAIChatLLMClient against {self.base_url!r} constructed "
                "without an API key. ADR 010 §2: empty-string / None keys are "
                "fail-fast at construction, not at first network call."
            )

    def complete(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int | None = None,
        temperature: float = 0.6,
        top_p: float = 0.95,
        reasoning: bool = False,
    ) -> LLMResponse:
        # Modal reasoning convention per ADR 005 amendment 2026-05-28,
        # preserved across backends (Nemotron honors the marker
        # server-side because the chat template is the same).
        if reasoning:
            effective_system_content = system_prompt
            effective_max_tokens = max_tokens if max_tokens is not None else 4096
        else:
            effective_system_content = "/no_think " + system_prompt
            effective_max_tokens = max_tokens if max_tokens is not None else 512
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": effective_system_content},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": effective_max_tokens,
        }
        if self.seed is not None:
            payload["seed"] = self.seed
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        url = self.base_url.rstrip("/") + "/chat/completions"
        # ADR 010 §2: single attempt. Transport-level connection /
        # timeout failures map to LLMEndpointUnreachableError to
        # match the contract surface used by the rest of the agent.
        import httpx

        try:
            response = self._post(url, payload, headers)
        except httpx.RequestError as exc:
            raise LLMEndpointUnreachableError(
                f"OpenAI-Chat-API endpoint {self.base_url!r} unreachable: "
                f"{type(exc).__name__}: {exc}"
            ) from exc
        if response.status_code >= 400:
            body_excerpt = response.text[:500]
            raise LLMServerError(
                f"OpenAI-Chat-API endpoint {self.base_url!r} returned "
                f"HTTP {response.status_code}: {body_excerpt!r}"
            )
        try:
            response_json = response.json()
        except ValueError as exc:
            raise LLMResponseFormatError(
                f"OpenAI-Chat-API endpoint {self.base_url!r} returned a "
                f"non-JSON body: {response.text[:300]!r}"
            ) from exc
        try:
            content = response_json["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMResponseFormatError(
                f"OpenAI-Chat-API response from {self.base_url!r} missing "
                f"`choices[0].message.content`: {response_json!r}"
            ) from exc
        usage = response_json.get("usage")
        if usage is None:
            raise MissingUsageError(
                f"OpenAI-Chat-API response from {self.base_url!r} missing "
                "the documented `usage` block."
            )
        prompt_tokens = usage.get("prompt_tokens")
        completion_tokens = usage.get("completion_tokens")
        if prompt_tokens is None or completion_tokens is None:
            raise MissingUsageError(
                f"OpenAI-Chat-API response from {self.base_url!r} `usage` "
                "block missing required fields: "
                f"prompt_tokens={prompt_tokens!r}, "
                f"completion_tokens={completion_tokens!r}."
            )
        proposal = _parse_setpoint_json(content)
        return LLMResponse(
            proposal=proposal,
            raw_text=content,
            prompt_tokens=int(prompt_tokens),
            completion_tokens=int(completion_tokens),
        )

    def _post(self, url: str, payload: dict[str, Any], headers: dict[str, str]) -> Any:
        """Indirection layer so unit tests can monkeypatch the HTTP call."""
        import httpx

        with httpx.Client(timeout=self.request_timeout_s) as client:
            return client.post(url, json=payload, headers=headers)


# ---------------------------------------------------------------------------
# Backend factory (ADR 011).
# ---------------------------------------------------------------------------


def build_llm_client(backend: str, *, seed: int | None = None) -> LLMClient:
    """Build the appropriate :class:`LLMClient` per ADR 011.

    Loads ``.env`` from project root via ``python-dotenv`` (idempotent
    across calls — ``load_dotenv()`` is safe to invoke repeatedly).

    Required env vars
    -----------------
    ``backend="nim"``
        ``NVIDIA_API_KEY``, ``NVIDIA_BASE_URL``, ``NVIDIA_MODEL``.
    ``backend="mac-studio"``
        ``MAC_STUDIO_BASE_URL``, ``MAC_STUDIO_MODEL`` (both optional,
        defaulting to the values from
        :class:`_MLXServerConfig` so the ablation path works without
        ``.env`` edits).

    Raises
    ------
    MissingBackendConfigError
        If one or more required vars are missing for the selected
        backend. All missing vars are listed in a single message so
        the operator can fix the ``.env`` in one pass.
    ValueError
        If ``backend`` is not one of the known values.
    """
    from dotenv import load_dotenv

    load_dotenv()
    if backend == "nim":
        required = ("NVIDIA_API_KEY", "NVIDIA_BASE_URL", "NVIDIA_MODEL")
        missing = [name for name in required if not os.environ.get(name)]
        if missing:
            raise MissingBackendConfigError(
                f"backend='nim' requires env vars {required!r} but the "
                f"following are missing or empty: {missing!r}. Populate "
                "them in .env (project root) per ADR 011."
            )
        return OpenAIChatLLMClient(
            base_url=os.environ["NVIDIA_BASE_URL"],
            api_key=os.environ["NVIDIA_API_KEY"],
            model=os.environ["NVIDIA_MODEL"],
            seed=seed,
        )
    if backend == "mac-studio":
        # MAC_STUDIO_* default to the MLXServerLLMClient defaults so
        # the ablation path works without .env edits — but the env
        # var, if set, overrides.
        base_url = os.environ.get("MAC_STUDIO_BASE_URL") or _MLXServerConfig.base_url
        model = os.environ.get("MAC_STUDIO_MODEL") or _MLXServerConfig.model
        return MLXServerLLMClient(base_url=base_url, model=model, seed=seed)
    raise ValueError(
        f"unknown backend {backend!r}; expected one of 'nim', 'mac-studio' (per ADR 011)."
    )


def _load_jinja_template(path: str) -> Any:
    """Load the model's ``chat_template.jinja`` with HuggingFace-compatible Jinja semantics.

    Matches the ``trim_blocks``/``lstrip_blocks`` defaults used by
    HuggingFace's ``apply_chat_template`` so the client-side
    rendering produces byte-identical output to the bundled
    tokenizer's render path.
    """
    import jinja2

    with open(path, encoding="utf-8") as fh:
        template_text = fh.read()
    env = jinja2.Environment(
        extensions=["jinja2.ext.loopcontrols"],
        trim_blocks=True,
        lstrip_blocks=True,
        autoescape=False,
    )
    return env.from_string(template_text)
