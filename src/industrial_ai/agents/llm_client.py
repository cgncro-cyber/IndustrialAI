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
    LLMResponseParseError,
    MockLLMClientMisuseError,
)
from industrial_ai.agents.state import SETPOINT_BOUNDS
from industrial_ai.agents.tools import SetpointProposalInput

__all__ = [
    "LLMClient",
    "LLMResponse",
    "LMStudioLLMClient",
    "MockLLMClient",
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
    """Abstract LLM client interface used by the agent graph."""

    name: str

    @abstractmethod
    def complete(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int = 256,
        temperature: float = 0.6,
        top_p: float = 0.95,
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
        max_tokens: int = 256,
        temperature: float = 0.6,
        top_p: float = 0.95,
    ) -> LLMResponse:
        del system_prompt, max_tokens, temperature, top_p

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
        max_tokens: int = 256,
        temperature: float = 0.6,
        top_p: float = 0.95,
    ) -> LLMResponse:
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
                max_tokens=max_tokens,
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
    """
    import json
    import re

    match = re.search(r"\{[^{}]*\}", text, flags=re.DOTALL)
    if match is None:
        raise LLMResponseParseError(f"could not find a JSON object in LLM response: {text!r}")
    try:
        payload = json.loads(match.group(0))
    except json.JSONDecodeError as exc:
        raise LLMResponseParseError(
            f"invalid JSON in LLM response: {match.group(0)!r} ({exc})"
        ) from exc
    return SetpointProposalInput(**payload)
