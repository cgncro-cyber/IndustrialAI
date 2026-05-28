"""Phase 3 — Schritt 4 LLM smoke check against the native ``mlx_lm.server``.

Empirically motivated by the post-Schritt-4 diagnosis chain:

- LM Studio bundled mlx-engine 1.8.5 (stable + beta) does not load
  the nemotron-nas architecture (``PreTrainedConfig`` rope-standardization
  regression in transformers 5.x; bug #704).
- A clean ``mlx-lm`` 0.31.3 install in an isolated venv hits the same
  ``transformers`` regression on the chat-completions path *and* a
  separate ``apply_chat_template(tokenize=True)`` regression that
  returns a batched ``Encoding`` list which the server cannot consume,
  producing a degenerate ``a a a ...`` repetition loop on
  ``/v1/chat/completions``.
- ``/v1/completions`` with a client-side jinja2-rendered prompt
  (template pinned at
  ``data/reference/nemotron_super_v1_5_chat_template.jinja``)
  produces coherent Nemotron output including the expected
  ``<think>``/``</think>`` reasoning block.

This smoke check exercises that final path. It validates:

- A prose sanity request (Test A) — quick "Hello world" of the
  reasoning path.
- N=10 tool-call requests (Test B) against a propose-setpoint
  prompt, parsed through the same ``_parse_setpoint_json`` that the
  agent graph uses. The reliability rate is the headline metric the
  ADR 005 amendment is conditioned on.

Outputs ``data/reference/phase3_llm_smoke.json`` with pinned
versions, latency stats, tool-call reliability, and the
deployment-config breadcrumbs Phase 5 needs to regenerate.

Invocation (with the mlx_lm.server up on the Mac Studio):

    uv run python tools/phase3_llm_smoke_check.py
"""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from industrial_ai.agents.errors import LLMResponseParseError
from industrial_ai.agents.llm_client import (
    LLMResponse,
    MLXServerLLMClient,
)

_REPO_ROOT = Path(__file__).resolve().parent.parent
_OUTPUT = _REPO_ROOT / "data" / "reference" / "phase3_llm_smoke.json"
_TEMPLATE_PATH = _REPO_ROOT / "data" / "reference" / "nemotron_super_v1_5_chat_template.jinja"
_DEFAULT_BASE_URL = "http://192.168.178.81:8080/v1"
_TOOL_CALL_N = 10

_SYSTEM_TOOL_PROMPT = (
    "You are the supervisory layer of a binary distillation column. "
    "Choose a (y_D_target, x_B_target) pair that minimizes IAE under the "
    "observed plant state. Physical bounds: y_D_target ∈ [0, 1], "
    "x_B_target ∈ [0, 1], y_D_target must strictly exceed x_B_target. "
    "Reply with a SINGLE JSON object on its own line: "
    '{"y_D_target": <float>, "x_B_target": <float>, "rationale": "<short reason>"}. '
    "No surrounding markdown, no trailing prose, no JSON arrays."
)

_USER_PROMPTS = [
    "Cycle 0 at t=0.0 min. Plant: y_D=0.99, x_B=0.01, LT=2.706, VB=3.206, "
    "F=1.000, zF=0.500, qF=1.000. Run IAE so far: 0.0000.",
    "Cycle 1 at t=5.0 min. Plant: y_D=0.985, x_B=0.012, LT=2.71, VB=3.21, "
    "F=1.000, zF=0.500, qF=1.000. Run IAE so far: 0.0080.",
    "Cycle 2 at t=10.0 min. Plant: y_D=0.990, x_B=0.011, LT=2.70, VB=3.21, "
    "F=1.000, zF=0.500, qF=1.000. Run IAE so far: 0.0090.",
    "Cycle 3 at t=15.0 min. Plant: y_D=0.989, x_B=0.011, LT=2.70, VB=3.21, "
    "F=1.000, zF=0.500, qF=1.000. Run IAE so far: 0.0120.",
    "Cycle 4 at t=20.0 min. Plant: y_D=0.991, x_B=0.010, LT=2.71, VB=3.20, "
    "F=1.000, zF=0.500, qF=1.000. Run IAE so far: 0.0130.",
    "Cycle 5 at t=25.0 min. Plant: y_D=0.988, x_B=0.012, LT=2.70, VB=3.21, "
    "F=1.000, zF=0.500, qF=1.000. Run IAE so far: 0.0150.",
    "Cycle 6 at t=30.0 min. Plant: y_D=0.990, x_B=0.011, LT=2.70, VB=3.21, "
    "F=1.000, zF=0.500, qF=1.000. Run IAE so far: 0.0160.",
    "Cycle 7 at t=35.0 min. Plant: y_D=0.991, x_B=0.010, LT=2.71, VB=3.20, "
    "F=1.000, zF=0.500, qF=1.000. Run IAE so far: 0.0170.",
    "Cycle 8 at t=40.0 min. Plant: y_D=0.989, x_B=0.011, LT=2.70, VB=3.21, "
    "F=1.000, zF=0.500, qF=1.000. Run IAE so far: 0.0190.",
    "Cycle 9 at t=45.0 min. Plant: y_D=0.990, x_B=0.011, LT=2.70, VB=3.21, "
    "F=1.000, zF=0.500, qF=1.000. Run IAE so far: 0.0200.",
]


def _git_sha() -> str:
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=_REPO_ROOT).decode().strip()
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def _template_sha() -> str:
    import hashlib

    return hashlib.sha256(_TEMPLATE_PATH.read_bytes()).hexdigest()


def _ssh_capture(cmd: str, host: str = "gamba@192.168.178.81") -> str:
    """Capture stdout of a single SSH command on the Mac Studio."""
    try:
        return subprocess.check_output(["ssh", host, cmd], text=True, timeout=30).strip()
    except subprocess.CalledProcessError as exc:
        return f"ssh-error:{exc.returncode}"
    except subprocess.TimeoutExpired:
        return "ssh-timeout"


def _run_test_a(client: MLXServerLLMClient) -> dict[str, Any]:
    """Single prose sanity call to confirm reasoning path is alive."""
    t0 = time.perf_counter()
    try:
        # NOTE: prose intentionally fails the strict JSON parse below;
        # this Test A's value is the wall-clock + coherent text check.
        client.complete(
            system_prompt="You are a process control engineer. Answer in one sentence.",
            user_prompt="In one sentence: what does the RGA quantify?",
            max_tokens=256,
        )
        elapsed = time.perf_counter() - t0
        return {
            "skipped_because": (
                "Test A is informational; the prose JSON-parser failure proves "
                "the request reached the model but is not a useful KPI here."
            ),
            "wall_clock_seconds": elapsed,
            "outcome": "parsed-as-json-as-expected-for-prose-rejection",
        }
    except LLMResponseParseError as exc:
        elapsed = time.perf_counter() - t0
        # This is the expected branch for prose: no JSON object → parse error.
        return {
            "wall_clock_seconds": elapsed,
            "outcome": "prose_returned_no_json",
            "snippet": str(exc)[:300],
        }


def _run_test_c(client: MLXServerLLMClient) -> dict[str, Any]:
    """One revision-mode call to verify ``reasoning=True`` path is healthy.

    Sends a deliberately ambiguous prompt that benefits from chain-of-
    thought reasoning; budget is the 4096-token default for revision
    mode. Demonstrates that the Critic-revision path produces clean
    JSON after the reasoning block (no silent fallback).
    """
    user_prompt = (
        "Cycle 3 at t=15.0 min. Plant: y_D=0.720, x_B=0.005, LT=2.706, VB=3.206, "
        "F=0.800, zF=0.450, qF=1.000. Run IAE so far: 6.4500. "
        "Critic feedback on previous proposal: target acquisition stalled "
        "after 10 min; reconsider whether a stepped interim setpoint would "
        "be more reachable than the nominal product spec at this OP."
    )
    t0 = time.perf_counter()
    try:
        response = client.complete(
            system_prompt=_SYSTEM_TOOL_PROMPT,
            user_prompt=user_prompt,
            reasoning=True,
        )
    except LLMResponseParseError as exc:
        return {
            "parsed": False,
            "wall_clock_seconds": time.perf_counter() - t0,
            "parse_error": str(exc)[:300],
        }
    elapsed = time.perf_counter() - t0
    return {
        "parsed": True,
        "wall_clock_seconds": elapsed,
        "had_think_block": "<think>" in response.raw_text and "</think>" in response.raw_text,
        "proposal": {
            "y_D_target": response.proposal.y_D_target,
            "x_B_target": response.proposal.x_B_target,
            "rationale": response.proposal.rationale[:200],
        },
        "raw_text_length": len(response.raw_text),
    }


def _run_test_b(client: MLXServerLLMClient) -> dict[str, Any]:
    """N tool-call requests; measure parse reliability + latency.

    Calls with ``reasoning=False`` (the modal default for tool-call
    cycles per ADR 005 amendment 2026-05-28). Critic-revision rounds
    use ``reasoning=True`` and are exercised separately by
    :func:`_run_test_c`.
    """
    results: list[dict[str, Any]] = []
    parse_ok = 0
    schema_ok = 0
    for i, user_prompt in enumerate(_USER_PROMPTS):
        t0 = time.perf_counter()
        parse_error: str | None = None
        proposal: dict[str, Any] | None = None
        raw_text = ""
        try:
            response: LLMResponse = client.complete(
                system_prompt=_SYSTEM_TOOL_PROMPT,
                user_prompt=user_prompt,
                reasoning=False,
            )
        except LLMResponseParseError as exc:
            elapsed = time.perf_counter() - t0
            parse_error = str(exc)[:300]
            results.append(
                {
                    "i": i,
                    "wall_clock_seconds": elapsed,
                    "parsed": False,
                    "parse_error": parse_error,
                }
            )
            continue
        elapsed = time.perf_counter() - t0
        raw_text = response.raw_text
        parse_ok += 1
        schema_ok += 1
        proposal = {
            "y_D_target": response.proposal.y_D_target,
            "x_B_target": response.proposal.x_B_target,
            "rationale": response.proposal.rationale[:200],
        }
        # Diagnostic: did the raw text contain a <think>...</think> block?
        had_think = "<think>" in raw_text and "</think>" in raw_text
        results.append(
            {
                "i": i,
                "wall_clock_seconds": elapsed,
                "parsed": True,
                "schema_ok": True,
                "had_think_block": had_think,
                "proposal": proposal,
                "raw_text_length": len(raw_text),
                "raw_text_snippet": raw_text[:400],
            }
        )
    wall_clocks = [r["wall_clock_seconds"] for r in results]
    return {
        "n": _TOOL_CALL_N,
        "parsed": parse_ok,
        "schema_ok": schema_ok,
        "reliability_rate": parse_ok / _TOOL_CALL_N,
        "schema_rate": schema_ok / _TOOL_CALL_N,
        "wall_clock_seconds": {
            "mean": statistics.mean(wall_clocks),
            "p50": statistics.median(wall_clocks),
            "p95": sorted(wall_clocks)[max(0, int(0.95 * len(wall_clocks)) - 1)],
            "max": max(wall_clocks),
            "min": min(wall_clocks),
        },
        "per_call": results,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default=_DEFAULT_BASE_URL)
    parser.add_argument("--output", type=Path, default=_OUTPUT)
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)

    print(f"=== Smoke check against {args.base_url} ===")
    print(f"chat_template.jinja SHA-256: {_template_sha()}")

    print("Querying mac-studio runtime versions ...")
    versions = {
        "lms_cli": _ssh_capture(
            "PATH=$HOME/.lmstudio/bin:$PATH lms version | grep -i 'commit:' | head -1"
        ),
        "lm_studio_app": _ssh_capture(
            "defaults read /Applications/LM\\ Studio.app/Contents/Info CFBundleShortVersionString"
        ),
        "mlx_lm_runtime_selected": _ssh_capture(
            "PATH=$HOME/.lmstudio/bin:$PATH lms runtime ls 2>&1 | awk '/mlx-llm.*✓/{print $1}'"
        ),
        "python_venv": _ssh_capture(
            "~/mlx_test_venv/bin/python -c 'import sys; print(sys.version.split()[0])'"
        ),
        "mlx_lm": _ssh_capture(
            "~/mlx_test_venv/bin/python -c '"
            'import importlib.metadata as m; print(m.version("mlx-lm"))\''
        ),
        "mlx": _ssh_capture(
            "~/mlx_test_venv/bin/python -c '"
            'import importlib.metadata as m; print(m.version("mlx"))\''
        ),
        "transformers": _ssh_capture(
            "~/mlx_test_venv/bin/python -c '"
            'import importlib.metadata as m; print(m.version("transformers"))\''
        ),
    }
    for k, v in versions.items():
        print(f"  {k}: {v}")

    client = MLXServerLLMClient(base_url=args.base_url)

    print("\n=== Test A — prose sanity ===")
    test_a = _run_test_a(client)
    print(json.dumps({k: v for k, v in test_a.items() if k != "snippet"}, indent=2))

    print(f"\n=== Test B — {_TOOL_CALL_N} tool-call requests (reasoning=False) ===")
    test_b = _run_test_b(client)
    print(
        f"reliability: {test_b['parsed']}/{test_b['n']} = {test_b['reliability_rate'] * 100:.1f} %"
    )
    print(f"wall_clock p50: {test_b['wall_clock_seconds']['p50']:.2f} s")
    print(f"wall_clock p95: {test_b['wall_clock_seconds']['p95']:.2f} s")
    print(f"wall_clock max: {test_b['wall_clock_seconds']['max']:.2f} s")

    print("\n=== Test C — single revision-mode call (reasoning=True) ===")
    test_c = _run_test_c(client)
    print(f"  parsed: {test_c['parsed']}  wall_clock: {test_c['wall_clock_seconds']:.1f} s")
    if test_c["parsed"]:
        print(f"  proposal: {test_c['proposal']}")

    payload: dict[str, Any] = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(tz=UTC).isoformat(),
        "git_sha": _git_sha(),
        "endpoint": {
            "base_url": args.base_url,
            "transport": "/v1/completions with client-side jinja2 template render",
            "chat_template_path": "data/reference/nemotron_super_v1_5_chat_template.jinja",
            "chat_template_sha256": _template_sha(),
            "trust_remote_code": True,
            "stop_tokens": ["<|eot_id|>"],
        },
        "versions": versions,
        "test_a_prose": test_a,
        "test_b_tool_calls": test_b,
        "test_c_revision_mode": test_c,
    }
    with args.output.open("w") as fh:
        json.dump(payload, fh, indent=2)
    print(f"\nWrote {args.output}")

    rate = test_b["reliability_rate"]
    if rate < 0.9:
        print(
            f"\n!! ADR-005 GATE: tool-call reliability {rate * 100:.0f} % < 90 %. STOP and review."
        )
        return 2
    p95 = test_b["wall_clock_seconds"]["p95"]
    if p95 > 30.0:
        print(f"\n!! ADR-005 GATE: per-call P95 {p95:.1f} s > 30 s. STOP and review.")
        return 3
    if p95 > 10.0:
        print(
            f"\n* Open item §5.1: per-call P95 {p95:.1f} s in 10-30 s band. "
            "Document concrete numbers in pre_submission_checklist."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
