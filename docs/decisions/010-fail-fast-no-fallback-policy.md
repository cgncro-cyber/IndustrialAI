# ADR 010 — Fail-Fast, No-Fallback Error Policy

**Status:** Accepted
**Date:** 2026-05-28
**Refines:** ADR 006 (hierarchical control architecture), ADR 009 (regulatory backend)

## Context

This project is a scientific evaluation pipeline. Every C2 / C3 run produces numbers that, after Phase 5, enter the paper's Results section and the bucket-classification decision tree in `docs/kpis.md` §6. The integrity of those numbers is the project's headline asset.

A silent fallback — a substitution that masks an unexpected failure and lets the run continue with degraded state — produces output that looks like a valid run but is in fact a fallback artifact. There is no way for a downstream KPI computation, an autograder, or a reviewer to know the difference. The contamination flows into the published bucket classification.

The temptation to add silent fallbacks is real: a missing LLM endpoint, an unreachable LM Studio instance, a transient network blip on a remote provider, a regulatory-backend that fails to construct. The convenient response is *"return a default setpoint and keep going"*, *"swap in the Mock client if the real one is unavailable"*, *"retry until it works or pick something sensible"*. Each of these is a contamination vector.

Reproducibility and experimental integrity require the opposite default: a run either produces valid data or fails loudly. A loud failure is informative — it points to a real problem (LM Studio crashed, endpoint URL is wrong, model not loaded, GPU OOM) — and prevents data contamination downstream. A silent fallback hides the same problem and lets it propagate into the analysis.

This is also a *trust* contract for human review: a Phase-5 reader of the per-cycle decision log must be able to assume that every cycle on the log was driven by the real configuration named in `manifest.json`. A run that quietly downgraded to a different configuration breaks that contract.

## Decision

### 1. No silent fallbacks anywhere in the pipeline

Errors propagate as named, typed exceptions. The pipeline does not catch a low-level error, substitute a default, and keep going. Every catch site must either fully resolve the error (rare) or re-raise with additional context.

Named exception types are preferred over generic `Exception` or `ValueError`. The Phase-3 / Phase-4 / Phase-5 codebase introduces (at minimum):

- `LLMEndpointUnreachableError` — the configured LLM endpoint refused the connection or timed out on a single attempt. The exception message must include the endpoint URL so the operator can fix it.
- `LLMResponseParseError` — the LLM returned a response that does not parse to the expected schema (already covered functionally by `_parse_setpoint_json`; the wrapper raises a named subtype).
- `RegulatoryBackendError` — the regulatory backend (MPC or PID, per ADR 009) failed to construct or step. The message must name which backend and which operation.
- `CriticLoopLimitError` — the agent graph hit the hard recursion limit without an `accept` verdict and could not fall back to a previous accepted target. (See §4 below for the *narrow* exception: when a previous accepted target exists, the graph's documented `escalate` path applies; only the *exhaustion-without-fallback* case raises this error.)

### 2. Unreachable external dependencies abort the run immediately

Specifically:

- One connection attempt per external call. No retry-until-default loops. (Bounded retries against transient network glitches may be re-introduced *only* via explicit configuration, with a clear log line per attempt and a final named exception if all attempts fail. Default = no retries.)
- No silent provider switch. If the configured LLM endpoint is unreachable, the run dies. There is no automatic substitution of a mock, a smaller local model, or a different remote provider.
- No silent setpoint default. If the agent cannot produce a setpoint, the supervisor does *not* emit `(0.99, 0.01)` and pretend the agent ran. The run dies and the cycle is missing from the per-cycle log.

### 3. MockLLMClient is exclusively a test double

Constructing the `MockLLMClient` outside of a test context must raise. The guard mechanism is one of:

- A check that `pytest` is currently active (via `sys.modules` membership or an environment variable that `pytest` sets, e.g. `PYTEST_CURRENT_TEST`).
- An explicit `allow_mock=True` constructor flag that lives only in test fixtures, never in production config files or notebook cells outside of `tests/`.

The combination of the two is fine; either alone is sufficient. The forbidden case is *any* path in production code that constructs a `MockLLMClient` because a real client could not be initialised. There must be zero wiring from `LMStudioLLMClient.__init__` failure to a `MockLLMClient` instance.

### 4. Explicit, logged safe-state transitions are exempt

The fallback prohibition is about *masking unexpected failures*. It does not constrain *designed responses to detected conditions*.

The canonical example is the C3 safety gate (Phase 4): an agent proposal that the gate classifies as unsafe is *blocked* and a documented safe setpoint is held instead. This is not a fallback — it is the designed function of the gate, and the only way the gate can usefully exist. The same applies to other future safe-state transitions (operator override, hard physical-bound clamp at the actuator, etc.) provided they meet all of the following:

- They respond to a *detected* condition with an explicit predicate, not to a generic exception.
- They emit a structured log line per transition, including the trigger condition, the rejected proposal, and the substitute action.
- They are counted in the relevant intercept / detection KPI from `docs/kpis.md` §3.

A safe-state transition that does not meet all three is a silent fallback under another name and is forbidden.

### 5. Critic-loop limit is a named failure, not a default

The agent graph's `max_critic_optimizer_rounds` budget already has a documented `escalate` path that *re-uses the previous accepted target*. This is a safe-state transition under §4: it logs the verdict, counts toward an escalation rate, and responds to a detected condition (budget exhaustion *with* a previous-accepted target available).

The case that this ADR adds is the *initial* one — the very first cycle of a run, where no previous accepted target exists yet. The current skeleton silently substitutes `(0.99, 0.01)` in `_safe_fallback_target()`. Under this ADR that substitution is a silent fallback and must be replaced by a `CriticLoopLimitError`. A run that cannot pass its very first cycle has *no* valid data to produce, and the operator wants to know that immediately, not after the bucket-classification table is computed against a column of nominal-spec defaults.

## Distinguishing forbidden fallbacks from permitted safe-state transitions

| Pattern | Forbidden / permitted? | Why |
|---|---|---|
| LLM endpoint refused connection → return `(0.99, 0.01)` | **Forbidden.** | Unexpected error masked. Operator never learns the endpoint is down. |
| LLM endpoint refused connection → raise `LLMEndpointUnreachableError` with URL | **Permitted (mandatory).** | Loud failure, root cause visible. |
| LM Studio client init fails → instantiate `MockLLMClient` and continue | **Forbidden.** | Configuration drift; downstream KPIs are mock numbers but look real. |
| `MockLLMClient(allow_mock=True)` in a test fixture | **Permitted.** | Test context, explicit opt-in, never in production code path. |
| Safety gate blocks an unsafe proposal → emit logged safe setpoint, count in `intercept_rate` | **Permitted.** | Designed response to a detected condition, logged, counted. |
| Optimizer raises → catch and emit previous accepted target | **Forbidden** if previous-accepted does not exist; **permitted** (and logged + counted) if it does and the substitution is part of the documented `escalate` verdict path. | The verdict path is a *designed* response; a silent catch is not. |
| Sim integration fails → carry forward last state, set `success=False`, continue with KPI flag | **Permitted.** | Existing simulator pattern, logged in `SimulationResult.message`, KPI computation treats it as `iae=inf` — this is a *structured* result, not a silent default. |
| LLM JSON parse error → swallow and retry until a clean parse appears | **Forbidden.** | Either the model is misconfigured (operator must know) or the prompt is wrong (operator must know); either way, fail loudly with `LLMResponseParseError`. Bounded retries with logged failure are an explicit config option (default off). |

## Rationale

- **Reproducibility.** A failed run that died loudly produces no contaminated artifacts and a clear root-cause trace. A run that silently degraded produces a misleading artifact and no signal that anything went wrong. The first is recoverable in minutes; the second is recoverable only by tearing down and re-running the entire evaluation pass.
- **Reviewer trust.** SAFEPROCESS reviewers and arXiv readers will assume that the per-cycle decision log corresponds to the configuration named in `manifest.json`. A pipeline that quietly substituted components breaks that contract and, once discovered, casts doubt on every reported number in the paper.
- **Operator feedback loop.** The faster a run dies on a bad configuration, the faster the operator fixes it. Silent fallbacks defer the feedback until KPI inspection, by which point the operator does not remember which run was misconfigured.
- **Aligns with the existing simulator design.** The simulator already uses structured failure (`SimulationResult.success=False` + `message`) rather than silent recovery. This ADR generalizes that pattern to every component in the pipeline.
- **Mock isolation.** A test double that can be instantiated in production code is a footgun. The guard prevents the most common kind of accidental contamination — a notebook or evaluation script that constructs the mock and forgets to swap it.

## Consequences

- The Phase-3 skeleton (commit `74f9517`) was built before this ADR. A targeted audit-and-harden pass follows immediately (Schritt 2 of the ADR-010 work): rename / introduce typed exceptions, guard `MockLLMClient` construction, replace `_safe_fallback_target` for the no-prior-accept case, repo-wide sweep for `or <default>` substitution patterns and bare-except-pass anti-patterns.
- Runs will crash more often during Phase 3 prompt iteration. Every crash is informative. The development loop sees the error directly instead of degrading to noise.
- Phase-4 safety-gate work inherits the exemption framework above. The gate's intercept events are logged + counted, not caught + masked.
- Future external dependencies (a remote KPI service, a cloud anomaly detector, a hosted dataset) inherit the same contract: one attempt, named exception on failure, no silent retry-to-default.

## Reversibility

- Reversibility as a *policy* is high — the change is a documentation decision and a series of source-level guards.
- Reversibility as a *codebase audit* is low. Re-introducing silent fallbacks would require auditing every error-handling site reintroduced under this policy and confirming the silent path is intentional rather than a leak. Not anticipated.
- The narrow exemption for *bounded, logged, opt-in retries against transient network glitches* (decision §2 last sentence) is the only sanctioned escape valve and is intentionally narrow: explicit configuration, per-attempt logging, named final exception. Anything beyond that requires an amendment to this ADR.
