# ADR 012 — Cross-Family Ablation Model: DeepSeek (NIM-Hosted)

**Status:** Accepted
**Date:** 2026-06-01
**Supersedes:** ADR-005's "Ablation model" specification (Qwen3.6-27B on Mac Studio)
**Depends on:** ADR-011 (Inference Stack Split)

## Context

ADR-005 specified `Qwen3.6-27B` (Mac-Studio, MLX) as the Phase-5 cross-family
ablation model. With ADR-011 moving the *primary* inference path off
Mac Studio to NIM-hosted Nemotron, the ablation slot has to be reconsidered
for two independent reasons:

1. **Capability ceiling.** With the primary now at NIM-hosted full-precision
   Nemotron-Super-49B-v1.5 (or potentially upgraded to Nemotron-Ultra-253B),
   Qwen3.6-27B is no longer a credible cross-family contrast — it is
   smaller and weaker than the primary, which undermines the "architecture
   generalizes across LLM families" claim the ablation is supposed to support.
   A reviewer reads "ablated against a weaker model" as biasing the
   comparison toward the primary.

2. **Reproducibility-chain alignment.** ADR-011 establishes the paper's
   methodological position as "open weights + open API + two independent
   reproduction paths" (NIM-hosted and self-hosted via published weights).
   Keeping the ablation on Mac-Studio-MLX would re-introduce the closed-stack
   reproducibility caveat (8-bit MLX, custom jinja render, transformers
   regression workaround, template SHA pinning) that ADR-011 just removed
   from the primary path. The ablation must inherit the primary's
   reproducibility properties.

## Decision

The Phase-5 cross-family ablation model is the **DeepSeek family**, served
through the same NIM-hosted OpenAI-Chat-API surface as the primary.

- **Specific model version:** deferred to Phase-5 kickoff, picked from
  NIM's then-current DeepSeek catalog. Likely candidates as of 2026-06:
  DeepSeek-V3.x, DeepSeek-R1, DeepSeek-V4-Pro (subject to NIM
  availability and release status at Phase-5 kickoff).
- **Selection criteria at Phase-5 kickoff:** the strongest DeepSeek on NIM
  that is comparable-or-larger in capacity to the chosen primary Nemotron
  variant. If primary remains Nemotron-Super-49B-v1.5, ablation should be
  at minimum DeepSeek-V3.x; if primary is upgraded to Nemotron-Ultra-253B,
  ablation should be V4-Pro or contemporary successor.

## Rationale

### Why DeepSeek as the cross-family contrast

- **Different post-training lineage.** Nemotron carries NVIDIA's
  RPO/RLVR/iterative-DPO stack targeting tool-calling and RAG; DeepSeek
  carries reasoning-focused post-training optimized for chain-of-thought
  and multi-step inference. If both produce comparable C2 KPI numbers,
  the architecture's value is post-training-agnostic — a strong paper claim.
- **Strongest open-weight cross-family option.** As of mid-2026, the
  DeepSeek family is the strongest open-weights reasoning lineage outside
  the Llama/Nemotron axis. A reviewer who asks "why not the strongest
  alternative?" has no fallback objection.
- **NIM-hosted on the same protocol.** Same API surface, same chat-template
  handling server-side, same reproducibility chain. The ADR-011 framing
  applies verbatim to the ablation.
- **Cross-region origin.** PRC-origin (DeepSeek) vs US/NVIDIA-origin
  (Nemotron). Not methodologically load-bearing but pre-empts the
  "Western-models-only" reviewer objection without manufacturing diversity.

### Why not Qwen (the original ADR-005 specification)

- Qwen3.6-27B is below the primary's capacity tier. With the move from
  Mac-Studio 8-bit Nemotron-49B to NIM BF16 Nemotron-49B (and possible
  Ultra-253B upgrade), the gap widens.
- Larger Qwen variants (Qwen3.5-72B, Qwen3.5-122B-A10B) would address the
  capacity issue but lose to DeepSeek on reasoning benchmarks at comparable
  parameter counts. The "natural strongest cross-family alternative"
  reviewer expectation is DeepSeek, not Qwen.
- Christian's personal local stack uses Qwen (Qwen3.5-122B-A10B-IQ4_XS)
  but that is general-purpose tooling, not the paper's evaluation model;
  no project-coherence reason to use the same family for the paper.

### Why not pin DeepSeek-V4-Pro now

- V4-Pro uses an FP4+FP8 mixed weight format that makes the self-hosted
  reproduction path non-trivial (per ADR-005's "Why not DeepSeek V4-Flash"
  section, which still applies for self-hosting). NIM abstracts the
  quantization, but a reproducibility-conscious reader self-hosting may
  find the path harder than for DeepSeek-V3.x.
- NIM catalog state in May 2026 ≠ NIM catalog state at Phase-5 kickoff.
  Pinning the exact version now risks obsolescence by drafting time.

## Consequences

- **Mac Studio's role in the project's evaluation pipeline is now zero.**
  The `MLXServerLLMClient` remains in the codebase as a reference
  implementation and as a contingency for any future on-workstation
  evaluation but does not run a primary or ablation route from this
  ADR forward. The Schritt-4 transport-amendment work is preserved as
  a methodological vignette in the paper (it produced the data that
  motivated ADR-011 and is therefore not sunk cost).
- ADR-005's "Ablation model" section is superseded. ADR-005 remains in
  force as historical audit trail and as documentation of the
  on-Mac-Studio path that the project explored and retired.
- `tools/run_c2_smoke.py` retains the `--backend {nim,mac-studio}` flag;
  `mac-studio` becomes a contingency switch rather than a routine
  ablation target. The ablation will use `--backend nim` with the
  ablation-model identifier (e.g. `NVIDIA_ABLATION_MODEL` env var
  added in Phase-5 prep) rather than `mac-studio`.
- `pre_submission_checklist.md` §2.2 bucket-probability re-estimate
  may shift slightly: a stronger ablation model raises the probability
  that the architecture generalizes across families (paper-positive) but
  raises the bar that the primary must clear before the cross-family
  claim is credible.
- `paper/outline.md` §3.4 acquires a cross-family-ablation sentence at
  Phase-5 drafting time. No edit required now.

## Reversibility

**Medium.** Swapping the specific DeepSeek version at Phase-5 kickoff is a
config-file change. Reverting to Qwen would re-invoke the capability-ceiling
problem in §Rationale; not recommended unless DeepSeek availability on NIM
changes catastrophically. Reverting the host to Mac-Studio is possible via
the retained `mac-studio` backend flag but undoes the reproducibility-chain
properties.

## Open items at decision time

1. **Specific DeepSeek model version** — pinned at Phase-5 kickoff against
   the NIM catalog state. Update this ADR with an Amendment at that time.
2. **Multi-family ablation (3+ models)** — possible to add a third family
   (Qwen3.5-Max, Llama-3.3-70B-Instruct) for a strengthened generalization
   claim. Not currently planned (4× ablation compute, dilutes contrast).
   Reserved as Phase-5 extension if reviewer feedback warrants.
3. **DeepSeek-R1 vs DeepSeek-V3.x for the specific Phase-5 run** —
   R1's reasoning trace is verbose and may stress the same output-discipline
   property we tightened for Nemotron. Methodological note for Phase-5
   prep: test both on the nominal_baseline smoke before committing to one.

## Changelog

- 2026-06-01 — Initial decision. Triggered by ADR-011's reproducibility-chain
  framing and the direction to use DeepSeek (not Qwen) as the cross-family
  ablation. Mac-Studio ablation role retires.
