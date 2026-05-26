# ADR 003 — Novelty Positioning: Safety-Gated Agentic Control

**Status:** Accepted
**Date:** 2026-05-26

## Context

The phrase "LLM agent controls a distillation column" is no longer novel in 2026. Multiple Kaggle notebooks, blog posts, and preprints already exist. A paper claiming this alone would be rejected as incremental.

## Decision

The paper's novelty claim is **the co-design of agentic setpoint optimization with an anomaly-detection safety gate** that filters agent proposals before they reach the process. The agent and the safety layer are *not* separate prior art — their integration is the contribution.

## Rationale

Three properties make this defensible novelty:

1. **Gap in literature**: Existing agentic-process-control work treats safety as a post-hoc concern (rule-based limit checks). Anomaly detection on industrial benchmarks (TEP, NoBOOM) treats detection as the end goal, not a control input.
2. **Practical relevance**: Industrial adoption of agentic control is blocked precisely by safety auditability. A learned safety gate that operates on the same data stream as the controller addresses this directly.
3. **Falsifiable claims**: The four-way comparison (PID / Agent / Agent+Safety / Disturbance Recovery) yields effect sizes that are either positive and significant, or not. Either outcome is publishable.

## Consequences

- The paper title leads with "Safety-Gated Agentic Control," not "LLM Agent for Distillation."
- The safety layer cannot be skipped or treated as a nice-to-have — it is the novelty.
- The evaluation must explicitly report what the safety layer *blocked* and what it would have *missed* if absent.

## Reversibility

Low. The architecture, evaluation design, and paper outline are all organized around this novelty claim.
