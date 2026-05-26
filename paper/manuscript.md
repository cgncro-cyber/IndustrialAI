# Paper Manuscripts

This directory will hold two deliverables, both derived from the same underlying work:

1. **`arxiv_preprint.md`** — full-length preprint, primary career-positioning asset.
2. **`safeprocess_paper.md`** — 6-page IFAC conference paper for SAFEPROCESS 2027.

Drafting begins in Phase 5 (see [`../PROJECT_PLAN.md`](../PROJECT_PLAN.md)). The conference paper is derived *from* the preprint, not authored separately.

## Working Title

Safety-Gated Agentic Control of Coupled Distillation Trains:
A Case Study in Industrial-AI Methodology Transfer

## Working Abstract (placeholder — to be replaced with real results)

Agentic large-language-model controllers have emerged as a promising paradigm for industrial process optimization, but their adoption is limited by the absence of auditable safety mechanisms. We propose a co-designed architecture in which a LangGraph-based multi-agent controller proposes setpoint adjustments and a downstream anomaly-detection layer — trained on industrial process benchmarks — gates these proposals before execution. We evaluate the architecture on a coupled C3/C4 distillation train modeled in IDAES, comparing against a classical PID baseline across {N} disturbance scenarios. Results report energy consumption per kilogram of product, yield, constraint-violation frequency, and disturbance-recovery time with 95 % confidence intervals. We discuss methodology-transfer implications for semiconductor process control, pharmaceutical continuous manufacturing, and other coupled multivariable industrial processes.

## Section Outline

1. Introduction
2. Related Work
   - Classical multivariable process control
   - Reinforcement-learning approaches to process control
   - Agentic LLM systems
   - Anomaly detection on industrial benchmarks (TEP, NoBOOM)
3. Methodology
   - Process twin (IDAES)
   - Baseline controller (PID)
   - Agentic layer (LangGraph)
   - Safety gate (anomaly detector)
4. Experimental Setup
   - Disturbance scenarios
   - KPIs and statistical methodology
5. Results
6. Discussion
   - Transfer to non-distillation domains
   - Limitations and threats to validity
7. Conclusion

## Conference Target: SAFEPROCESS 2027

- **Venue:** TU Delft, Netherlands
- **Dates:** 29 June – 2 July 2027
- **Theme:** AI for Safety
- **Submission:** PaperCept, 6-page IFAC two-column format
- **Hard deadline:** 31 October 2026
- **Proceedings:** IFAC-PapersOnline, Scopus-indexed

## arXiv Submission

- **Categories:** `cs.LG` (primary), `eess.SY` (cross-list)
- **Timing:** posted same day as SAFEPROCESS submission, to maximize citation runway
- **License:** CC-BY 4.0 (compatible with IFAC proceedings policy — verify before submission)

## Deferred: Journal Version

Only pursued post-job-search. Candidate venues, in order of preference:

1. Computers & Chemical Engineering (Elsevier)
2. Journal of Process Control (Elsevier)
3. Engineering Applications of Artificial Intelligence (Elsevier) — if AI angle dominates
