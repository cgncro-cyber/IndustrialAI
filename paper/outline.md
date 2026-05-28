# Paper Outline — Safety-Gated Agentic Supervisory Control

Content skeleton for the arXiv preprint and the SAFEPROCESS 2027 6-page
derivative. Section structure plus 2–3 sentences of intent per section.
Not prose — the drafting happens in Phase 5 (`PROJECT_PLAN.md`). This file
exists to **scope Phase 3 / 4**: each section names the artifact that must
exist for it to be writable, so empirical gaps surface now rather than in
October.

**Relationship to other paper files.** `manuscript.md` is the venue / logistics
/ submission-mechanics document (its inline section list predates the build and
is stale — IDAES, C3/C4 train, energy/yield KPIs were never built; treat this
outline as the authoritative content structure). `methods_phase3_buckets.md`
holds the three pre-registered Methods paragraphs for the §5 outcome; this
outline references them rather than restating them. `docs/figures.md` is the
figure registry.

**Authoritative sources this outline must stay consistent with.** ADR 005
(LLM + transport amendment), ADR 006 (two-layer hierarchy, C0–C3 contract,
5–15 min cadence), ADR 007 (Skogestad Column A, LV config), ADR 008 (deferred
real-data validation), ADR 009 (C2/C3 regulatory backend: MPC primary, PID
deployment-economics branch), ADR 010 (fail-fast / no-fallback), `docs/kpis.md`
(five KPIs + bucket decision tree), `docs/pre_submission_checklist.md` §4
(disclosed limitations).

---

## Title (working)

Safety-Gated Agentic Supervisory Control of a Coupled Distillation Column:
A Methodology-Transfer Case Study

Note: the prior "coupled distillation trains" / "C3/C4" title overclaims — the
study is a **single** Skogestad Column A (ADR 007, disclosed limitation §4.3).
Title must not promise multiple columns or a train.

## Abstract

State the gap (agentic LLM controllers lack auditable safety mechanisms),
the contribution (co-designed two-layer architecture: LLM supervisor proposing
composition setpoints + downstream anomaly gate vetting them before execution),
and the evaluation (four-way C0/C1/C2/C3 comparison on Skogestad Column A,
apples-to-apples per ADR 006). Report the headline KPI outcome as the selected
bucket from `methods_phase3_buckets.md` once C2/C3 runs exist — leave `[VALUE]`
placeholders until then. Close on transfer implications (semiconductor, pharma,
battery) framed as architecture argument, not empirical coverage.

## 1. Introduction

Motivate industrial supervisory control as a setting where LLM agents could
add value (regime adaptation, no per-OP retuning) but cannot be deployed without
a safety mechanism an operator can audit. State the three contributions: (i) the
safety-gated supervisory architecture, (ii) the apples-to-apples four-way
evaluation isolating the agent's and the gate's marginal value, (iii) the
cross-domain-trained gate. End with the explicit claim scope: one canonical
column, transfer argued architecturally.

## 2. Related Work

- Classical multivariable / supervisory process control (RGA, decentralized vs
  MPC, two-layer hierarchies) — anchors the C0/C1 baselines as industrial-standard.
- RL for process control — position the agent against learned-policy approaches;
  contrast auditability.
- Agentic LLM systems and tool-calling — the Observer/Optimizer/Critic lineage.
- Anomaly detection on industrial benchmarks (TEP) — sets up the cross-domain gate.
- Gap statement: none combine an agentic supervisor with an auditable,
  cross-domain safety gate evaluated apples-to-apples against MPC.

## 3. Methodology

### 3.1 Process twin
Skogestad Column A, LV configuration, nominal OP F=1, zF=0.5, qF=1 (ADR 007).
State why canonical (recognized benchmark, high RGA ≈ 36 → genuinely coupled —
§4.1). Level closure (M_D, M_B) held identical across all four configs — the
basis of the apples-to-apples contract (ADR 006).

### 3.2 Four-way comparison contract
C0 = PID composition control; C1 = linear MPC (replaces composition PID);
C2 = agentic supervisor over the regulatory layer; C3 = C2 + safety gate
(ADR 006). C2/C3 supervise the MPC backend as primary (Option A), with the
PID-backend "MPC-free deployment" branch as a secondary analysis (ADR 009).
Emphasize: same plant, same level closure, same scenarios, same seeds.

### 3.3 Baseline controllers (C0, C1)
C0 tuning provenance (TL shootout, Outcome A — `pre_submission_checklist.md`
§2.1). C1 linear MPC, per-OP linearization from the sweep cache. Report the
C1 regularization Pareto front (Figure 9): no fixed-weight tuning wins both
nominal and off-nominal — the structural motivation for an adaptive layer
(§4.6). This subsection carries the "why an agent at all" argument.

### 3.4 Agentic supervisor (C2)
LangGraph Observer → Optimizer → Critic; the Critic is rule-based (no LLM call).
Agent proposes composition targets (y_D, x_B) per supervisory cycle; target
sequencing across cycles is the mechanism for navigating the near-singular
low-F regime to a *feasible* target (feasibility verified: LT≈2.79, VB≈3.15 at
F=0.8). LLM stack and modal reasoning policy per ADR 005 amendment
(`reasoning=False` default tool-call mode, `reasoning=True` on Critic revision).
Hard Critic-round limit; fail-fast on unreachable LLM (ADR 010).

### 3.5 Safety gate (C3)
Anomaly detector trained cross-domain on TEP, applied to Column A without
per-plant retraining. Forked-twin counterfactual: a proposal is gated if its
30-min forked trajectory violates any of the 9 pinned safety constraints
(`docs/kpis.md` §3.3). Explicit logged safe-state on block (not a silent
fallback — ADR 010). [Artifact gap: detector built in Phase 4.]

## 4. Experimental Setup

### 4.1 Disturbance scenarios
Canonical 5-scenario set (aggregate IAE) + 16-point off-nominal grid
(off-nominal robustness) + 4-point screening grid for prompt iteration
(`docs/kpis.md` §2). State seeds, N ≥ 10, bootstrap 95 % CI protocol.

### 4.2 KPIs and statistical methodology
The five KPIs (`docs/kpis.md`): aggregate_iae; off_nominal target-acquisition
and disturbance-rejection sub-metrics; constraint intercept + detection rates;
linearization_consistency; supervisory_cycle_wallclock. Bucket classification
requires bootstrap-CI separation, not point estimates (§7.1 resolution). State
the pre-registration: buckets + Methods paragraphs fixed before C2 numbers
exist (`methods_phase3_buckets.md`).

## 5. Results

Select exactly one outcome bucket via the `docs/kpis.md` §6 decision tree;
import the matching paragraph from `methods_phase3_buckets.md` (current pre-run
estimate: Bucket B ~60 %, C ~25 %, A ~15 %). Bucket-B comparison is against the
best gate-passing C1 tuning (×100 Pareto reference), not the under-regularized
×1 — anti-strawman. Report C0/C1/C2/C3 KPI table, the off-nominal grid result,
and the C3 gate case studies. [Artifact gap: C2/C3 runs are Phase 3/4.]

## 6. Discussion

- **MPC-free deployment branch (ADR 009 Option B):** PID + agent vs standalone
  MPC — the operator-economics result; outcome publishable either way.
- **Transfer to non-distillation domains** — architecture argument
  (cross-domain anomaly detector + supervisory abstraction), not empirical
  coverage; enumerate semiconductor / pharma / battery / HVAC / water as
  future targets (§4.3).
- **Limitations and threats to validity** — pull the disclosed set verbatim-in-
  spirit from `pre_submission_checklist.md` §4: high RGA (§4.1), bounded local-
  LLM reasoning (§4.2), single column (§4.3), C0 fixed-gain non-extrapolation
  (§4.4), SIMC 2DoF scenario-mix (§4.5), LV near-singularity (§4.6). Frame as
  anchor points, not apologies.

## 7. Conclusion

Restate the architecture, the selected-bucket headline, and the single
strongest transfer claim. One sentence on the deferred real-data validation
(ADR 008) as the bridge to deployment credibility.

---

## Artifact gap ledger (what must exist before each section is writable)

| Section | Blocking artifact | Phase |
|---|---|---|
| 3.3 | C1 regularization sweep + Figure 9 | done (Phase 2) |
| 3.4 | C2 agent runs (Observer/Optimizer/Critic, real LLM) | Phase 3 |
| 3.5 | TEP-trained anomaly detector | Phase 4 |
| 5 | C2/C3 KPI tables, bucket classification | Phase 3 / 4 |
| 5 | C1 disturbance-rejection-only off-nominal baseline (Bucket-B comparator) | Phase 3 prep — pending |
| 6 (MPC-free) | PID-backend C2 secondary run | Phase 5 ablation |
| 6 (transfer) | none — architecture argument only | writable now |

## SAFEPROCESS 6-page derivation note

The 6-page IFAC version is cut *from* the preprint. Likely casualties for the
page limit: §2 compressed to one paragraph, §6 MPC-free branch and full
limitations list trimmed to the arXiv version, Related Work reduced to the gap
statement. The four-way comparison (3.2), the Pareto-front motivation (3.3),
the selected bucket (5), and the gate case studies (3.5/5) are the load-bearing
content that must survive the cut.
