# ADR 006 — Hierarchical Control Architecture: Agent as Supervisory Layer

**Status:** Accepted
**Date:** 2026-05-26

## Context

Local LLM inference on the Mac Studio (Llama-3.3-Nemotron-Super-49B v1.5, ADR 005) operates on the order of seconds per agent decision. A naive paper framing — *"LLM agent controls distillation column"* — would invite immediate rejection from IFAC reviewers on the grounds that no LLM, local or remote, can manipulate valve actuators at the millisecond cadence required for stable closed-loop process control.

The industrial reality of model-based process control since the 1980s is a strict **two-layer hierarchy**:

- A **regulatory layer** (classical PID, single-loop, fast, deterministic) manipulates the actual physical actuators.
- A **supervisory layer** (advanced control: linear MPC, nonlinear MPC, RTO, today increasingly RL or LLM agents) computes optimal *setpoints* for the regulatory layer at a much slower cadence.

This architecture is the technical and rhetorical foundation that lets the paper sidestep the latency objection entirely.

## Decision

The project implements a strict two-layer hierarchical control architecture. The LangGraph-based multi-agent system operates exclusively as a **supervisory controller**, computing setpoints at a 5–15 minute cadence. The regulatory layer is a classical PID stack running at a few-seconds cadence, identical across all baseline and agent configurations.

```
┌──────────────────────────────────────────────────────────────────────┐
│  Supervisory Layer (cadence: 5–15 min)                               │
│  ────────────────────────────────────────────────────────────────    │
│   PID-only*  │  Linear MPC  │  Agent  │  Agent + Safety Gate         │
│   (baseline) │  (baseline)  │ (ours)  │  (ours, full system)         │
└──────────────────────────────────────────────────────────────────────┘
                                  │
                                  │ Setpoints (rate-limited / ramped)
                                  ▼
┌──────────────────────────────────────────────────────────────────────┐
│  Regulatory Layer: classical multi-loop PID (cadence: ~1–5 s)        │
│  Identical across all four supervisory configurations.               │
└──────────────────────────────────────────────────────────────────────┘
                                  │
                                  │ Manipulated variables (valve positions, flows)
                                  ▼
┌──────────────────────────────────────────────────────────────────────┐
│  Process Twin: IDAES dynamic flowsheet of C3/C4 distillation train   │
└──────────────────────────────────────────────────────────────────────┘

* PID-only configuration uses fixed manual setpoints. It is the
  do-nothing baseline that quantifies the value of any supervisory layer.
```

## The Four-Way Comparison

The supervisory layer is the only thing that varies across experimental configurations. The regulatory PID stack, the process twin, the disturbance scenarios, and the KPI definitions are all held constant.

| Configuration | Supervisory Layer | What it Demonstrates |
|---|---|---|
| **C0 — PID-only** | Fixed setpoints, no supervisor | Baseline performance of the regulatory layer alone |
| **C1 — Linear MPC** | `do-mpc` linear MPC | Industrial state-of-the-art baseline |
| **C2 — Agent** | LangGraph agentic controller | Agent vs. industrial baseline |
| **C3 — Agent + Safety Gate** | LangGraph + anomaly-detection gate | Full proposed system |

The paper's central claims live in the C2 vs. C1 and C3 vs. C2 comparisons.

## Rationale

- **Latency decoupling.** A 5–15 minute supervisory cadence is two to three orders of magnitude slower than typical distillation-train time constants for setpoint changes. LLM inference latency of even tens of seconds is negligible at this cadence. This is the entire point of the hierarchy and it must be stated explicitly in the paper's Methods section.
- **Industrial credibility.** This mirrors the topology of every deployed APC system from Honeywell DMC and AspenTech DMCplus onward. Reviewers recognize it; defending it requires no exotic claims.
- **Fail-safe behavior.** If the agent crashes, returns no proposal, or violates the safety gate, the regulatory layer simply holds the last accepted setpoint. The plant remains stable. No degraded mode involves the agent directly manipulating actuators.
- **Apples-to-apples comparison.** Holding the regulatory layer constant across configurations isolates the supervisory-layer effect cleanly. Any KPI difference between C0/C1/C2/C3 is attributable to the supervisory choice alone.

## Consequences

- **IDAES twin must implement both layers explicitly.** The dynamic simulation includes the PID regulatory stack as part of the twin itself. Setpoint inputs to the twin are accepted from any of the four supervisory configurations through a uniform interface.
- **Observer agent reads downsampled, aggregated state.** Not raw high-frequency live feeds. Typically the most recent 5–15 minutes of state are summarized into a structured snapshot for the Observer prompt.
- **Setpoint trajectories must be rate-limited / ramped.** Abrupt setpoint jumps from supervisor to regulatory layer cause IPOPT solver instability in IDAES dynamic mode and are also physically unrealistic. A rate limiter sits at the handoff between supervisor and regulatory layer in all configurations.
- **Linear MPC baseline is a deliverable, not optional.** See PROJECT_PLAN Phase 2.
- **Computational cost stays bounded.** Even with N seed replicates per scenario (Phase 5), a 5–15 min supervisory cadence over a few hours of simulated process time keeps total LLM calls per scenario in the tens, not thousands.

## Reversibility

Low. Switching to a direct-actuator agent architecture would invalidate the entire paper framing, the Phase 2 MPC baseline rationale, and the latency-defense argument with reviewers. Decision is permanent for this paper.
