# ADR 002 — C3/C4 Distillation Train over Chlorosilane Surrogate

**Status:** **Superseded by [ADR 007](./007-skogestad-column-a-over-c3c4-train.md)** (2026-05-26)
**Original Date:** 2026-05-26

> **Note (2026-05-26):** This ADR is superseded. The C3/C4 distillation train was found to be infeasible as a *dynamic* case study because IDAES `TrayColumn` is hard-coded `dynamic=False` (IDAES Issue #96), affecting any multi-component distillation case study including C3/C4, HDA, depropanizer, and debutanizer. The case study has been replaced with Skogestad's Column A benchmark per ADR 007. The strategic argument below (broad audience reach, non-silicones framing, no proprietary-data risk) still applies and is satisfied even more strongly by Skogestad's Column A, which is *the* canonical distillation control benchmark.

---

## Context

The case study must be (a) credible to industrial reviewers, (b) reproducible from public data, (c) maximally reachable for the project's career-positioning purpose. Two candidates:

1. **C3/C4 distillation train** (depropanizer + debutanizer) — petrochemical standard, taught worldwide, abundant comparative literature.
2. **Close-boiling chlorosilane surrogate** — niche, deeply tied to owner's prior domain (silicones), minimal existing repos for differentiation.

## Decision (Superseded)

**C3/C4 train**, with one optional appendix reference to the methodology's applicability to close-boiling separations (no chlorosilane data used).

## Rationale (Superseded but Still Informative)

The strategic intent of the project is to broaden the owner's industrial-AI positioning *away from silicones*. A chlorosilane case study would reinforce a domain identity that the owner is actively moving away from. The C3/C4 train:

- Reaches an estimated 10–20× larger audience (refineries, NGL, petrochemicals).
- Has dense comparative literature for honest benchmarking.
- Carries no proprietary-data risk relative to former employer.
- Reinforces the framing that the **methodology is the contribution**, with the chemistry as an interchangeable demonstration.

## Why This Was Superseded

The C3/C4 train requires *dynamic* multi-component distillation with rigorous thermodynamics. Investigation during Phase 1 implementation revealed:

1. IDAES `TrayColumn` does not support dynamic simulation (`dynamic=False` hard-coded, IDAES Issue #96 pending for years).
2. The custom Peng-Robinson property package for C2–C4 was workable but stalled at condenser initialization, consistent with the documented IDAES limitation.
3. The strategic objection above does not point at C3/C4 specifically — it points away from chlorosilanes. Any sufficiently audience-reaching distillation benchmark satisfies the argument equally well.

Skogestad's Column A satisfies the original strategic argument *more* strongly (it is the most-cited distillation control benchmark in the IFAC community), supports dynamic simulation, has literature-validated reference trajectories, and removes the framework-level blocker. Full rationale: ADR 007.

## Consequences (Superseded)

- No silicones-specific terminology, parameters, or framing appears in public deliverables. *(Still applies under ADR 007.)*
- The owner's chlorosilane domain experience remains an *interview-stage* asset, not a *paper-stage* one. *(Still applies under ADR 007.)*
- We accept reduced uniqueness in exchange for broader audience reach. *(Still applies — Skogestad Column A is even broader.)*

## Reversibility

The strategic argument *away from chlorosilanes* is locked. The specific case-study choice (C3/C4 → Skogestad Column A) is reversible at moderate cost and has now been exercised once.
