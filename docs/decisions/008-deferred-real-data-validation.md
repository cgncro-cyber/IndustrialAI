# ADR 008 — Deferred Phase 6: Industrial Validation via Anonymized Plant Data

**Status:** Accepted (deferred — not active until Phase 5 closes)
**Date:** 2026-05-27

## Context

The Phase 1–5 plan establishes the methodology — safety-gated agentic supervisory control on top of a classical regulatory PID stack per ADR 006 — and validates it on Skogestad's Column A, a canonical public benchmark per ADR 007. This is sufficient for the SAFEPROCESS 2027 submission and the primary career-positioning arXiv preprint per ADR 004.

A follow-on real-data validation would qualitatively elevate the contribution from *"methodology on a public benchmark"* to *"methodology with industrial validation"*, which is a strong differentiator for downstream career positioning, journal-version expansion, and methodology credibility in industrial-AI hiring conversations. It is, however, a high-friction undertaking with non-trivial career, IP, and approval-process risks if executed at the wrong point in time or via the wrong channel.

The original `CLAUDE.md` hard boundary — *"No Momentive proprietary data, parameters, catalysts, or process details. Ever. Public literature only."* — was set to protect the Phase 1–5 scope from IP entanglement, accidental disclosure, and approval-process drag that would jeopardize the SAFEPROCESS deadline. That rationale remains correct for Phase 1–5.

For a deferred Phase 6 executed after the Phase 5 arXiv preprint is public, the trade-offs change materially:

- The methodology is already published and IP-clean, independent of any second-party data.
- The negotiation position with any data-providing plant shifts from *"please give me data for an idea"* to *"please validate this published method on your plant"*.
- Career-positioning risk drops because the first paper is already a credential.
- Approval-process drag is no longer on the critical path to the SAFEPROCESS deadline.

## Decision

A deferred **Phase 6 — Industrial Validation** is added to the plan, governed by this ADR, under the following preconditions and contract.

### Preconditions

1. The Phase 5 arXiv preprint is publicly live with a stable identifier.
2. The SAFEPROCESS submission has been completed.
3. No Phase 6 work begins until both of the above are met.

### Sourcing strategy

Approach is **local-operational** (e.g., plant manager / Werksleiter direct), not via corporate R&D, strategy, or central legal. A symmetric barter is offered: fully anonymized historized data of one column in exchange for an **Advisory-Mode controller trial** on the same plant as Gegenleistung. No co-authorship is sought or accepted from the data provider.

### Scope of data

A single column, four weeks of historized continuous operation, with at least two documented disturbance events. Required tags: compositions (online analyzer or interpolated lab values), flows, temperatures, levels, valve positions, setpoints, mode and operator-action markers. Sampling rate ≤ 1 min for regulatory loops, ideally < 10 s for composition loops. Sensor calibration status per tag.

### Anonymization standard (binding floor, not aspirational ceiling)

Before any data enters the repository:

- Strip all material identities, catalyst references, exact pressures and temperatures, plant location, customer or product names, capacity figures, and trade-named products.
- Disguise the unit as *"a continuous binary separation column in chemical manufacturing"*. No *"specialty chemicals"* or *"silicones"* framing anywhere in the manuscript, code, or supplementary material.
- Normalize variables; publish ratios and time constants in absolute form, but no absolute compositions, flows, or temperatures.
- Run an automated sanitization pipeline that enforces these rules as a precondition for the data crossing into version control.

### Co-authorship

Explicitly declined. The data provider remains unnamed and has no editorial role on the manuscript. This keeps IP allocation clean and prevents retroactive co-ownership claims on any methodology refinement done with their data.

### Documentation of consent

A short post-conversation email exchange documenting the agreement is the minimum written record. No formal NDA or Data Use Agreement is sought, because the data scope is intentionally below the threshold that would require one: no live access, no proprietary identifiers retained, fully anonymized historized data only, no continuing relationship implied. A template email is committed to the repository at Phase 6 kickoff.

### Internal-data-owner verification

Before any data flows, the plant manager confirms in writing that they have the authority to release the requested data scope, or names the additional internal approval needed. The project does not proceed on assumed authority. If corporate-level approval is required and granted, the same anonymization standard applies; only the email-trail expands to include the corporate sign-off.

### Gegenleistung — Advisory Mode only

The reciprocal benefit offered to the data provider is an **Advisory-Mode controller trial**: parallel hardware reads plant tags and computes setpoint recommendations, which are written to a comparison log only. Nothing is written back to the DCS; operators retain full control. This avoids triggering HAZOP re-review, IEC-61511 SIL re-classification, OT/IT integration commitments, or any safety-recertification overhead.

**Closed-loop deployment is explicitly out of scope** for the Gegenleistung commitment. Closed-loop deployment is a multi-quarter engineering project with compliance implications that could outlive the author's tenure at the data provider and is therefore not offered as part of the barter.

### Pre-submission courtesy review

The data provider sees the manuscript before submission. This is a relationship safeguard, **not** an approval gate. The anonymization standard above ensures there is nothing the data provider could legitimately object to on confidentiality grounds; the review exists to preserve goodwill and protect the data provider from retroactive exposure.

### Current-employer special case

If the data provider is the author's current employer at the time of Phase 6 execution, an additional check against the employment contract for moonlighting clauses, IP-assignment clauses, and external-publication-approval clauses is required before the email exchange documenting consent is sent.

## Rationale

- **Sequenced timing is the core insight.** Approaching a plant manager *after* the first paper is public reverses the asking position. The data provider sees validated methodology, working code, and reviewer-positive momentum, not a vague request. Approval likelihood and quality of cooperation both improve.
- **Local-operational sourcing avoids the corporate-approval-chain death-spiral.** Werksleiter-level decisions can be made in weeks; corporate-level decisions routinely take months and die in legal review. The local route is fast or fails fast — both are acceptable.
- **Advisory Mode contains the long-tail commitment risk.** Closed-loop edge deployment is a multi-quarter engineering project. Advisory Mode delivers the validation evidence at < 10 % of the engineering effort and zero safety-recertification overhead.
- **No co-authorship keeps IP allocation clean.** With no author from the data provider, no employer can later claim IP entanglement based on the data exchange.
- **Anonymization is binding because the readership is small.** In specialty chemicals, loose anonymization risks retroactive identification of the data provider. The standard above is the floor.

## Consequences

- The `CLAUDE.md` hard boundary on company-specific data is amended: the *"Ever"* prohibition applies to Phase 1–5 absolutely. Phase 6 is governed by this ADR and may use anonymized data under the contract above.
- Phase 6 produces its own artifact — either an arXiv v2 expansion of the Phase 5 preprint or a separate journal-version manuscript. The Phase 5 arXiv record is not modified retroactively.
- Phase 6 inherits the supervisory architecture (agent, safety gate, KPIs, statistical guardrails) from Phase 1–5 unchanged. Only the regulatory PID layer, the linear model, and the operating-point context are re-identified from the real-plant data.
- The Phase 1 `column_a/assumptions.md` does not change. A separate `phase6/assumptions.md` documents the real-plant context with the same rigor, but anonymized.
- If the data provider's identity becomes inferable post-publication despite the anonymization standard, the manuscript is amended on arXiv and the data provider is notified. This is treated as a defect to be corrected, not a relationship-ending event.

## Fallback options

If Phase 6 cannot be sourced within Q1 2027:

- **Public process-control datasets.** Tennessee Eastman Process (already in the safety-gate training pipeline), Eastman Pittsburgh Challenge, NoBOOM. Lower glamour than real-plant data, but publication-safe.
- **Academic-partner sourcing.** A chemical-engineering chair (TU Dortmund DYNAMICS, RWTH Aachen AVT, TU München) may hold anonymized industrial datasets cleared for academic use. Adds a co-authorship dimension but moves the IP locus to academia, which is cleaner than a current-employer route.
- **Skip Phase 6 entirely.** The Phase 5 deliverable stands as the primary credential independently. Phase 6 is upside, not load-bearing.

## Reversibility

High. The decision is *"defer until Phase 5 closes"*, and Phase 6 may be abandoned at that decision point without affecting any earlier deliverable. Until Phase 5 is complete, this ADR has no operational impact on the active project.
