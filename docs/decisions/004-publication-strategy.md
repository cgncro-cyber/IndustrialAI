# ADR 004 — Publication Strategy: arXiv-First, SAFEPROCESS-Second, Journal Deferred

**Status:** Accepted
**Date:** 2026-05-26

## Context

This project serves two distinct purposes for the owner:

1. **Career positioning** — visible, citable, technically credible evidence of industrial-AI capability for job applications and Wacker / Siemens / Honeywell / industrial-AI-startup interview pipelines.
2. **Academic credentialing** — a peer-reviewed publication contributing to the long-term DBA narrative.

These two purposes have fundamentally different time scales. The career-positioning need is *now* (target: arXiv-live and conference-submitted by Q4 2026). The DBA narrative is paced over years. Optimizing for both in the same publication path would compromise the more urgent one.

The owner's DBA thesis at Walsh College is separately planned around the Health-01 multi-omics project, not this work. This removes the DBA constraint from the IndustrialAI publication strategy.

## Decision

Three-tier publication path, sequenced for speed:

1. **arXiv preprint** (primary deliverable) — posted immediately upon Phase 5 completion. Cross-listed in `cs.LG` and `eess.SY`. This is the *real* career-positioning asset.
2. **SAFEPROCESS 2027 conference paper** (secondary) — 6-page IFAC paper derived from the preprint, submitted by 31 October 2026.
3. **Journal version** (deferred, optional) — only pursued if time allows after the job-search phase concludes. Computers & Chemical Engineering or Journal of Process Control as targets.

## Rationale

### Why arXiv is the primary, not preliminary, deliverable

- Recruiters do not browse Elsevier databases. They Google candidates, scan GitHub, and check LinkedIn. arXiv links surface in all three.
- Time-to-citability: arXiv = same day. Journal = 6–12 months. For a candidate actively applying, this difference dominates everything else.
- Industrial-AI hiring managers care about *whether the work exists and can be inspected*, not about journal-impact-factor signaling.

### Why SAFEPROCESS 2027 specifically

- **Theme alignment**: the conference theme is literally "AI for Safety" — a near-perfect match for the project's novelty claim.
- **Deadline alignment**: 31 October 2026 is 22 weeks from project start, comfortably accommodating the 10–12 week project plan plus buffer.
- **Venue accessibility**: Delft is a 3-hour train ride from Cologne, low-friction for in-person presentation.
- **Audience**: heavily industrial, including process-control practitioners from Shell, BP, BASF, Siemens — exactly the recruiter pool the project targets.
- **Indexing**: IFAC-PapersOnline, Scopus-indexed; reputable citation venue.

### Why ADCHEM 2027 is backup, not primary

- Stronger fit on control theory specifically, but weaker fit on the safety angle.
- Hong Kong venue significantly raises travel cost and visa friction.
- Deadline not yet announced (likely November/December 2026), creating less predictable planning.

### Why journal submission is explicitly deferred

- A journal version would require an additional ~3–4 weeks of ablation studies, extended literature review, and revision cycles to compete in C&CE or JPC.
- Pursuing journal acceptance before the job-search phase concludes adds 6–12 months of dead time during which the work is unpublished to a journal audience but already public on arXiv.
- The marginal value of a journal acceptance over an arXiv preprint plus a SAFEPROCESS proceedings paper is modest in the industrial-AI hiring context.

## Consequences

- Phase 5 effort drops from 3–4 weeks to 2–3 weeks because the journal-grade ablations are deferred.
- The conference paper must be derived *from* the arXiv preprint, not authored separately. The preprint is the source of truth.
- The owner must accept that the work appearing on arXiv and IFAC-PapersOnline — not in a high-impact journal — is the intended deliverable, and that this is appropriate for the job-search goal.
- If a journal submission later happens, it will be a substantially extended version, not a clone of the preprint.

## Reversibility

Medium. Adding a journal submission later is always possible. Removing the SAFEPROCESS commitment after the deadline passes is not.

## Anchor Dates

| Date | Milestone |
|---|---|
| 26 May 2026 | Project start (today) |
| ~10 Aug 2026 | Phase 4 complete (12 weeks from start) |
| ~24 Aug 2026 | Phase 5 complete, internal version ready |
| Sep – Oct 2026 | Buffer / polish / co-author review (if any) |
| 31 Oct 2026 | SAFEPROCESS 2027 paper submission deadline (hard) |
| Same day | arXiv preprint goes live |
| 28 Feb 2027 | SAFEPROCESS acceptance notification |
| 1 Apr 2027 | SAFEPROCESS final paper + early registration |
| 29 Jun – 2 Jul 2027 | SAFEPROCESS 2027 in Delft |
