# ADR 007 — Skogestad's Column A over C3/C4 Distillation Train

**Status:** Accepted
**Date:** 2026-05-26
**Supersedes:** ADR 002 (C3/C4 over chlorosilane)
**Refines:** ADR 001 (IDAES over DWSIM)

## Context

Phase 1 implementation of the C3/C4 distillation train (per ADR 002) revealed a framework-level blocker that invalidates the original case-study choice.

**The chain of discoveries:**

1. **Property-package side-track.** First attempt used IDAES `HC_PR` (Hydrocarbon Peng-Robinson) with `TrayColumn`. Initialization failed at the feed tray. Root cause: `HC_PR` is built for flash calculations, not coupled rigorous distillation — no IDAES test combines `HC_PR` with `TrayColumn`. Required writing a custom C2–C4 Modular Generic property package following the `BT_PR` (binary benzene/toluene PR) architecture pattern, with component data from Reid/Prausnitz/Poling 4th ed. (1987).

2. **Custom property package partially worked.** The custom PP was accepted by the feed tray, but `TrayColumn` then failed at the condenser-init step. This failure mode matches IDAES Issue #96 ("Improvements to distillation unit model"), which documents the missing linear-temperature-profile initialization for multi-component/cubic-EOS columns — listed as a pending enhancement for years without resolution.

3. **Framework-level finding.** Inspecting `idaes/models_extra/column_models/tray_column.py` lines 55–61 revealed the root issue:

   > `"dynamic": ... doc="Indicates whether this model will be dynamic or not, **default** = False. Tray column units do not support dynamic behavior."`

   **IDAES `TrayColumn` is hard-coded steady-state-only.** This is not a limitation of our property package — it is a framework constraint that affects *every* distillation case study one could attempt in IDAES, including HDA, depropanizer, debutanizer, Petlyuk, or dividing-wall configurations.

4. **Implication for the paper.** All KPIs in PROJECT_PLAN Phase 5 depend on dynamic simulation: settling time, recovery time, integrated absolute error, disturbance-rejection trajectories, safety-gate timeline. Without dynamics, the SAFEPROCESS 2027 theme *"AI for Safety"* collapses to a steady-state feasibility analysis, which is not publishable as a safety-control paper.

## Decision

**Replace the C3/C4 distillation train case study with Skogestad's "Column A" benchmark, implemented as a self-contained Python port of the original MATLAB code.**

Concretely:

- **Process model:** Re-implementation of `colamod.m` (nonlinear dynamic) and supporting MATLAB files from `https://skoge.folk.ntnu.no/book/1st_edition/matlab_m/cola/` in Python.
- **Simulator runtime:** `scipy.integrate.solve_ivp` (LSODA or Radau for stiff init phase) for the column ODE; CasADi optional in Phase 2 for symbolic gradients to support Linear MPC.
- **IDAES retained** for steady-state property-package usage where helpful (per refined ADR 001), but *not* used for the distillation column itself.

## Rationale

### Why Skogestad's Column A

| Criterion | Custom C3/C4 in IDAES | Skogestad Column A (Python port) | Winner |
|---|---|---|---|
| Dynamic simulation possible | ✗ (IDAES Issue #96) | ✓ (literature-validated dynamic ODE) | Skogestad |
| Reviewer recognition | Generic depropanizer/debutanizer | *The* canonical distillation control benchmark since 1988 | Skogestad |
| Literature validation anchor | None — would be first PR+TrayColumn+C2–C4 in IDAES | Published reference trajectories in Skogestad & Morari 1988, Skogestad 1997 | Skogestad |
| Phase 1 implementation effort | Indeterminate — blocked by Issue #96 | ~8 working days (Claude Code feasibility recherche) | Skogestad |
| Reproducibility | High (if it converged) | Higher (well-known model, citable reference trajectories) | Skogestad |
| Multivariable control complexity | 2-column train, but framework-blocked | 4×4 MIMO single column, documented PID suboptimality | Comparable |
| Cited by SAFEPROCESS-relevant prior work | Sparse | Hundreds of process-control papers cite Column A | Skogestad |

### Why not the alternatives considered

- **Quasi-steady-state cascade in IDAES.** Rejected — incompatible with the *"AI for Safety"* theme. A safety gate that cannot reason about trajectories is a feasibility check, not an anomaly detector.
- **Custom dynamic distillation in Pyomo.DAE on top of IDAES property packages.** Rejected for Phase 1 — 2–4 weeks of framework-level engineering with high convergence risk, and no literature anchor for validation. Remains a viable extension path if needed later.
- **Pivot to Tennessee Eastman Process (TEP) Python port.** Rejected — exits the distillation case-study storyline entirely, requires rewriting the case-study narrative across all paper sections, and weakens the relationship to ADR 002's distillation framing. Strong alternative if Column A turns out to be insufficient, but not the first choice.
- **Adopting an existing Python port of Column A.** Rejected after Claude Code feasibility recherche: of four candidates evaluated (`marcosfelt/distill.py`, `alchemyst/Skogestad-Python`, `a1pat/Distillation`, `dejac001/distillation`), none satisfies the joint criteria of (a) 40 stages, (b) LV configuration with liquid-flow dynamics, (c) MIT-compatible license, (d) literature-validated trajectories, (e) Pyomo-compatible numerical stack. The closest candidate (`marcosfelt`) is 30 stages, GEKKO-based, unlicensed, and uncited.

### Why single-column Column A is enough complexity

Phase 1 starts with a single Column A in LV configuration. This is a 4×4 MIMO system (manipulated: L, V, D, B; controlled: y_D, x_B, M_D, M_B) with documented control challenges at the 99% purity specifications. Skogestad's own work (Skogestad & Morari 1988, IECR 27:1848) and subsequent literature establish that decentralized PID achieves "reasonable but not optimal" performance — leaving genuine room for MPC and supervisory agent control to demonstrate value.

If Phase 2 baseline results show PID and Linear MPC performance too close to leave room for agent differentiation, the Column A architecture extends naturally to a two-column direct sequence (Top product of Column 1 → feed of Column 2), preserving the "distillation train" framing of the original ADR 002 while staying entirely within validated Skogestad-pattern modeling. This extension is deferred to a Phase 2 decision junction, not pre-committed.

## Consequences

### What changes

- **Phase 1 deliverables** (PROJECT_PLAN) shift from `c3c4_train.py` + custom PR property package to a `column_a/` Python module re-implementing the Skogestad MATLAB code, with explicit tests against published reference trajectories.
- **ADR 002 is marked Superseded** by this ADR. The strategic argument for choosing a non-silicones case study (audience reach, no Momentive-data risk) still applies — Skogestad's Column A is *more* generic and *more* audience-reaching than C3/C4, not less.
- **ADR 001 is refined**, not overturned. IDAES remains the project's chosen simulator for steady-state work and property packages where useful. Dynamic distillation moves to `scipy.integrate` because IDAES `TrayColumn` does not support dynamics. This is consistent with ADR 001's original spirit (Python-native, reproducible, macOS-compatible).
- **CLAUDE.md Section 5** architecture table updates the case-study row and adds clarification to the process-simulator row.
- **`docs/figures.md`** data-logging paths rename from `c3c4_train` to `column_a`. The hierarchical-control architecture (ADR 006) and the eight-figure plan are otherwise unchanged.

### What does not change

- **Novelty positioning** (ADR 003) — unchanged. The contribution is safety-gated agentic control; the chemistry is the demonstration vehicle.
- **Publication strategy** (ADR 004) — unchanged. arXiv preprint primary, SAFEPROCESS 2027 secondary.
- **Local LLM selection** (ADR 005) — unchanged.
- **Hierarchical control architecture** (ADR 006) — unchanged. The supervisory layer (5–15 min cadence) and regulatory PID layer (~1–5 s cadence) apply identically to Column A as they did to the C3/C4 train.
- **Four-way configuration comparison** C0 / C1 / C2 / C3 — unchanged.
- **Phase 5 statistical guardrails** (N ≥ 10 seeds, bootstrap CIs, Cohen's d, practical-significance thresholds) — unchanged.
- **All Phase 2–5 deliverables** — unchanged in spirit; only the underlying twin model changes.

### Validation anchor

The port's correctness is anchored against published Skogestad/Morari results. The pytest suite in `tests/test_column_a_against_matlab.py` regresses against:
- Steady-state stage compositions and holdups at the nominal operating point (Skogestad & Morari 1988, IECR 27(10):1848–1862, Tables therein).
- Open-loop step responses to L, V, F, and z_F changes (Skogestad 1997 Trans IChemE 75:539–562, Section 3 reference trajectories).

This makes twin validation citable and reviewer-defensible from the start — a stronger position than a self-validated custom IDAES configuration would have offered.

### Licensing

Skogestad's MATLAB code is published as educational supplementary material to the book *Multivariable Feedback Control* (Wiley, 1996, 2005). A clean-room Python re-implementation derived from the published equations (not line-by-line from the MATLAB source) under MIT license, with explicit attribution to Skogestad & Morari 1988, Skogestad 1997, and Skogestad & Postlethwaite 1996, is standard practice in the IFAC community. An e-mail confirmation from the NTNU Process Systems Engineering Group is sent in parallel for documentation purposes; the port proceeds while the response is pending, with the understanding that any clarification from NTNU triggers a license-section update before publication.

## Reversibility

**Medium.** The Column A architecture is process-agnostic at the supervisory level — the agent and safety gate operate against an abstract twin interface (`step(setpoints, disturbances) → state, kpis`). Switching to a different process simulator later (TEP, custom Pyomo.DAE column, or a future dynamic IDAES TrayColumn once Issue #96 is resolved) requires reimplementing only the twin module, not the agent, safety gate, or evaluation layers.

## Sources

- IDAES Issue #96 — *Improvements to distillation unit model*: https://github.com/IDAES/idaes-pse/issues/96
- IDAES `tray_column.py` source — `dynamic=False` hard-coded: https://github.com/IDAES/idaes-pse/blob/main/idaes/models_extra/column_models/tray_column.py
- Skogestad's Column A — NTNU page with MATLAB files: https://skoge.folk.ntnu.no/book/1st_edition/matlab_m/cola/cola.html
- Skogestad, S. & Morari, M. (1988). *Understanding the Dynamic Behavior of Distillation Columns*. Ind. Eng. Chem. Res. 27(10):1848–1862.
- Skogestad, S. (1997). *Dynamics and Control of Distillation Columns — A Tutorial Introduction*. Trans IChemE 75(A):539–562.
- Skogestad, S. & Postlethwaite, I. (1996). *Multivariable Feedback Control: Analysis and Design*. Wiley.
- Claude Code Python-port feasibility recherche (internal artifact, 2026-05-26) — evaluated `marcosfelt/distill.py`, `alchemyst/Skogestad-Python`, `a1pat/Distillation`, `dejac001/distillation`; none meets the Phase 1 quality criteria.
