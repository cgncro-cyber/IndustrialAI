# ADR 001 — IDAES over DWSIM as Process Simulator

**Status:** Accepted (refined 2026-05-26 — see Refinement section below)
**Date:** 2026-05-26

## Context

The project requires a process simulator that integrates with a Python-native agentic control layer (LangGraph) and anomaly-detection stack (PyTorch / scikit-learn). Two realistic options were considered:

1. **DWSIM** — open-source, Windows-native, COM-API access from Python.
2. **IDAES** — open-source, Python-native, equation-oriented, Pyomo-based, DOE/NREL-backed.

## Decision

**IDAES**, with optional DWSIM screenshots in README/paper for visual storytelling only.

## Rationale

| Criterion | DWSIM | IDAES | Winner |
|---|---|---|---|
| Native platform fit (macOS dev) | Windows-only, needs VM | macOS/Linux native | IDAES |
| Integration with Python ML/agent stack | Indirect via COM bridge | Direct, in-process | IDAES |
| Academic credibility | Moderate | High (DOE-funded, frequent in CACE) | IDAES |
| Reproducibility for peer review | Requires Windows reproducer | Pure Python, `pip install` reproducible | IDAES |
| Out-of-the-box GUI visuals | Strong | Weak | DWSIM |
| Learning curve | Flat (GUI) | Steeper (equation-oriented) | DWSIM |

The integration friction of DWSIM (COM bridge, Windows dependency, file-mediated handoff) would cost more time than IDAES's learning curve, and would significantly degrade the reproducibility story the paper needs to make.

## Consequences

- All twin code lives in pure Python and is unit-testable with pytest.
- The reproducibility story is strong: `pip install -e .` plus `idaes get-extensions` and the entire pipeline runs.
- We accept the cost of building our own plots and visualizations rather than using DWSIM's GUI.
- A small set of DWSIM screenshots may be added to the README purely for visual communication; they are not part of the experimental pipeline.

## Reversibility

Low. Switching to DWSIM later would require rewriting the twin layer and the entire data pipeline.

---

## Refinement (2026-05-26) — Scope Boundary for Dynamic Distillation

This ADR is **refined**, not overturned, by the following clarification:

**IDAES is retained as the project's chosen process simulator for steady-state work and property-package usage. For *dynamic* distillation column simulation, the project uses `scipy.integrate` on top of a self-contained Python re-implementation of Skogestad's Column A model (see ADR 007).**

### Why this refinement was needed

During Phase 1 implementation, inspection of `idaes/models_extra/column_models/tray_column.py` revealed that IDAES `TrayColumn` is hard-coded `dynamic=False`:

> `doc="Indicates whether this model will be dynamic or not, **default** = False. Tray column units do not support dynamic behavior."`

IDAES Issue #96 documents this as a pending enhancement that has not been resolved for years. Because the paper's KPIs (settling time, recovery time, disturbance trajectories) and the SAFEPROCESS theme *"AI for Safety"* both depend on dynamic simulation, the project cannot accept this framework limitation for its distillation case study.

### What is preserved from the original decision

- **Python-native, macOS-compatible, `pip install`-reproducible.** `scipy.integrate.solve_ivp` satisfies all three.
- **No regression toward DWSIM, Aspen Dynamics, or other closed/Windows-bound tools.** The dynamic Column A port runs in the same Python process as the agent, anomaly detector, and evaluation layer.
- **IDAES remains available for any steady-state lookup, property-package access, or future non-distillation unit operations** (reactors, heat exchangers, flash drums) the project may need.

### Why this refinement is consistent with ADR 001's original rationale

The original decision criteria (native platform fit, Python integration, academic credibility, reproducibility) all favored IDAES *because they are properties of the Python-native equation-oriented ecosystem*, not properties of IDAES specifically. `scipy.integrate` and the Pyomo ecosystem (of which IDAES is a part) share these properties. Using `scipy.integrate` for the dynamic column ODE is a *narrower* tool choice within the same ecosystem, not a departure from it.

### Reversibility of the refinement

High. If IDAES Issue #96 is resolved in a future IDAES release and dynamic `TrayColumn` becomes a tested capability, migrating the Column A model into native IDAES is a localized refactor of the twin module that does not touch the agent, safety gate, or evaluation layers.
