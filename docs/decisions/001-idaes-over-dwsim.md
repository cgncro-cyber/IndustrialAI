# ADR 001 — IDAES over DWSIM as Process Simulator

**Status:** Accepted
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
