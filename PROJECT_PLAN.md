# Project Plan — Five Phases

**Hard anchor:** SAFEPROCESS 2027 (Delft) paper submission deadline **31 October 2026**. All scheduling works backward from there.

Total estimated effort: **11–13 weeks** at ~10–15 h/week. With the SAFEPROCESS deadline 22 weeks out, this allows ~9 weeks of buffer for slippage.

Each phase has an **artifact** (the thing that ships) and a **gate** (the criterion that must be met before the next phase starts).

The project follows a strict two-layer hierarchical control architecture — see [ADR 006](./docs/decisions/006-hierarchical-control-architecture.md). The supervisory layer (cadence: 5–15 min) is the variable under study; the regulatory PID layer (cadence: ~1–5 s) is held constant across all four experimental configurations (C0 PID-only / C1 Linear MPC / C2 Agent / C3 Agent + Safety Gate).

---

## Phase 1 — Foundation: Skogestad Column A Dynamic Twin

**Effort:** 8–10 working days (extended by ~3 days vs. original C3/C4 plan due to the case-study pivot — see ADR 007)

**Goal:** A literature-validated Python re-implementation of Skogestad's "Column A" dynamic distillation model, in MIT-licensed code, with regulatory PID layer, setpoint interface, and operating-data logging fully instrumented for the figures in `docs/figures.md`.

**Case study choice:** Skogestad's Column A — 40-stage binary distillation, relative volatility 1.5, 99 % purity products, with LV, DV, and L/D-V/B configurations (see ADR 007). Single column for Phase 1; extension to a two-column direct sequence is a deferred Phase 2 decision-junction if PID and Linear MPC results turn out too close.

### Progress snapshot (Day 8 closed, Phase 1 gate cleared, 2026-05-27)

Day 1–4 deliverables landed in commits `6f88ce9` / `1079c39` / `7e15787`; Day 5 in `aec4b9d` (DV configuration); Day 6 in `efd7add` (LDVB configuration). The Day-4 mini-gate is green: G^LV(0) matches Skogestad 1997 Eq. (31) within 0.01 %, τ₁ within 0.04 %, τ₂/τ₃ within 0.5 %/0.3 %, and three Octave-cross-checked step trajectories (L +1 %, z_F −10 %, F +10 %) match within 1e-6.

Day 8 (the gate-consolidation tag) ships the remaining Phase-1 acceptance items in a single commit:

- `column_a/assumptions.md` — full citation of every modeling assumption (binary mixture, constant α, constant molar overflow, linearized hydraulics + K2 effect, P-only level loops, numerical method choices, what is *not* modeled).
- `column_a/balances.py` + tests — algebraic mass-balance closure check; both overall (F = D + B) and light-component (F·zF = D·y_D + B·x_B) balances close to << 0.1 % at the published SS and at a re-converged perturbed SS at zF = 0.6.
- `test_rate_limiter_divergence_guard.py` — ±20 % rate-limited steps on LT integrate to bounded compositions and re-converge to the Newton SS within 5e-4 (3 cases).
- `simulate.py` + `test_lv_disturbance_scenarios.py` — tick-based LV closed-loop driver with two regulatory PIDs, slew-limited setpoint interface, and full RunLogger integration. Three end-to-end disturbance scenarios (F +20 %, zF −10 %, y_D setpoint +0.5 %) each produce the complete 7-artifact data-logging contract (`timeseries/tray_profile/setpoints.parquet`, `kpis/latency/manifest.json`, `config.yaml`).
- `column_a/operating_window.py` + `tools/run_operating_window_sweep.py` — LV-closed Newton-Krylov sweep with long-time integration fallback, scoped to the realistic ±10 % LT/VB · ±20 % F/zF window. The 1080-point baseline sweep converges 100 % in 82.6 s on the dev machine and is materialized to `data/baseline_operating_window.csv`.
- `notebooks/01_twin_walkthrough.ipynb` — guided tour: parameters → SS profile → open-loop L+1 % step → closed-loop F-step → operating-window summary.
- pyproject.toml: added `N802` to the Skogestad-notation exemption so domain identifiers like `y_D`, `x_B`, `zF` survive ruff.

Test suite at the close of Day 8: **72 pytest cases pass**, twin-layer coverage ≥ 96 %, ruff + format + mypy --strict all clean.

### Deliverables — `src/industrial_ai/twin/column_a/`

Module layout, derived from the published MATLAB code at `https://skoge.folk.ntnu.no/book/1st_edition/matlab_m/cola/` (clean-room re-implementation, citing Skogestad & Morari 1988 IECR 27:1848 and Skogestad 1997 Trans IChemE 75:539):

| Status | Python Module | MATLAB Source | Contents |
|---|---|---|---|
| ✓ | `column_a/model.py` | `colamod.m` | Core nonlinear ODE `f(t, x, u, p) → dx/dt` with 82 states (41 stage compositions + 41 stage holdups). |
| ✓ | `column_a/integrator.py` | `cola4.m` | `scipy.integrate.solve_ivp` wrapper (LSODA or Radau for stiff initialization phase). |
| ✓ | `column_a/steady_state.py` | `cola_init.m` | Newton–Krylov solver for steady-state initialization. |
| ✓ | `column_a/configurations/lv.py` | `cola_lv.m` | LV configuration: P-controllers on D and B level loops. **Phase 1 primary.** |
| ✓ | `column_a/configurations/dv.py` | `cola_dv.m` | DV configuration: P-controllers on D and B level loops, MD→LT and MB→B. |
| ✓ | `column_a/configurations/ldvb.py` | `cola_rr.m` | L/D–V/B double-ratio configuration: LV-style level loops, LT=LR·D, VB=VR·B. |
| ✓ | `column_a/linearize.py` | `cola_linearize.m` | Numerical Jacobian for Phase 2 Linear MPC (do-mpc). |

### Deliverables — supporting infrastructure

- ✓ `src/industrial_ai/twin/regulatory_pid.py` — embedded multi-loop PID, identical across configurations per ADR 006 (top-composition loop, bottom-composition loop, condenser-level loop, reboiler-level loop). Used by all four supervisory configurations C0/C1/C2/C3.
- ✓ `src/industrial_ai/twin/setpoint_interface.py` — uniform setpoint ingress with rate-limiter / ramping logic per ADR 006.
- ✓ `tests/test_column_a_against_matlab.py` — pytest regression against 3–4 published Skogestad reference trajectories (steady-state stage profiles + open-loop step responses to L, V, F, z_F changes).
- ✓ `notebooks/01_twin_walkthrough.ipynb` — guided tour: column construction, initialization, a representative LV-configuration step response.
- ✓ `data/baseline_operating_window.csv` — 1080 logged steady-state points across the LV operating window (100 % convergence via Newton-Krylov with long-time integration fallback).
- ✓ `src/industrial_ai/twin/column_a/assumptions.md` — every modeling assumption documented: binary mixture, constant pressure, constant relative volatility, equilibrium on all stages, total condenser, constant molar flows, no vapor holdup, linearized liquid dynamics with K2-vapor-flow effect, plus all numerical-method choices.
- ✓ `src/industrial_ai/twin/simulate.py` — tick-based LV closed-loop driver (regulatory PIDs + slew-limited setpoint interface + RunLogger integration).
- ✓ `src/industrial_ai/twin/column_a/operating_window.py` + `tools/run_operating_window_sweep.py` — Newton-Krylov sweep with integration fallback for ill-conditioned (LT, VB) combinations.
- ✓ `src/industrial_ai/twin/column_a/balances.py` — algebraic mass-balance closure check (overall + light component).

### Data-logging contract (required for Phases 3–5 figures)

Every run produced by the Phase 1 twin must populate `data/runs/<config>/<scenario>/<seed>/` with the following files per `docs/figures.md` Data-Logging Contract:

- `timeseries.parquet` — high-frequency state for Figure 3
- `tray_profile.parquet` — per-stage compositions and holdups over time for Figure 2
- `setpoints.parquet` — commanded setpoints with timestamps
- `kpis.json` — scalar KPIs for Figures 4 and 7
- `latency.json` — wall-clock per supervisory cycle for Figure 6
- `safety_log.parquet` — anomaly scores and decisions for Figure 5 (C3 only)
- `config.yaml` — full hyperparameter snapshot
- `manifest.json` — input data hashes, model versions, seed

Without this logging contract in place at the end of Phase 1, no later phase's figures can be regenerated.

### Optional Phase 1 add-on (defer-or-include decision at start)

- **CasADi symbolic-gradient layer** (~1.5 additional days). Makes Phase 2 Linear MPC via `do-mpc` significantly cleaner than finite-difference Jacobians. Optional — does not block Phase 2 but recommended if Phase 1 has buffer.

### Gate

- ✓ Pytest regression tests pass against the published Skogestad reference trajectories within agreed numerical tolerances (steady-state values: ±1 %, step-response shape: visual + L2 norm under threshold). *Day-4 mini-gate cleared with margin: G^LV(0) Eq. (31) ±5 % spec → 0.01 % actual; τ₁ ±2 % spec → 0.04 % actual; three Octave trajectories rel ≤ 1e-6.*
- ✓ Twin converges across the full intended LV operating window without manual intervention. *1080-point sweep, 100 % convergence in 82.6 s; small-grid pytest enforces ≥99 % convergence.*
- ✓ Energy and mass balances close to within 0.1 % at steady state. *Both balances close to << 0.1 % at the published SS and at a re-converged zF = 0.6 SS (pytest enforces).*
- ✓ Three independent disturbance scenarios (feed-rate step, feed-composition step, reflux step) run end-to-end and write the full data-logging contract. *F +20 %, zF −10 %, y_D setpoint +0.5 %; each scenario produces the 7-artifact contract and pytest re-opens every artifact.*
- ✓ Setpoint rate-limiter prevents solver divergence on ±20 % step changes. *±20 % LT step at 0.1 kmol/min² slew rate integrates to bounded compositions and re-converges to the Newton SS within 5e-4.*
- ✓ `assumptions.md` lists every modeling assumption with citation back to Skogestad & Morari 1988 / Skogestad 1997 where applicable.

---

## Phase 2 — Baselines: Classical PID and Linear MPC (non-negotiable)

**Effort:** 1.5 weeks

**Goal:** Establish an honest, two-tier baseline. The agent must outperform both the textbook PID-only configuration (C0) *and* a standard industrial linear MPC supervisor (C1) under nonlinear disturbance scenarios. Without the MPC baseline, the paper is vulnerable to the strongest reviewer objection: *"but linear MPC would beat your agent."*

**Decision junction at Phase 2 start:** Evaluate whether the single Column A in LV configuration provides sufficient agent-vs-PID/MPC differentiation. If preliminary C0 and C1 KPIs come too close to leave room for supervisory agent value, extend to a two-column direct sequence (Top product of Column 1 → feed of Column 2). Each Column A is a self-contained integrable instance, so the extension is mechanical at the simulation layer; the agent layer is unaffected.

**Deliverables**
- `src/industrial_ai/control/c0_pid_only.py` — Configuration C0: PID regulatory layer with static manual setpoints. Tuned via relay-feedback / Ziegler–Nichols.
- `src/industrial_ai/control/c1_linear_mpc.py` — Configuration C1: Linear MPC supervisor via the `do-mpc` package, sitting above the same PID regulatory layer. Linearization point from `column_a/linearize.py`.
- KPI definitions (used identically for all four configurations): energy per kg product, yield, constraint-violation count, settling time after disturbance, manipulated-variable activity, integrated absolute error (IAE).
- `notebooks/02_baselines_benchmark.ipynb` — side-by-side comparison of C0 and C1 across all disturbance scenarios.

**Gate**
- All KPIs computed for ≥5 disturbance scenarios across both C0 and C1.
- Results documented with mean ± 95 % CI (deterministic configurations require a single run, but spec the CI machinery here so Phase 5 reuses it).
- C1 (Linear MPC) demonstrably outperforms C0 (PID-only) on at least 3 of the 5 scenarios — otherwise the MPC implementation is broken and Phase 2 is not complete.
- Decision-junction outcome documented (single Column A retained, OR two-column direct sequence adopted with a short addendum to `column_a/assumptions.md`).

---

## Phase 3 — Agentic Controller

**Effort:** 3 weeks

**Goal:** LangGraph multi-agent system that proposes setpoint adjustments based on downsampled twin state, operating at the supervisory cadence defined in ADR 006.

**Architecture**
- `Observer` — reads downsampled, aggregated twin state, summarizes deviations from spec.
- `Optimizer` — proposes setpoint changes against KPI objectives.
- `Critic` — challenges proposals against historical operating envelope.

**Deliverables**
- `src/industrial_ai/agents/` — graph definition, prompts, state schema.
- `notebooks/03_agent_runs.ipynb` — illustrative trajectories.
- Config-driven runs (YAML), all hyperparameters versioned.
- Hard recursion / iteration limits in the LangGraph to prevent runaway loops.

**Gate**
- Configuration C2 (Agent, no safety gate) runs end-to-end without manual intervention for ≥3 disturbance scenarios.
- Output setpoints are within physical limits 100 % of the time (pre-safety-layer).
- Average wall-clock time per supervisory cycle is at least 10× faster than the supervisory cadence (i.e., ≤30–90 s per decision for a 5–15 min cadence).

---

## Phase 4 — Safety Gate with Cross-Domain Validation

**Effort:** 2 weeks

**Goal:** An anomaly detector that filters agent proposals, evaluated via cross-domain transferability to ensure honest generalization claims.

**Methodology (execution order is part of the contribution)**
1. **Primary evaluation — cross-domain.** Train the anomaly detector on established public benchmarks (TEP and/or NoBOOM). Test its zero-shot detection performance on the Skogestad Column A twin under simulated process anomalies (e.g., column flooding, tray drying, feed-composition drift). This is the stronger generalization story and a publishable sub-result either way.
2. **Secondary — in-domain fallback.** If cross-domain transfer fails to detect at acceptable rates, document the limitation transparently as a paper finding and train an in-domain detector on a held-out slice of Column A anomaly data, with strict train/test separation across distinct disturbance profiles. Avoid co-training and co-testing on the same disturbance type — this is the data-leakage trap reviewers will catch.

**Deliverables**
- `src/industrial_ai/safety/anomaly_gate.py`
- Detector(s) trained per the methodology above.
- `notebooks/04_safety_layer.ipynb` — ROC curves, FP/FN analysis, cross-domain detection rates.
- Documented operating-point justification.

**Gate**
- Documented FP/FN trade-off at the chosen operating point.
- Cross-domain transfer result reported (positive or negative) — both are publishable.
- Configuration C3 (Agent + Safety Gate) blocks ≥1 demonstrably-bad agent proposal in a controlled disturbance scenario.

---

## Phase 5 — arXiv Preprint + SAFEPROCESS Submission

**Effort:** 2–3 weeks

**Goal:** Public arXiv preprint as primary career-positioning deliverable; conference paper as secondary credential. A polished journal version is explicitly **deferred** to a later sprint, not blocking the job-search use case.

**Statistical Guardrails (anti-reviewer-kill-switches)**

These are non-negotiable, because LLM outputs are stochastic and a single-run comparison is not falsifiable:

- **Stochastic accounting.** Every C2/C3 disturbance scenario is evaluated over **N independent seed runs** (target: N ≥ 10). KPIs are reported with bootstrap confidence intervals, not point estimates.
- **Practical significance threshold.** Effect-size thresholds are defined *before* running the final experiments — e.g., >5 % energy reduction or >10 % settling-time improvement counts as practically significant; anything below is reported as null. This prevents the "statistically significant but industrially trivial" trap.
- **Effect-size reporting.** Cohen's d (or equivalent) reported alongside p-values for all primary comparisons. Means alone are insufficient.
- **C0 and C1 baselines may use a single run.** They are deterministic — re-running adds no information. Only the stochastic supervisory layers (C2, C3) need replicates.

**Deliverables**
- `src/industrial_ai/evaluation/` — KPI computation, bootstrap CI machinery, statistical tests, plot generation.
- `paper/arxiv_preprint.md` (or `.tex`) — full preprint, paper-grade.
- `paper/safeprocess_paper.md` — 6-page IFAC conference version derived from the preprint.
- `make reproduce` — single command that regenerates every figure.
- arXiv submission (cs.LG + eess.SY cross-listing).
- SAFEPROCESS 2027 submission via PaperCept.

**Gate**
- arXiv preprint live with a DOI-equivalent identifier.
- SAFEPROCESS submission confirmation received before 31 October 2026.
- Every reported effect comes with a CI and a Cohen's d.
- LinkedIn post + CV update with arXiv link.

**Explicitly deferred (post-job-search, optional)**
- Journal version with additional ablation studies for *Computers & Chemical Engineering* or *Journal of Process Control*.

---

## Conference Targeting — Locked-In

| Venue | Date | Deadline | Fit |
|---|---|---|---|
| **SAFEPROCESS 2027** (primary) | 29 Jun – 2 Jul 2027, Delft NL | **31 Oct 2026** | Conference theme "AI for Safety" matches the novelty claim exactly |
| ADCHEM 2027 (backup) | 13–16 Jul 2027, Hong Kong | TBA (likely Nov/Dec 2026) | Strong process-control fit; weaker safety angle; expensive travel |

Decision rationale: see `docs/decisions/004-publication-strategy.md`.

---

## Key Decisions Already Locked

- Simulator: **IDAES for steady-state and property packages**, `scipy.integrate` for dynamic column ODE — see `docs/decisions/001-idaes-over-dwsim.md` (refined 2026-05-26) and `docs/decisions/007-skogestad-column-a-over-c3c4-train.md`.
- Case study: **Skogestad's Column A dynamic distillation benchmark**, single column for Phase 1, extensible to two-column direct sequence — see `docs/decisions/007-skogestad-column-a-over-c3c4-train.md` (supersedes ADR 002).
- Novelty claim: **safety-gated agentic control**, not "agent steers column" — see `docs/decisions/003-novelty-positioning.md`.
- Publication path: **arXiv preprint as primary, SAFEPROCESS as secondary, journal deferred** — see `docs/decisions/004-publication-strategy.md`.
- Local LLM: **Llama-3.3-Nemotron-Super-49B v1.5** primary, **Qwen3.6-27B** ablation, on LM Studio — see `docs/decisions/005-local-model-selection.md`.
- **Two-layer hierarchical control architecture** with four-way comparison C0/C1/C2/C3 — see `docs/decisions/006-hierarchical-control-architecture.md`.

## Open Decisions

- Whether to release trained anomaly-detector weights publicly or only training code.
- Whether to submit an invited-session proposal for SAFEPROCESS 2027 (deadline 30 Sep 2026, requires coordinating ≥6 thematically coherent papers — likely too ambitious for a first submission, default: regular session).
- Exact value of N for seed replicates (target ≥ 10, finalize after Phase 3 wall-clock measurements).
- Whether to include the optional CasADi symbolic-gradient layer in Phase 1 (defer-or-include decision at Phase 1 start).
- Whether Phase 2 promotes the twin to a two-column direct sequence (decision-junction at Phase 2 start, based on C0/C1 baseline differentiation).
