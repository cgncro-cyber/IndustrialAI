# Project Plan — Five Phases

**Hard anchor:** SAFEPROCESS 2027 (Delft) paper submission deadline **31 October 2026**. All scheduling works backward from there.

Total estimated effort: **10–12 weeks** at ~10–15 h/week. With the SAFEPROCESS deadline 22 weeks out, this allows ~10 weeks of buffer for slippage.

Each phase has an **artifact** (the thing that ships) and a **gate** (the criterion that must be met before the next phase starts).

---

## Phase 1 — Foundation: Process Twin

**Effort:** 2 weeks

**Goal:** A reproducible IDAES model of a C3/C4 distillation train (depropanizer + debutanizer in series), with deterministic operating data logging.

**Deliverables**
- `src/industrial_ai/twin/c3c4_train.py` — IDAES flowsheet
- `notebooks/01_twin_walkthrough.ipynb` — guided tour for reviewers
- `data/baseline_operating_window.csv` — ≥1000 logged steady-state points
- Documentation of every assumption (feed composition, column dimensions, thermo package)

**Gate**
- Twin converges across the full intended operating window
- Energy and mass balances close to within 0.1 %
- Three independent disturbance scenarios run end-to-end without intervention

---

## Phase 2 — Baseline: Classical Control (non-negotiable)

**Effort:** 1 week

**Goal:** Honest PID baseline. Without it, every later claim about the agent is unfalsifiable and the paper is unpublishable.

**Deliverables**
- `src/industrial_ai/control/pid_baseline.py` — multi-loop PID with relay-feedback tuning
- KPI definitions: energy per kg product, yield, constraint-violation count, settling time after disturbance
- `notebooks/02_pid_benchmark.ipynb` — benchmark table the agent must beat

**Gate**
- All KPIs computed for ≥5 disturbance scenarios
- Results documented with mean ± 95 % CI

---

## Phase 3 — Agentic Controller

**Effort:** 3 weeks

**Goal:** LangGraph multi-agent system that proposes setpoint adjustments based on twin state.

**Architecture**
- `Observer` — reads twin state, summarizes deviations from spec
- `Optimizer` — proposes setpoint changes against KPI objectives
- `Critic` — challenges proposals against historical operating envelope

**Deliverables**
- `src/industrial_ai/agents/` — graph definition, prompts, state schema
- `notebooks/03_agent_runs.ipynb` — illustrative trajectories
- Config-driven runs (YAML), all hyperparameters versioned

**Gate**
- Agent runs end-to-end without manual intervention for ≥3 disturbance scenarios
- Output setpoints are within physical limits 100 % of the time (pre-safety-layer)

---

## Phase 4 — Safety Gate

**Effort:** 2 weeks

**Goal:** An anomaly detector that sits between Optimizer and Twin, blocking setpoints whose downstream state would resemble known anomaly patterns.

**Deliverables**
- `src/industrial_ai/safety/anomaly_gate.py`
- Detector trained on TEP and/or NoBOOM
- `notebooks/04_safety_layer.ipynb` — false-positive / false-negative analysis
- ROC curves with operating-point justification

**Gate**
- Documented FP/FN trade-off at the chosen operating point
- Safety layer blocks ≥1 demonstrably-bad agent proposal in a controlled disturbance scenario

---

## Phase 5 — arXiv Preprint + SAFEPROCESS Submission

**Effort:** 2–3 weeks (down from original 3–4)

**Goal:** Public arXiv preprint as primary career-positioning deliverable; conference paper as secondary credential. A polished journal version is explicitly **deferred** to a later sprint, not blocking the job-search use case.

**Deliverables**
- `src/industrial_ai/evaluation/` — KPI computation, statistical tests, plot generation
- `paper/arxiv_preprint.md` (or `.tex`) — full preprint, paper-grade
- `paper/safeprocess_paper.md` — 6-page IFAC conference version derived from the preprint
- `make reproduce` — single command that regenerates every figure
- arXiv submission (cs.LG + eess.SY cross-listing)
- SAFEPROCESS 2027 submission via PaperCept

**Gate**
- arXiv preprint live with a DOI-equivalent identifier
- SAFEPROCESS submission confirmation received before 31 October 2026
- LinkedIn post + CV update with arXiv link

**Explicitly deferred (post-job-search, optional)**
- Journal version with additional ablation studies for Computers & Chemical Engineering or Journal of Process Control

---

## Conference Targeting — Locked-In

| Venue | Date | Deadline | Fit |
|---|---|---|---|
| **SAFEPROCESS 2027** (primary) | 29 Jun – 2 Jul 2027, Delft NL | **31 Oct 2026** | Conference theme "AI for Safety" matches the novelty claim exactly |
| ADCHEM 2027 (backup) | 13–16 Jul 2027, Hong Kong | TBA (likely Nov/Dec 2026) | Strong process-control fit; weaker safety angle; expensive travel |

Decision rationale: see `docs/decisions/004-publication-strategy.md`.

---

## Key Decisions Already Locked

- Simulator: **IDAES**, not DWSIM — see `docs/decisions/001-idaes-over-dwsim.md`
- Case study: **C3/C4 train**, not chlorosilanes — see `docs/decisions/002-c3c4-over-chlorosilane.md`
- Novelty claim: **safety-gated agentic control**, not "agent steers column" — see `docs/decisions/003-novelty-positioning.md`
- Publication path: **arXiv preprint as primary, SAFEPROCESS as secondary, journal deferred** — see `docs/decisions/004-publication-strategy.md`

## Open Decisions

- Whether to release trained anomaly-detector weights publicly or only training code
- Whether to submit an invited-session proposal for SAFEPROCESS 2027 (deadline 30 Sep 2026, requires coordinating ≥6 thematically coherent papers — likely too ambitious for a first submission, default: regular session)
