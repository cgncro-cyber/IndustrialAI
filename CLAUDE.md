# CLAUDE.md — Project Context for Claude Code

> Read this file first whenever you start a session in this repo.

## 1. Project Identity

**Name:** IndustrialAI — Safety-Gated Agentic Control for Coupled Multivariable Processes
**Type:** Public research project (GitHub repo → arXiv preprint → conference paper)
**Owner:** Christian Rosenthal
**Status:** Phase 2 in progress (Day 2.5 closed, 2026-05-27). Phase 1 closed with the Skogestad Column A dynamic twin including CasADi symbolic Jacobians. Phase 2 progress: Day 1 — relay-feedback Tyreus-Luyben C0; Day 2 — 6-KPI suite + 5 canonical disturbance scenarios; Day 2.5 — 6-candidate PID tuning shootout ({TL, SIMC-1DoF, SIMC-2DoF} × {no decoupler, simplified decoupler with retuned SIMC against g_eff = g_ii / λ_ii}). Winner is TL_no_decoupler with aggregate IAE 0.8362 mole-fraction·min over 5 scenarios; decoupled variants are competitive (SIMC-1DoF+D: 1.18) but do not beat the relay-tuned SISO, matching the known structural limitation of static decoupling on high-RGA plants (Skogestad & Postlethwaite 1996 §10.8). 154 pytest cases pass; F-perturbed robustness OPs were deferred to Phase 5 due to LV plant's numerical conditioning. Next: Day 3 — C1 Linear MPC via do-mpc on the CasADi LV-closed model.

## 2. Strategic Framing — Why This Project Exists

This is a **career-positioning project**, not a DBA-thesis project. The DBA work is separately planned around Health-01 (multi-omics). The chemistry here is the *demonstration vehicle*; the *product* is a sector-agnostic methodology (agentic control + anomaly-based safety gating) that transfers to semiconductor, pharma, battery manufacturing, energy, and industrial-AI domains.

**Never write README or paper text that frames this as "a distillation project."** Always frame as "a methodology for coupled multivariable process control, demonstrated on a distillation case study."

## 3. About the Owner — How to Communicate

- 15+ years industrial experience (Lean Six Sigma Black Belt, Product Manager EMEA at Momentive Performance Materials, prior Automation Engineer at ProLeit, real BB-era work on distillation and MCS-synthesis reactors).
- DBA candidate AI/ML at Walsh College — that thesis is on Health-01, NOT this project.
- MIT Applied Data Science alumnus.
- Strong domain intuition for chemical processes — does NOT need basic explanations of distillation, VLE, or PID tuning.
- **Prefers brief, to-the-point explanations.** No fluff, no hedging, no over-apologizing.
- German native.

**Language policy (strict):**
- **Chat in this session: German.** Reply to Christian in German unless he switches.
- **Everything written to disk or to GitHub: English.** This includes source code, comments, docstrings, commit messages, PR titles and descriptions, issue text, branch names, ADRs, README, paper drafts, notebook markdown cells, and any other artifact that lives in the repo or on GitHub.
- If a chat answer references code or a commit message, the surrounding explanation stays in German but the quoted artifact stays in English.

## 4. Hard Boundaries

- **No Momentive proprietary data, parameters, catalysts, or process details.** Ever. Public literature only.
- **No silicones-specific framing in public deliverables.** Christian is exiting silicones — the chemistry is a generic case study.
- **No real customer or supplier names** in code, comments, commits, or docs.

## 5. Technical Architecture (locked-in decisions)

| Layer | Choice | Rationale |
|---|---|---|
| Process simulator (steady-state, property packages) | **IDAES** (Pyomo-based) | Python-native, runs on macOS, DOE-backed, paper-friendly — see ADR 001 |
| Process simulator (dynamic distillation) | **`scipy.integrate.solve_ivp`** on a Python port of Skogestad's Column A | IDAES `TrayColumn` is hard-coded `dynamic=False` (Issue #96); dynamics required for SAFEPROCESS "AI for Safety" KPIs — see ADR 001 Refinement and ADR 007 |
| Case study | **Skogestad's "Column A"** — 40-stage binary distillation, LV/DV/L/D-V/B configurations, literature-validated trajectories | The canonical distillation control benchmark; supports dynamics; reviewer recognition; reproducible — see ADR 007 (supersedes ADR 002) |
| Control architecture | **Two-layer hierarchical**: supervisory (5–15 min) over regulatory PID (~1–5 s) | Industrial APC standard; decouples LLM latency from real-time control — see ADR 006 |
| Regulatory layer (held constant across all configurations) | Classical multi-loop PID (top composition, bottom composition, condenser level, reboiler level) with relay-feedback tuning | Identical under PID-only / MPC / Agent / Agent+Safety — see ADR 006 |
| Supervisory baseline C0 | PID-only with fixed manual setpoints | Do-nothing baseline that quantifies the value of any supervisor |
| Supervisory baseline C1 | **Linear MPC via `do-mpc`**, linearization point from `column_a/linearize.py` | Industrial state-of-the-art baseline; pre-empts the strongest reviewer objection |
| Supervisory C2 / C3 | Agentic LangGraph controller (with optional safety gate) | The contribution under study |
| Agent framework | **LangGraph** | Stateful multi-agent, good Python integration |
| LLM runtime | **LM Studio** on Mac Studio M3 Ultra (96 GB) | Native MLX acceleration for Apple Silicon, OpenAI-compatible server on `http://localhost:1234/v1` |
| LLM (primary) | **Llama-3.3-Nemotron-Super-49B v1.5** (MLX preferred, GGUF Q4 fallback) | Agentic post-training (RAG + tool calling), NVIDIA brand, comfortable RAM fit — see ADR 005 |
| LLM (ablation) | **Qwen3.6-27B** dense, Apache 2.0 (MLX preferred) | Robustness check across model family / license / architecture |
| LLM client library | `langchain-openai` against the LM Studio endpoint | Framework-agnostic — swappable to vLLM, Ollama, or remote provider without code change |
| Safety layer | Anomaly detector, cross-domain validation (TEP → Skogestad Column A) primary, in-domain fallback | Gates agent setpoints before execution; cross-domain transfer is itself a publishable sub-result |
| Eval/Plots | matplotlib + seaborn, no plotly | Paper-print compatible |
| License | MIT | Maximum reach, no fake formality |

See `docs/decisions/` for ADR-style rationales.

## 6. Five-Phase Plan

See `PROJECT_PLAN.md`. Do not skip phases. **Phase 2 (PID + Linear MPC baselines) is non-negotiable** — without the MPC baseline, the paper is vulnerable to the strongest reviewer objection.

**Hard anchor: SAFEPROCESS 2027 submission deadline 31 October 2026.**

## 7. Code Standards

- Python 3.11+
- Formatting: `ruff format` (line length 100)
- Linting: `ruff check`
- Type checking: `mypy --strict` for `src/`, relaxed for `notebooks/`
- Tests: `pytest`, target ≥70% coverage on `src/industrial_ai/`
- Docstrings: NumPy style, every public function
- Pre-commit hooks enforce all of the above

## 8. Commit Discipline

- Conventional Commits (`feat:`, `fix:`, `docs:`, `refactor:`, `test:`, `chore:`)
- One logical change per commit
- Reference the phase in commit body when relevant: `Phase 2: baseline PID tuning for Column A LV configuration`

## 9. Reproducibility Rules

Every experimental run must produce:
1. A versioned config file (YAML) capturing all hyperparameters.
2. Deterministic seeds where applicable.
3. A logged manifest of input data hashes.
4. Output plots regenerable from a single `make reproduce` target.
5. **Stochastic accounting for any LLM-in-the-loop configuration**: each scenario evaluated over N ≥ 10 independent seed runs; KPIs reported with bootstrap confidence intervals and Cohen's d effect sizes, never as point estimates. Practical-significance thresholds are defined *before* the final runs, not after. See PROJECT_PLAN Phase 5.
6. **Data-logging contract** per `docs/figures.md`: every Phase 1+ run must populate `data/runs/<config>/<scenario>/<seed>/` with `timeseries.parquet`, `tray_profile.parquet`, `setpoints.parquet`, `kpis.json`, `latency.json`, `safety_log.parquet` (C3 only), `config.yaml`, `manifest.json`. Without this contract, later phases cannot regenerate figures.

## 10. Publication Strategy (locked)

Three-tier path, sequenced for speed:

1. **arXiv preprint** — *primary* career-positioning asset. Cross-listed `cs.LG` + `eess.SY`. Goes live the day Phase 5 completes.
2. **SAFEPROCESS 2027** — IFAC conference, Delft NL, 29 Jun – 2 Jul 2027. Theme "AI for Safety" matches novelty claim. **Paper deadline: 31 October 2026.** 6-page IFAC format, derived from the arXiv preprint. Proceedings on IFAC-PapersOnline (Scopus-indexed).
3. **Journal version** — *explicitly deferred*, optional. *Computers & Chemical Engineering* or *Journal of Process Control* as later targets. Not blocking the job-search use case.

Working title: *"Safety-Gated Agentic Control of Multivariable Distillation: A Case Study in Industrial-AI Methodology Transfer"*.

See `docs/decisions/004-publication-strategy.md` for the full rationale, and `paper/manuscript.md` once Phase 5 begins.

## 11. What to Do When Uncertain

- If a design decision is irreversible or affects the paper's claim of novelty → STOP and ask Christian.
- If a decision is minor and reversible → make it, document in `docs/decisions/`, move on.
- If Momentive-relevant context creeps in → strip it, do not commit.
- **If something in IDAES, Skogestad's MATLAB code, or any external dependency looks broken or unsupported** → verify with primary sources (IDAES GitHub issues, NTNU code, peer-reviewed papers) *before* writing a workaround. Hidden framework limitations like IDAES Issue #96 are exactly the kind of trap that wastes days if discovered late.

## 12. Out of Scope (do not implement, do not suggest)

- Real-time DCS/PLC integration (this is a simulation study).
- Direct actuator manipulation by the agent (violates ADR 006 hierarchy).
- Reinforcement learning from scratch (use LLM-agent paradigm only — that's the novelty).
- Web dashboards / Streamlit apps (paper-grade plots only).
- Cloud deployment, Docker images for production (Dockerfile for *reproducibility* is fine).
- Journal submission before the SAFEPROCESS deadline (explicitly deferred — see ADR 004).
- Multi-component distillation in IDAES `TrayColumn` (framework does not support dynamics — see ADR 001 Refinement, ADR 007).
