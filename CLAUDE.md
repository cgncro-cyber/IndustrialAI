# CLAUDE.md — Project Context for Claude Code

> Read this file first whenever you start a session in this repo.

## 1. Project Identity

**Name:** IndustrialAI — Safety-Gated Agentic Control for Coupled Multivariable Processes
**Type:** Public research project (GitHub repo → arXiv preprint → conference paper)
**Owner:** Christian Rosenthal
**Status:** Phase 0 (scaffolding)

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
| Process simulator | **IDAES** (Pyomo-based) | Python-native, runs on macOS, DOE-backed, paper-friendly |
| Case study | **C3/C4 distillation train** (depropanizer + debutanizer) | Standard textbook system, max audience reach |
| Baseline controller | Classical PID with relay-feedback tuning | Required for honest benchmarking |
| Agent framework | **LangGraph** | Stateful multi-agent, good Python integration |
| LLM runtime | **LM Studio** on Mac Studio M3 Ultra (96 GB) | Native MLX acceleration for Apple Silicon, OpenAI-compatible server on `http://localhost:1234/v1` |
| LLM (primary) | **Llama-3.3-Nemotron-Super-49B v1.5** (MLX preferred, GGUF Q4 fallback) | Agentic post-training (RAG + tool calling), NVIDIA brand, comfortable RAM fit — see ADR 005 |
| LLM (ablation) | **Qwen3.6-27B** dense, Apache 2.0 (MLX preferred) | Robustness check across model family / license / architecture |
| LLM client library | `langchain-openai` against the LM Studio endpoint | Framework-agnostic — swappable to vLLM, Ollama, or remote provider without code change |
| Safety layer | Anomaly detector trained on TEP and/or NoBOOM | Gates agent setpoints before execution |
| Eval/Plots | matplotlib + seaborn, no plotly | Paper-print compatible |
| License | MIT | Maximum reach, no fake formality |

See `docs/decisions/` for ADR-style rationales.

## 6. Five-Phase Plan

See `PROJECT_PLAN.md`. Do not skip phases. **Phase 2 (PID baseline) is non-negotiable** — without it the agent claims are unfalsifiable and the paper is unpublishable.

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
- Reference the phase in commit body when relevant: `Phase 2: baseline PID tuning for column 1`

## 9. Reproducibility Rules

Every experimental run must produce:
1. A versioned config file (YAML) capturing all hyperparameters
2. Deterministic seeds where applicable
3. A logged manifest of input data hashes
4. Output plots regenerable from a single `make reproduce` target

## 10. Publication Strategy (locked)

Three-tier path, sequenced for speed:

1. **arXiv preprint** — *primary* career-positioning asset. Cross-listed `cs.LG` + `eess.SY`. Goes live the day Phase 5 completes.
2. **SAFEPROCESS 2027** — IFAC conference, Delft NL, 29 Jun – 2 Jul 2027. Theme "AI for Safety" matches novelty claim. **Paper deadline: 31 October 2026.** 6-page IFAC format, derived from the arXiv preprint. Proceedings on IFAC-PapersOnline (Scopus-indexed).
3. **Journal version** — *explicitly deferred*, optional. Computers & Chemical Engineering or Journal of Process Control as later targets. Not blocking the job-search use case.

Working title: *"Safety-Gated Agentic Control of Coupled Distillation Trains: A Case Study in Industrial-AI Methodology Transfer"*.

See `docs/decisions/004-publication-strategy.md` for the full rationale, and `paper/manuscript.md` once Phase 5 begins.

## 11. What to Do When Uncertain

- If a design decision is irreversible or affects the paper's claim of novelty → STOP and ask Christian.
- If a decision is minor and reversible → make it, document in `docs/decisions/`, move on.
- If Momentive-relevant context creeps in → strip it, do not commit.

## 12. Out of Scope (do not implement, do not suggest)

- Real-time DCS/PLC integration (this is a simulation study)
- Reinforcement learning from scratch (use LLM-agent paradigm only — that's the novelty)
- Web dashboards / Streamlit apps (paper-grade plots only)
- Cloud deployment, Docker images for production (Dockerfile for *reproducibility* is fine)
- Journal submission before the SAFEPROCESS deadline (explicitly deferred — see ADR 004)
