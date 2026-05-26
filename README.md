# IndustrialAI

**Safety-Gated Agentic Control for Coupled Multivariable Processes**

A reproducible research project demonstrating how LLM-based agentic controllers, combined with an anomaly-detection safety layer, can outperform classical control on coupled multivariable industrial processes — with a distillation train as the case study.

> The chemistry is the demonstration. The methodology is the contribution. The architecture transfers to semiconductor process control, pharmaceutical continuous manufacturing, battery electrode lines, energy systems, and beyond.

---

## Status

🚧 **Phase 1 — Foundation in progress.** Skogestad Column A twin port (per [ADR 007](./docs/decisions/007-skogestad-column-a-over-c3c4-train.md)). Day-4 mini-gate passed against published Skogestad 1997 references (G^LV(0) to within 0.01 %, dominant time constant τ₁ to within 0.04 %, three Octave-cross-checked step-response trajectories within 1e-6). DV / L-D-V-B configurations, operating-window sweep, and the walkthrough notebook still pending. See [`PROJECT_PLAN.md`](./PROJECT_PLAN.md) for the full five-phase roadmap and remaining Phase-1 deliverables.

## Why this matters beyond distillation

This architecture transfers to any process with:

- Coupled multivariable loops where classical PID hits its limits
- High constraint-violation costs (quality, safety, energy)
- Significant gains available from anticipatory rather than reactive control

Concrete transfer targets:

- **Semiconductor wafer fabs** — multi-zone temperature/gas control with tight spec windows
- **Pharma continuous manufacturing** — coupled crystallization/drying/granulation trains
- **Battery electrode coating** — coupled slot-die, drying, calendering with quality gates
- **HVAC and district heating** — multi-loop energy optimization with safety constraints
- **Water treatment plants** — coupled chemical dosing with effluent-spec safety

## Architecture (high level)

```
                    ┌────────────────────────┐
                    │  Process Twin (IDAES)  │
                    │  C3/C4 Distillation    │
                    └───────────┬────────────┘
                                │ state
                  ┌─────────────┴─────────────┐
                  ▼                           ▼
        ┌──────────────────┐        ┌──────────────────┐
        │  Baseline: PID   │        │  Agentic Layer   │
        │  (benchmark)     │        │  (LangGraph)     │
        └──────────────────┘        └────────┬─────────┘
                                             │ proposed setpoints
                                             ▼
                                  ┌──────────────────────┐
                                  │  Safety Gate         │
                                  │  (Anomaly Detector)  │
                                  └────────┬─────────────┘
                                           │ accepted setpoints
                                           ▼
                                     back to Twin
```

## Tech Stack

- **Process simulation:** IDAES (Pyomo)
- **Control baseline:** Python PID with relay-feedback tuning
- **Agent framework:** LangGraph multi-agent (Observer / Optimizer / Critic)
- **LLM:** Llama-3.3-Nemotron-Super-49B v1.5 (primary) / Qwen3.6-27B (ablation), served locally via LM Studio (MLX) — see [ADR 005](./docs/decisions/005-local-model-selection.md)
- **LLM client:** `langchain-openai` against the local OpenAI-compatible endpoint — provider-agnostic, swappable to vLLM, Ollama, or remote APIs without code changes
- **Anomaly detection:** Trained on TEP and/or NoBOOM benchmark datasets
- **Evaluation:** matplotlib, seaborn, statistical effect-size reporting

## Getting Started

```bash
# Prerequisites: macOS (Apple Silicon) or Linux, Homebrew (macOS only)

# Install uv (one-time)
brew install uv                                    # macOS
# curl -LsSf https://astral.sh/uv/install.sh | sh  # Linux

# Clone and set up
git clone <repo-url>
cd IndustrialAI
make setup       # uv sync + idaes get-extensions

# Verify
make smoke       # IDAES + ipopt sanity check
```

For Apple-Silicon-specific notes and known pitfalls, see
[`docs/setup/idaes_on_macos.md`](./docs/setup/idaes_on_macos.md).

## Repository Layout

```
src/industrial_ai/
├── twin/          # IDAES process models
├── control/       # PID baselines + tuning
├── agents/        # LangGraph agents
├── safety/        # Anomaly detection gate
└── evaluation/    # KPIs, plots, statistical tests

tests/             # Unit + integration tests
notebooks/         # Exploratory work, never source of truth
data/              # Benchmark datasets (TEP, NoBOOM references)
paper/             # LaTeX/Markdown manuscript
docs/decisions/    # ADRs — architecture decision records
```

## Citation

Once published:

```bibtex
@article{rosenthal2026safety,
  title  = {Safety-Gated Agentic Control of Coupled Distillation Trains},
  author = {Rosenthal, Christian},
  year   = {2026},
  journal= {tbd},
}
```

## License

MIT — see [`LICENSE`](./LICENSE).

## About the Author

Christian Rosenthal is a Product Manager and Lean Six Sigma Black Belt with 15+ years of experience optimizing chemical processes at industrial scale, currently completing a DBA in AI/ML. This project bridges classical process engineering with contemporary agentic AI.
