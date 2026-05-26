# ADR 005 — Local Model Selection: Nemotron-Super-49B + Qwen3.6-27B on LM Studio

**Status:** Accepted
**Date:** 2026-05-26

## Context

The project's agentic layer (LangGraph: Observer / Optimizer / Critic) requires a local LLM running on the owner's Mac Studio M3 Ultra (**96 GB unified memory**). The Mac Studio already runs **LM Studio** as the local inference engine, which provides native MLX acceleration on Apple Silicon and exposes an OpenAI-compatible HTTP server.

The choice has to satisfy four constraints simultaneously:

1. **Capability** — sufficient reasoning and tool-use quality for multi-step process-control decisions.
2. **Reproducibility** — open-weight, downloadable, version-pinnable; preferably permissive license.
3. **Hardware fit** — leaves enough headroom (≥30 GB) for KV cache, macOS, and concurrent dev tools (VS Code, browser, Claude Code, LM Studio itself).
4. **Paper credibility** — recognizable provenance that IFAC-domain reviewers respect.

The frontier closed-weight Qwen 3.7-Max (announced 20 May 2026) and Kimi K2.6 (~520 GB at INT4) and DeepSeek V4-Pro (~864 GB official weights) are categorically out of reach. The decision is therefore between mid-scale open-weight models.

## Decision

**Inference runtime:** LM Studio, OpenAI-compatible server on `http://localhost:1234/v1`.

**Primary model:** `nvidia/Llama-3.3-Nemotron-Super-49B-v1.5` (~25–30 GB at Q4, MLX preferred)
**Ablation model:** `Qwen3.6-27B` dense (~15–20 GB at Q4, MLX preferred)

Both will be used in the final paper. The primary handles all main experimental runs; the ablation provides a single comparative run-series in Section 5 to demonstrate that the architecture generalizes across model families.

The Python client uses `langchain-openai` pointed at the LM Studio endpoint — framework-agnostic by design, so a later swap to vLLM, Ollama, llama.cpp server, or a remote provider does not require code changes.

## Rationale

### Why LM Studio over Ollama as the runtime

- **Native MLX support.** On Apple Silicon, MLX-format models run noticeably faster than GGUF via Ollama because they leverage the unified-memory Metal backend directly.
- **GUI for model management.** Easier iteration during early Phase 3 when swapping models for quick comparison.
- **OpenAI-compatible server.** Standard endpoint at `http://localhost:1234/v1`, identical contract to remote OpenAI/Anthropic APIs, makes the client code provider-agnostic.
- **Already installed on the Mac Studio.** No tooling migration needed.

### Why Nemotron-Super-49B v1.5 as primary

- **Explicitly agentic post-training.** The model card documents RPO/RLVR/iterative-DPO stages targeting RAG and tool-calling behavior — exactly the failure modes that matter for a LangGraph multi-agent controller.
- **Comfortable hardware fit.** At 25–30 GB it leaves >60 GB of headroom for KV cache during long agent traces, plus concurrent IDE / browser / Claude Code sessions without memory pressure.
- **Strong reasoning benchmarks.** MATH500 97.4, AIME-2025 82.71, GPQA 71.97, LiveCodeBench 73.58 (NVIDIA-reported, NeMo-Skills, pass@1).
- **Recognizable provenance.** NVIDIA + Llama-3.3 lineage gives the paper credibility with IFAC-domain reviewers who tend to be skeptical of unfamiliar Chinese-lab models.
- **128K context.** Sufficient for any realistic agent trace this project produces.
- **MLX build available.** Native Apple Silicon acceleration via LM Studio.

### Why Nemotron-3-Super-120B was rejected for this hardware

The 120B-A12B variant has a 64 GB Q4 minimum *model footprint*. Adding KV cache for long agent traces, macOS overhead (~15 GB), and concurrent dev tools pushes total memory use to 85–95 GB. On a 96 GB system this causes swap pressure and unpredictable inference latency during development. The marginal capability gain over the 49B does not justify the operational fragility.

### Why Qwen3.6-27B as ablation

- **Different model family.** Alibaba vs NVIDIA → reviewers see family-independent results.
- **Different architecture.** Dense Transformer vs NAS-compressed MoE-ish.
- **Different license.** Apache 2.0 vs Llama Community License → covers the licensing-sensitivity question reviewers may raise.
- **Fits trivially.** 15–20 GB leaves room to run primary and ablation models concurrently if helpful.
- **Strong agentic benchmark performance.** Public reports describe the 27B dense as outperforming 397B MoE peers on agentic coding tasks.

### Why not DeepSeek V4-Flash

Technically the 158B model (~160–170 GB at Q4) does not fit on 96 GB. Even on larger Mac Studio configurations, the FP4+FP8 mixed source weights make sub-Q4 quantization unreliable, and llama.cpp/GGUF support is still maturing as of May 2026. Reproducibility risk too high for a paper that must survive 5+ months until SAFEPROCESS review.

### Why not Kimi K2.6

Out of hardware budget by an order of magnitude (~350 GB minimum at lowest quant).

## Consequences

- Phase 3 (Agentic Controller) implementation targets Nemotron-Super-49B v1.5 as the default backend in all configs.
- Phase 5 evaluation runs include one ablation series with Qwen3.6-27B as the LLM backend, all other parameters held constant.
- The paper's "Methods" section reports the LLM swap as a robustness test; the "Results" section reports primary vs ablation KPIs side by side.
- Both model versions are pinned in `configs/` to ensure reproducibility.
- If Nemotron-Super gets a v1.6 or later during the project, the version stays pinned until Phase 5 is complete — no mid-project upgrades.
- The Python client uses `langchain-openai` only. No code path imports `langchain-ollama`.

## Reversibility

Medium. Swapping the primary model is technically a config change, but every benchmark and plot would need re-running, costing ~1 week of recompute. Decision should not be revisited unless a fundamental capability gap surfaces in Phase 3.

## Setup

### LM Studio side

1. Open LM Studio → Discover (search) → download:
   - `nvidia/Llama-3.3-Nemotron-Super-49B-v1.5` — prefer MLX variant; if unavailable, Q4_K_M GGUF
   - `Qwen/Qwen3.6-27B` (or community MLX conversion) — prefer MLX variant; Q4_K_M GGUF fallback
2. Load Nemotron-Super-49B v1.5 in the Server tab.
3. Start the local server. Default endpoint: `http://localhost:1234/v1`. Verify with:
   ```bash
   curl http://localhost:1234/v1/models
   ```
4. Note the exact model identifier LM Studio assigns (varies by build). Pin it in the project config.

### Python client (minimal example)

```python
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(
    base_url="http://localhost:1234/v1",
    api_key="lm-studio",                 # any non-empty string
    model="nemotron-3-super-49b",        # exact ID from LM Studio
    temperature=0.6,                     # NVIDIA-recommended for Reasoning ON
    top_p=0.95,
)
```

## Open Question (revisit at end of Phase 3)

If Nemotron-Super-49B v1.5 turns out to be insufficient on tool-call reliability or multi-step coherence during Phase 3 experiments, the fallback escalation path is:

1. Try Nemotron-3-Super-120B-A12B with aggressive quantization (Q3_K_M) — accept tight memory, gain capability.
2. If still insufficient → reconsider DeepSeek V4-Flash on a different machine (HP Elite Mini won't help; would need cloud GPU rental).

This is a contingency, not a plan.
