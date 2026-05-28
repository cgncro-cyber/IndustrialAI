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

---

## Amendment — 2026-05-28: Transport switched from LM Studio to native `mlx_lm.server`; modal reasoning policy added

The decision named *LM Studio* as the inference runtime and *langchain-openai* as the Python client. After Schritt-4 implementation, both choices have been amended on empirically motivated grounds. The primary model selection (`Llama-3.3-Nemotron-Super-49B v1.5`, `splats/Llama-3_3-Nemotron-Super-49B-v1_5-mlx-8Bit` MLX 8-bit quant) is unchanged.

### Empirical chain that motivated the amendment

1. **LM Studio bundled mlx-engine cannot load `nemotron-nas`.** Both stable and beta channels' latest `mlx-llm-mac-arm64-apple-metal-advsimd@1.8.5` (selected via `lms runtime ls` after `lms runtime update --all`) hit `AttributeError: 'PreTrainedConfig' object has no attribute 'max_position_embeddings'` during model load. Root cause: bundled `transformers==5.5.4` lacks a registered `nemotron_nas` model type; the `splats` MLX conversion provides custom `configuration_decilm.py` / `modeling_decilm.py` via `auto_map`, but the LM Studio MLX backend does not honour `trust_remote_code`. Reproduces LM Studio bug #704. **Not fixable by an LM Studio version bump in any currently available channel.**

2. **Native `mlx_lm.server` 0.31.3 with `--trust-remote-code` loads the model successfully** (config and tokenizer use the bundled custom remote-code modules). Loading takes ~30 s on the Mac Studio M3 Ultra; resident set size ~16 GB (model file is mmap'd).

3. **The `/v1/chat/completions` endpoint produces degenerate output** even with `--chat-template` set to the bundled `chat_template.jinja`. Empirical root cause: `transformers==5.9.0`'s `apply_chat_template(tokenize=True)` returns a list of two `tokenizers.Encoding` objects instead of a flat token-id list for this tokenizer (regression separate from the rope-standardization one). `mlx_lm.server` cannot consume that structure and feeds the model malformed input → `"a a a a a ..."` repetition loop. The flag `--chat-template-args '{"tokenize": false}'` is rejected with `dict() got multiple values for keyword argument 'tokenize'` (server hardcodes `tokenize` and merges args naively).

4. **The `/v1/completions` endpoint with a client-side jinja2-rendered prompt works correctly**, producing coherent output including Nemotron's expected `<think>...</think>` reasoning block followed by the actual response. Byte-identical client-side render verified against `tokenizer.apply_chat_template(tokenize=False)`.

### Amended transport

- **Server**: `mlx_lm.server --model <path-to-splats-bundle> --host 0.0.0.0 --port 8080 --trust-remote-code`, started as a `nohup`-detached process so it survives SSH disconnects. PID and log path are recorded in `~/mlx_server_runs/last_run.txt` on the Mac Studio.
- **Client**: `MLXServerLLMClient` in `src/industrial_ai/agents/llm_client.py`. Renders the chat template client-side via jinja2 (pinned template fixture below); sends prompts to `/v1/completions` with `stop=["<|eot_id|>"]`; parses replies with `_parse_setpoint_json`. The `LMStudioLLMClient` class is retained for any future LM-Studio-backed configuration but is no longer the default.
- **Endpoint URL**: `http://192.168.178.81:8080/v1` (LAN address of the Mac Studio). Local clients on the Mac Studio itself can use `http://localhost:8080/v1`. The `MLXServerLLMClient` accepts `base_url` as a constructor argument so the same code runs both locally and over LAN.

### Modal reasoning policy

Nemotron-Super-49B v1.5 supports a runtime reasoning-on/off toggle through `/think` / `/no_think` markers in the system content (detected by the chat template). The agent uses both modes:

- **`reasoning=False` (default)** — the modal default for tool-call cycles. The system prompt is prefixed with `/no_think `; the template appends `<think>\n\n</think>\n\n` after the assistant header, the model skips chain-of-thought, and emits the JSON directly. Empirical Schritt-4 smoke check: **10 / 10 = 100 % tool-call reliability** at P50 = 5.6 s, P95 = 6.1 s — comfortably inside the 5-minute supervisory cadence.
- **`reasoning=True`** — reserved for Critic-revision rounds inside the agent graph. `_optimizer_node` sets `reasoning = critic_feedback is not None`, so revision rounds get chain-of-thought; the typical first-round path is fast. `max_tokens` budget rises from 512 to 4096. The smoke check's single worst-case revision call (off-nominal F=0.8 with verbose critic feedback) ran 331.8 s and hit the 4096-token cap before emitting JSON. This is documented as an open item in `pre_submission_checklist.md` §5.1 (Phase-3-iteration tuning question: budget vs. cadence), not as a blocker — the gate is on `reasoning=False` reliability.

### Pinned versions and artifacts

| Component | Pinned value |
|---|---|
| Mac Studio host | `gamba@192.168.178.81` (mDNS `admins-mac-studio.local`), Darwin 25.4.0 arm64, macOS 26.4 |
| Python venv | CPython 3.12.13 (installed via `uv` 0.11.16) |
| `mlx-lm` | 0.31.3 |
| `mlx` | 0.31.2 |
| `transformers` | 5.9.0 |
| `tokenizers` | 0.22.2 |
| Model identifier | `splats/Llama-3_3-Nemotron-Super-49B-v1_5-mlx-8Bit` (architecture `nemotron-nas` / `DeciLMForCausalLM`, 8-bit MLX, ~53 GB on disk) |
| Server bind | `0.0.0.0:8080`, `--trust-remote-code`, stop tokens `["<|eot_id|>"]` |
| Chat template fixture | `data/reference/nemotron_super_v1_5_chat_template.jinja`, SHA-256 `1b13a386b158bb4033ad9960032530554c47a92d1735c7dfce715efcabf30e5c` |
| Smoke-check report | `data/reference/phase3_llm_smoke.json` (Schritt-4 run, 2026-05-28) |
| Sampling defaults | `temperature=0.6`, `top_p=0.95`, `max_tokens=512` (reasoning OFF) / `4096` (reasoning ON) |

### Reproducibility procedure (replaces the LM Studio recipe above)

```bash
# On the Mac Studio (one-off setup):
curl -LsSf https://astral.sh/uv/install.sh | sh
uv venv --python 3.12 ~/mlx_test_venv
uv pip install --python ~/mlx_test_venv mlx-lm    # pins the versions above

# Server launch (re-run after reboot or for a fresh session):
MODEL_DIR=~/.lmstudio/models/splats/Llama-3_3-Nemotron-Super-49B-v1_5-mlx-8Bit
LOG=~/mlx_server_runs/mlx_lm_server_$(date +%Y%m%d_%H%M%S).log
nohup ~/mlx_test_venv/bin/mlx_lm.server \
    --model "$MODEL_DIR" --host 0.0.0.0 --port 8080 --trust-remote-code \
    > "$LOG" 2>&1 &
echo "$!" > ~/mlx_server_runs/last_run.pid
```

### Python client (replaces the langchain-openai snippet above)

```python
from industrial_ai.agents.llm_client import MLXServerLLMClient

llm = MLXServerLLMClient(
    base_url="http://192.168.178.81:8080/v1",  # or localhost on the Mac itself
    model="default_model",
    request_timeout_s=600.0,
)
# Round-1 tool call:
response = llm.complete(system_prompt=SYS, user_prompt=USER, reasoning=False)
# Critic-revision call:
response = llm.complete(system_prompt=SYS, user_prompt=USER + "\n\nCritic: ...", reasoning=True)
```

### Decision on the Open Question (above) remains active

The original `Open Question` clause stays in force: a Phase-3 evaluation that surfaces insufficient tool-call reliability or multi-step coherence will trigger the documented escalation path (Nemotron-3-Super-120B-A12B / external endpoint). The amendment changes *how* Nemotron-Super-49B v1.5 is served, not whether it is the primary candidate.
