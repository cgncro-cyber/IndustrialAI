# ADR 011 — Inference Stack Split: Hosted NIM Primary, Mac Studio Ablation Host

**Status:** Accepted
**Date:** 2026-06-01

## Context

Phase-3 work on the Mac-Studio + Nemotron-Super-49B-v1.5 (8-bit MLX) path
surfaced two independent failure modes in the 2026-06-01 post-canonical-IAE
smoke (`data/runs/c2_smoke/nominal_baseline/seed0/smoke.json`):

1. **Over-purification bias.** Agent systematically proposes
   `(y_D_target=0.995, x_B_target=0.005)` on a no-disturbance nominal-baseline
   scenario instead of holding the on-spec defaults `(0.99, 0.01)`.
   Canonical IAE 0.16756 mole-fraction·min on 5 cycles before abort.
2. **Output-discipline defect.** Long markdown bullet preambles before the JSON
   object hit the 512-token completion cap on 3/5 cycles; cycle 6 was
   JSON-truncated and triggered an ADR-010 named abort.

Both reproduce deterministically at `seed=0` after Claude Code's
`e92a7e9` thread-through of the seed parameter. A cross-model probe
with Claude Sonnet 4.6 against the *identical* prompt returns correct
on-spec hold proposals, isolating the cause as model-side
(Nemotron-49B-8bit-MLX), not prompt-side.

ADR-005's §5.1 explicit revisit trigger ("multi-step coherence
insufficiency") is empirically met, with reproducible evidence.

Independently, the on-Mac-Studio path has accumulated three layers of
amendment cost: LM Studio mlx-engine `nemotron-nas` load failure →
native `mlx_lm.server` chat-completions tokenizer regression →
client-side jinja2 chat-template render with pinned template SHA. None
of these constraints exist on a hosted vLLM-backed OpenAI-Chat-API
endpoint.

The IndustrialAI paper note already commits the production deployment
vision to *"NVIDIA NIM microservice with fallback to a hosted endpoint
when local inference is unavailable"*. Aligning the evaluation stack
with the documented deployment vision is methodologically preferable
to bridging a Mac-Studio MLX stack to an industrial reviewer.

## Decision

The Phase-3 **primary inference path** moves from on-Mac-Studio
`mlx_lm.server` to **NVIDIA NIM hosted endpoints**
(`https://integrate.api.nvidia.com/v1`) over the standard
OpenAI-Chat-API.

- **Initial primary model:** `nvidia/llama-3.3-nemotron-super-49b-v1.5`
  served by NIM in BF16 (full precision). Same model family as ADR-005,
  hosted at full precision rather than local 8-bit MLX. Provides a
  clean one-variable delta against the Mac-Studio results.
- **Phase-5 ablation host:** Per ADR-012 (2026-06-01), the cross-family
  ablation model is the DeepSeek family on the *same* NIM-hosted
  endpoint as the primary. ADR-005's original Mac-Studio + Qwen3.6-27B
  ablation plan is superseded. The Mac-Studio MLX stack and
  `MLXServerLLMClient` remain in the codebase as a reference
  implementation and contingency switch (`--backend mac-studio`) but no
  routine evaluation route uses them.
- **Open: capability-ceiling upgrade.** Whether to escalate the
  primary to `nvidia/llama-3.1-nemotron-ultra-253b-v1` (5× the
  capacity, same agentic post-training lineage) is deferred to a
  fork after the first NIM smoke result. The infrastructure decision
  here does not pre-commit the model decision.

The Python client surface is unified behind the existing `LLMClient`
protocol. `OpenAIChatLLMClient` joins `MLXServerLLMClient` in
`src/industrial_ai/agents/llm_client.py`; a small `build_llm_client(backend)`
factory selects between them based on a `.env`-controlled config or
CLI flag.

## Rationale

### Reproducibility chain

This ADR's strongest methodological asset is not the choice of NIM as the
platform but the **reproducibility chain** that the platform participates in.
The chain has six links, every one of them open or fixed in code:

| Link | Status | Reproduction path |
|---|---|---|
| Model weights (`nvidia/llama-3.3-nemotron-super-49b-v1.5`) | Open-weights (Llama Community License) on HuggingFace | Anyone can download |
| Chat template (jinja2) | Public in the HuggingFace repo | Anyone can render |
| Sampling parameters (T=0.6, top_p=0.95, max_tokens=512/4096) | Pinned in `industrial_ai.agents.llm_client` | Anyone can read the source |
| Seeds (0, 1, 2 for the N=3 smoke; pinned for Phase-3 evaluation) | Pinned in `tools/run_c2_smoke.py` and successor drivers | Anyone can rerun |
| Inference protocol (OpenAI Chat Completions) | Open standard | Any compatible server |
| Inference host (NIM) | Free-tier hosted, no credit card | Anyone with a developer account |

A third-party replication study therefore has **two independent
reproduction paths**:

1. **NIM-hosted replication** — sign up at `build.nvidia.com`, point
   `base_url` and `model` at the same identifiers we used, run the same
   driver. Five minutes of setup, $0 cost, byte-identical reproduction
   modulo seed-driven sampling variance.
2. **Self-hosted replication** — download the model weights, run vLLM
   (or SGLang, or llama.cpp server, any OpenAI-Chat-compatible inference
   engine), swap `NVIDIA_BASE_URL` in `.env` to localhost. Independent
   of NVIDIA's hosted service availability.

This is a categorically stronger reproducibility position than a paper
that used a closed-weight commercial API (Anthropic, OpenAI, Google).
Closed-weight APIs degrade on three axes simultaneously: model can be
deprecated, model can drift silently between calls, and the per-token cost
bars cost-constrained replicators. The Nemotron-on-NIM stack has none of
those properties.

**Implication for the paper.** §3.4 Methods (Agentic Supervisor) names the
reproducibility chain explicitly rather than burying it as an implementation
detail. The paper's methodological claim *"this architecture works with an
open-weight agentic-post-trained LLM, reproducibly so"* is the contribution
that the ablation slot then generalizes across LLM families.

### Why NIM as the hosted-primary platform

- **Hosts the exact ADR-005 model**, preserving the ADR-005 model-pinning
  argument verbatim.
- **Hosts the Nemotron-Ultra-253B variant**, providing a same-family
  capacity-escalation path without a methodology refactor if the
  -49B does not meet the Phase-3 acceptance criteria.
- **Free tier is unlimited in token volume, rate-limit-bound at 40 RPM**
  (raisable to 200 RPM via developer-forum request). NVIDIA abolished the
  credit-based system in April 2026; the current free tier is permanent,
  requires no credit card, and covers Phase-3 + Phase-5 evaluation in full.
  Reproducibility-strong: any reader can replicate on a free NVIDIA
  developer account.
- **OpenAI-Chat-API native.** No tokenizer regressions, no
  client-side template rendering. `messages: [...]` arrays go in,
  parsed completions come out.
- **Vision-aligned.** The IndustrialAI paper note explicitly names
  NIM as deployment target; the paper's Methods section can describe
  the inference stack honestly without anti-realistic deployment claims.

### Why not AWS (Bedrock / SageMaker / EC2)

- Bedrock catalog lacks the Nemotron-Super and Nemotron-Ultra families;
  Llama 3.3 70B vanilla (without NVIDIA's reasoning/agentic post-training)
  is not equivalent.
- SageMaker JumpStart bills per minute of endpoint uptime, ill-suited
  to bursty smoke iteration.
- EC2 + manual vLLM setup is high-friction with no upside over
  purpose-built LLM hosting providers.

### Why not Together AI / RunPod / Fireworks as primary

These remain valid alternatives and may be reached for if NIM
availability or pricing changes. NIM's family advantage (Nemotron-49B
*and* Nemotron-Ultra-253B both directly available) and vision-alignment
(paper deployment narrative names NIM) make it the preferred default.

### Why server-side chat-template render rather than client-side

The Mac-Studio amendment chain reached client-side jinja2 render
because `mlx_lm.server`'s `/v1/chat/completions` had a `transformers`-5.x
regression on `nemotron-nas` tokenizer behavior. NIM's vLLM backend
has no such regression — its `/v1/chat/completions` correctly renders
Nemotron chat templates server-side. `OpenAIChatLLMClient` therefore
passes plain `messages` arrays and lets the server handle template
expansion. **Net code reduction**, not addition.

`MLXServerLLMClient` keeps its client-side render because for the
Phase-5 Qwen ablation on Mac Studio that render remains necessary
(or at least not yet refuted). One client per backend, each correct
for its target.

## Consequences

- A second `LLMClient`-protocol implementation lands; tests cover both.
- `tools/run_c2_smoke.py` (and successor `tools/run_c2_agent_scenarios.py`)
  gain a `--backend {nim,mac-studio}` flag, default `nim`.
- ADR-005 is **not** rescinded. Its Amendment 2026-05-28 stays in
  force as governance for the *ablation host* and as historical
  audit trail. ADR-005 §5.1's revisit trigger ("multi-step coherence")
  is partially fulfilled and explicitly addressed by this ADR; the
  remaining open question (which model, including possible Ultra-253B
  escalation) is left to the downstream fork.
- Phase-3 evaluation wall-clock improves substantially. Hosted
  Nemotron-49B at ~5 s/cycle removes most of the Mac-Studio 30–40 s
  decode overhead, making the N ≥ 10 seed protocol from `kpis.md`
  §1.3 economically feasible.
- `paper/outline.md` §3.4 (Methodology — Agentic Supervisor)
  acquires an inference-stack sentence at Phase-5 drafting time.
  No edit required now.
- `pre_submission_checklist.md` §5.1 will receive a Changelog entry
  recording this ADR's relationship to the §5.1 trigger.
  (Edit handled separately when the first NIM smoke result lands.)

## Reversibility

**High.** The `LLMClient` protocol is the abstraction surface; either
backend becomes primary with a config-file or CLI flag flip. If NIM
availability, pricing, or model catalog changes mid-project, the
Mac-Studio stack is one flag away.

## Setup

### One-time NIM account setup (manual, owner-side)

1. Account at `https://build.nvidia.com`, generate API key.
2. Key stored in project-root `.env` (gitignored — verify with
   `grep -E '^\.env$' .gitignore`):
   ```
   NVIDIA_API_KEY=nvapi-...
   NVIDIA_BASE_URL=https://integrate.api.nvidia.com/v1
   NVIDIA_MODEL=nvidia/llama-3.3-nemotron-super-49b-v1.5
   ```
3. Exact model identifier may differ from the placeholder above
   depending on NIM's catalog naming at access time;
   the value above is the expected canonical form. Owner verifies
   in the NIM dashboard before the first call and updates `.env`
   if needed.

### Python client side

```python
from industrial_ai.agents.llm_client import build_llm_client

llm = build_llm_client(backend="nim")
# .env loaded, OpenAIChatLLMClient constructed with NVIDIA_* values
response = llm.complete(system_prompt=SYS, user_prompt=USER, reasoning=False)
```

For the ablation path:

```python
llm = build_llm_client(backend="mac-studio")
# MLXServerLLMClient constructed against the Mac-Studio endpoint per ADR-005
```

### ADR-010 compliance for missing config

Missing `NVIDIA_API_KEY` (when `backend="nim"`) or missing Mac-Studio
endpoint config (when `backend="mac-studio"`) raises a named
`MissingBackendConfigError` at `build_llm_client` time. No silent
defaults, no fallback to a different backend.

## Open items at decision time

1. **Model choice for the first NIM smoke** — defaults to
   `nvidia/llama-3.3-nemotron-super-49b-v1.5` (same as ADR-005, full
   precision) to isolate the quantization-vs-prompt variable. Upgrade
   to Nemotron-Ultra-253B (or other NIM Nemotron) is the next fork
   after the first smoke's data is in hand.
2. **System-prompt** — the v1 reformulation in
   `docs/prompts/2026-06-01_phase3_system_prompt_reformulation.md`
   is *not* applied for the first NIM smoke. The first NIM smoke
   uses the *same* system prompt as the 2026-06-01 Mac-Studio smoke,
   isolating the backend/quantization variable. Prompt reformulation
   lands in a separate downstream pass.
3. **Rate-limit headroom** — free tier is 40 RPM. Smoke iteration (≤ 12 RPM
   peak per smoke) is unconstrained. Headline-run parallelism (multiple
   seeds or scenarios in flight simultaneously) could approach 40 RPM and
   trigger 429s. Two-step escalation if reached: (a) request 200 RPM
   upgrade via NVIDIA developer forum, then (b) if 200 RPM is still tight,
   batch seed-runs sequentially in the driver. Per-token-billing
   alternatives (Together AI, Fireworks) remain in reserve but are not
   currently anticipated as needed.

## Changelog

- 2026-06-01 (initial) — Initial decision. Triggered by the post-canonical-IAE
  smoke's two-failure-mode evidence, the cumulative ADR-005 amendment
  cost, and the strategic decision to maximize paper-result strength
  by removing the Mac-Studio capability ceiling from the primary path.
- 2026-06-01 (free-tier correction) — Rationale and Open-Items sections
  updated. The original draft anticipated a credit-based free tier with
  a paid-tier transition for headline runs. Web-verified state
  (post-April-2026): NVIDIA abolished credit accounting; free tier is
  unlimited tokens with a 40 RPM rate limit (200 RPM on forum request).
  No paid account required for any phase of this project. The
  "Paid-tier transition" open item is replaced by a "Rate-limit
  headroom" item documenting the escalation path if 40 RPM becomes
  binding during parallel headline runs.
