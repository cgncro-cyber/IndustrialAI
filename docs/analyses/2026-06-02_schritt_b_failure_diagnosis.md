# Schritt-B Failure Diagnosis — Worst-Case Corner Cluster (F=0.8 / zF=0.45 / F_step_+20pct / target_acquisition)

**Date:** 2026-06-02
**Authority:** Per docs/prompts/2026-06-02_schritt_b_failure_diagnosis_and_restart.md — Outcome B path (genuine failure mode).
**Status:** Sweep PID 88452 SIGTERM'd. Manifest preserved; no relaunch pending direction.

---

## Sweep state at SIGTERM

| Status | Count |
|---|---|
| failed | 5 |
| interrupted | 1 |
| pending | 394 |

All 5 failed + the 1 interrupted are at the **same factor cell** `(F=0.8, zF=0.45, F_step_+20pct, target_acquisition)`, seeds 0 through 5. The remaining 4 seeds (6–9) of the same factor cell are pending.

## Step 1 diagnostic table

| seed | manifest_status | manifest_error | smoke.json | cycles_done | last_cycle_wall_s | cell wall (s) |
|---|---|---|---|---|---|---|
| 0 | failed | timeout | **no** | 0 | n/a | 900 |
| 1 | failed | timeout | **no** | 0 | n/a | 900 |
| 2 | failed | timeout | **no** | 0 | n/a | 900 |
| 3 | failed | timeout | **no** | 0 | n/a | 900 |
| 4 | failed | smoke_parse | yes | 6/12 | 31.4 | 106 |
| 5 | interrupted | n/a | no | 0 | n/a | ~396 (SIGTERM'd) |

**4 of 5 failed cells have NO smoke.json at all.** Per the prompt's Outcome B definition ("majority of failed cells have smoke.json with < 4 cycles completed or no smoke.json at all"), this is the genuine-failure-mode branch.

## Wall-clock pattern of the 4 silent timeouts

| seed | started (UTC) | completed (UTC) | duration |
|---|---|---|---|
| 0 | 17:56:47 | 18:11:47 | 15:00 (= 900 s) |
| 1 | 18:11:49 | 18:26:49 | 15:00 |
| 2 | 18:26:51 | 18:41:51 | 15:00 |
| 3 | 18:41:53 | 18:56:53 | 15:00 |

Exactly 900s per cell. The smoke subprocess was SIGKILLed at the timeout boundary, before it could write the partial smoke.json. Smoke driver writes JSON at the END of the cycle loop (or at abort); a subprocess SIGKILLed mid-LLM-call writes nothing.

seed=4 broke this pattern by hitting `LLMResponseParseError` at cycle 6 (within 106 s of subprocess start), which triggered the smoke's atomic partial-output write (cbbbff4 mechanism) and exited 2.

## seed=4 — the only smoke.json we have

The single surviving data point shows what's actually happening when an off-nominal F=0.8/zF=0.45 cell does emit output.

### config (full)

```json
{
  "scenario": "F_step_+20pct",
  "op_F": 0.8,
  "op_zF": 0.45,
  "submetric": "target_acquisition",
  "x0_source": "lookup_lv_ss",
  "LT0": 2.70629,
  "VB0": 3.20629,
  "horizon_min": 60.0,
  "tick_min": 5.0,
  "n_ticks": 12,
  "seed": 4,
  "backend": "nim",
  "base_url": "https://integrate.api.nvidia.com/v1",
  "model": "nvidia/nemotron-3-super-120b-a12b",
  "reasoning_protocol": "nemotron_extra_body",
  "reasoning_mode": "off",
  "reasoning_budget": 4096,
  "temperature": 0.3,
  "top_p": 0.95,
  "max_tokens": 512,
  "regulatory_backend": "mpc"
}
```

### aggregate (partial — 6 of 12 cycles)

```json
{
  "iae_mole_fraction_min": 23.936,
  "internal_tracking_iae_mole_fraction_min": 14.336,
  "total_wall_clock_seconds": 101.68,
  "cycle_wall_clock_seconds_p50": 8.88,
  "cycle_wall_clock_seconds_p95": 29.85,
  "cycle_wall_clock_seconds_max": 31.43,
  "prompt_tokens_total": 1200,
  "completion_tokens_total": 1347,
  "completion_tokens_p50": 217,
  "completion_tokens_p95": 277,
  "completion_tokens_max": 283,
  "completed_cycles": 6,
  "aborted_at_cycle": 6,
  "abort_reason": "LLMResponseParseError"
}
```

### Per-cycle trajectory — the plant is collapsing

| cycle | t_min | y_D_pre | x_B_pre | y_D* | x_B* | y_D_post | x_B_post | wall_s | rationale snippet |
|---|---|---|---|---|---|---|---|---|---|
| 0 | 0 | 0.7197 | 0.0005 | 0.72 | 0.001 | **0.0857** | 4.9e-14 | 4.7 | "plant at steady state ... small buffer ... 0.001" |
| 1 | 5 | 0.0857 | 0 | 0.95 | 0.05 | 0.8202 | 0.1516 | 6.6 | "y_D=0.0857 far below desired ... 0.95 ... 0.05" |
| 2 | 10 | 0.8202 | 0.1516 | 0.85 | 0.12 | 0.6164 | 0.0886 | 8.5 | "increase driving force ... 0.85 ... 0.12" |
| 3 | 15 | 0.6164 | 0.0886 | ? | ? | ? | ? | ~14 | (mid-recovery) |
| 4 | 20 | ? | ? | ? | ? | 0.7355 | **1.000** | ~22 | (column completely failed) |
| 5 | 25 | 0.7355 | 1.000 | 0.75 | 0.25 | 0.8623 | 0.9962 | 31.4 | "x_B=1.0000 ... near-total failure ... reduce x_B to 0.25" |

Cycle 6 produced a malformed JSON string (the abort_reason includes the truncated rationale at the parse failure):

> `{ "y_D_target": 0.8700, "x_B_target": 0.9900, "rationale": "Current y_D (0.8623) is below the desired purity for distillate, indicating under-separation; increasing y_D_target to 0.8700 pushes for higher light component recovery. Current x_B (0.9962) is very high, suggesting the bottoms is overly rich in light component — likely due to disturbance or reflux imbalance; reducing x_B_target to 0.9900 allows for slightly more light component in` *[truncated]*

`y_D_target = 0.87` and `x_B_target = 0.99` is meaningless — x_B above y_D violates physics, and the agent's reasoning has gone off-rails.

## Diagnosis — this is not a wallclock-tax issue

The cell wall_clock numbers say it's not timeout-bound:

- seed=4's max per-cycle wall: **31.4 s** (cycle 5). P95 = 29.9 s.
- 12 cycles × 31.4 s ≈ 377 s — well within 900 s. **The smoke could complete 12 cycles even at the slowest observed pace.**

The 4 silent timeouts at exactly 900 s indicate the smoke driver was **hanging on a single LLM call**, not slowly chewing through 12 cycles. The hang is upstream of the smoke's per-cycle log line, which would have given us partial progress otherwise.

Hypothesis: under the catastrophic plant state at this corner (`y_D` collapses to 0.09, then `x_B` saturates at 1.0 — both physically meaningless), the agent's prompt body becomes increasingly long. The `Run IAE so far:` field accumulates fast (IAE = 23.9 over 6 cycles), and the Observer report becomes a description of an impossible plant state. Nemotron's response to this prompt is either:

1. A coherent JSON inside the 512-token cap that gets parsed (seed=4 cycles 0–5);
2. A response that exceeds the 512-token cap mid-JSON, gets truncated, fails the JSON parser (seed=4 cycle 6 — abort_reason rationale ends mid-sentence);
3. A response that goes into an internal model loop NIM's edge proxy doesn't terminate within 900 s on our timeout boundary (the 4 silent timeouts).

The pre_submission_checklist §4.6 LV near-singularity at F=0.8/zF=0.45 (cond(G_mv) ≈ 6800 vs 150 nominal) is the root cause of the plant collapse the agent is reading. The agent is **responding to a physically meaningless plant signal** because LT/VB stayed pinned at nominal (2.706, 3.206) — those MVs are not appropriate for F=0.8 operation. The MPC's QP at this near-singular linearization point amplifies the agent's tiny proposed targets into wrong-direction bounded-saturating commands, and the plant goes into the catastrophic regime by cycle 1.

## What this means for the screening

- This is the documented LV-singularity behavior C1 already exposes (C1 IAE at this same factor cell = 160.7 mole-fraction·min). The agent doesn't have a way to escape it at the smoke's tick cadence + cycle budget.
- Raising `--cell-timeout` to 1500 s would not help: per-cycle wall is already < 32 s. The hang is either inside a single LLM call (NIM-side latency at long-prompt regime) or inside the JSON parser's regex at malformed long output. Neither is fixed by waiting longer.
- The screening can't make Bucket-B claims at this corner with the current architecture: cells fail at a 5/5 rate so far, and even the one cycle-6 abort had y_D_post = 0.86, x_B_post = 0.996 — nowhere near operator-spec (0.99, 0.01).

## Path forward — decisions for docs-side

This needs a strategic decision before the sweep restarts. Three plausible directions:

**1. Skip the LV-singular OPs and report Schritt-B on the remaining 3 corners.**
The §4.6 disclosure already documents that fixed-weight linearized supervisory layers cannot handle this regime. Restrict the screening grid to `(0.8, 0.55), (1.2, 0.45), (1.2, 0.55)` — three corners where C1 numbers are well-defined and the agent has a chance. Bucket-B claim becomes "Bucket B on the 3 stable corner OPs; the 4th corner is structurally infeasible per §4.6 for both C1 and C2."

**2. Add an MV pre-stage to the target_acquisition setup.**
Currently target_acquisition starts at LT0 = nominal, VB0 = nominal at off-nominal F/zF. That's why the plant collapses: nominal MVs at off-nominal OP land in the catastrophic LV-singular regime immediately. If we pre-stage to LT*, VB* (the same pair the disturbance_rejection submetric already uses), the agent reads a sensible plant state and can probably operate normally. But this would change the §2.3 sub-metric definition — currently §2.3 measures "drive composition on-spec from off-spec starting point", which depends on LT/VB starting at nominal-spec values. Changing it would need a §2.3 amendment.

**3. Accept high failure rate as a Bucket-B data signal.**
If C1 also struggles here (IAE 160), C2 failing entirely at this corner is consistent with the §4.6 architectural argument. The screening continues to completion (with 4 of 50 cells at this corner expected to fail mostly), the analyzer aggregates over the 36 successful cells per submetric, and the Bucket-B verdict is computed on the remaining grid. Cost: ~50 cells × 900 s = ~12 hours of wallclock on cells that produce no usable data.

## Sweep recovery commands (when direction is decided)

Manifest is preserved at:
```
data/runs/c2_offnominal_screening/nemotron-3-super-120b-a12b/sweep_manifest.json
```

Same nohup chain with idempotent restart will pick up from 394 pending + retry the 5 failed + 1 interrupted under whichever new configuration:

```bash
nohup bash -c '
  uv run python tools/run_offnominal_screening.py \
    --model nvidia/nemotron-3-super-120b-a12b \
    --output-root data/runs/c2_offnominal_screening/nemotron-3-super-120b-a12b \
    --retry-failed && \
  uv run python tools/analyze_offnominal_screening.py \
    --output-root data/runs/c2_offnominal_screening/nemotron-3-super-120b-a12b
' > ~/offnominal_screening_$(date +%Y%m%d_%H%M%S).log 2>&1 &
```

(The `--retry-failed` flag includes the 5 failed cells in the next run; without it they stay failed and only the 394 pending + 1 interrupted are attempted.)
