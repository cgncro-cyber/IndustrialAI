# KPI Definitions — IndustrialAI

This document defines the Key Performance Indicators used to compare the four configurations C0 / C1 / C2 / C3 (per ADR 006) and to discriminate between the Phase 3 outcome buckets (per `docs/pre_submission_checklist.md` §2.2). It is the authoritative source for KPI definitions; if a notebook, script, or paper section disagrees with this document, this document wins until amended here.

The KPIs are intentionally simple, with explicit computation pseudocode where ambiguity could leak in. Where a KPI depends on configuration choices (window lengths, integration intervals, percentile choices), those choices are pinned here.

All KPIs are evaluated on the same disturbance scenario sets, with the same regulatory layer (LV-level closure per ADR 006), the same plant (Skogestad Column A in LV configuration per ADR 007), and the same nominal operating point (F = 1, zF = 0.5, qF = 1) unless explicitly noted otherwise. This is the apples-to-apples contract.

---

## 1. Primary KPI — `aggregate_iae`

The headline performance metric used since Phase 2 Day 2.5. Sum of per-scenario Integral Absolute Error of composition variables over the canonical 5-scenario disturbance set, evaluated against the nominal-OP composition targets.

### 1.1 Definition

For each scenario *s* in the canonical scenario set *S*, with simulation horizon *T_s*:

```
IAE_s = ∫_{0}^{T_s} ( |y_D(t) - y_D_target| + |x_B(t) - x_B_target| ) dt
```

Aggregate over the scenario set:

```
aggregate_iae = Σ_{s ∈ S}  IAE_s
```

Units: mole-fraction · minutes.

### 1.2 Canonical scenario set (frozen as of Phase 2 close, commit bb2bcf4)

| # | Name | Disturbance | Magnitude | Horizon | Initial X0 |
|---|---|---|---|---|---|
| 1 | `F_step_+20pct` | Feed-flow step | F: 1.0 → 1.2 | 240 min | nominal SS |
| 2 | `F_step_-20pct` | Feed-flow step | F: 1.0 → 0.8 | 240 min | nominal SS |
| 3 | `zF_step_+10pct` | Feed-composition step | zF: 0.5 → 0.55 | 240 min | nominal SS |
| 4 | `zF_step_-10pct` | Feed-composition step | zF: 0.5 → 0.45 | 240 min | nominal SS |
| 5 | `yD_setpoint_+0.5pct` | Distillate-setpoint step | y_D_target: nominal → +0.5 % | 240 min | nominal SS |

Targets:
- For scenarios 1–4 (disturbance rejection): y_D_target and x_B_target remain at their nominal SS values; IAE counts the deviation from nominal.
- For scenario 5 (setpoint tracking): y_D_target steps at t = 0; IAE counts the deviation from the new target.

### 1.3 Implementation contract

- Composition signals sampled at the simulator integration rate (sub-minute), not the supervisory cadence. The agent's slower decision frequency does not change the KPI's resolution.
- Numerical integration via trapezoidal rule on the sampled trajectory. No additional filtering before IAE computation.
- No anti-windup or wind-up correction applied to the KPI itself; controllers handle their own integral state.
- Each scenario produces a deterministic IAE under a fixed seed (per Reproducibility Rule 1). For configurations with stochastic components (C2/C3 with non-deterministic LLM sampling), the KPI is reported as `mean ± bootstrap 95 % CI over N ≥ 10 seeds` per Phase 5 statistical guardrails.

### 1.4 Per-scenario disaggregation (mandatory for paper)

The aggregate is the headline; the per-scenario breakdown is what tells the story. The paper Results section presents a table with one row per scenario and one column per configuration (C0/C1/C2/C3), with the aggregate as the final row. Phase-2 Day 3 numbers already follow this format; Phase 3 extends it with the C2 column.

---

## 2. Bucket B KPI — `off_nominal_robustness_iae`

Discriminates between Bucket A (agent dominates C1 in aggregate) and Bucket B (agent ≈ C1 in aggregate, dominates off-nominal). Required for Phase 3 evaluation.

### 2.1 Rationale

The Phase 2 Day 3 result (C1 wins 5/5, aggregate ratio 6.8×) shows that linear MPC dominates on the nominal-OP scenario set. The remaining performance gap for an agentic layer is off-nominal robustness — operating points where C1's linear model assumptions degrade. This KPI quantifies that.

A direct empirical anchor: in Phase 2 Day 2.6, fixed-gain TL did not extrapolate to F ± 20 % OPs (per `pre_submission_checklist.md` §4.4). C1 inherits a milder version of the same locality through its per-OP linearization, which is itself a snapshot. An agent that re-triggers linearization or shifts setpoint targets under regime change is exactly what this KPI rewards.

### 2.2 Definition

Define the off-nominal evaluation grid as the Cartesian product:

```
G = { (F, zF) :  F ∈ {0.8, 0.9, 1.1, 1.2},
                 zF ∈ {0.45, 0.475, 0.525, 0.55} }
```

16 off-nominal operating points. The nominal OP (F=1.0, zF=0.5) is excluded by construction — it is already covered by `aggregate_iae`.

For each off-nominal OP *g ∈ G*, run the same five canonical disturbance scenarios *S*, with initial X0 sourced from `data/operating_window_states.parquet` via `lookup_lv_ss(F, zF)`. Compute the per-OP aggregate IAE:

```
iae_g = Σ_{s ∈ S}  IAE_{s, g}
```

The headline robustness KPI is the 95th-percentile of the per-OP aggregate IAEs over the grid:

```
off_nominal_robustness_iae = P95( {iae_g : g ∈ G} )
```

P95 (not max) is used to dampen the influence of a single pathological OP. Max is reported alongside as a secondary diagnostic.

### 2.3 Implementation contract

- X0 for each off-nominal OP is loaded from the Phase-1 sweep cache. No runtime Newton-Krylov re-solve (per `pre_submission_checklist.md` §1.1).
- For C0: the same fixed-gain TL controller is applied at each off-nominal OP without re-tuning. NaN trajectories (per §4.4) are counted as IAE = infinity for percentile computation; report a `infeasibility_count` alongside.
- For C1: the linearization is recomputed at each off-nominal OP via the same `lookup_lv_ss` X0 (this is how the MPC stays valid at off-nominal OPs). Report `linearization_recompute_count` to make the C1-vs-C2 mechanism transparent.
- For C2/C3: no special handling; the agent and safety gate operate identically to nominal-OP runs.
- Seeds: same per-configuration N ≥ 10 protocol as `aggregate_iae`.

### 2.4 Discriminating threshold (for bucket classification)

Bucket B classification uses a three-band interpretation of the C2 vs C1 P95 ratio, rather than a binary cutoff:

| C2 P95 vs C1 P95 | Classification |
|---|---|
| ≥ 2.0× improvement | Bucket B with **strong evidence** |
| 1.5× – 2.0× improvement | Bucket B with **moderate evidence** (minimum threshold) |
| 1.0× – 1.5× improvement | Ambiguous → §6 Decision Tree Step 5 (methodology revisit) |
| < 1.0× (C2 worse than C1 off-nominal) | Phase 3 failure, not silent reclassification |

The 1.5× minimum is calibrated against the Day-3 zF_step_+10 % 1.5× C1-over-C0 gap — the empirical signal that the agent has room to exploit on coupled disturbances. The 2.0× "strong evidence" band exists so that the paper Discussion can characterize the evidence honestly rather than collapsing a marginal result into a binary claim.

The threshold is **not** a Phase 3 success gate. Phase 3 succeeds when results map unambiguously to one bucket, regardless of which.

### 2.5 Screening grid for Phase 3 prompt iteration

The 16-point grid in §2.2 is the headline evaluation grid for the final paper. For iterative Phase 3 prompt and graph development, a coarser 4-point screening grid is used to cut C2 iteration compute by approximately 75 %:

```
G_screening = { (F, zF) :  F ∈ {0.8, 1.2},  zF ∈ {0.45, 0.55} }
```

Four off-nominal extremes (corners of the full grid). Results on the screening grid are **diagnostic only** — they inform prompt iteration during Phase 3 development but are never reported as the headline KPI. Final Phase 3 evaluation always runs the full 16-point grid from §2.2.

---

## 3. Bucket C KPI — `constraint_violation_intercept_rate`

Discriminates between Bucket A/B (agent improves performance) and Bucket C (agent is enabled by the safety gate to explore aggressively, with the gate as the load-bearing contribution). Required for Phase 3/4 evaluation.

### 3.1 Rationale

If the agent's contribution is *aggressive exploration that the safety gate makes safe*, the KPI that captures this is not aggregate performance but the rate at which the safety gate intercepts setpoint proposals that would have led to plant-physically-unsafe states.

The "what would have happened" counterfactual is essential — without it, a high intercept rate could mean either "the gate is doing important work" or "the agent is incompetent and the gate is just papering over it."

### 3.2 Definition

For each agent proposal *p* during a scenario run *s*, define:

- *blocked_p* ∈ {0, 1}: did the safety gate block this proposal?
- *counterfactual_unsafe_p* ∈ {0, 1}: if the proposal had been executed on a forked twin trajectory, would the resulting state have violated a documented safety constraint within a defined counterfactual horizon?

The intercept rate is the proportion of blocked proposals that the counterfactual confirms as truly unsafe:

```
constraint_violation_intercept_rate
    = ( count of (blocked_p = 1 AND counterfactual_unsafe_p = 1) )
      ÷ ( count of (blocked_p = 1) )
```

Range: [0, 1]. Higher means the gate is blocking real threats; lower means the gate is over-conservative.

A second derived metric captures gate sensitivity:

```
constraint_violation_detection_rate
    = ( count of (blocked_p = 1 AND counterfactual_unsafe_p = 1) )
      ÷ ( count of (counterfactual_unsafe_p = 1) )
```

This is the true-positive rate. Together, intercept_rate and detection_rate characterize the gate; reporting both is required.

### 3.3 Counterfactual implementation

- The forked twin is a deep copy of the plant state at the moment of the agent proposal.
- The counterfactual horizon is 30 minutes (six 5-minute supervisory cycles) post-proposal, integrated on the forked twin under the proposed setpoint without further agent intervention. The regulatory PI continues to operate normally on the forked twin.
- **Rationale for 30 min.** ≈ 6 supervisory cycles ≈ 3× the dominant coupled-plant Pu (11 min, from Phase 2 Day 2.5 relay tests). Captures immediate consequences without testing long-term recovery dynamics that next-cycle agent intervention would handle anyway.
- **Fast-fail sub-check at 5 min.** Constraint violations that occur within the first 5 minutes of the counterfactual (i.e., before a single supervisory cycle could intervene) are flagged separately and aggregated as `fast_fail_count`, reported alongside the 30-min counterfactual results. A proposal that leads to immediate harm is categorically worse than one that drifts slowly toward a constraint; the paper Discussion separates these cases.
- Safety constraints (definitive list, pinned here):
    - y_D > 0.99 (distillate purity upper bound)
    - y_D < 0.97 (distillate purity lower bound)
    - x_B < 0.005 (bottoms purity upper bound)
    - x_B > 0.03 (bottoms purity lower bound)
    - M_D outside [0.2 × M_D_nominal, 1.8 × M_D_nominal] (accumulator overflow / runs dry)
    - M_B outside [0.2 × M_B_nominal, 1.8 × M_B_nominal] (sump overflow / dry boiling)
    - any composition outside [0, 1] (physical infeasibility)
    - any negative flow rate (physical infeasibility)
    - any LT or VB outside the operating envelope documented in `column_a/assumptions.md`
- Rate-of-change bounds on flows (mechanical valve damage) deliberately excluded for the methodology paper; reintroduce in Phase 6 (ADR 008) if real-plant data motivates them.
- The counterfactual evaluation is itself a CI-bracketed quantity; report mean ± bootstrap 95 % CI per Phase 5 protocol.

### 3.4 Discriminating threshold

Bucket C classification requires: *C2 aggregate_iae* is within 20 % of *C1 aggregate_iae* (i.e., the agent does not dominate on raw performance) AND *constraint_violation_intercept_rate* > 0.7 (i.e., the gate is meaningfully active and accurate) AND at least 3 documented false-negative case studies show the gate catching specific unsafe proposals (per PROJECT_PLAN Phase 4 binding deliverable).

---

## 4. Auxiliary diagnostic — `linearization_consistency`

Not a primary discriminating KPI but required for the Bucket B Methods paragraph. Quantifies the OP-locality of C1's linearization, making the Bucket B story empirically grounded rather than rhetorical.

### 4.1 Definition

For each off-nominal OP *g ∈ G* (same grid as §2.2), compute the spectral norm of the difference between C1's nominal linearization *A_nom* and the linearization re-identified at *g* (*A_g*):

```
linearization_drift_g = || A_g - A_nom ||_2
```

Aggregate as the 95th percentile:

```
linearization_consistency = 1 - P95( linearization_drift_g / ||A_nom||_2 )
```

Range: (-∞, 1]; closer to 1 means C1's linearization is consistent across the OP grid (locality is mild); closer to 0 (or negative) means strong locality (Bucket B opportunity).

### 4.2 Implementation contract

- A matrices via the same CasADi linearization pipeline used by C1 (`build_lv_closed_rhs` → symbolic Jacobian → numeric evaluation at the OP-specific X0).
- The metric is reported in Phase 5 Methods as a structural property of the LV-closed plant, not as a configuration-comparison KPI. It informs the Bucket B narrative without being scored against.

---

## 5. Auxiliary diagnostic — `supervisory_cycle_wallclock`

Required for the deployability/production-readiness argument in the paper, and as a soft gate for Phase 3 viability.

### 5.1 Definition

For each supervisory cycle (every 5 min by default per ADR 006) during a scenario run, record the wallclock time from the moment the agent receives its observation to the moment it returns a decision (or the safety gate returns its verdict for C3). Aggregate:

```
supervisory_cycle_wallclock = { mean, P50, P95, P99, max } over all cycles in the scenario set
```

### 5.2 Soft gate

A configuration is **deployable** if `P95 ≤ 60 s` (i.e., 95 % of supervisory decisions complete in ≤ 20 % of the 5-min cadence). The Phase 2 C1 value is 345 ms max (well within budget). Phase 3 C2 and Phase 4 C3 inherit this gate.

If C2's P95 exceeds 60 s, this is reported as a known limitation in the Methods section, not as a Phase 3 failure. The gate is a viability signal, not a publication blocker.

---

## 6. Bucket classification decision flow (Phase 3 evaluation)

Phase 3 produces C2 numbers; bucket classification follows this decision tree:

1. Compute `aggregate_iae` (§1) and `off_nominal_robustness_iae` (§2) for C2.
2. If C2 `aggregate_iae` < 0.85 × C1 `aggregate_iae` (i.e., agent dominates by ≥ 15 % in aggregate): **Bucket A** candidate. Validate by checking that this dominance holds in ≥ 4 of 5 individual scenarios. If yes, classify Bucket A.
3. Else if C2 `off_nominal_robustness_iae` < 0.67 × C1 `off_nominal_robustness_iae` (i.e., 1.5× improvement at P95 over the off-nominal grid): **Bucket B**. Validate by checking that the improvement is concentrated at the OPs with highest `linearization_drift_g` (§4) — i.e., agent's advantage is where MPC's linear model is most stressed.
4. Else if C3 `constraint_violation_intercept_rate` > 0.7 AND `constraint_violation_detection_rate` > 0.7 AND ≥ 3 false-negative case studies documented: **Bucket C**.
5. Else: results are ambiguous. Phase 3 has not produced a publishable outcome and the methodology requires revisiting. This is a documented failure mode, not a hidden one.

Step 5 is the unambiguous-mapping criterion from `pre_submission_checklist.md` §2.2. If Phase 3 lands here, the response is to extend the off-nominal grid, refine the safety constraints, or revisit ADR 005 LLM choice — not to massage the data into a bucket it does not fit.

---

## Review resolutions (2026-05-27)

The four open items at initial draft time were reviewed and resolved:

- **Counterfactual horizon (§3.3): 30 min retained.** Rationale tightened (≈ 3× coupled-plant Pu). Fast-fail sub-check at 5 min added to discriminate immediate-harm from slow-drift proposals.
- **Off-nominal grid size (§2.2): 16 points retained as headline evaluation grid.** Coarser than a Phase-1-style sweep but sufficient for Phase-3-evaluation purposes; if resulting CIs are too wide, the grid can be densified post-hoc without invalidating earlier runs. 4-point screening grid (§2.5) added for Phase 3 prompt iteration only — diagnostic, never reported as KPI.
- **Safety constraint list (§3.3): expanded from 7 to 9 items.** Added M_D and M_B holdup bounds (accumulator overflow / dry, sump overflow / dry boiling). Citation corrected from "ADR 005" to `column_a/assumptions.md`. Rate-of-change bounds on flows deliberately excluded; reintroduce in Phase 6 if real-plant data motivates.
- **Bucket B threshold (§2.4): 1.5× minimum retained.** Three-band interpretation added (≥ 2.0× strong evidence, 1.5–2.0× moderate, 1.0–1.5× ambiguous → Decision Tree Step 5, < 1.0× Phase 3 failure). Forces the paper Discussion into an honest characterization of evidence strength rather than a binary claim.

---

## Changelog

- 2026-05-27 (initial draft) — KPI definitions for the four configurations and three outcome buckets, drafted at the Phase 2 / Phase 3 boundary. Four open items flagged for review.
- 2026-05-27 (review resolved) — Four review items resolved (see "Review resolutions" above). §2.4 augmented with three-band interpretation; §2.5 added (screening grid); §3.3 augmented with fast-fail sub-check, M_D/M_B bounds, citation fix. Document is now the authoritative source for Phase 3 evaluation; further changes require an entry in this changelog and, where structural, an ADR or `pre_submission_checklist.md` update.
