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

## 2. Bucket B KPIs — off-nominal performance, split into two sub-metrics

Discriminates between Bucket A (agent dominates C1 in aggregate) and Bucket B (agent ≈ C1 nominal, dominates off-nominal). Required for Phase 3 evaluation.

**2026-05-28 amendment.** The original §2.2 defined a single composite metric, `off_nominal_robustness_iae`, that mixed two distinct effects: (a) supervisory target acquisition from a composition-off-target steady state, and (b) disturbance rejection at the off-nominal OP. The post-Schritt-2 diagnosis (`data/reference/c1_off_nominal_baseline.json`) showed that the worst-cluster IAEs (F=0.8 row, ~770) are dominated by (a) — the column starts at `y_D ≈ 0.72` and must drive 0.27 mole-fraction to the product spec while the linearized MPC saturates at the bound. Conflating (a) and (b) lets one effect mask the other and makes the Bucket-B mechanism ambiguous.

This section now defines *two* sub-metrics, both pre-registered. Bucket B can be triggered on one or both; the paper Results section reports each separately and the Methods text states which mechanism the C2 agent exploits.

### 2.1 Rationale

The Phase 2 Day 3 result (C1 wins 5/5, aggregate ratio 6.8×) shows that linear MPC dominates on the nominal-OP scenario set. The remaining performance gap for an agentic layer is off-nominal behavior — operating points where C1's linear model assumptions degrade. The Pareto sweep in `data/reference/c1_regularization_sweep.json` (see `docs/figures.md` Figure 9) confirms this is a structural limitation of fixed-weight linear MPC: no single regularization multiplier resolves both regimes while preserving the Phase-2 gate. The agentic layer's value proposition is precisely to break that trade-off — either by recognizing a near-singular regime and choosing a reachable interim target (target-acquisition variant) or by re-planning around the disturbance once the OP is composition-on-target (disturbance-rejection variant).

A direct empirical anchor: in Phase 2 Day 2.6, fixed-gain TL did not extrapolate to F ± 20 % OPs (per `pre_submission_checklist.md` §4.4). Symmetrically, at F=0.8 cluster the Skogestad LV configuration's `G_mv` becomes near-singular (`cond(G_mv) = 6800` vs 150 nominal, see `pre_submission_checklist.md` §4.6) — a structural property of LV in this regime that constrains *any* fixed-weight linearized supervisor.

### 2.2 Off-nominal grid (shared by both sub-metrics)

Both sub-metrics evaluate on the same Cartesian-product grid:

```
G = { (F, zF) :  F ∈ {0.8, 0.9, 1.1, 1.2},
                 zF ∈ {0.45, 0.475, 0.525, 0.55} }
```

16 off-nominal operating points. The nominal OP (F=1.0, zF=0.5) is excluded by construction — it is already covered by `aggregate_iae` (§1).

### 2.3 Sub-metric A — `off_nominal_target_acquisition_iae`

**What it measures.** Whether the supervisor can drive the column from its *natural* off-nominal LV-closed steady state (composition-off-target at nominal LT/VB) to the nominal product spec (y_D = 0.99, x_B = 0.01) while a canonical disturbance is applied. This is the existing Schritt-2 metric, renamed and made explicit.

**Setup.** For each OP `g = (F, zF) ∈ G`:

- `X0_g = lookup_lv_ss(F, zF)` — the LV-closed SS at nominal `LT = L0 = 2.706`, `VB = V0 = 3.206`. The composition at `X0_g` is off-target by construction (e.g., `y_D ≈ 0.72` at F=0.8/zF=0.45).
- Run all five canonical disturbance scenarios `S` with the same relative magnitudes from the OP (see `industrial_ai.control.off_nominal_scenarios`).
- IAE measured against the *nominal* product spec `(y_D = 0.99, x_B = 0.01)`.

**Headline KPI.** P95 of per-OP aggregate IAEs (max reported alongside).

**Feasibility verified (2026-05-28).** At the worst-cluster OP F=0.8/zF=0.45 the target `(0.99, 0.01)` is reachable inside the MV box `(LT, VB) ∈ [0, 10]^2` — empirical search lands at `(LT ≈ 2.79, VB ≈ 3.15)` producing `(y_D = 0.995, x_B = 0.004)`. So the target-acquisition variant is mathematically well-posed (target is not outside the operating window). What fails for C1 at this OP is the *route* — `G_mv` near-singularity (§4.6) breaks the linearized MPC's gradient direction and the QP solution drives LT/VB to the wrong bound.

### 2.4 Sub-metric B — `off_nominal_disturbance_rejection_iae`

**What it measures.** Disturbance rejection *only*, without the initial target-acquisition transient. Isolates the pure regime-change-response question.

**Setup.** For each OP `g = (F, zF) ∈ G`:

- Pre-stage `X0_g_onspec`: the steady state at OP `g` *with `(LT, VB)` adjusted* so that the composition is on-spec (`y_D = 0.99 ± 5e-4`, `x_B = 0.01 ± 5e-4`). Found by inverse search over the MV box (Newton iteration on the LV-closed residual with the composition-on-target constraint).
- Run all five canonical disturbance scenarios `S` with the same relative magnitudes from the OP. The supervisor under test (C0/C1/C2/C3) starts at `X0_g_onspec` with its MV state initialized at the corresponding `(LT*, VB*)` — i.e., the column is already running on-spec at this OP before the step.
- IAE measured against the nominal product spec `(y_D = 0.99, x_B = 0.01)`.

**Headline KPI.** P95 of per-OP aggregate IAEs.

**OPs that cannot be pre-staged.** If a particular OP has no `(LT, VB)` inside `[0, 10]^2` producing a composition-on-target SS (none currently observed in the 16-point grid, but defensive against future grid extensions), the OP is excluded from the sub-metric B aggregate with an explicit `infeasible_op_count` alongside.

**Implementation note.** The pre-stage Newton iteration is run *once per OP* and the resulting `(X0_g_onspec, LT*, VB*)` cached in `data/reference/off_nominal_on_spec_pre_stages.json`. Computation cost: one Newton solve per OP × 16 = ~5 s total, run as a Phase-3 prep tool when the C2 evaluation begins.

### 2.5 Implementation contract (both sub-metrics)

- Sub-metric A `X0_g` comes from the Phase-1 sweep cache (`operating_window_states.parquet`). No runtime Newton-Krylov re-solve.
- Sub-metric B `X0_g_onspec` and `(LT*, VB*)` come from a one-off Newton search and are cached after first computation; subsequent C0/C1/C2/C3 runs use the cache (per `pre_submission_checklist.md` §1.1).
- For C0: the same fixed-gain TL controller is applied at each off-nominal OP without re-tuning. Trajectories that NaN-out or trip the wall-clock cap are counted as IAE = infinity for percentile computation; an `infeasibility_count` is reported alongside.
- For C1: the linearization is recomputed at each off-nominal OP. For sub-metric B, the linearization is built around `X0_g_onspec`, not the natural off-nominal SS, so the operating point and the supervisor's internal model agree on where the column already is. Report `linearization_recompute_count` to make the C1-vs-C2 mechanism transparent. The Pareto-reference C1 (`r_lt = r_vb = 10`, the strongest gate-passing variant from §6 Step 3) is used in the headline comparison; the under-regularized `r = 0.1` variant is also reported for the Pareto figure but not used as the head-to-head reference.
- For C2/C3: no special handling; the agent and safety gate operate identically to nominal-OP runs on both sub-metrics.
- Seeds: same per-configuration N ≥ 10 protocol as `aggregate_iae`.

### 2.6 Discriminating threshold (for bucket classification)

Bucket B classification uses a three-band interpretation of the C2 vs C1 P95 ratio on *either or both* sub-metrics:

| C2 P95 vs C1 (Pareto-reference) P95 | Classification |
|---|---|
| ≥ 2.0× improvement | Bucket B with **strong evidence** on that sub-metric |
| 1.5× – 2.0× improvement | Bucket B with **moderate evidence** on that sub-metric (minimum threshold) |
| 1.0× – 1.5× improvement | Ambiguous on that sub-metric → §6 Decision Tree Step 5 input |
| < 1.0× | C2 worse than C1 on that sub-metric — sub-metric does not support Bucket B |

The paper Results section reports the table separately for both sub-metrics and the Methods text states whether the agent's contribution is (a) target acquisition under near-singular regime, (b) disturbance rejection under regime change, or both.

The 1.5× minimum is calibrated against the Day-3 zF_step_+10 % 1.5× C1-over-C0 gap — the empirical signal that the agent has room to exploit on coupled disturbances. The 2.0× "strong evidence" band exists so that the paper Discussion can characterize the evidence honestly rather than collapsing a marginal result into a binary claim. The threshold is **not** a Phase 3 success gate. Phase 3 succeeds when results map unambiguously to one bucket, regardless of which.

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

Phase 3 produces C2 numbers; bucket classification follows this decision tree. All comparisons use `mean ± bootstrap 95 % CI` per the Phase 5 statistical protocol (N ≥ 10 independent seed runs per configuration, bootstrap resampling at the seed level — see §1.3 and `PROJECT_PLAN.md` Phase 5 statistical guardrails). Point estimates without CIs are insufficient for classification: the small absolute scale of the Phase-2 C1 aggregate IAE (0.122 mole-fraction · min) means a 5 % point-estimate gap between C2 and C1 is not necessarily statistically distinguishable at N = 10 seeds, and the tree below would otherwise risk mapping noise to a Bucket-A or Bucket-B claim.

1. Compute `aggregate_iae` (§1) and `off_nominal_robustness_iae` (§2) for C2, each as a mean and 95 % bootstrap CI over the N ≥ 10 seed runs. C1's headline numbers from Phase 2 Day 3 stand as point estimates (deterministic MPC, no seed-driven stochasticity) and are used as the comparison baseline.
2. **Bucket A candidate.** C2's 95 % bootstrap CI on `aggregate_iae` must lie *entirely below* C1's point estimate **and** the 0.85× ratio threshold must hold against C2's CI *upper bound* (not its mean). If both conditions hold, validate that the dominance also holds in ≥ 4 of 5 individual scenarios (per-scenario CIs likewise entirely below their C1 counterparts). If yes, classify Bucket A.
3. **Bucket B candidate.** Otherwise, if C2's 95 % bootstrap CI on *either or both* of the two Bucket-B sub-metrics — `off_nominal_target_acquisition_iae` (§2.3) and `off_nominal_disturbance_rejection_iae` (§2.4) — lies *entirely below* 0.67 × the *Pareto-reference* C1 point estimate on the same sub-metric (i.e., the 1.5× improvement at P95 must clear C2's CI upper bound): **Bucket B**. Validate by checking that the improvement is concentrated at the OPs with highest `linearization_drift_g` (§4) — i.e., agent's advantage is where MPC's linear model is most stressed.

    **Sub-metric labeling.** The Methods paragraph for Bucket B (see `paper/methods_phase3_buckets.md`) must explicitly name which sub-metric drove the classification: "Bucket B (target acquisition)", "Bucket B (disturbance rejection)", or "Bucket B (both)". Reporting only the union obscures the agent's mechanism.

    **Pareto-reference C1 for Bucket B.** The C1 baseline used in this comparison is *not* the Phase-2-nominal-baseline variant (`r_lt = r_vb = 0.1`, the strongest nominal performer that collapses off-nominal). It is the *strongest gate-passing fixed-weight C1 variant* from the regularization sweep in `data/reference/c1_regularization_sweep.json` — i.e., the variant that minimizes off-nominal aggregate IAE among multipliers that still win ≥ 3 / 5 nominal scenarios against C0. In the Day-5 build this is the ×100 variant (`r_lt = r_vb = 10`); per `docs/figures.md` Figure 9, no fixed-weight tuning that satisfies the Phase-2 gate produces a lower off-nominal IAE on either sub-metric. Comparing C2 against this Pareto-reference point (rather than the under-regularized ×1 variant) pre-empts the "strawman MPC" reviewer objection — C2 must beat the strongest gate-passing fixed-weight competitor, not the weakest.
4. **Bucket C candidate.** Otherwise, if C3 `constraint_violation_intercept_rate` > 0.7 AND `constraint_violation_detection_rate` > 0.7 (both reported with bootstrap 95 % CIs whose lower bounds exceed 0.7) AND ≥ 3 false-negative case studies documented: **Bucket C**.
5. **Ambiguous (publishable failure mode).** Otherwise — including the case where any of the CIs in Steps 2-4 overlap their respective thresholds rather than clearing them — the result is classified as ambiguous. This is the explicit landing zone when a C2-vs-C1 difference is real but not statistically separated, and it is a *publishable* outcome under the unambiguous-mapping criterion of `pre_submission_checklist.md` §2.2: the methodology paper reports the negative result honestly, identifies which step's CI overlap drove the classification, and proposes which next moves (denser off-nominal grid, larger N, refined safety constraints, ADR 005 LLM swap) the evidence motivates. *Ambiguous is an outcome to be reported, not avoided.*

The bootstrap-CI-driven gate is a strengthening of the original point-estimate decision tree, not a change in spirit: Steps 2-4 already framed bucket classification as a strong claim, and CIs are what make the strength quantitative. If a CI is too wide to clear its threshold, the appropriate response is N enlargement or grid densification (mechanical, not methodological), not threshold relaxation. Step 5 stays the landing pad whenever neither route restores separation.

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
- 2026-05-28 (bootstrap-CI gate) — §6 decision tree rewritten to require bootstrap 95 % CI separation rather than point-estimate comparison for Bucket A (`aggregate_iae` CI entirely below C1 point, 0.85× threshold cleared on C2's CI upper bound) and Bucket B (`off_nominal_robustness_iae` CI entirely below 0.67 × C1 point). Bucket C similarly requires both rates' CI lower bounds above 0.7. Step 5 (ambiguous) is now explicit about being a publishable failure mode, not an outcome to avoid. Origin: `pre_submission_checklist.md` §7.1; trigger: the small absolute scale of the Phase-2 C1 aggregate IAE (0.122) means a 5 % point-estimate gap risks classifying noise as Bucket A at N = 10 seeds.
- 2026-05-28 (Pareto-reference C1 for Bucket B) — §6 Step 3 amended so the Bucket-B comparison uses the *strongest gate-passing fixed-weight* C1 variant (from `data/reference/c1_regularization_sweep.json`, Day-5 build: ×100, `r_lt = r_vb = 10`) instead of the Phase-2-nominal baseline (×1, `r_lt = r_vb = 0.1`). The under-regularized ×1 variant collapses MVs at the F=0.8 cluster (IAE ≈ 770 worst OP), which would let any reasonable C2 score a hollow Bucket-B win against a strawman C1. Comparing against the Pareto-reference forces C2 to beat the strongest fixed-weight competitor that survives the Phase-2 gate. Motivated by the irreducible nominal/off-nominal trade-off documented in `docs/figures.md` Figure 9.
- 2026-05-28 (off-nominal metric split — pre-registered amendment) — §2 restructured. The original single composite `off_nominal_robustness_iae` mixed two distinct mechanisms (supervisory target acquisition from a composition-off-target SS, and disturbance rejection at the off-nominal OP). The post-Schritt-2 diagnostic showed target acquisition dominates the worst-cluster IAEs, masking the disturbance-rejection signal. §2 now defines two sub-metrics: §2.3 `off_nominal_target_acquisition_iae` (the previous mixed metric, renamed; X0 from `lookup_lv_ss` at nominal LT/VB) and §2.4 `off_nominal_disturbance_rejection_iae` (new; X0 pre-staged to composition-on-target at the OP via a one-off Newton search, then disturbance applied). Bucket B can be triggered on either or both; the Methods paragraph must name which sub-metric drove the classification. Feasibility verified at worst-OP F=0.8/zF=0.45: target `(0.99, 0.01)` reachable inside MV bounds at `(LT≈2.79, VB≈3.15)`; both sub-metrics are well-posed.
