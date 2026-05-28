# Pre-Submission Checklist — IndustrialAI

Rolling tracker for items that must be resolved, pre-committed, or deliberately acknowledged before the Phase 5 arXiv preprint and SAFEPROCESS submission. Distinct from `PROJECT_PLAN.md`, which captures the durable five-phase architecture; this file is a working ledger updated as decisions accumulate and external feedback is processed.

Update this file at every phase boundary and whenever a deferred item is resolved or a new pre-commit is needed.

---

## 1. Deferred technical items (with planned fix path)

### 1.1 Off-nominal operating-point evaluation via Phase-1-Sweep X0 lookup

- **Origin.** Phase 2 Day 2.5 robustness spot-check failed with NaN under per-OP Newton-Krylov re-solve at F ± 20 %.
- **Root cause.** Newton-Krylov re-solve at perturbed OPs is ill-conditioned for the LV-closed plant with RGA(1,1) ≈ 36.
- **Fix path.** Replace per-OP re-solve with X0 lookup from `data/baseline_operating_window.csv` (1080 Phase-1-sweep-converged points).
- **Status. RESOLVED 2026-05-27 (commit d575387).** `operating_window_states.parquet` written with 1125 sweep-converged state vectors (100 % convergence, 82 s sweep time). `lookup_lv_ss(F, zF)` utility is now the canonical off-nominal X0 source. Reused immediately by the Day-2.6 robustness spot-check; will be reused by Day-3 MPC linearization and Phase-5 disturbance-scenario seeding.

### 1.2 F ± 20 % robustness under fixed-gain TL — deferred to Phase 3 by design

- **Origin.** Day-2.6 robustness spot-check found that fixed-gain TL (Kp / Ti tuned at nominal SS) does not extrapolate to F ± 20 % OPs even when X0 is correctly seeded from the sweep cache.
- **Why this is not a bug.** Fixed-gain TL is a single-OP tuning; extrapolation failure is the expected behavior, not a numerical artifact. Recovery would require gain scheduling, deliberately excluded as overkill for the C0 baseline.
- **Treatment.** Logged as a publishable limitation of C0, not as a deferred fix. Becomes a motivating argument for C1 (Linear MPC with per-OP linearization from the sweep cache).
- **Status. RESOLVED-BY-FRAMING, no further action.** See §4.4 below for the matching paper-disclosure note.

---

## 2. Pre-commits for paper discussion sections

These exist so empirical results land in a pre-defined narrative slot, not a hastily-improvised one. Write the matching paragraph variant only after the run; discard the others.

### 2.1 TL retune on decoupled plant — RESOLVED, Outcome A confirmed (2026-05-27)

Empirical result: TL_with_decoupler_retuned reaches IAE 2.81 (vs 2.96 naive, vs 0.84 TL_no_decoupler). Outcome A applies.

**Locked discussion paragraph (Outcome A):**
*"Decoupling does not simplify tuning even with consistent relay calibration on the effective plant. The decoupler slows the dominant time constant by approximately one order of magnitude (Pu_top: 11 → 107 min), which RGA-driven gain reduction does not fully compensate. Tyreus-Luyben's fixed Kp / Ti formula, derived from coupled-plant relay data, does not adapt to this slowdown; it produces a controller that is well-matched to the coupled plant and poorly matched to the decoupled plant."*

**Additional methodological insight from the run (worth a sentence in Discussion):** model-based tuning rules (SIMC) adapt to the decoupler-induced slowdown automatically because they use the updated model time constant; model-free tuning rules (TL, relay-only) cannot. This explains the ranking pattern — SIMC+Decoupler reasonable (1.18), TL+Decoupler catastrophic (2.81) even after fair retune. The paper-grade framing: *"Decoupling places a structural requirement on the tuning method, not only on the plant model."*

Outcome B paragraph discarded.

### 2.2 Phase 3 — agent vs C1 (Linear MPC) outcome buckets

Do **not** pre-commit to a binary "agent must beat MPC by X %" success threshold. That framing forecloses the most interesting paper variants. Three outcome buckets, each publishable, each with a pre-drafted Methods paragraph at Phase-3 kickoff:

- **Bucket A — agent dominates MPC in aggregate IAE.** Story: *"Agentic supervisory control outperforms industrial-baseline linear MPC across the disturbance scenario set."*
- **Bucket B — agent ≈ MPC in aggregate, agent dominates at off-nominal OPs / regime changes / mode switches.** Story: *"Agentic layer adds robustness rather than raw performance; benefits emerge under regime change and off-nominal operation, conditions where linear MPC degrades by design."*
- **Bucket C — agent ≈ MPC in aggregate, safety gate catches MPC-feasible-but-plant-unsafe setpoints.** Story: *"The safety gate, not the agent, is the load-bearing contribution; agentic exploration is enabled because the gate constrains the action space safely."*

**Process.** At Phase 3 kickoff, draft one Methods paragraph per bucket. After empirical results, select the matching one and discard the others. The acceptance criterion for Phase 3 is *not* "Bucket A reached"; it is "results unambiguously map to exactly one bucket."

**Updated bucket-probability assessment after Day-3 C1 results (2026-05-27).** Day-3 produced an aggregate C1/C0 ratio of 6.8× (C1 IAE 0.1224 vs C0 IAE 0.8362), winning all 5 of 5 scenarios. The C1 baseline is therefore much stronger than originally anticipated, which makes Bucket A harder and Bucket B more attractive. Pre-Phase-3 probability re-estimate:

- Bucket A: ~15 % (down from ~35 %). 0.12 aggregate IAE is a high bar for an agentic layer to beat across the same nominal-OP scenario set.
- Bucket B: ~60 % (up from ~40 %). Two empirical signals point here: (i) the zF_step_+10 % scenario shows only a 1.5× C1-over-C0 gap, indicating that C1's linear-model assumptions weaken under coupled disturbances; (ii) C0's fixed-gain TL does not extrapolate to F ± 20 % OPs (§4.4), and C1's per-OP linearization inherits a milder version of the same locality. An agent that re-evaluates the linearization point or switches setpoint targets under regime change is well-positioned to exploit this gap.
- Bucket C: ~25 % (unchanged). Orthogonal to the agent-vs-MPC gap; depends on Phase-4 detector design.

**Implication for Phase 3 design.** Engineer the disturbance scenario set for C2 to include explicit off-nominal and regime-change evaluation, not only the five nominal-OP scenarios used for C0/C1. The off-nominal-robustness KPI (§5.2) needs to be operationalized before the agent is built, not after.

### 2.3 Phase 4 — safety-gate cross-domain transfer outcome handling

Already enumerated in `PROJECT_PLAN.md` Phase 4 ("cross-domain primary, in-domain fallback"). No further pre-commit needed; documented here for visibility.

---

## 3. Binding deliverables added post-PROJECT_PLAN

### 3.1 Phase 4 — false-negative case studies (binding)

External review (May 2026) flagged that aggregate ROC curves are weak paper currency for safety contributions. Three to five documented false-negative case studies are required as a binding Phase-4 deliverable. For each case:

- The unsafe setpoint proposed by the agent
- The physical danger if executed (mass-balance, energy-balance, or constraint argument)
- Whether the safety gate caught it
- Which detector signal triggered, and how close to the threshold

Bind into `notebooks/04_safety_layer.ipynb` and into a dedicated section of the paper.

### 3.2 Paper outline skeleton

Not yet drafted. To be created **before** Phase 3 prompt iteration consumes attention. Purpose:

1. Surface empirical gaps early, so Phase 3 / 4 can be scoped to fill them.
2. Save a week of structure debate in October.

Output: `paper/outline.md` with section structure and 2–3 sentences per section. Not full prose.

---

## 4. Known limitations to disclose transparently in the paper

These are not weaknesses to hide; they are anchor points that make the paper more defendable when called out upfront.

### 4.1 High RGA of the LV configuration

RGA(1,1) ≈ 36 at the nominal operating point. Inherent to the LV configuration; DV or L/D-V/B would show better steady-state decoupling. LV is retained because the regulatory layer must be held constant across C0/C1/C2/C3 (ADR 006) and LV is the canonical comparison configuration in the literature. **Disclose in Methods, not in a footnote.**

### 4.2 Local-LLM reasoning quality bounded

Local LLM (per ADR 005) bounds the agent's multi-step reasoning depth. Mitigated by the supervisory cadence of 5–15 min (ADR 006), which permits chain-of-thought time, and by the structured Observer / Optimizer / Critic decomposition. **Disclose in Methods. Do not overclaim agent reasoning ability.**

### 4.3 Single column case study

Phase 1–5 use Skogestad's Column A only. Cross-process transfer is demonstrated through the safety-gate cross-domain training (TEP → Column A), not through additional process case studies. Phase 6 (ADR 008) is the optional follow-up that adds real-plant validation. Future-work section explicitly enumerates transfer targets (semiconductor, pharma, battery, HVAC, water) without claiming empirical coverage.

### 4.4 Fixed-gain TL does not extrapolate to off-nominal OPs

The Tyreus-Luyben-tuned C0 baseline is calibrated at the nominal SS (F = 1, zF = 0.5). Day-2.6 robustness evaluation found that the same fixed gains do not maintain stability at F ± 20 % OPs, even when X0 is correctly seeded from the operating-window sweep. This is the expected behavior of single-OP fixed-gain tuning, not a numerical artifact.

**Disclose in Methods as a deliberate property of the C0 baseline.** Frame it as a motivating argument for C1 (Linear MPC with per-OP linearization from the sweep cache) and downstream for the agentic supervisor (which adapts setpoints without re-tuning the regulatory PI).

### 4.5 SIMC 2DoF benefit is scenario-mix dependent

Day-2.6 verification confirmed that the 2DoF setpoint filter is active (lower peak LT swing, slower y_D rise on the pure y_D-tracking scenario). The Aggregat-IAE near-identity between 1DoF and 2DoF (1.5696 vs 1.5656) is a scenario-mix effect: 4 of 5 shootout scenarios are disturbance-dominated, where the 2DoF filter does not contribute.

**Disclose in Methods.** Acknowledge that 2DoF was tested, that the filter is active and behaves as designed on tracking-only scenarios, and that the disturbance-dominated scenario set is the reason aggregate IAE does not separate. Do not present 2DoF as ineffective in general.

---

## 5. Open decisions tracked here, not yet ADR-worthy

### 5.1 LLM model choice — alignment between ADR 005 and current local stack

- ADR 005 names **Llama-3.3-Nemotron-Super-49B v1.5** primary, **Qwen3.6-27B** ablation.
- The project author's general-purpose local stack on the Mac Studio (as of the May 2026 strategic reset) is **Qwen3.5-122B-A10B IQ4_XS** (~65 GB).
- The two are not in conflict: ADR 005 pins project-scoped models via `configs/`; the personal stack can be different.
- **Decision deferred.** Do not amend ADR 005 until Phase 3 prompt iteration reveals concrete reasoning or tool-call failures. ADR 005 already specifies the revisit trigger ("end of Phase 3, if Nemotron-Super-49B v1.5 turns out to be insufficient on tool-call reliability or multi-step coherence").
- **Cheap to swap.** The `langchain-openai` client against the LM Studio endpoint is provider-agnostic; model swap is a config-file change, not code.
- **Action at Phase 3 kickoff.** Verify Nemotron-Super-49B v1.5 is downloaded and runs on the Mac Studio with acceptable latency. If not, the discrepancy with the personal stack becomes a forcing function for amendment.

**External recommendation considered and rejected (2026-05-27).** An external advisor recommended pre-emptively swapping the primary to vanilla Llama 3.3 70B and the ablation from Qwen to Mistral Small 3 for "Western industry acceptance". Rejected for three reasons: (a) the advisor conflated Nemotron-Super-49B v1.5 with vanilla Llama 3.3 70B — Nemotron is NVIDIA's reasoning- and agentic-post-trained variant built on Llama 3.3, specifically targeting the tool-calling and multi-step coherence needs of this project; vanilla Llama 3.3 70B lacks the post-training layer. (b) The "too Chinese" framing for Qwen is provincial in 2026; the Qwen ablation is a methodological *strength* (cross-family diversity), not a credibility weakness; swapping to Mistral would reduce ablation diversity (both Mistral and Llama are Western dense reasoning-tuned models). (c) The recommendation violates ADR 005's explicit "revisit only on Phase 3 evidence" rule. Decision: stay the course; re-evaluate per ADR 005 trigger.

### 5.2 KPI set and outcome buckets — RESOLVED 2026-05-27

Open item flagged by external review and internal critique: the KPI set used for Phase 2 comparison (aggregate IAE over five scenarios) was the *baseline*, but Phase 3 outcome buckets (see §2.2) required additional KPIs to discriminate between buckets.

**Resolved by `docs/kpis.md` (initial draft 2026-05-27).** Five KPIs defined:

1. `aggregate_iae` — primary headline metric, frozen against the canonical 5-scenario set.
2. `off_nominal_robustness_iae` — P95 over a 16-point off-nominal OP grid; Bucket B discriminator.
3. `constraint_violation_intercept_rate` + `constraint_violation_detection_rate` — with forked-twin counterfactual; Bucket C discriminator.
4. `linearization_consistency` — auxiliary diagnostic supporting the Bucket B story.
5. `supervisory_cycle_wallclock` — deployability/production-readiness diagnostic with soft gate.

The bucket classification decision tree (`kpis.md` §6) operationalizes the *"unambiguous mapping to one bucket"* criterion from §2.2 above, including the explicit Step-5 failure mode (results ambiguous → methodology revisit, not data massage).

**Action at Phase 3 kickoff.** Review of the four open items at the end of `docs/kpis.md` is **complete** (2026-05-27, see `kpis.md` "Review resolutions" section). The KPI set is locked: 30-min counterfactual horizon + 5-min fast-fail sub-check, 16-point off-nominal grid + 4-point screening grid for prompt iteration, 9-item safety constraint list including M_D / M_B holdup bounds, three-band Bucket B threshold (1.5× minimum, 2.0× strong evidence). Phase 3 implementation can target stable definitions.

---

## 6. Pre-push hygiene patterns (apply at every phase boundary)

- Provenance JSON written for any tuning / decision artifact (e.g., `c0_pid_tuning_shootout.json`)
- ADR-level documentation for irreversible decisions
- `PROJECT_PLAN.md` gates explicitly verified
- `pytest` green, coverage ≥ 96 % on `src/industrial_ai/`
- This checklist updated with any new deferred items, pre-commits, or known limitations

---

## 7. Queued doc edits (apply after current Phase-3 kickoff commits land)

### 7.1 `kpis.md` §6 — add statistical-significance component to bucket classification — RESOLVED 2026-05-28

**Origin.** External review (2026-05-27) flagged that the Phase 2 C1 baseline (aggregate IAE 0.1224) is small enough that a 5 % gap between C2 and C1 would not be statistically distinguishable at N ≥ 10 seeds. The original Decision Tree (`kpis.md` §6) compared point estimates only, which would have risked classifying a non-significant difference as "Agent dominates".

**Resolved by `kpis.md` §6 rewrite (2026-05-28).** Bucket assignment now requires bootstrap-CI separation, not point-estimate comparison:

- **Bucket A** triggers only if C2's 95 % bootstrap CI on `aggregate_iae` lies entirely below C1's point estimate AND the 0.85× ratio test holds against C2's CI upper bound (not its mean).
- **Bucket B** triggers only if C2's 95 % bootstrap CI on `off_nominal_robustness_iae` lies entirely below 0.67 × C1's point estimate.
- **Bucket C** triggers only if both intercept-rate and detection-rate CI *lower bounds* exceed 0.7.
- **Step 5 (ambiguous)** is now explicitly the publishable-failure-mode landing zone for CI overlap. The text states this directly: ambiguous is an outcome to be reported, not avoided.

The decision tree also references the bootstrap convention explicitly (`mean ± bootstrap 95 % CI per Phase 5 protocol, N ≥ 10 seeds`) so the statistical pipeline is unambiguous from the KPI document alone.

**Landed in commit.** See the `docs:` commit that resolved this item; `kpis.md` changelog entry dated 2026-05-28 records the structural shift.

---

## Changelog

- 2026-05-27 (initial) — Initial version. Captures Day-2.5 shootout outcomes, external review feedback (Phase-3 outcome buckets, false-negative deliverable), and the LLM-stack alignment question raised at Phase-2 close.
- 2026-05-27 (Day 2.6 close) — §1.1 resolved (sweep-cache lookup landed, commit d575387). §1.2 added: F±20 % TL extrapolation reframed from "deferred fix" to "publishable C0 limitation". §2.1 resolved: Outcome A confirmed empirically (TL_decoupled_retuned IAE 2.81), Outcome B paragraph discarded. §4.4 added: fixed-gain-TL non-extrapolation as paper-disclosed limitation. §4.5 added: SIMC 2DoF filter active but scenario-mix-masked. §5.1 augmented: audit-trail for the considered-and-rejected external LLM-swap recommendation.
- 2026-05-27 (Phase 2 close / Day 3 results) — §2.2 augmented with empirical bucket-probability re-estimate after Day-3 C1 results. Aggregate C1/C0 ratio 6.8× (5/5 scenarios won by C1, max wall-clock 345 ms per supervisory tick). Bucket B becomes the most likely outcome; Bucket A becomes harder. Phase 3 scenario design must include off-nominal/regime-change evaluation, not only nominal-OP disturbances.
- 2026-05-27 (KPI session) — §5.2 resolved. `docs/kpis.md` drafted with five KPIs (aggregate_iae, off_nominal_robustness_iae, constraint_violation_intercept_rate + detection_rate, linearization_consistency, supervisory_cycle_wallclock) and a bucket-classification decision tree. Four open items at the end of `kpis.md` require yes/keep review before Phase 3 kickoff.
- 2026-05-27 (KPI review closed) — Four review items in `kpis.md` resolved: 30-min counterfactual horizon retained + 5-min fast-fail sub-check added; 16-point off-nominal grid retained + 4-point screening grid for prompt iteration; safety constraint list expanded from 7 to 9 (added M_D / M_B holdup bounds, fixed citation to `column_a/assumptions.md`); Bucket B threshold three-band interpretation (1.5× minimum / 2.0× strong evidence). KPI set is locked. §5.2 above updated to reflect closure.
- 2026-05-27 (Phase 3 kickoff / external review) — §7 added: queued doc edits to apply after current Phase-3 kickoff commits. §7.1: augment `kpis.md` §6 with bootstrap-CI separation for bucket classification (point-estimate comparison risks classifying non-significant differences as "Agent dominates" given the small C1 aggregate IAE baseline). Edit deferred to avoid simultaneous-edit conflict with Claude Code's in-flight work in `paper/methods_phase3_buckets.md` and `src/industrial_ai/agents/`. External-review recommendation to start the Qwen ablation in parallel during Phase 3 was considered and rejected: ADR 005 already makes the swap a config-file change via `langchain-openai` against the LM-Studio endpoint, and Phase 5 is the methodologically correct slot for the ablation run; building it before the Primary is empirically validated quadruples the debug surface without earlier insight.
- 2026-05-28 (§7.1 resolved) — `kpis.md` §6 rewritten to require bootstrap 95 % CI separation for all bucket-classification thresholds (Bucket A: C2 CI entirely below C1 point and 0.85× cleared on CI upper bound; Bucket B: CI entirely below 0.67 × C1 point; Bucket C: both rates' CI lower bounds above 0.7). Step 5 (ambiguous) text strengthened to make publishable-failure-mode framing explicit. `kpis.md` changelog updated. §7.1 status flipped from "queued" to "RESOLVED".
