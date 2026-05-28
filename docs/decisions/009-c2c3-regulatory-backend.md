# ADR 009 — C2/C3 Regulatory Backend: MPC Primary, PID as Deployment-Economics Branch

**Status:** Accepted
**Date:** 2026-05-28
**Supersedes:** none
**Refines:** ADR 006 (hierarchical control architecture)

## Context

ADR 006 fixed the two-layer hierarchy: a slow supervisory layer (5–15 min cadence) emits composition setpoints to a fast regulatory layer (~1–5 s) that drives the plant. ADR 006 named PID as the regulatory layer in the Phase-1 build and left the door open for richer regulatory backends in later phases.

Phase 2 closed with two regulatory configurations:

- **C0** — supervisory PID-only (manual setpoints, no MPC, no agent).
- **C1** — Linear MPC at the supervisory cadence, on top of the same LV-level closure as C0.

The Phase-2 Day-3 gate confirmed C1 dominates C0 on the nominal-OP scenario set (5/5 wins, aggregate 6.8× margin). The Day-5 diagnostic added two structural findings:

- The Skogestad LV configuration is *near-singular in the low-F regime* (`cond(G_mv)` grows from 150 at the nominal OP to 6800 at F=0.8, see `pre_submission_checklist.md` §4.6). C1's QP solution saturates MV bounds in the wrong direction at those OPs and the column lands in a worse SS than the start.
- The regularization sweep (`data/reference/c1_regularization_sweep.json`, `docs/figures.md` Figure 9) confirms the nominal/off-nominal trade-off is *irreducible* for fixed-weight linear MPC: no single multiplier resolves both regimes while preserving the Phase-2 gate.

The Phase-3 agent (C2) and the safety-gated variant (C3) sit on top of *some* regulatory backend. The open question is which one drives the C0/C1/C2/C3 four-step ladder that the paper compares. Two coherent positionings exist:

- **Option A — Agent supervises MPC.** C2/C3 emit composition setpoints; the MPC consumes them at the supervisory cadence and drives the plant. The ladder reads C0 (PID) → C1 (MPC) → C2 (Agent + MPC) → C3 (Agent + Safety Gate + MPC). The agent's contribution is measured *over the strongest regulatory baseline*.
- **Option B — Agent supervises PID directly.** C2/C3 emit composition setpoints that the PID loops track without MPC in the loop. The ladder is C0 (PID) → C2 (Agent + PID) → C3 (Agent + Safety Gate + PID). The agent's contribution is measured *over the weaker baseline*, but the deployment story is materially different — no MPC license, no MPC commissioning, no model-maintenance burden.

Both options have merit on different axes. Option A is the harder test of marginal agent value; Option B is the better proxy for the real-world deployment that an industrial-AI methodology paper plausibly motivates.

## Decision

C2/C3 supervise the **MPC** (Option A) as the **primary** four-step ladder. The PID-direct variant (Option B) is implemented as a named **deployment-economics branch**, evaluated as a secondary analysis in Phase 5, *not* as a generic ablation.

### Primary configuration (Option A)

- Four-step ladder: **C0 (PID-only) → C1 (Linear MPC) → C2 (Agent + MPC) → C3 (Agent + Safety Gate + MPC)**.
- C2 / C3 emit `(y_D_target, x_B_target)` at the supervisory cadence. The Linear MPC (per ADR 006 + the Day-5 regularization-sweep Pareto-reference variant for the off-nominal comparison) tracks those targets at the same cadence and drives the LV-level closure underneath.
- This isolates the **marginal agent value over the strongest gate-passing regulatory baseline**. The Bucket-B mechanism (per `docs/kpis.md` §6) is cleanest in this configuration: the agent's contribution is target sequencing across the ill-conditioned operating-window region where the linearized MPC's QP saturates against the bounds.
- Implementation default: `regulatory_backend: mpc` in the agent config.

### Deployment-economics branch (Option B)

- Ladder: **C0 (PID-only) → C2_pid (Agent + PID) → C3_pid (Agent + Safety Gate + PID)**.
- C2_pid / C3_pid emit `(y_D_target, x_B_target)` at the supervisory cadence; the same two-loop PID stack from C0 tracks those targets directly without an MPC layer in between.
- Framing in the paper: **"MPC-free deployment."** Motivation is industrial: a plant operator skips the MPC license, commissioning, and ongoing model-maintenance burden, and pays for that simplification only if the agent can compensate for the missing model-predictive layer.
- The outcome is open in both directions and publishable either way:
    - If C2_pid ≈ C2 (agent recovers most of the MPC's value via target sequencing), the contribution sharpens — "the agent makes the MPC layer optional in this class of plants".
    - If C2_pid ≪ C2 (agent cannot replace MPC at this RGA), the contribution is calibrated honestly — "MPC remains necessary; the agent's role is target sequencing on top, not replacement".
    - If C2_pid > C2 in the *low-F regime specifically* (plausible: PID + agent may be the harder test there, because PID does not collapse the way MPC does at near-singular OPs), the result is a publishable surprise — the simpler stack outperforms the more complex one in a specific regime, and the paper explicitly says so.
- Pre-commit on framing: this is a *deployment-economics question*, not a "did the agent beat baseline?" question. The Methods section names it as such and does not present Option B as a fallback in case Option A is unfavorable.

### Implementation contract

- The regulatory backend is a single config flag, `regulatory_backend: mpc | pid`, threaded through the agent and safety-gate node graph. No code branch lives in the agent itself; the same Observer / Optimizer / Critic decomposition runs against either backend.
- Default at C2/C3 instantiation time: `regulatory_backend = "mpc"` (matches the primary ladder).
- The PID backend reuses the regulatory PID layer from C0 directly. No retuning of the regulatory gains across backends — that is the *point* of the deployment-economics framing (the operator does not retune; they remove the MPC).
- The MPC backend uses the Pareto-reference `r_lt = r_vb = 10` variant from `data/reference/c1_regularization_sweep.json` for the off-nominal evaluation, and the Phase-2-baseline `r_lt = r_vb = 0.1` variant for the nominal evaluation (per `docs/kpis.md` §6 Step 3 — comparing C2 against the strongest gate-passing fixed-weight C1 on each sub-metric is what the Bucket-B classification requires).
- Both backends produce the same data-logging-contract artifacts (`docs/figures.md` §Data-Logging Contract). The `manifest.json` includes the `regulatory_backend` field so downstream figures can disaggregate.

### Phase-5 evaluation contract

- The headline Results table (`paper/methods_phase3_buckets.md` post-empirics version, Methods + Results) reports the primary ladder C0 / C1 / C2 / C3 with `regulatory_backend = mpc`. Bucket classification uses these numbers.
- A **separate** "Deployment-economics analysis" section reports the C0 / C2_pid / C3_pid PID-backend ladder, with the explicit framing above. The figures from this section may live in the main body or supplementary depending on space, but the analysis itself is *not* optional.
- The off-nominal sub-metrics from `docs/kpis.md` §2.3 / §2.4 are computed on both backends. Bucket-B classification is on the primary (MPC backend); the PID-backend numbers are reported alongside and discussed in the Deployment-economics section but do not enter the bucket decision.
- No retroactive bucket reclassification based on the PID-backend numbers. If the primary lands in Bucket B (target acquisition) and the PID-backend lands in Bucket A on the same sub-metric, both are reported as-is; the Methods text describes both findings.

## Rationale

- **Hardest test, isolated agent value.** Option A measures the agent's marginal contribution above the strongest gate-passing regulatory baseline (Linear MPC, Pareto-reference variant). Any positive result from C2 over C1 in this configuration is unambiguously about the agent's role. The "did the agent beat a strawman regulatory layer?" objection is closed by construction.
- **Cleanest Bucket-B mechanism story.** §4.6 documents that Linear MPC's QP saturates at the near-singular F=0.8 cluster and produces a *worse* SS than the start. An agent that recognizes the ill-conditioned region and chooses a reachable interim target *sequences* the supervisor into the operating-window region where MPC's linearization is well-conditioned, then hands back to the MPC at the new operating point. This narrative is sharpest with an MPC backend — the agent and the MPC are doing complementary jobs, not the agent replacing the MPC.
- **PID-direct as deployment-economics, not as ablation.** A "C2 with PID instead of MPC" ablation framed as a robustness check would invite the framing question *"is the agent supposed to replace MPC?"* — which is not the methodology's claim. Framing Option B as a deployment-economics question reframes the same data into a publishable industrial-relevance question: under what conditions can an industrial operator skip MPC and rely on agent + PID? That question is independently interesting and has a clean publication story regardless of the empirical outcome.
- **PID + Agent may be the harder test in the low-F regime.** PID does not collapse the same way MPC does at near-singular OPs (no QP, no linearization to mis-condition). With coupled PIDs, RGA ≈ 36 nominal, and the same near-singular `G_mv` at F=0.8, the PID-direct backend is its own distinct test. The agent's setpoint-sequencing role may be more or less useful there than on the MPC backend; the paper should not pre-judge the direction.
- **Single config flag means no debugging-surface tax.** Both backends share the agent code, the data-logging contract, the safety gate, the KPI computation. The marginal engineering cost of supporting both is the config flag + a thin regulatory-backend interface. The Phase-3 agent skeleton (`src/industrial_ai/agents/`) is built around a `RegulatoryBackend` protocol from day one so the cost stays low.

## Consequences

- Phase 3 implementation: the agent skeleton (state schema, Observer / Optimizer / Critic node graph, mock LLM client, LM Studio client) is built with a pluggable `RegulatoryBackend` protocol. The first runs use `regulatory_backend = mpc` (Option A); the PID backend is a config-flag change.
- Phase 4: the safety gate operates identically on both backends. The forked-twin counterfactual (per `docs/kpis.md` §3.3) integrates with whichever backend the proposal was emitted into.
- Phase 5: two evaluation passes — the primary four-step MPC-backend ladder for bucket classification, and the deployment-economics PID-backend secondary analysis. Both pre-committed here.
- Compute cost: roughly 2× the Phase-3/Phase-4 evaluation runtime versus a single-backend study, but no extra implementation surface area beyond the config flag and the PID-backend adapter (which already exists as C0).
- The pre-submission checklist will get a §7.2 follow-up to track Phase-5 Deployment-economics section drafting once the empirical numbers are in.

## Open items

- The exact PID retuning policy if the off-nominal grid surfaces a regime where the C0 fixed-gain TL diverges *and* C2_pid is required to operate there. The current C0 fixed-gain non-extrapolation (§4.4) suggests that no retuning is the honest answer (the deployment-economics framing forbids it — the operator removes MPC and accepts the locality of fixed-gain PID; the agent's job is to compensate via setpoints). This is the natural interpretation but is recorded here as an open item to revisit if Phase-3 empirics call it out.
- Whether the deployment-economics section warrants its own outcome-bucket pre-commit document analogous to `paper/methods_phase3_buckets.md`. Likely yes — track in the pre-submission checklist when Phase-5 drafting begins.
