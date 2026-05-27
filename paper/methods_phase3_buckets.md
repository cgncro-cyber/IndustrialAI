# Phase 3 — Outcome Bucket Methods Paragraphs (pre-drafted)

Three pre-drafted Methods paragraphs, one per outcome bucket from
`docs/kpis.md` §6. After Phase 3 empirical results, exactly one is
selected as the final Methods text; the other two are discarded
(remain in git history for audit trail).

Pre-registration purpose: by drafting the three candidate narratives
*before* the C2 numbers exist, the eventual bucket classification
cannot be reverse-engineered from the data. The decision tree in
`docs/kpis.md` §6 maps the empirical KPIs onto exactly one of these
paragraphs, with no degrees of freedom left for narrative drift.

KPI placeholders are written as `[VALUE]` and are filled in only
once the runs are complete. All paragraphs assume the apples-to-apples
evaluation contract: same plant, same regulatory layer, same scenario
set, same seeds, per `docs/kpis.md` §1.

---

## Bucket A — Agentic supervisor dominates linear MPC

The agentic supervisor C2 reduced the aggregate IAE on the canonical
five-scenario set to [VALUE] mole-fraction·min, compared with [VALUE]
for the linear-MPC baseline C1, an improvement below the 0.85×
threshold pre-registered in `docs/kpis.md` §6. The dominance was
consistent across at least four of the five individual scenarios,
ruling out a single-scenario artefact. Mechanistically, C1 relies on
a fixed linearization at the nominal operating point, so its setpoint
trajectories are optimal only in a neighbourhood of that point; C2
re-plans from twin observations at every supervisory cycle and is not
constrained by the locality of any single linearization. The auxiliary
diagnostic `linearization_consistency` (`docs/kpis.md` §4), reported
alongside, characterizes the size of the locality region that C1
operates in and against which C2's plan space is unconstrained. The
LLM was held at the configuration documented in ADR 005, with
temperature and top-p fixed across seeds per Phase 5 reproducibility
rules. The performance gain over C1 was achieved without changes to
the regulatory PI layer, the plant, the disturbance scenarios, or the
operating point — i.e., the supervisor itself is the load-bearing
contribution.

## Bucket B — Agentic supervisor adds off-nominal robustness

On the nominal-operating-point scenario set, C2 and C1 were
statistically indistinguishable in aggregate IAE ([VALUE] vs [VALUE]
mole-fraction·min, within the bootstrap 95 % confidence interval).
On the 16-point off-nominal grid defined in `docs/kpis.md` §2.2, the
95th-percentile per-OP aggregate IAE was [VALUE] for C2 against
[VALUE] for C1, an improvement of [VALUE]× that places the result in
the [strong / moderate] evidence band of the three-band
classification in `docs/kpis.md` §2.4. The C2 advantage was
concentrated at the operating points with the largest
`linearization_drift_g` values (`docs/kpis.md` §4), confirming that
the gain originates where C1's per-OP linearization is most stressed
rather than from a uniform shift across the grid. A symmetric piece
of evidence on the C0 side comes from Phase 2 Day 2.6: the
relay-tuned, fixed-gain Tyreus–Luyben controller did not extrapolate
to F ± 20 % operating points (documented as a publishable C0
limitation in `docs/pre_submission_checklist.md` §4.4), establishing
that locality is a structural feature of the LV-closed Skogestad
Column A plant and not a Bucket-B-specific framing. C2 mitigates that
structural locality by re-evaluating setpoint targets against twin
observations at each cycle rather than committing to a single
operating-point linearization for the full horizon.

## Bucket C — Safety gate is the load-bearing contribution

C2 and C1 aggregate IAEs were within 20 % of each other ([VALUE] vs
[VALUE] mole-fraction·min), placing the contribution outside the
performance-dominance bands of Bucket A and Bucket B. The
configuration C3 (C2 augmented with the anomaly-based safety gate)
reported a `constraint_violation_intercept_rate` of [VALUE] and a
`constraint_violation_detection_rate` of [VALUE], both above the 0.7
threshold pre-registered in `docs/kpis.md` §3.4, against the
counterfactual horizon of 30 min and the safety-constraint list
pinned in `docs/kpis.md` §3.3. At least three documented
false-negative case studies (Phase 4 binding deliverable) show the
gate blocking specific proposals whose forked-twin trajectories would
have violated a documented safety constraint within the
counterfactual horizon. Mechanistically, the agentic supervisor
explores a wider region of the setpoint space than a linear MPC by
construction — its proposals are not constrained by a convex QP over
a fixed linearization — and the safety gate is what makes that
exploration tolerable for closed-loop operation. The gate is trained
cross-domain on Tennessee Eastman anomaly signatures (Phase 4) and
applied to the Skogestad Column A twin without per-plant retraining,
so the contribution is the safety architecture rather than the
plant-specific configuration of the agent or the controller.

---

## Disposal rule (post-empirics)

After Phase 3 KPIs are computed and the `docs/kpis.md` §6 decision
tree classifies the result, the two non-selected paragraphs are
removed from this file in a single commit, with the commit message
recording (a) which bucket was selected, (b) the KPI values that
drove the classification, and (c) a `git show` reference to this
pre-draft for audit-trail traceability.
