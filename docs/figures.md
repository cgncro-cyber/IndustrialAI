# Paper Figure Plan

> This file is the contract for which figures the paper will contain and what data each requires.
> It is also a forcing function for instrumentation: Phases 1–4 must log everything Phase 5 needs to plot.

Eight mandatory figures total. The SAFEPROCESS 6-page paper includes 5–6 of them; the arXiv preprint includes all eight plus optional add-ons. Generation code lives in `src/industrial_ai/evaluation/figures/`.

For all figures: matplotlib + seaborn only, no plotly. Output formats: PDF (vector, for paper) and PNG (raster, for README / LinkedIn). All figures regenerable via `make reproduce`.

---

## Figure 1 — Hierarchical Architecture Diagram

**Purpose.** Establish the two-layer architecture from ADR 006 visually. This is the most-viewed figure of any paper after the abstract.

**Type.** Block diagram (not data plot). Authored once, kept static across runs.

**Tooling.** TikZ source in `paper/figures/fig1_architecture.tex` for the paper; matplotlib stand-in version for the README.

**Content.**
- Top swim lane: supervisory layer with four boxes labeled C0 (PID-only), C1 (Linear MPC), C2 (Agent), C3 (Agent + Safety Gate). Cadence label: 5–15 min.
- Middle: setpoint handoff with rate-limiter symbol.
- Bottom swim lane: regulatory PID layer (identical across all configurations). Cadence label: 1–5 s.
- Below: process twin (Skogestad Column A dynamic model, integrated via `scipy.integrate`).
- Color: greyscale-readable; reviewers print papers.

**Data source.** None (conceptual figure).

**Inclusion.** SAFEPROCESS ✓ · arXiv ✓ · README ✓

---

## Figure 2 — Tray-by-Tray State Profile (Side-by-Side Configurations)

**Purpose.** Show how each supervisory configuration handles disturbance propagation through the column. Chemical engineers recognize front migration immediately; this differentiates the paper from generic ML papers.

**Type.** Three or four heatmaps arranged horizontally, one per configuration.

**Axes (per panel).**
- Y: tray index from bottom (reboiler) to top (condenser).
- X: time over a representative disturbance scenario (e.g., feed-composition step).
- Color: light-component mole fraction (Skogestad Column A is binary). Same color scale across all panels.

**Content.**
- Disturbance onset marked with vertical line.
- Each panel labeled with configuration ID and headline KPI (e.g., "Time-to-spec: 47 min").
- Colorbar shared, with operating-window band marked.

**Data source.** `src/industrial_ai/twin/column_a/integrator.py` must log per-stage composition and holdup at every regulatory-layer timestep. Log to `data/runs/<config>/<scenario>/tray_profile.parquet`.

**Inclusion.** SAFEPROCESS ✓ · arXiv ✓

---

## Figure 3 — Process-vs-Control Dashboard

**Purpose.** The "EKG" of the plant. Workhorse Results figure for one canonical disturbance scenario.

**Type.** Four stacked subplots with shared x-axis (time).

**Subplots, top to bottom.**
1. **Disturbance.** Step or ramp profile in the disturbed variable (feed flow F, feed composition z_F, feed enthalpy q_F).
2. **Controlled variables (CVs).** Top-product composition y_D and bottom-product composition x_B, with spec bands (99 % purities for Column A). One trace per configuration in distinct colors.
3. **Setpoints (commanded by supervisor).** Stair-step lines showing the cadence difference: agent setpoints update every 5–15 min; PID alone has constant setpoints. Makes the hierarchy visible.
4. **Manipulated variable (MV) activity.** Reflux L and boilup V over time (LV configuration). Critical for the "but the agent oscillates the valves" reviewer objection. One trace per configuration.

**Data source.** `data/runs/<config>/<scenario>/timeseries.parquet` with columns: t, disturbance_value, y_D, x_B, setpoint_y_D, setpoint_x_B, L, V, D, B.

**Inclusion.** SAFEPROCESS ✓ · arXiv ✓

---

## Figure 4 — KPI Violin Plots with Effect Sizes

**Purpose.** Aggregate statistical comparison across all N seed runs and all scenarios. The "final accounting" plot. Pre-empts the "you cherry-picked one good run" objection.

**Type.** Grid of violin plots (one column per KPI, one violin per configuration).

**KPIs as separate panels.**
- Specific energy consumption (boilup V per kg light-component product)
- Yield (% of theoretical maximum)
- Integrated absolute error (IAE) over disturbance window
- Constraint-violation count (excursions outside the 99 % purity band)
- Settling time

**Visual elements per violin.**
- Violin shape from N ≥ 10 seed runs (only for stochastic configurations C2, C3; deterministic C0, C1 shown as single horizontal line plus shaded band for run-to-run noise floor).
- Bootstrap 95 % CI marked as inner box.
- Practical-significance threshold as horizontal dashed line, labeled.
- Cohen's d printed above each pair-wise comparison that clears the threshold.

**Data source.** `data/runs/<config>/<scenario>/<seed>/kpis.json`. Aggregated by `evaluation/aggregate_kpis.py`.

**Inclusion.** SAFEPROCESS ✓ · arXiv ✓

---

## Figure 5 — Safety-Gate Decision Timeline

**Purpose.** Demonstrate the novelty of the safety gate. Show what it accepted, what it blocked, and why.

**Type.** Two-panel plot, shared x-axis.

**Panel A (top).** Continuous anomaly score from the detector over time, with the decision threshold marked. Shaded region above threshold = "would-be blocked."

**Panel B (bottom).** Agent proposal markers along the time axis:
- Green circle: proposed and accepted.
- Red cross: proposed and blocked by gate.
- Annotation: which proposed setpoint values triggered the block.

**Why not the t-SNE / PCA phase space plot.** Gemini suggested a 2D phase-space scatter with an anomaly zone overlay. Rejected because (a) t-SNE layouts are not deterministic and break figure reproducibility, and (b) the detector decides on time-series state, not 2D points — drawing a 2D zone misrepresents the actual decision boundary. The score-over-time plot is honest.

**Data source.** `src/industrial_ai/safety/anomaly_gate.py` logs score, threshold, and decision for every proposal to `data/runs/<config>/<scenario>/safety_log.parquet`.

**Inclusion.** SAFEPROCESS ✓ · arXiv ✓

---

## Figure 6 — Supervisory Cycle Latency Profile

**Purpose.** Defend against the "LLMs are too slow for control" reviewer objection. Show that agent decisions complete well within the supervisory cadence budget.

**Type.** Boxplot or histogram of wall-clock time per supervisory cycle.

**Content.**
- One distribution per configuration (PID-only and Linear MPC will be near-zero; agent in the seconds-to-minutes range; safety-gated agent slightly higher).
- Horizontal line marking the supervisory cadence (e.g., 10 min) — visual confirmation that all configurations stay well below the budget.
- Median, IQR, and worst case annotated.

**Data source.** Each supervisor logs `wall_clock_seconds` per cycle to its run metadata.

**Inclusion.** SAFEPROCESS ✓ · arXiv ✓

---

## Figure 7 — Disturbance-Recovery Summary Heatmap

**Purpose.** Compress the entire experimental matrix into one figure. Quick reviewer takeaway.

**Type.** Two heatmaps side by side, sharing axes.

**Axes (both panels).**
- Y: disturbance scenarios (5–8 rows: F-step ±20 %, z_F-step ±10 %, L-step, V-step, q_F-step, ...).
- X: configurations (C0, C1, C2, C3 — 4 columns).

**Color encoding.**
- Left panel: time-to-recovery (lower is better, blue-to-red colormap reversed).
- Right panel: peak deviation from spec (lower is better).
- Cell annotations: numerical value with bootstrap CI in parentheses for stochastic configurations.

**Data source.** Aggregated from Figure 4's KPI data.

**Inclusion.** SAFEPROCESS (if space) · arXiv ✓

---

## Figure 8 — Cross-Domain Transfer Confusion Matrix

**Purpose.** Visualize the cross-domain story from Phase 4 / ADR-006-aligned methodology: anomaly detector trained on TEP / NoBOOM, tested on Skogestad Column A simulated anomalies.

**Type.** 2×2 confusion matrix per evaluation pairing.

**Content.**
- Confusion matrix with TP / FP / FN / TN counts.
- Below: precision, recall, F1, ROC-AUC.
- One panel per anomaly category (column flooding, tray drying, feed-composition drift, etc.).
- Comparison: detector trained on TEP vs. detector trained in-domain (fallback per Phase 4) — shown side by side. Both bars expected if cross-domain transfer is imperfect.

**Data source.** `data/runs/safety_eval/cross_domain.json` populated by Phase 4 evaluation script.

**Inclusion.** SAFEPROCESS (if cross-domain story is positive) · arXiv ✓

---

## Optional / Add-On Figures (arXiv only, if space and time)

- **A1 — Setpoint trajectory overlay.** Per-configuration plot of the setpoint sequences for a single disturbance, all configurations on the same axes. Shows the "personality" of each controller.
- **A2 — ROC curves of the anomaly detector** at different operating-point choices.
- **A3 — Agent reasoning excerpts.** Short tables of Observer / Optimizer / Critic exchanges for one disturbance. Qualitative, but reviewers like it.
- **A4 — Twin validation against Skogestad reference trajectories.** Side-by-side plot of the Python Column A port vs. the published Skogestad 1997 step-response trajectories, with L2-norm of the residual annotated. Establishes twin credibility for skeptical reviewers.

---

## Data-Logging Contract (read before starting any Phase)

For every experimental run, the following data must be logged with consistent file structure:

```
data/runs/
  <config>/                   # c0_pid / c1_lin_mpc / c2_agent / c3_agent_safety
    <scenario>/               # e.g., feed_step_+20pct, zF_step_-10pct, reflux_drift, ...
      <seed>/                 # only for stochastic configs (c2, c3)
        timeseries.parquet    # high-frequency state for Fig 3
        tray_profile.parquet  # per-stage composition + holdup for Fig 2
        setpoints.parquet     # commanded setpoints with timestamps
        kpis.json             # scalar KPIs for Fig 4 / 7
        latency.json          # wall-clock per supervisory cycle for Fig 6
        safety_log.parquet    # anomaly scores and decisions for Fig 5 (c3 only)
        config.yaml           # full hyperparameter snapshot
        manifest.json         # input data hashes, model versions, seed
```

If any of the above is missing at the end of a phase, the corresponding figure cannot be regenerated — meaning that phase's gate is not met.

---

## Generation Workflow

1. `src/industrial_ai/evaluation/figures/` contains one script per figure (`fig1_architecture.py`, ..., `fig8_cross_domain.py`).
2. Each script reads only from `data/runs/` and writes to `paper/figures/`.
3. `make reproduce` runs all eight scripts in sequence.
4. Each figure script is unit-tested with synthetic data in `tests/test_figures.py` so the plotting code itself is verified independently of the experimental results.
