# Skogestad Column A — Modeling Assumptions

This file enumerates every assumption baked into the Python port at
`industrial_ai.twin.column_a`, with citations back to the primary
sources. The port is a clean-room re-implementation of the equations,
not a line-by-line translation of the MATLAB code, but it reproduces
the same canonical system so reviewers can cross-check.

**Primary sources.**

- **[Skogestad & Morari 1988]**
  Skogestad, S. and Morari, M. (1988). *Understanding the Dynamic
  Behavior of Distillation Columns.* Ind. Eng. Chem. Res. 27(10),
  1848–1862. Defines the underlying linear-hydraulics + constant-molar-
  overflow distillation model.
- **[Skogestad 1997]**
  Skogestad, S. (1997). *Dynamics and Control of Distillation Columns
  — A Tutorial Introduction.* Trans IChemE 75(A), 539–562. Source for
  the Column A specification and the analytic time-constant /
  steady-state-gain expressions used as Phase 1 validation anchors
  (Eq. 31, §4.4).
- **[Skogestad & Postlethwaite 1996]**
  Skogestad, S. and Postlethwaite, I. (1996). *Multivariable Feedback
  Control: Analysis and Design.* Wiley. Hosts the canonical numerical
  values for Column A and the three control-pairing configurations
  (LV, DV, L/D-V/B).
- **[MATLAB code]**
  Reference MATLAB source at
  https://skoge.folk.ntnu.no/book/1st_edition/matlab_m/cola/ (files
  `colamod.m`, `cola4.m`, `cola_init.m`, `cola_lv.m`, `cola_dv.m`,
  `cola_rr.m`). Used as semantic reference and validation oracle
  (`cola_init.mat` provides the SS state vector and `cola4.m` produces
  the step-response trajectories used by
  `tests/test_column_a_against_matlab.py`).

---

## 1. Mixture and thermodynamics

1. **Binary mixture.** Light and heavy component only. State vector
   tracks the light-component mole fraction on each stage.
   *Source: Skogestad & Morari 1988, §2; Skogestad 1997, §2.*
2. **Constant relative volatility.** `alpha = 1.5` everywhere in the
   column, independent of temperature and composition. VLE is
   approximated by `y = alpha x / (1 + (alpha - 1) x)`.
   *Source: Skogestad 1997, Table 1.*
3. **Constant pressure.** No pressure dynamics, no pressure drop across
   stages. Pressure is not a state.
   *Source: Skogestad 1997, §2.*
4. **Ideal saturated VLE on equilibrium stages.** Liquid and vapor
   leaving stages 1 through NT-1 are in thermodynamic equilibrium.
   *Source: Skogestad & Morari 1988, §2.*

## 2. Stage and hydraulic model

5. **41 theoretical stages.** Reboiler at stage 1 (0-indexed: 0),
   total condenser at stage NT = 41 (0-indexed: 40), feed at stage
   NF = 21 (0-indexed: 20).
   *Source: Skogestad 1997, Table 1.*
6. **Total condenser is not an equilibrium stage.** All vapor from
   stage NT-1 condenses; the liquid splits into reflux LT and
   distillate D. No phase equilibrium is computed at the condenser.
   *Source: Skogestad & Morari 1988, §2; cola_lv.m.*
7. **Single-stage reboiler at equilibrium.** Vapor leaves the reboiler
   in equilibrium with the liquid on that stage.
   *Source: Skogestad & Morari 1988, §2.*
8. **Linearized tray hydraulics.** Liquid flow leaving any tray
   depends linearly on holdup deviation from nominal:
   `L_i = L_i0 + (M_i - M_i0) / tau_L`, with `tau_L = 0.063 min`. The
   total liquid holdup on each tray is therefore a state.
   *Source: Skogestad & Morari 1988, §2; Skogestad 1997, §3.*
9. **K2 (vapor-flow) effect on liquid dynamics.** Liquid flow is also
   sensitive to vapor-flow deviations:
   `L_i = L_i0 + (M_i - M_i0) / tau_L + lambda (V_{i-1} - V_{i-1,0})`.
   Default `lambda = 0` matches the published Column A nominal case;
   non-zero lambda is available for parameter studies.
   *Source: Skogestad 1997, §3.*
10. **No vapor holdup.** Vapor flow is algebraic, not dynamic. There
    is no vapor-side mass storage anywhere in the column.
    *Source: Skogestad & Morari 1988, §2.*

## 3. Flow assumptions

11. **Constant molar overflow.** Vapor flow above the feed is constant
    `V_t = V_b + (1 - qF) F`, and vapor flow below the feed equals the
    boilup `V_b = VB`. No energy balance is solved explicitly; the
    constant-molar-overflow assumption substitutes for it under the
    classical Lewis approximation (equal molar enthalpies of
    vaporization for both components).
    *Source: Skogestad & Morari 1988, §2; Skogestad 1997, §2.*
12. **Feed enters the feed stage with composition zF and liquid
    fraction qF.** The liquid portion `qF * F` joins the liquid stream
    below the feed; the vapor portion `(1 - qF) * F` joins the vapor
    stream above the feed. There is no additional vaporization or
    heat-input modeling around the feed.
    *Source: Skogestad 1997, §2.*

## 4. Configuration assumptions

13. **Level loops are P-only with the published gains.** Condenser and
    reboiler holdups are closed with proportional controllers at
    `Kc = 10` and setpoints `0.5 kmol`. This is identical across the
    LV, DV, and L/D-V/B configurations.
    *Source: cola_lv.m, cola_dv.m, cola_rr.m.*
14. **Configuration choice affects only the supervisor-to-input
    mapping.** The underlying ODE, integrator, steady-state solver,
    and linearization are all configuration-agnostic. Switching
    configuration changes which pair of `{LT, VB, D, B}` is set by the
    supervisor versus computed by the level controllers.
    *Source: Skogestad & Postlethwaite 1996, §10.*

## 5. Numerical-method choices

15. **`scipy.integrate.solve_ivp` with method `LSODA`.** LSODA handles
    automatic stiff / non-stiff switching, which is appropriate
    because the Column A system is mildly stiff during fast
    composition transients but non-stiff at steady state. `Radau` and
    `BDF` are available as fully-implicit fallbacks for difficult
    cases.
    *Choice rationale: ADR 001 Refinement and ADR 007.*
16. **Default tolerances.** Open-loop integration: `rtol = 1e-8`,
    `atol = 1e-10`. Steady-state via integration: `rtol = 1e-9`,
    `atol = 1e-11` (tighter, because steady-state identification is
    more demanding than transient simulation).
    *Source: this code, calibrated against `cola_init.mat`.*
17. **Newton–Krylov for steady-state perturbations.** When a steady
    state at a perturbed input is needed, `scipy.optimize.newton_krylov`
    is used with `f_tol = 1e-8` and a previously-computed SS as the
    initial guess. Long-time integration (~20 000 min) is used for the
    cold-start nominal SS, matching `cola_init.m`.
    *Source: cola_init.m, this code.*
18. **Symbolic Jacobian via CasADi.** Linearization for the Phase 2
    Linear MPC baseline uses exact algorithmic-differentiation
    Jacobians produced by a CasADi symbolic re-implementation of the
    ODE (`column_a/casadi_model.py`). The numpy implementation in
    `model.py` remains the integration target and the validation
    oracle; the CasADi build is purely the differentiation backend.
    Parity with central-difference Jacobians is enforced by
    `tests/twin/test_casadi_model.py` to ~1e-5, and the
    Skogestad 1997 mini-gate scalar invariants (G^LV(0), τ₁, τ₂, τ₃)
    pass identically under either backend. Switch via
    `linearize_lv(backend="finite_difference" | "casadi")`.
    *Source: CasADi 3.7 algorithmic differentiation, this code.*

## 6. What is *not* modeled

- Pressure dynamics (column pressure, condenser/reboiler pressures).
- Real (rate-limited) heat transfer in the reboiler and condenser.
- Vapor holdup.
- Non-ideal VLE (activity coefficients, azeotropes).
- Multi-component mixtures.
- Heat losses, jacket dynamics, sensor and actuator dynamics.
- Tray flooding, weeping, or any hydraulic non-idealities beyond the
  linearized model in §2.

These omissions are deliberate. The case-study purpose of Column A is
to provide a literature-validated, reproducible coupled-MV benchmark
for the supervisory-control methodology. Adding heat-exchanger
dynamics or multi-component VLE would complicate validation without
strengthening the methodology claim. Real-world deployment of the
agentic + safety-gated architecture (which is the contribution under
study) is out of scope for the SAFEPROCESS submission.
