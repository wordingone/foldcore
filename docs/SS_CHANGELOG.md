# CHANGELOG

## Session 23 — Meta-Cognitive Review: Blind Spots, ACE/rk Discovery, ANIMA Architecture Ceiling (2026-03-02)

### Context
Session 22 closed the ANIMA parameter landscape: all parameters characterized, none both binding AND adaptable from within. The conclusion from Session 22's adversarial review was unambiguous — ANIMA has reached an architecture ceiling for Stage 3. Session 23 was scheduled as a meta-cognitive review: no experiments, no code, no new implementations. Pure analysis. The goal: what are we missing? Where are the blind spots?

No teammates deployed for implementation. Analysis conducted by [researcher] (team lead).

### Blind Spot 1: ACE.py — Third Substrate, Never Formally Tested

`src/ACE.py` (Autopoietic Computation Engine, 325 lines) exists as a third substrate that has never been run against the constitution's ground truth test.

**Architecture:** Each process has three components — dynamic state x (concerns), membrane (autonomy scalar), and concern vector c. Concerns evolve from the computation itself:
```
dc/dt = α · (dx/dt ⊙ tanh(x) - c) · max(0, 1 - ‖c‖²)
```
This is a self-referential update: concern direction is driven by the interaction of state velocity and current state. Membrane autonomy gates information exchange between processes. Topology: concern-weighted adjacency, not fixed ring.

**Why this matters:** ACE has per-process adaptive parameters that emerge FROM the computation rather than being tuned externally. The concern update rule is exactly the kind of Principle-II-compliant self-generated signal ANIMA lacks. Whether it passes the ground truth test is unknown.

**Status:** File exists, reads correctly, has not been tested. Not in the known substrate history. Not in state.md architecture section.

### Blind Spot 2: rk.py — Fourth Substrate, Directly Addresses Stage 7

`src/rk.py` (Reflexive Kernel, 564 lines) is a fourth substrate where **the state IS the transformation**. A matrix M self-applies:
```
Φ(M) = tanh(α·M + β·M²/k)
```
Eigenform convergence: the system finds fixed points in transformation space, not state space. The representation IS the computation.

**Why this matters:** Stage 7 requires the system to represent and modify its own update rule as first-class data. The Living Seed's ceiling (c037) was precisely that its equation is hardcoded Python — not data the system can modify. rk.py's architecture is the equation-as-data structure Stage 7 requires. The Reflexive Kernel's self-application structure is a candidate for exactly the self-representation mechanism that every prior substrate lacked.

**Status:** File exists, 564 lines, never formally tested against constitution. Not connected to stage progression. Not referenced in state.md.

### Blind Spot 3: ANIMA Separation Theorem — Formal Proof, Never Connected to Experiments

`anima/ANIMA_SEPARATION_THEOREM_V2.md` contains a formal proof that reactive systems (where parameters are independent of current state) require Ω(k) state for k-way routing, versus O(log k) for state-conditioned systems.

**ANIMA as implemented is reactive.** W predicts neighbor interactions; I accumulates error. Neither W nor I's current value conditions how the parameters (w_lr, tau, gamma) behave. The theorem predicts this failure mode directly: a reactive system cannot route K signals efficiently from within its own dynamics. The separation theorem predicts the entire Session 19–22 finding (no Principle-II-compliant signal can locate the w_lr optimum) as a necessary consequence of ANIMA's architecture class.

**Why this is critical:** This is not a coincidence or a post-hoc explanation — it is a theorem. A formal proof was sitting in the codebase that predicts the exact pattern observed across five sessions of ANIMA Stage 2/3 work. The theorem was never cited in any session. The theorem implies that state-conditioned dynamics are the necessary next step, not a minor variant.

### Blind Spot 4: No ANIMA Frozen Frame Inventory

`frozen_frame.json` was created in Session 13 for the Living Seed only. Five sessions of ANIMA characterization (Sessions 18–22) produced 51 constraints, identified the architecture ceiling, and characterized all parameters — but no formal ANIMA frozen frame inventory has been maintained.

The state.md frozen frame table still lists Living Seed elements. The ANIMA frozen frame is characterized across multiple constraint entries but has no single authoritative document. This gap is being filled in Session 23 housekeeping (entry 069, Rune).

### Blind Spot 5: anima/ Parallel Codebase — Relationship Unclear

The `anima/` directory contains 80+ files across `core/`, `training/`, `eval/`, `tests/`, `variants/`, and `archive/`. The relationship between this codebase and `src/anima_organism.py` (the canonical active substrate) has never been formally documented.

Whether `anima/` is an earlier version, a parallel development track, or experimental variants that were superseded is unknown from session records. The canonical ANIMA organism (src/anima_organism.py) was developed in Sessions 17–18; the relationship to the anima/ directory predates formal tracking.

### Blind Spot 6: Calibration Test Unrun — Most Consequential Open Question

`src/anima_organism.py` contains the ALIVE vs STILL ground truth test. The test compares an organism with resp_z-driven adaptation (ALIVE) against an identical organism with adaptation frozen at converged alpha values (STILL-with-converged-alpha).

**This specific comparison has never been run.** The existing ground truth tests (`_gt7.py` and related) compare ALIVE against STILL with IDENTICAL starting alphas. The calibration question — whether resp_z provides ongoing adaptive benefit or merely calibrates alpha to a good initialization — remains empirically open.

**Why this is the most consequential unrun experiment:** If ALIVE beats STILL-with-converged-alpha, Stage 2 has genuine ongoing adaptation. If they tie, Stage 2 is a calibration artifact — the benefit comes from reaching good alpha values, not from maintaining the adaptation mechanism during evaluation. This is the difference between a Principle-II-compliant adaptive system and a sophisticated initialization procedure.

### Deep Analysis: The Information-Theoretic Ceiling

The pattern across 22 sessions reveals a structural constraint embedded in Principle II itself.

**The asymmetry:** The ground truth test measures sequence-level mutual information — a property of the system's response to the FULL K-signal protocol. But Principle II requires adaptation signals to arise from the computation itself, per-step. The per-step signal cannot encode sequence-level structure without storing the entire sequence history (which requires external memory, violating Principle II) or computing MI directly (which requires an evaluator, violating Principle II).

This is not a failure of any particular signal. It is a structural gap between the timescale of adaptation and the timescale of evaluation. The only escape is a self-generated sequence-level signal — and the only candidates that have been tested (resp_z, dual-timescale I) are either insufficient or harmful.

**resp_z's position:** After 22 sessions, resp_z remains the ONLY working Principle-II-compliant adaptation signal discovered. It measures signal sensitivity (cause), not MI gap (effect), but the correlation is sufficient for Stage 1 calibration. No signal has demonstrated ongoing Stage 2 benefit beyond initialization.

### ANIMA Regression from Living Seed

The Living Seed had one confirmed Stage 2 capability: alpha adaptation via resp_z produces genuine improvement over STILL (d=2.46 confirmed, Session 19 revalidation). ANIMA was introduced as the path to Stage 7 capability.

**ANIMA's current Stage 2 status is vacuous under Amendment 1** — the mechanism works (w_lr is modulated) but no adaptation signal can improve over the calibrated optimum. This means ANIMA regressed from the Living Seed's Stage 2 capability while gaining Stage 7 potential.

The Separation Theorem explains this: the Living Seed's per-cell alpha (state-conditioned) is not reactive. Alpha values condition each cell's computation locally. ANIMA's w_lr is global and reactive — the same for all cells, driven by population-level signals. The theorem predicts per-cell parameters outperform global ones for routing tasks, and this is precisely what the Living Seed vs ANIMA comparison shows.

### Housekeeping Completed

- state.md stale frozen frame inventory (Living Seed-only) identified
- ANIMA frozen frame inventory entry 069 being created (Rune, parallel task)
- Total constraints: 51 (c051 added Session 22, confirmed)
- Forward path decision deferred to Session 24 based on ANIMA frozen frame inventory

### Session Summary

No code written. No experiments run. No new constraints. No new knowledge base entries (entry 069 pending from Rune).

**What changed:** Five blind spots identified that have been present in the codebase without formal acknowledgment. Two untested substrates (ACE.py, rk.py) discovered that directly address the current blockers. One formal theorem (Separation Theorem) identified that retroactively predicts Sessions 18–22 findings. One critical unrun experiment identified (calibration test). One parallel codebase (anima/) with unknown relationship to canonical substrate.

**The forward decision:** Three options remain for Session 24.
1. Run calibration test: ALIVE vs STILL-with-converged-alpha. Resolves whether Stage 2 is genuine or artifact.
2. Test ACE.py against ground truth: Third substrate with per-process adaptive parameters, never evaluated.
3. Test rk.py against ground truth: Fourth substrate with self-representation structure, directly addresses Stage 7.

Option 1 is necessary regardless — if Stage 2 is a calibration artifact, all stage declarations must be revisited. Options 2 and 3 are substrate evaluations that could replace ANIMA without the Separation Theorem ceiling.

Next: Session 24 — Calibration test execution. ACE/rk ground truth evaluation. Formal decision on ANIMA architecture ceiling and substrate path.

---

## Session 22 — Additive Dual-Timescale I Hurts MI Gap; ANIMA Architecture Ceiling (2026-03-02)

### Context
Session 21 confirmed MI-err structural decoupling: no I-based signal can locate the w_lr interior optimum. The forward path was tau_slow — implement a slow-timescale memory accumulator (mem_slow) that feeds back into organism dynamics and characterize whether tau_slow is a binding parameter for Stage 3.

Full team deployed: [engineer] (implementation + sweep script), Jude (code review), Tess (revalidation + sweep execution).

### Deliverable 1: Dual-Timescale I Implementation (Task #1, [engineer]; Task #2, Jude)
`src/anima_organism.py` modified (172→193 lines, +21 lines).

**Mechanism:** mem_slow accumulates tanh(err) with tau_slow EMA rate, parallel to existing mem (tau=0.3). Feedback: `i_slow_drive = mem_slow[i][k] * xs[i][k]`, additive to pre_act alongside i_drive. Principle II compliant: signal from organism's own prediction error, feeds back into state.

**Backward compatibility:** tau_slow=0.0 (default): mem_slow stays zero, i_slow_drive=0.0, dynamics bit-for-bit identical to original. Verified by [engineer] (noise=0 comparison), Jude (code review), Tess (revalidation).

**Review verdict (Jude):** APPROVED. All checks pass. One non-blocking note: no range validation on tau_slow (consistent with existing code style).

### Deliverable 2: tau_slow Binding Sweep Script (Task #3, [engineer])
`src/run_tau_slow_sweep.py` created (257 lines). Protocol: 7 tau_slow values × 10 seeds × K=[4,6,8,10] × n_perm=8 × n_trials=6. 280 total conditions.

### Deliverable 3: Stage 1 Revalidation (Task #4, Tess)
- tau_slow=0.0: alive_gap = +0.1299 (canonical +0.1256, delta=+0.0043, within ±0.005 tolerance). **PASS.**
- tau_slow=0.005: alive_gap = +0.0847 (>0, ground truth passes but 35% below baseline). **PASS but degraded.**

### Deliverable 4: tau_slow Sweep Results (Task #5, Tess)

| tau_slow | mean_gap | std    |
|----------|----------|--------|
| 0.000    | +0.1299  | 0.0410 |
| 0.001    | +0.1067  | 0.0483 |
| 0.003    | +0.1010  | 0.0401 |
| 0.005    | +0.0847  | 0.0352 |
| 0.010    | +0.0966  | 0.0435 |
| 0.030    | +0.1093  | 0.0457 |
| 0.100    | +0.1186  | 0.0520 |

**Shape:** Non-monotone trough at tau_slow=0.005 (-34.8% from baseline). Partial recovery at higher tau_slow but never reaches baseline. No tau_slow > 0 improves over baseline.

**Effect size:** baseline vs trough d=+1.19 (significant). Baseline vs best nonzero d=-0.24 (modest harm).

**Binding verdict:** tau_slow is binding in the NEGATIVE direction only. Boundary-optimal at 0.0.

### Adversarial Review Conclusions ([researcher])

1. **The effect is real.** d=1.19 on 10 seeds × 4 K values (40 conditions per tau_slow) is well above noise. The trough at 0.005 makes physical sense: tau_slow=0.005 creates a ~200-step integration window, while signal presentation is 90 steps. This integrates across half a signal cycle — maximally confusing current-signal processing with previous-signal residue.

2. **Why additive dual-timescale I hurts.** i_slow_drive adds bias to pre_act that the fast I already handles optimally. The slow accumulator carries information from PREVIOUS signals into current-signal processing, reducing sequence discrimination. The fast I (tau=0.3) is the right timescale; adding a second timescale degrades the balance.

3. **ANIMA Architecture Ceiling for Stage 3.** ALL parameters now characterized:
   - w_lr: Calibration (interior optimum 0.0003, not adaptable from within) — c044, c047-c050
   - tau: Non-binding [0.1, 0.7] — c045
   - tau_slow: Harmful, boundary-optimal at 0.0 — c051
   - gamma: Calibration (boundary-optimal at 3.0) — c046
   - delta: Calibration (boundary-optimal at 1.0) — c035
   - w_clip: Non-binding — c025 analog
   - noise: Cosmetic

   No ANIMA parameter is both binding AND adaptable from within. The W+I organism's dynamics are a fixed-point system where the fast I channel (tau=0.3) captures all available sequence information.

### New Constraint
- **c051**: Additive dual-timescale I reduces MI gap for all tau_slow > 0. Boundary-optimal at tau_slow=0.0. Fast I (tau=0.3) is sufficient; slow I adds misleading temporal blurring.

### Forward Path Options
1. Declare ANIMA Stage 3 vacuously passed under Amendment 1 (all params tested, mechanism works, no adaptive target exists)
2. Try multiplicative/gating coupling instead of additive — but Session 21 showed I_slow/I_fast ratio is <1.5% variation, making this unlikely to help
3. Declare architecture ceiling for ANIMA at Stage 2+3 vacuous, advance to Stage 4 structural adaptation or new substrate

---

## Session 21 — Stage 3 Phase 1: MI-Err Structural Decoupling Confirmed (2026-03-02)

### Context
Session 20 produced the Stage 3 forward plan (entry 066): dual-timescale I with cycle-boundary adaptation for w_lr. The central question for Session 21 Phase 1: can ANY combination of I-based signals detect the w_lr interior optimum (0.0003) from within the organism's own computation? Or is the MI inverted-U invisible from prediction-error space?

Full team deployed: Sage (analytical derivation), [engineer] (diagnostic script), Jude (code review), Tess (execution), Rune (analysis + verdict).

### Deliverable 1: Analytical Signal Shape Predictions (Task #1, Sage)
Sage derived expected shapes of I_fast_mean(w_lr), I_slow_mean(w_lr), and their ratio analytically.

**Predictions:**
- I_fast_mean: inverted-U (cancellation at high w_lr from sign-alternating err between signals)
- I_slow_mean: inverted-U (stronger cancellation from longer memory window)
- Ratio: not cleanly peaked at 0.0003 (confirmed entry 066 adversarial review)
- Anti-signal risk: HIGH — signals are unsigned, cannot encode direction

### Deliverable 2: Phase 1 Diagnostic Script (Task #2, [engineer]; Task #3, Jude)
`src/run_phase1_diagnostic.py` — InstrumentedAnimaOrganism wrapping AnimaOrganism with mem_slow (tau_slow=0.005) tracking. Measures 5 signals + alive_gap at 5 w_lr values × 2 seeds × 4 K values (40 combinations).

**Two critical bugs found by Jude during review:**
1. **Per-permutation organism reset** — measure_alive_gap() created a new organism for every permutation instead of one across all. Caused alive_gap magnitudes 3-10× too low and peak shifted from 0.0003 to 0.003. Fix: moved org creation outside perm loop.
2. **I_fast_mean K-normalization** — I_fast divided by cycle_steps introduced reverse K-confound. mem values saturate via tanh and don't grow with K. Fix: removed normalization.

### Deliverable 3: Phase 1 Empirical Results (Task #4, corrected data)

| w_lr   | I_fast | I_slow  | ratio   | I_curv  | W_vel | alive_gap |
|--------|--------|---------|---------|---------|-------|-----------|
| 0.0001 | 0.364  | 2.72e-4 | 7.4e-4  | 8.0e-3  | 0.066 | +0.018    |
| 0.0003 | 0.376  | 2.82e-4 | 7.5e-4  | 8.1e-3  | 0.205 | +0.061    |
| 0.001  | 0.381  | 2.87e-4 | 7.4e-4  | 8.3e-3  | 0.701 | +0.101    |
| 0.003  | 0.427  | 3.11e-4 | 7.2e-4  | 9.0e-3  | 2.143 | +0.189    |
| 0.01   | 0.453  | 2.97e-4 | 6.5e-4  | 9.5e-3  | 4.900 | +0.090    |

**The result that matters:** A 100× change in w_lr produces <25% variation in ALL I-based signals. The organism is blind to its own learning rate from within.

Alive_gap peaks at w_lr=0.003 in this 2-seed run — confirmed as noise (Jude). Session 19's 10-seed × n_trials=6 remains authoritative for peak at 0.0003.

### Deliverable 4: Formal Analysis and Verdict (Task #5, Entry 067, Rune)

**Verdict: C — Structurally Vacuous.** All five Amendment 1 conditions met.

Signal-by-signal:
- **I_fast_mean**: Monotone increasing (+24%). Anti-signal — pushes w_lr upward toward instability.
- **I_slow_mean**: Flat (14% variation). Tiny peak at 0.003, not 0.0003. No usable gradient.
- **Ratio**: Flat (<1.5% variation). Technically peaks at 0.0003 but signal-to-noise is ~0.
- **I_curvature**: Monotone increasing (+19%). Anti-signal — same as I_fast.
- **W_velocity**: Monotone increasing (74×). Strong signal but trivially ∝ w_lr. Anti-signal for MI quality.

**Equilibrium hypothesis (entry 066): FALSIFIED.** Boundary forces exist at the W level (W_vel diverges at high w_lr) but are invisible to I-based signals. There is no crossover point, no natural fixed point, no self-stabilizing dynamics.

**Analytical predictions vs empirical:**
- Sage predicted I_fast inverted-U: WRONG (monotone increasing — no cancellation at tau=0.3)
- Sage predicted I_slow non-monotone: PARTIALLY RIGHT (negligible amplitude, wrong location)
- Entry 066 predicted ratio not peaked: CONFIRMED
- Entry 066 predicted MI-err structural decoupling: FULLY CONFIRMED

### New Constraints
- **c049**: No combination of I_fast, I_slow, I_curvature, or their ratios can locate the w_lr interior optimum via dual-timescale I. All signals show <25% variation across 100× w_lr range. MI-err structural decoupling empirically confirmed.
- **c050**: W_velocity is monotone increasing in w_lr and is not available as an internal organism adaptation signal. It cannot serve as a Principle-II-compliant Stage 3 signal.

### Adversarial Review Conclusions ([researcher])

1. **The MI-err structural decoupling is the session's key scientific result.** The w_lr inverted-U exists in MI space (alive_gap peaks at 0.0003) but has NO gradient in prediction-error space. err-based signals (which is all the organism has) cannot detect position on the MI landscape. This is not a measurement problem — it is a structural property of the relationship between prediction error and mutual information.

2. **Why err is blind to w_lr:** Prediction error (err = actual - w_pred) conflates predictive accuracy with dynamic magnitude. At high w_lr, W is large → w_pred is large → err can be large even when W is "doing well." The absolute magnitude of err scales with the magnitude of the dynamics, which scale with w_lr. A normalized signal would help, but normalization requires an external reference — violating Principle II.

3. **Rune's tau_slow recommendation is the right direction but premature.** Before adapting tau_slow, we need: (a) implement dual-timescale I in anima_organism.py (mem_slow feeds back into dynamics), (b) show I_slow changes MI gap at any tau_slow, (c) show tau_slow is binding. c045 blocks tau in [0.1, 0.7] — tau_slow=0.005 is outside that range but the principle may hold.

### Stage 3 Forward Path
w_lr is declared a **calibration constant** for ANIMA — binding but not adaptable from within. This is the MI-err structural decoupling.

Stage 3 target shifts to **tau_slow** (slow I integration rate), contingent on:
1. Session 22: Implement dual-timescale I in organism dynamics (mem_slow drives state, not just measurement)
2. Show I_slow has nonzero effect on MI gap
3. Characterize tau_slow binding landscape
4. If binding → design adaptation mechanism

### Session Summary
1 new entry (067), 2 new constraints (c049, c050). The central Phase 1 question answered definitively: dual-timescale I signals cannot locate the w_lr optimum. MI-err structural decoupling confirmed — the most important theoretical result since the timescale barrier identification (Session 19). Stage 3 for w_lr is structurally vacuous under Amendment 1. Forward path: implement dual-timescale I as organism mechanism, target tau_slow.

Next: Session 22 — Implement dual-timescale I in anima_organism.py. Stage 1 revalidation with I_slow feedback. tau_slow binding characterization.

---

## Session 20 — Meta-Cognitive Review: 19 Sessions, 48 Constraints, and the Path to Stage 3 (2026-03-02)

### Context
Session 19 closed Stage 2 as vacuous (Amendment 1) — w_lr has interior optimum at 0.0003 but no Principle-II-compliant signal can detect position on the curve per-step. The timescale barrier (c036, c047, c048) is the central unsolved problem. Session 20 is the scheduled meta-cognitive review: full audit of 19 sessions, 48 constraints, and 63 knowledge entries, plus integration of the Han 7+2 Framework paper.

Full team deployed: Rune (constraint taxonomy), Mira (7+2 mapping), Jude (KB consistency), [engineer] (compile.py fix), Tess (baseline validation), Sage (Stage 3 forward plan).

### Deliverable 1: Constraint Taxonomy (Entry 065, Rune)
All 48 constraints categorized into 8 failure modes:

| Category | Count | Description |
|----------|-------|-------------|
| Non-binding | 9 | Parameter doesn't affect performance |
| Anti-signal | 3 | Signal drives adaptation in wrong direction (c020, c022, c036) |
| Statistical | 5 | False positive from insufficient seeds — SOLVED by protocol |
| Architectural | 6 | Structural impossibility (beta/gamma globality, Living Seed ceiling) |
| Timescale mismatch | 3 | Per-step signal cannot capture sequence-level property — CENTRAL BLOCKER |
| Protocol/methodology | 7 | Evaluation procedure was wrong — SOLVED by infrastructure |
| Calibration-only | 4 | Binding but at optimization boundary |
| Vacuous | 3 | Mechanism works, zero effect |

The pattern: 12 constraints are solved (statistical + protocol), 9 are uninformative (non-binding), 6 are hard walls (architectural). The remaining 21 are the actual search frontier, and 3 of those (timescale mismatch) point at the same wall.

### Deliverable 2: 7+2 Framework Mapping (Entry 065, Mira)
Han's 7+2 Framework defines sufficient state space: 7 individual variables (Vm, [Ca²⁺]ᵢ, {g}, {w}, θ, x, M) + 2 collective (Φ, Wᵢⱼ).

**ANIMA implements 3/7 core variables:**
- Vm → x (cell state)
- {g} → ring topology (conductances)
- {w} → W (world model weights)

**Missing critical variables:**
- [Ca²⁺]ᵢ → dual-timescale calcium. ANIMA's I has single timescale (tau=0.3). 7+2 requires fast error detection + slow amplitude×duration integral. This is exactly the timescale barrier.
- x → spatial embedding (not yet relevant)
- M(t) → neuromodulatory signal (maps to T mechanism)

**Key insight:** The degeneracy finding (most parameters non-binding) is predicted by 7+2: ~10⁵ molecular configurations → ~10² functional states. Non-binding parameters ARE degeneracy.

### Deliverable 3: KB Consistency (Entry 065, Jude)
Zero anomalies. 48 constraints sequential c001-c048. All cross-references valid. Canonical params consistent across state.md, anima_organism.py, run_anima_stage1.py.

### Deliverable 4: compile.py ANIMA-Aware (Entry 064, [engineer])
Fixed compile.py to handle ANIMA architecture:
- Active-substrate vs archived status detection
- format_decision() fallback (content.decision → content.finding)
- Architecture section: active first, archived second
- Historical Summary now includes "analysis" type entries
- Current Stage detection sorts by (session, timestamp)
- Entry 064 created: ANIMA architecture with [ACTIVE SUBSTRATE] marker
- Entry 001 updated: Living Seed marked [ARCHIVED — ceiling 6/8]

### Deliverable 5: Baseline Validation (Entry 065, Tess)
ANIMA Stage 1 baseline confirmed exactly reproducible:
- alive_gap = +0.1256 (d=2.46, 10 seeds × K=[4,6,8,10])
- Uses calibrated params: w_lr=0.0003, gamma=3.0
- Exact match to Session 19 calibration result

### Deliverable 6: Stage 3 Forward Plan (Entry 066, Sage)

**Central problem:** Stage 3 requires self-generated adaptation signal for w_lr. The timescale barrier means no per-step signal can detect optimal w_lr. Need a sequence-timescale signal.

**Approach A (primary): T as cycle-quality signal.**
- Implement T as cycle-level accumulator integrating err variance (not mean) over each K-signal sequence
- At cycle boundaries, T produces cycle_quality signal that adapts w_lr between sequences
- Why variance not mean: mean err is ~constant (~0.04) regardless of w_lr (phase-dominated). Variance captures prediction quality fluctuations that depend on w_lr
- Principle II compliant: T's state computed by same dynamics
- Avoids timescale barrier: fires at cycle boundaries only

**Phase gate:**
1. Phase 1 (theory): Derive expected err variance as f(w_lr) analytically. Define variance dimension. Define cycle boundary.
2. Phase 2 (binding): 5-seed fixed T-cycle signal. Does T-driven w_lr show non-trivial values?
3. Phase 3 (validation): 10-seed multi-K paired design. T-adaptive vs fixed w_lr=0.0003.

**Vacuousness criterion:** If Phase 1 shows no signal has information content and no natural equilibrium exists, Stage 3 for w_lr is structurally vacuous. The MI-err structural decoupling means the inverted-U has no intrinsic gradient detectable from within.

### Adversarial Review Findings ([researcher] + Sage)

The adversarial exchange produced three key theoretical results:

1. **Across-K err variance is monotonically decreasing in w_lr, not peaked.** High w_lr → steep err decay within signal windows → high across-K variance. Low w_lr → flat err → low variance. Raw err variance cannot locate the interior optimum.

2. **The I_slow/I_fast_mean ratio does not cleanly peak.** Both low w_lr (large I_slow from persistence) and high w_lr (small I_fast from fast decay) produce high ratios. The ratio is U-shaped or monotonic, not peaked at 0.0003.

3. **MI-err structural decoupling hypothesis.** The interior optimum may exist in MI space (sequence distinguishability) but NOT in prediction-error space. If true, w_lr is a calibration parameter at ALL stages — not an adaptation target.

**The rescue: natural equilibrium from boundary forces.** At high w_lr, W oscillates/diverges (downward pressure). At very low w_lr, err is noise (upward pressure). A monotonic signal with boundary constraints may have a natural fixed point near the optimum. Key question for Session 21.

**Architecture decision: dual-timescale I, not T-as-Φ.** Stage 3 = augment I with slow timescale. T-as-Φ violates Principle IV (adds frozen frame element). Deferred to Stage 4+.

### Six Meta-Lessons ([researcher])

1. **The landscape is almost entirely flat.** Of all parameters tested across two substrates, almost none are binding. The system is massively degenerate. 7+2 explains: ~10⁵:10² compression.

2. **The timescale barrier is THE problem.** Three constraints (c036, c047, c048) converge on the same wall. 7+2 names it precisely: plasticity direction requires amplitude × duration (2D discriminant) at intermediate timescale.

3. **Negative results ARE the output.** 48 constraints in 19 sessions = 2.5 closed doors per session. The pattern of elimination IS the search. The remaining solution volume shrinks monotonically.

4. **Statistical infrastructure matured correctly.** Multi-K averaging (CV 14.8%→5.1%), 10-seed minimum, phased validation. No false positive has survived since Session 15.

5. **Amendment 1 was essential.** Vacuous stage declaration prevented deadlock at Stage 3 (Living Seed) and Stage 2 (ANIMA). Both correctly identified: mechanism works, landscape flat, theoretical explanation exists.

6. **The substrate transition was correct.** Amendment 2 (forward viability check) caught Living Seed ceiling before wasted optimization. ANIMA has 4.6× MI gap. Constraints transferred as design requirements.

### Session Summary
4 new entries (063-066), 0 new constraints. Meta-cognitive review complete. Stage 3 forward plan synthesized. Infrastructure improved (compile.py ANIMA-aware, baseline validated). The timescale barrier remains the central problem; the T mechanism operating at cycle boundaries with err variance signal is the proposed solution.

Next: Session 21 — Stage 3 Phase 1 (analytical). Derive expected err variance as f(w_lr). Define cycle boundary concretely. Determine if variance of prediction error contains information that mean does not.

---

## Session 19 — ANIMA Stage 2: The Timescale Barrier (2026-03-02)

### Context
Session 18 confirmed ANIMA Stage 1 (MI gap +0.0274, d=0.84). Stage 2 requires: the system's own computation produces a signal that, if used to modify a parameter, improves performance. Target parameter: w_lr (world model learning rate). Three parameters to characterize (w_lr, tau, gamma), then test adaptation.

### Deliverable 1: Parameter Binding Diagnostic (Entry 059)
5-seed screening → 10-seed extended sweep.

**w_lr: INTERIOR OPTIMUM at 0.0003.** Classic inverted-U:
| w_lr | alive_gap | interpretation |
|------|-----------|----------------|
| 0.0001 | +0.0819 | W too slow — err noisy, I accumulates garbage |
| **0.0003** | **+0.0948** | **Sweet spot — W tracks structure, err persists for I** |
| 0.001 | +0.0616 | W faster — err shrinks, I partially starved |
| 0.01 | +0.0274 | Canonical — W too fast, I starved |

3.5× improvement over canonical. The W-I tension mechanism: W and I compete for the same error signal. Lower w_lr → W learns slower → prediction error persists → I accumulates richer sequence history → better MI gap. But too low → W doesn't learn → err is noise → I accumulates garbage.

**tau: NON-BINDING.** Flat across [0.1, 0.7]. Consistent with Living Seed c032.

**gamma: BOUNDARY-OPTIMAL.** Monotonically increasing to 3.0. Calibration-only (c035 pattern).

w_lr optimum is K-independent (acts on intra-signal 60-step windows, not inter-signal K-dependent timescale).

New constraints: c044 (w_lr=0.0003 canonical), c045 (tau non-binding), c046 (gamma calibration-only).

### Deliverable 2: Stage 2 Adaptation Tests (Entries 060-061)

**Attempt 1 — Gradient-based w_lr (Entry 060): FAIL.**
err EMA drives gradual w_lr push. Push magnitude ~1e-8/step — functionally inert. w_lr moved from 0.01000 to 0.01003 over entire run.

**Attempt 2 — Reactive phase-dependent w_lr (Entry 061): FAIL.**
Rule: `w_lr_eff = w_lr_base / (1 + err_scale * prev_mean_abs_err)`. High err → low w_lr (preserve err for I). Low err → high w_lr (W consolidates). 5 configurations tested at 10 seeds:

| Config | Adaptive gap | d vs fixed 0.0003 | Result |
|--------|-------------|-------------------|--------|
| base=0.001, scale=3 | +0.0756 | -0.511 | Hurts |
| base=0.001, scale=10 | +0.0831 | -0.307 | Hurts |
| base=0.001, scale=30 | +0.0890 | -0.149 | Hurts |
| **base=0.0003, scale=3** | **+0.0960** | **+0.031** | **Ties** |
| base=0.0003, scale=10 | +0.0830 | -0.311 | Hurts |

Pattern: starting above optimum → adaptation hurts. Starting at optimum → ties. The signal encodes phase presence (signal vs settling), not W-I quality balance. The ANIMA analog of c036.

**Theoretical rejection — I-state variance:** Non-monotonic w_lr mapping (both too-high and too-low w_lr produce low I variance). Cannot determine push direction. c005 analog.

New constraint: c047 (per-step signals can't capture sequence-level w_lr optimality), c048 (Stage 2 vacuous).

### Deliverable 3: Stage 2 Vacuous Verdict (Entry 062)

**ANIMA Stage 2 declared vacuously passed under Amendment 1.** All four criteria satisfied:
1. Multiple approaches tested with adequate power (gradient, reactive×5, theoretical rejection)
2. Mechanism works (w_lr modulates 3.5× range, non-degenerate)
3. No performance difference (best d=+0.031, p=0.863)
4. Theoretical explanation: per-step ↔ sequence-level timescale gap

**Key insight — ANIMA's vacuousness is different from Living Seed's:**
- Living Seed Stage 3: eta landscape is FLAT (parameter non-binding, nothing to optimize)
- ANIMA Stage 2: w_lr landscape has INVERTED-U (interior optimum exists!) but no Principle-II-compliant signal can detect position on the curve per-step

The information about optimal w_lr exists only at the sequence level (~1000 steps). No per-step measurement can access it. This is a fundamental timescale barrier, not a landscape shape issue.

### Calibration
Canonical w_lr updated from 0.01 to 0.0003. ANIMA MI gap improved from +0.0274 to +0.0948 (3.5×).

### Session Summary
4 new entries (059-062), 5 new constraints (c044-c048). Stage 2 vacuous — w_lr calibrated but not adaptable via self-generated signal. The per-step vs sequence-level timescale barrier is the dominant architectural constraint for adaptation in ANIMA.

New files: `src/run_anima_binding.py`, `src/anima_organism_adaptive.py`, `src/run_anima_stage2.py`.

Next: Session 20 should address the meta-cognitive review (due at Session 20) and consider Stage 3 forward — or whether the timescale barrier applies to all future stages.

---

## Session 18 — ANIMA Stage 1: The New Substrate Computes (2026-03-02)

### Context
Session 17 closed the Living Seed chapter — frozen frame minimum 6/8, architecture ceiling declared (c037). 37 constraints = design spec for next substrate. ANIMA's W⊕I⊕T decomposition (World model + Internal memory + Temporal phase) was the candidate. Three questions: Does ANIMA pass Amendment 2 forward check? How do 37 constraints map? Does it compute (MI gap > 0)?

### Deliverable 1: Amendment 2 Forward Check (Entry 056)
**ANIMA PASSES — in principle.** The Living Seed failed Stage 7 because its equation is Python source code (not data). Category error. ANIMA's update rules are parameterized by weight tensors — tensors ARE first-class data. Path to Stage 7: hypernetwork sublayer H within I reads I's current state, outputs weight deltas for I's own update gates. Self-modification applies at T cycle boundaries (not per-step) to prevent c008-analog bang-bang instability.

New constraints: c038 (hypernetwork required for Stage 7), c039/c041 (cycle-boundary gating), c040 (novel-input exit criterion), c042 (characterize CV before seed count), c043 (guard against false positives/anti-signals).

### Deliverable 2: Constraint Audit (Entry 057)
All 37 Living Seed constraints classified against ANIMA:
- **5 addressed-by-design**: c003 (no local proxy needed — W is global), c006/c007 (per-cell decomposition not needed — W inherently distributed), c009 (meta-rate external to eta — I dynamics separate from W), c035 (delta at boundary — I provides memory, no blending needed)
- **1 open wall**: c008 (bang-bang oscillation risk persists for self-referential meta-rates)
- **13 methodology carry-forward**: Statistical protocol constraints (c010, c014-c019, c024, c027, etc.) — substrate-independent
- **18 not-applicable**: Living Seed-specific (beta/gamma parameters, resp_z derivatives, etc.)

### Deliverable 3: Stage 1 Ground Truth Test (Entry 058)
**MI gap > 0. GROUND TRUTH PASSES.**

New files: `src/anima_organism.py` (AnimaOrganism class, pure Python W+I dynamics) and `src/run_anima_stage1.py` (runner script).

Results (10 seeds × K=[4,6,8,10], n_perm=8, n_trials=6):
- STILL gap: -0.0006 (std=0.0156) — no I accumulation, no sequence memory
- ALIVE gap: +0.0274 (std=0.0393) — I accumulates prediction error
- d-statistic: 0.84 (medium effect)
- Ground truth: **PASS** (alive_gap > 0)

d=0.84 vs Living Seed d~20+ reflects uncalibrated prototype, not architecture limitation. The critical result is gap > 0 — ANIMA computes on this task class. Parameter calibration deferred to Session 19.

### Session Summary
All three deliverables complete. ANIMA is a viable substrate:
1. Passes forward check (Stage 7 reachable in principle)
2. Resolves 5 constraints by design, 1 open wall (c008), 13 carry forward
3. Computes — MI gap > 0 confirmed empirically

Stage 2 (W-conditioned adaptation) ready for Session 19.

---

## Session 17 — Architecture Ceiling: The Living Seed Chapter Closes (2026-03-02)

### Context
Session 16 closed Stage 4 as a characterization result: delta=1.0 is binding (+6%) but calibration-only; the Living Seed is a memoryless signal processor. All structural parameters exhaustively characterized across 16 sessions. Amendment 2 (Session 15) requires a forward viability check before any further stage progression.

### The Forward Viability Check
Amendment 2 asks: can the Living Seed in principle satisfy Stage 7 — represent and modify its own update rule as first-class data?

The answer is no. The equation phi[k] = tanh(alpha*x + beta*(x[k+1]+gamma*s)*(x[k-1]+gamma*s)) is hardcoded Python. No self-representation mechanism exists. This is not a performance ceiling or a signal search problem — the mechanism category does not exist in the substrate.

### Architecture Ceiling Declaration
Living Seed frozen frame minimum: 6/8. Stage progression halted under Amendment 2.

The 16-session journey produced:
- 2 genuine frozen frame reductions (alpha adaptive Session 4, eta vacuous Session 12)
- 1 binding calibration (delta=1.0, Session 16)
- 36 constraints characterizing every testable aspect of the architecture
- 3 constitutional amendments governing the research process itself
- The discovery that the Living Seed is a memoryless signal processor

Each constraint is a design requirement for the next substrate.

### Next Substrate: ANIMA
The Living Seed's failure pattern diagnoses what the next substrate must have. See `src/session_blindspots.md` Section 8 for the full requirements table derived from the constraint set.

---

## Session 16 — Stage 4 Closure: The Living Seed is a Memoryless Signal Processor (2026-03-02)

### Context

Session 15 established that ALL structural parameters are resolved: delta=1.0 is optimal (+6%, d=+1.289), beta/gamma are architecturally impossible, and every other parameter (eta, tau, eps, threshold, clip bounds) is non-binding. Session 16 had two tasks: (1) update the canonical implementation to delta=1.0, and (2) test whether an adaptive delta mechanism can discover the optimum.

### Task 1: Canonical Update — delta=1.0 (Entry 053)

`the_living_seed.py` updated: `self.delta = 1.0` (was 0.35). The state update equation `xs_new = (1-delta)*xs + delta*phi` simplifies to `xs_new = phi` — pure replacement. State blending eliminated. This is the canonical implementation going forward. +6% MI gap improvement baked in.

### Task 2: Adaptive Delta Phase 2 — Decisive Null (Entry 054)

**Experiment design:** Four conditions tested at 10 seeds, K=[4,6,8,10], n_perm=4, n_trials=3:
- fixed_0.35 (old canonical, baseline)
- fixed_1.0 (new canonical, calibration upper bound)
- adaptive_full (starts at 0.35, adapts via divergence signal)
- adaptive_reversed (starts at 1.0, adapts via divergence signal)

**Adaptive mechanism:** State-output divergence per cell `||p[k] - xs[k]||` — Principle-II-compliant (arises from the same computation). High divergence → push delta toward replacement. Leaky integrator (lr=0.01).

**Results:**

| Condition | Mean | d vs 0.35 | p | Final delta |
|-----------|------|-----------|---|-------------|
| fixed_0.35 | +0.6501 | BASE | — | fixed |
| fixed_1.0 | +0.6895 | +1.289 | 0.0000 | fixed |
| adaptive_full | +0.6033 | -1.968 | 0.0000 | 0.162 |
| adaptive_reversed | +0.5949 | -1.857 | 0.0000 | 0.161 |

**Critical finding:** Both adaptive conditions converge to **~0.16 regardless of initialization**. Started at 0.35 → converges to 0.16. Started at 1.0 → converges to 0.16. The divergence signal is an ANTI-SIGNAL — it drives delta toward a stable attractor that is NOT the performance optimum.

Adaptive conditions are significantly worse than the calibrated optimum (adaptive_full vs fixed_1.0: d=-2.414, p=0.0000).

**New constraint c036:** Adaptive delta via state-output divergence signal is an anti-signal. Never retry divergence-based adaptive delta.

### Stage 4 Closure Declaration

**[director]'s declaration:** Stage 4 is NOT vacuously passed. Delta IS binding (+6%). Declare closed as characterization result: **the Living Seed is a memoryless signal processor.**

What this means:
- Pure state replacement (delta=1.0) is architecturally optimal
- No self-generated signal can discover this optimum — the adaptive mechanism converges to the wrong value
- The system functions as a pure reactive processor: each step's computation replaces state entirely
- Alpha carries adaptation memory independently; xs carries no history

**All Stage 4 structural parameters resolved:**

| Parameter | Finding | Evidence |
|-----------|---------|---------|
| beta/gamma | Architecturally impossible (Principle II) | c002-c007, Entry 051 |
| eta | Non-binding (vacuous, Stage 3) | c011 |
| threshold | Non-binding | c023 |
| clip bounds | Non-binding under canonical protocol | c025, c029 |
| tau | Non-binding, K-dependent direction | c031-c033 |
| eps | Non-binding around canonical | c034 |
| delta | Binding (+6%) but calibration-only, anti-signal adaptive | c035, c036 |

No adaptive target remains. The frozen frame floor for this architecture is **6** — unchanged since Session 12.

### What Stage 4 Tells Us

Stage 4's answer is precise: the architecture has a frozen frame floor at 6. The path forward requires either:
1. **Stage 5 (topology)** — making the connectivity pattern adaptive, which may make beta/gamma structurally irrelevant
2. **Stage 6 (functional form)** — making the update rule's mathematical form adaptive, addressing the true bottleneck
3. **Architecture modification** — introducing lateral signaling to make beta/gamma locally accessible

### Session 16 Verdict

Two tasks, one session, decisive results. The Living Seed has been fully characterized as a Stage 4 system. The constitution advances to Stage 5.

---

## Session 15 — Tau False Positive & Beta/Gamma Impossibility (2026-03-01)

### Context
Session 14 (Entry 046) found tau=0.2 significantly beats canonical tau=0.3 (d=+1.202, p=0.007) at 5 seeds with multi-K protocol. This was the strongest Stage 4 target. Beta/gamma impossibility had been established experimentally across Sessions 3-5 (c002-c007) but never formally assembled.

### Path A: Adaptive Tau Investigation

**Experiment 1: Tau Meta-Signal Diagnostic (Entry 049)**
Three diagnostics (alpha distribution, MI trajectory, cell correlation) showed ZERO difference between tau=0.2 and tau=0.3 at 200 steps. But per-K decomposition revealed critical structure:
- K=4 (200 steps): d=+1.336, p=0.003 — strongest at short training
- K=8 (400 steps): d=-0.266 — REVERSED
- Attention entropy: tau=0.2 → H=1.43, tau=0.3 → H=1.52 (measurable, Principle II compliant)

The tau effect is TRANSIENT — helps early, hurts late. Multi-K average conceals this by mixing positive and negative K contributions.

**Experiment 2: Adaptive Tau Phase 1 (5 seeds)**
Designed plast-driven per-cell adaptive tau: `tau_cell = tau_min + (tau_max - tau_min) * plast`. Plasticity signal drives tau — settled cells get lower tau (sharper attention). Four conditions tested. C_adapt_narrow [0.15, 0.35] showed d=+1.944 vs canonical but only d=+0.615 vs fixed tau=0.2 (p=0.169).

**Experiment 3: Adaptive Tau Phase 2 — DEFINITIVE NULL (Entry 050)**
10-seed validation [42, 137, 2024, 999, 7, 10, 11, 12, 13, 14]:
- Adapt vs Fixed 0.2: d=-0.039, p=0.902 — dead null
- Fixed 0.2 vs Canon 0.3: d=+0.317, p=0.317 — NOT SIGNIFICANT
- Original 5 seeds ALL favored tau=0.2; new 5 seeds showed OPPOSITE pattern

**Entry 046 (d=+1.202, p=0.007) is a 5-seed false positive.** The tau effect does not survive 10-seed validation. Tau joins eta, threshold, clip bounds as non-binding. Adaptive tau is dead.

### Path C: Beta/Gamma Impossibility Argument (Entry 051)

Formal impossibility argument assembled from c002-c007 (document: src/beta_gamma_impossibility.md). Six constraints from three sessions close the argument: beta/gamma require global information to adapt, the computation produces only local information, no bridge exists within this architecture. Four escape conditions documented, one partially open (single scalar parameterizing beta/gamma family).

### Experiment 4: Eps/Delta 10-Seed Validation (Entry 052) — KEY RESULT

Validated all eps/delta conditions from Entry 046 at 10 seeds.

**Eps findings:**
- eps=0.05: d=-0.461, p=0.145 (was d=-3.092 at 5 seeds — THIRD 5-seed false positive)
- eps=0.0: d=-0.757, p=0.017 (zero coupling worse, confirming coupling helps)
- eps=0.5: d=+0.425, p=0.179 (not significant)
- Eps is non-binding around canonical (flat landscape near 0.15)

**Delta findings — SIGNIFICANT:**
- delta=0.1: d=-2.899, p=0.000 (-25.9% — degradation SURVIVES)
- delta=0.7: d=+0.914, p=0.004 (+4.9%)
- delta=0.95: d=+0.934, p=0.003 (+5.3%)
- delta=1.0: d=+1.289, p=0.000 (+6.0%, 9/10 seeds)
- Pattern: MONOTONIC improvement. Higher delta = better. Best at boundary (1.0).
- delta=1.0 means pure replacement — no state blending. `v = p[k]`.
- This is CALIBRATION: optimal at boundary, no room for adaptive delta to beat it.
- Canonical delta should be updated from 0.35 to 1.0.

### New Constraints
- **c031**: Tau's MI effect is K-dependent — positive at K=4, reverses at K=8
- **c032**: Tau is non-binding in [0.2, 0.5] range (Entry 046 was 5-seed false positive)
- **c033**: Plast-driven adaptive tau dead (d=-0.039 vs fixed)
- **c034**: Eps non-binding in [0.05, 0.5] range (Entry 046 eps=0.05 d=-3.092 was false positive)
- **c035**: Delta optimal at boundary (1.0), +6%. Calibration, not adaptive opportunity.

### Stage 4 Status After Session 15

ALL structural parameters are now resolved:

| Parameter | Finding | Constraint |
|-----------|---------|------------|
| eta | Non-binding, vacuously passed (Stage 3) | c011 |
| threshold | Non-binding | c023 |
| clip bounds | Non-binding under canonical protocol | c025, c029 |
| tau | Non-binding, K-dependent direction | c031-c033 |
| eps | Non-binding around canonical | c034 |
| delta | Calibration: optimal at 1.0, +6% | c035 |
| beta/gamma | Impossible under Principle II | c002-c007, Entry 051 |

No Stage 4 adaptive target exists. Every parameter is either non-binding (performance-insensitive), calibration-only (better fixed value exists at boundary), or architecturally impossible to make adaptive.

Stage 4 assessment: VACUOUS PASSAGE under Amendment 1. The structural constants are not binding constraints on performance. The frozen frame floor is in the EQUATION FORM itself (Stage 6), not in parameter values (Stage 4).

---

## Session 14 — Stage 4 MI-GT Validation & Parameter Binding (2026-02-25)

### Context
Session 13 identified clip bounds as binding at multi-K resolution (Entry 042, d=7.583) and resolved the birth confound (Entry 044, clip constraint accounts for 97% of MI improvement). An external reviewer (Kimi K2.5) raised a critical concern: MI gap had never been validated against the binary ground truth test (Principle V). Additionally, the researcher agent fabricated state parameter data (Entry 043), requiring real experiments.

### Experiment 1: Centroid MI-GT Validation (Entry 045)
First attempt used centroid cosine similarity (averaging across cells). Found MI gap correlates with GT (r=0.934) but VERY_NARROW was the WORST condition — opposite of Entry 042. Flagged as methodologically confounded: centroid vs per-cell measurement.

### Experiment 2: State Parameter Binding (Entry 046)
Swept tau, eps, delta at multi-K resolution (K=[4,6,8,10], 5 seeds, reduced protocol). All three confirmed BINDING:
- **tau=0.2**: d=+1.202, p=0.007 — significant improvement over canonical tau=0.3
- **delta=0.1**: d=-3.122, p=0.000 — strong degradation (lower mixing hurts)
- **eps=0.05**: d=-3.092, p=0.000 — strong degradation (weak coupling actively harmful)
Methodology caveat: these are calibration findings (better fixed values), not adaptive opportunities.

### Experiment 3: Per-Cell MI-GT Validation (Entry 047) — KEY RESULT
Re-ran MI-GT validation using birthconfound-exact protocol (per-cell cosine, Python random, same sequence structure as Entry 042/044). 10 seeds, 7 conditions.

**Results:**
- All 7/7 conditions pass Principle V (10/10 seeds each)
- VERY_NARROW: MI=+0.7657, d=+3.134, p=0.0000 — **Entry 042 REPLICATES**
- VERY_NARROW per-cell MI mean (+0.7657) exactly matches Entry 042's 5-seed mean
- d weaker with 10 seeds (3.134 vs 7.583) but highly significant

**Centroid anomaly resolved:** Per-cell cosine shows VERY_NARROW as BEST; centroid cosine showed it as WORST. Explanation: tight clip bounds improve cell-level specialization but may reduce population-level coherence. Per-cell captures the meaningful signal.

**External reviewer's concern addressed:** MI gap and ground truth are ALIGNED — both always positive across all conditions. Ground truth is a universal floor, not discriminative.

### Experiment 4: Protocol Confound Resolution (Entry 048) — CRITICAL CORRECTION

Entry 047 attributed the centroid/per-cell reversal to measurement method and declared per-cell canonical (c028). But neither Entry 045 nor 047 controlled for PROTOCOL: Entry 042 used a short protocol (50 steps, no warmup), while centroid validation used the long harness.py protocol (300+60+30+60).

Ran BOTH per-cell and centroid on the same organisms, same long protocol, same run. Two seed sets (SetA [10-19], SetB [42,137,2024,999,7]). 7 clip conditions.

**Results:**
- Per-cell and centroid give **IDENTICAL rankings** under the same protocol
- VERY_NARROW ranks LAST (7/7) on BOTH measurements, BOTH seed sets
- Per-cell/centroid ratio: 0.93-1.01x (essentially identical)
- VERY_NARROW vs CANONICAL: d=-1.654 to -2.912 (strongly negative on both methods)

**The real confound was protocol, not measurement:**
- Short protocol (50 steps): CANONICAL +0.6274, VERY_NARROW +0.7657 (VN best, 3.6-9.4x higher MI)
- Long protocol (300+60+30+60): CANONICAL +0.1745, VERY_NARROW +0.0818 (VN worst)

Entry 042's d=7.583 is real but specific to the short protocol. Under the canonical harness.py protocol, clip bounds are non-binding (c025 was correct all along). The clip bound effect is transient — matters in early computation before alpha self-organizes, washes out with training.

**c028 RETRACTED** — per-cell and centroid agree when protocol is controlled. Neither is inherently superior. New constraints: c029 (Entry 042 is protocol-specific), c030 (retract c028).

### Key Decisions
- c028 retracted — per-cell vs centroid is not the relevant variable; protocol length is
- Entry 042 d=7.583 is protocol-specific (short protocol only), not generalizable
- Clip bounds confirmed non-binding under canonical harness.py protocol (c025 restored)
- State parameters (tau, eps, delta) confirmed as Stage 4 targets (Entry 046)
- tau=0.2 is the strongest adaptive candidate (inverted-U landscape, d=+1.202, p=0.007)

## Session 12 — Constitutional Amendment 1 & Stage 3 Vacuously Passed (2026-02-24)

### Context
Session 11 completed Phase 2 validation for two Stage 3 signal candidates: delta_stability passed Phase 2 at alpha_meta=0.05/0.10 with tiny improvement (+0.0068, within noise floor); delta_correlation was eliminated (anti-signal). Session 12 did not run new experiments. Instead, it resolved the Stage 3 deadlock by addressing a structural problem with the constitution itself.

### Constitutional Amendment 1: Vacuous Stages

The original rule ("Do not skip stages") was designed to prevent unvalidated assumptions from accumulating. It did not account for a different failure mode: stages that are empirically vacuous — where the frozen element can be made adaptive but doing so produces no measurable effect.

**The deadlock:** After 7 sessions (Sessions 6-11) across 5 independent approaches, Stage 3 (eta adaptation) is exhaustively validated as non-binding. The mechanism works — eta becomes adaptive with healthy, non-degenerate distributions — but no approach produces measurable performance improvement. Under the original rule, Stage 3 cannot be passed (because "beats fixed" is undefined on a flat landscape), and Stage 4 cannot be entered (because Stage 3 has not passed). This is a deadlock with no resolution under the original constitution.

**The amendment:** A stage may be declared **vacuously passed** if ALL of the following hold:
1. Multiple independent approaches tested with adequate statistical power
2. Mechanism works — frozen element becomes adaptive with non-trivial, non-degenerate behavior
3. No measurable performance difference between adaptive and frozen versions
4. Theoretical explanation exists for why the element is not a binding constraint

**Evidence for Stage 3 vacuous passage:**

| Evidence | Source |
|----------|--------|
| c011: system insensitive to per-cell eta heterogeneity | Session 8 (10 seeds, p=0.999) |
| c012: resp_z derivative tower collapses at order 1 | Session 8 (autocorr 0.98 → 0.11 → -0.48) |
| c013: Stage 3 requires fundamentally different signal | Session 8 |
| Self-referential eta: bang-bang oscillation | Session 6 Exp A |
| delta_rz on canonical: p=0.999 training, p=0.688 novel | Session 8 (10 seeds) |
| delta_stability Phase 2: +0.0068 (within noise floor CV=29%) | Session 11 |
| delta_correlation: anti-signal, eliminated | Session 11 |

**Theoretical explanation:** Eta is a scaling parameter for alpha updates. On a flat performance landscape (already established: evolutionary search converges to noise ceiling, c018), scaling the update rate cannot improve performance. It modifies HOW FAST the system learns, not WHAT it learns. The binding constraints are structural (beta/gamma globality, rule form) — not the adaptation rate.

**Consequence:** Eta is marked as "adaptive (vacuous)" in the frozen frame. The frozen frame shrinks from 7/8 to 6/8. Stage 3 is vacuously passed. Stage 4 (structural constants become adaptive) is the active problem.

**Revalidation trigger:** If Stage 4+ experiments reveal a conditional dependency on adaptive eta (e.g., structural changes create sensitivity that was absent before), Stage 3 vacuous status must be revalidated.

### Stage 4 Entry

**Current frozen frame (6/8):**

| Element | Status | Notes |
|---------|--------|-------|
| Alpha | adaptive | Per-cell, Stage 2, Session 4 |
| Eta | adaptive (vacuous) | Stage 3, Session 12 — non-binding constraint |
| Beta | frozen | Global coupling, no local proxy (Stage 4 target) |
| Gamma | frozen | Global self-coupling, no local proxy (Stage 4 target) |
| Plasticity Rule Form | frozen | Stage 6 target |
| Topology | frozen | Stage 5 target |
| Activation Function | frozen | Stage 6 target |
| Ground Truth Test | frozen | Stage 8 only |

**Stage 4 definition:** "Structural constants become adaptive" — parameters that define WHAT the system computes (not just HOW STRONGLY), driven by the system's own dynamics. Beta and gamma are the primary Stage 4 targets. Per-cell decomposition failed (c006: 53% MI loss, Session 5), local proxy search failed (c003: best r=0.44, 7 proxies tested), and analytical gradient is vacuous (c002). A fundamentally new approach is required.

### Infrastructure Updates

**Paper compiler updated** for "adaptive (vacuous)" status — frozen_frame.json now correctly reflects eta as adaptive (vacuous), and compile_paper.py handles the new status value without requiring manual edits.

**Progress metric corrected:** The paper previously counted adaptive elements as a simple fraction of 8. This understates progress because vacuous stages represent genuine scientific knowledge (the element is not a binding constraint), not a missing step. The metric now tracks frozen frame count (currently 6/8 frozen) rather than stage-completion-as-binary.

### Session 12 Verdict

**Not an experimental session. A constitutional session.**

Seven sessions of evidence accumulated until the constitution itself became the bottleneck. Amendment 1 resolves the structural problem: the constitution now has a mechanism for handling empirically vacuous stages rather than deadlocking on them.

**What changed:**
- Stage 3: not_started → vacuously passed
- Eta: frozen → adaptive (vacuous)
- Frozen frame: 7/8 → 6/8
- Active stage: Stage 3 → Stage 4

**What did NOT change:**
- The ground truth test is unchanged
- Beta/gamma remain frozen — the structural problem that has blocked progress since Session 5
- The system is insensitive to eta variation — that finding is still true
- No new experiments were run; no new data was collected

### Constraints Added (3 in Session 11, none new in 12, total 22)
- c020: delta_correlation one-way eta adaptation is anti-signal for this system
- c021: delta_stability Phase 2 improvement within noise floor — Phase 3 required
- c022: One-way eta reduction designs misread successful adaptation as degradation

### Open for Stage 4
1. Classify all plasticity rule parameters by type (scaling vs structural) — which elements legitimately belong in Stage 4 vs Stage 6?
2. Audit infrastructure for Stage 4 readiness — harness.py, search_space.py, constraint_checker.py
3. Design Stage 4 experiment: what mechanism can drive structural adaptation from self-generated signal, given that per-cell decomposition (c006), local proxies (c003), and analytical gradients (c002) all failed?
4. Meta-cognitive review (completed this session — see progress.json)

---

## Session 11 — Phase 2 Validation of Stage 3 Signals (2026-02-25)

### Context
Session 10 completed research phase for Stage 3 signals: 5 candidates identified, 2 survived adversarial filtering (delta_stability, delta_correlation). Implementation existed in harness.py but was never validated. Session 11 objective: fix bugs, run Phase 2 validation for both candidates.

### Bug Fixes (6 total in harness.py)
1. **delta_stability formula inverted** — used `phi_change/N` instead of spec's `1/(1+sqrt(phi_change))`, reversing adaptation direction
2. **Contamination check vacuous** — `mean(resp_z)` is identically 0 by z-scoring; changed to `mean(|resp_z|)` (~0.82)
3. **Unfair eta starting point** — Stage 3 started eta=0.01 (33x canonical 0.0003), clip [0.001, 0.1] excluded canonical; fixed to eta=0.0003, clip [0.0001, 0.01]
4. **Phase 2 using Phase 3 compute** — Validation functions used n_perm=8, n_trials=6 instead of spec's n_perm=4, n_trials=3
5. **Eta bounds check wrong values** — Trivial-convergence check compared against old clip bounds 0.001/0.1 instead of actual 0.0001/0.01
6. **Unicode encoding crash** — `>=` character crashes on Windows cp1252

### Phase 2 Results

**delta_stability (2-way: settling -> decrease eta, destabilizing -> increase eta)**

| alpha_meta | Phase 2 | GT 3/3 | Contamination | Eta | Improvement |
|------------|---------|--------|---------------|-----|-------------|
| 0.01 | FAIL | FAIL (2/3) | +0.013 | TRIVIAL | +0.0149 |
| 0.05 | **PASS** | PASS | +0.013 | [0.000297, 0.000309] | +0.0068 |
| 0.10 | **PASS** | PASS | +0.015 | [0.000292, 0.000318] | +0.0037 |

**delta_correlation (1-way: degrading -> decrease eta, improving -> neutral)**

| alpha_meta | Phase 2 | GT 3/3 | Contamination | Eta | Improvement |
|------------|---------|--------|---------------|-----|-------------|
| 0.01 | FAIL | PASS | +0.004 | [0.000210, 0.000299] | -0.0034 |
| 0.05 | FAIL | PASS | +0.023 | [0.000100, 0.000297] | -0.0149 |
| 0.10 | FAIL | PASS | +0.040 | [0.000100, 0.000295] | -0.0231 |

### Analysis

**delta_stability advances to Phase 3** at alpha_meta=0.05 (recommended) or 0.10. All Phase 2 criteria met. However, the improvement (+0.0068) is tiny and within the noise floor (c014: CV=29%). Phase 3's 10+ seed validation will determine if this is real signal or noise.

**delta_correlation eliminated.** One-way design is fundamentally flawed: successful adaptation changes error-push correlation, which the signal misinterprets as degradation. More coupling (higher alpha_meta) = more suppression = worse performance. Monotonically harmful.

### Session 11 Verdict
**Phase 2 COMPLETE.** One candidate advances (delta_stability), one eliminated (delta_correlation). Six implementation bugs found and fixed. Team infrastructure broke during context compaction; validations run directly by team lead.

### Constraints Added (3 new, total 22)
- c020: delta_correlation one-way eta adaptation is anti-signal for this system
- c021: delta_stability Phase 2 improvement within noise floor — Phase 3 required
- c022: One-way eta reduction designs misread successful adaptation as degradation

### Open for Next Session
1. Phase 3 validation for delta_stability (10+ seeds, full protocol n_perm=8, n_trials=6, K=[4,6,8,10])
2. Meta-cognitive review (overdue since Session 10)
3. If Phase 3 passes: update frozen_frame.json, recompile paper, advance Stage 2->3

---

## Session 10 — Stage 3 Signal Search (2026-02-24)

### Context
Session 9 built the knowledge infrastructure. With resp_z derivative tower collapsed (c012) and no known signal family for Stage 3 (c013), Session 10 tackled the open question: what signal can drive eta adaptation?

### Stage 3 Signal Search (Research Phase — COMPLETED)
- **6-agent team** (strategist, engineer, researcher, analyst, reviewer, qa) deployed
- **Strategist** produced signal requirements specification: 5 required properties (R1-R5), 6 forbidden properties (F1-F6)
- **Researcher** identified 5 candidate signal families from literature and architecture analysis
- **Team lead (adversarial)** filtered candidates:
  - **REJECTED: Push Variance** — direction ambiguous, resp_z-derived (violates c013)
  - **REJECTED: Plasticity Consensus** — uses orthogonal variables, R5 violation
  - **REJECTED: Response Autocorrelation** — redundant with resp_z family (violates c012)
  - **SURVIVED: delta_stability** — `stability(t) = mean(||phi(t)-phi(t-1)||²)`; delta drives eta: settling → ↓eta, destabilizing → ↑eta
  - **SURVIVED: delta_correlation** — `corr(|bare_diff|, push_mag)`; delta drives eta: degrading → ↓eta, improving → neutral (one-way)
- **Key insight**: Temporal derivatives solve direction ambiguity — use delta of signal, not absolute value
- **Contamination check required**: Both signals must show |corr(signal, mean(resp_z))| < 0.7

### Implementation Phase — FAILED
- `src/harness.py` was **never modified** despite teammate claims of completion
- Phase 2 validation scripts produced 0-byte output files
- Contamination check returned all-zero correlations (bogus — ran on empty/placeholder data)
- Root cause: agent teammates hallucinated file modifications

### Deliverables on Disk
- 6 strategy documents (~2400 lines total) in `src/stage3_*.md`
- 9 helper/test scripts (unverified, may not be functional)
- Entry 034 ingested to .knowledge system

### Open for Next Session
1. Read strategy docs (real content, adversarially validated)
2. Implement delta_stability and delta_correlation in harness.py (team lead should do this directly)
3. Implement eval protocol upgrade (n_perm=8, n_trials=6)
4. Run Phase 2 validation with contamination check
5. If either signal passes Phase 2, proceed to Phase 3 search

### Constraints Added
None (no new experimental failures — research only).

---

## Session 9 — Knowledge Base & Paper System (2026-02-24)

### Context
Session 8 was a demolition session that debunked prior "breakthroughs", quantified the derivative tower collapse, and exposed the evaluation protocol as underpowered. With 19 constraints, 33 knowledge entries, and the project at Stage 2 of 8, Session 9 focused on infrastructure: building a structured knowledge base and an auto-compiled research paper.

### .knowledge System Built
- **33 structured JSON entries** across 4 types: architecture, experiment, discovery, decision
- **ingest.py / compile.py** workflow: add entries via CLI, recompile state.md automatically
- **constraints.json**: 19 machine-readable constraints with source entry links
- **index.json**: entries indexed by type, tag, and status for programmatic access
- **state.md**: single-source-of-truth project state, auto-compiled from entries
- **AGENT_GUIDE.md**: reference for entry format and usage across sessions

### Paper Compiler Built (compile_paper.py)
- Reads ALL content from `.knowledge/` — zero hardcoded values
- Outputs `paper/paper.html` — self-contained HTML, viewable in any browser
- **Dynamic values**: stage number, % complete, constraint count, session count, frozen frame status — all computed from knowledge base at compile time
- **Frozen frame tracking**: `.knowledge/frozen_frame.json` tracks all 8 elements (1 adaptive, 7 frozen)
- **Proof of dynamism**: simulated Stage 3 advancement (Eta → adaptive), recompiled — all values updated automatically (12% → 25%, 1/8 → 2/8, Stage 2 → Stage 3)
- Nobody edits paper.html. Content updates happen in `.knowledge/entries/`, then `python paper/compile_paper.py` regenerates everything.

### Session 9 Verdict
**INFRASTRUCTURE COMPLETE.** Two systems built:
1. `.knowledge/` — structured, machine-readable project memory that prevents repeated mistakes and enables automated constraint checking
2. `paper/compile_paper.py` — living research paper that reflects current knowledge base state, never goes stale, never needs manual editing

No new experiments run. No constraints added. No frozen frame changes. This session was about building the tools to make future sessions more efficient.

### Files Created/Modified
- `.knowledge/frozen_frame.json` — frozen/adaptive element tracking (8 elements)
- `paper/compile_paper.py` — HTML paper compiler from knowledge base
- `paper/paper.html` — generated output (71KB, 9 sections, 35 entry blocks)

---

## Session 8 — The Demolition Session (2026-02-24)

### Context
Session 7 reported a "breakthrough" — Candidate #2 from evolutionary search appeared to beat canonical on both training (+7.6%) and novel (+37.6%). Session 8 was planned to validate this statistically and build on it with Stage 3 delta_rz on the new baseline. Instead, Session 8 became a demolition session: both prior "breakthroughs" were debunked, a fundamental signal limitation was quantified, and the evaluation protocol was shown to be underpowered.

### Retraction: Candidate #2 "Breakthrough" (Entry 027)

**10-seed statistical validation of Session 7 Candidate #2:**
- Seeds: [42, 137, 2024, 7, 314, 1618, 2718, 3141, 9999, 31337]
- Training: p=0.295, d=0.35 — **NOT significant**
- Novel: p=0.972, d=0.02 — **NOT significant**
- 4 of 10 seeds show Candidate #2 WORSE than canonical (seeds 42, 2024, 314, 9999 have negative deltas)
- **Verdict: RETRACTED.** The +7.6% training / +37.6% novel gains were seed-specific noise amplified by insufficient sample size (Session 7 used 5 seeds)

### Stage 3 Delta_rz: FAILS 10-Seed Validation (Entry 028)

**Tested delta_rz eta adaptation (Session 6 Experiment B) on canonical baseline with 10 seeds:**
- Training: p=0.999, d=-0.001 — **statistically identical to fixed eta**
- Novel: p=0.688, d=0.19 — **no significant advantage**
- Eta distribution healthy (mean=0.00056, std=0.0003, no clipping) — mechanism works but has no effect
- Session 6's +55% generalization improvement was seed-specific noise (3 seeds insufficient)
- **Verdict: FAILED.** The system is INSENSITIVE to per-cell eta variation. Adaptive eta is solving a non-problem.

### Discovery: Derivative Tower Collapses at Order 1 (Entry 029)

**Measured lag-1 autocorrelation of the resp_z signal family across all 72 cells, 3 seeds:**

| Order | Signal | Autocorrelation | Interpretation |
|-------|--------|----------------|----------------|
| 0 | resp_z | +0.9823 | Extremely structured — drives Stage 2 |
| 1 | delta_rz (resp_z change) | +0.1090 | Too noisy — cannot drive reliable adaptation |
| 2 | delta_delta_rz (acceleration) | -0.4838 | Anti-correlated — tower has collapsed |

**Key insight:** The resp_z signal family can drive exactly ONE stage of adaptation (Stage 2, alpha). The derivative tower — the chain resp_z → delta_rz → delta_delta_rz — collapses at order 1. This is a fundamental property of the system dynamics, not a mechanism failure. It EXPLAINS why Stage 3 via delta_rz failed: the signal lacks the temporal structure to drive reliable adaptation.

**Implication:** Stages 3+ require a fundamentally different self-generated signal, not derivatives of resp_z. Candidate signal families: windowed/cumulative averages, frequency-domain analysis, inter-cell correlation patterns, competition/cooperation dynamics, information geometry of the alpha distribution.

### Discovery: Evaluation Protocol CV=29% (Entry 030)

**Canonical baseline gap measured across 10 seeds:**
- Mean: +0.1625, Std: 0.0475, CV: 29.2%
- Range: +0.0955 (seed 1618) to +0.2291 (seed 31337) — **2.4× range**
- Per-seed values: [42→+0.2216, 137→+0.1044, 2024→+0.1792, 7→+0.1327, 314→+0.2022, 1618→+0.0955, 2718→+0.1773, 3141→+0.1236, 9999→+0.1593, 31337→+0.2291]

**Power analysis:**

| Effect size to detect | Seeds needed (p<0.05, 80% power) |
|----------------------|----------------------------------|
| 7.6% (Candidate #2 claim) | ~116 seeds |
| 20% | ~17 seeds |
| 50% | ~4 seeds |

**ALL prior 3-5 seed comparisons were underpowered.** Only effects >50% could have been reliably detected. This is a methodological constraint, not a substrate limitation.

### Variance Reduction Analysis (Entry 031)

**Three approaches tested to reduce the 29% CV:**

| Approach | CV achieved | Verdict |
|----------|-----------|---------|
| Within-seed paired design | 91.9% | **DO NOT USE** — doubles variance when effect is near zero |
| 2× signal exposure (n_perm=8, n_trials=6) | 19.6% | **ADOPT** — 1.8× variance reduction |
| More K values (K=3,4,6,8,10,12) | 48.2% | **DO NOT USE** — extreme K adds noise |

**New standard protocol:** n_perm=8, n_trials=6 (4× runtime cost). Minimum detectable effect with 10 seeds: ~5% (down from ~8%).

### Local Evolutionary Search (Task #3 — Entry 032)
- Local search around Candidate #2: population 12, 5 generations, full eval, 3 seeds, sigma=0.1
- Best found: +0.2463 training, +0.0889 novel (60 total evaluations)
- **Best-of-60 with CV=29% predicts ~0.266 from pure noise. Observed +0.2463 fits perfectly.**
- Novel gap stayed at canonical level (~0.08-0.09) — training "gains" are seed-specific
- Mean fitness drifted up (0.168→0.190), diversity collapsed (0.116→0.076) — population converging to noise ceiling
- **Verdict: FLAT LANDSCAPE CONFIRMED.** Evolutionary search converges to the noise ceiling, not a genuine optimum. Strongest evidence yet that the parameter landscape is genuinely flat within 3-seed resolution.

### Meta-Cognitive Progress Review (Entry — researcher)

- ONE frozen frame reduction in 7 sessions (8→7 elements, Stage 2 alpha)
- Candidate #2 was Stage 2 optimization, NOT Stage 3 progress
- Progress estimate: 15-20% toward project goals
- Next review: Session 11

### Session 8 Verdict

**DEMOLITION COMPLETE. Three structural findings reshape the project:**

1. **The resp_z signal family is exhausted** (derivative tower collapses at order 1). Stage 3 via any resp_z derivative is a proven impossibility. A new signal family is required.

2. **The evaluation protocol was fundamentally underpowered** (CV=29%). All prior 3-5 seed "breakthroughs" were noise. Improved protocol (2× exposure, CV≈20%) adopted, but even this requires 50+ seeds for small effects.

3. **The parameter landscape around canonical is likely flat.** Neither evolutionary search (Candidate #2), adaptive eta (delta_rz), nor heterogeneous eta produced statistically significant improvements when properly validated.

**What this session achieved:** Not forward progress, but the elimination of false paths and the quantification of two fundamental limitations (signal tower depth, evaluation variance). These are hard-won negative results that prevent future wasted effort.

### Constraints Added (4 new, total 14)
- c011: Per-cell eta adaptation does not improve performance
- c012: resp_z derivative tower collapses at order 1
- c013: Stage 3+ requires fundamentally different signal family
- c014: Evaluation protocol CV=29% — need 50+ seeds or variance reduction

### Files Created
- `src/validate_candidate2.py` — 10-seed paired t-test validation
- `src/validate_candidate2_quick.py` — Quick validation variant
- `src/stage3_session8.py` — Stage 3 delta_rz 10-seed validation
- `src/stage3_session8_derivative_tower.py` — Autocorrelation analysis
- `src/variance_reduction_analysis.py` — Protocol variance reduction study
- `.knowledge/entries/027-031.json` — 5 new knowledge entries

### Open Questions for Session 9
1. **What new signal family can drive Stage 3?** The resp_z derivatives are exhausted. Candidates: windowed averages of delta_rz (recover structure via integration), frequency-domain resp_z features, inter-cell correlation dynamics, information geometry of the alpha distribution.
2. **Is the canonical parameter regime actually optimal?** The flat landscape finding assumes the gap metric is the right measure. Could a different fitness metric reveal structure?
3. **Can longer-horizon signals (cumulative delta_rz over N steps) recover the temporal structure lost in instantaneous delta_rz?** This is the most promising near-term experiment.
4. **Should the project pivot to a fundamentally different substrate?** The Living Seed has reached Stage 2. The derivative tower suggests resp_z can only drive one stage. Is a new substrate needed for Stage 3+, or can a new signal be found within the existing dynamics?

---

## Session 6 — Stage 3: The Adaptation Rate Adapts (2026-02-24)

### Context
Sessions 1-5 established: Stage 2 works for alpha (per-cell resp_z), fails for beta/gamma (3 approaches exhausted). Decision: accept beta/gamma as frozen, advance to Stage 3 for alpha. Stage 3 exit criteria: adaptive adaptation beats fixed adaptation, ground truth passes, rates converge to non-trivial values.

### Control Experiments

**Control A — Heterogeneous eta without adaptation:**
- Per-cell eta[i,k] ~ U(0.0001, 0.001), fixed (not adaptive)
- STILL: +0.121, ALIVE-s2 (global eta): +0.168, ALIVE-hetero: +0.156
- Hetero LOSES to global on training signals (-0.012)
- Hetero WINS on novel signals (+0.042 vs global)
- **Verdict**: Fixed heterogeneous eta does not beat uniform eta on training, but generalizes better

**Control B — resp_z autocorrelation:**
- Lag-1 autocorrelation: mean = +0.983, std = 0.008, min = +0.952, max = +0.995
- ALL 72 cells above 0.95 — extremely high temporal correlation
- **Verdict**: delta_rz is a clean, informative second-order signal (not noise)

### Experiment A — Self-Referential Eta (FAIL)
- eta[i,k] updated by SAME plasticity rule as alpha, meta-rate = eta itself
- Frozen frame: 7 → 6 (net -1)
- **Result: FAIL.** ALIVE-s3-A (+0.114) WORSE than STILL (+0.121)
- Eta distribution bimodal at clip bounds: 17-22 cells at lower, 5-7 at upper
- **Root cause**: Multiplicative self-reference (push_e ~ eta * f(rz)) creates exponential growth/decay → bang-bang oscillation between clip bounds
- Adversarial prediction confirmed: self-referential meta-rate is inherently unstable

### Experiment B — Delta_rz Second-Order Signal (PARTIAL)
- eta adapts via delta_rz = resp_z(t) - resp_z(t-1); push_eta = 0.1 * eta * tanh(delta_rz)
- Frozen frame: 7 → 6 (net -1)
- **Main benchmark**: ALIVE-s3-B (+0.165) vs ALIVE-s2 (+0.168) = -0.003 (within noise, effectively tied)
- **Novel signals**: ALIVE-s3-B (+0.115) vs ALIVE-s2 (+0.074) = **+0.041 (55% improvement)**
- Eta distribution HEALTHY: mean ~0.00056, no cells at clip bounds, genuine spread
- **Verdict**: PARTIAL. Does not beat s2 on training, but substantially better generalization

### Experiment C — Unified Branchless Rule (NOT RUN)
- Conditional on A or B passing. A failed, B partial. Deferred pending strategic decision.

### Topology Feasibility Study (Task 4)
- Assessed whether adaptive topology could replace beta/gamma
- **Verdict: NOT VIABLE** as Stage 2-4 solution. Frozen frame grows (2 removed, 70-140 added). Repeats local proxy failure. Topology learning is a Stage 5 problem.

### Files Created
- `src/stage3_control.py` — Control A+B: heterogeneous eta + autocorrelation
- `src/stage3_exp_a.py` — Experiment A: self-referential eta (FAIL)
- `src/stage3_exp_b.py` — Experiment B: delta_rz eta (PARTIAL)

### Session 6 Verdict
**Stage 3: INCONCLUSIVE.**

Self-referential eta (Exp A) is unstable — multiplicative meta-rates cause bang-bang oscillation. Dead end.

Delta_rz eta (Exp B) ties on training but improves generalization by 55%. The second-order signal works (high autocorrelation confirms it's informative), but the improvement doesn't meet the strict "adaptive beats fixed" criterion on training data. Whether the generalization advantage constitutes Stage 3 success depends on interpretation.

### Key Insights
1. **Autocorrelation = 0.98**: resp_z has massive temporal structure. Second-order signals are viable.
2. **Generalization vs training split**: Adaptive eta helps generalization but not training performance. This mirrors Control A (hetero eta also generalizes better).
3. **Multiplicative instability**: Self-referential meta-rates are fundamentally unstable. Any Stage 3 mechanism must use an EXTERNAL meta-rate (like the drift multiplier 0.1), not eta itself.
4. **The frozen frame floor**: Even with the best result (Exp B), the floor remains at 6 elements. Beta/gamma stay frozen. The question of whether this floor is fundamental remains open.

### Open Questions
1. Does the Exp B generalization advantage hold under more seeds / more novel worlds? (Statistical robustness)
2. Can Experiment C (unified branchless rule, -3 frozen elements) work with delta_rz despite B being only partial?
3. Is the training/generalization split an artifact of the measure_gap protocol, or a genuine architectural property?
4. Strategic: Should the project continue pushing on this architecture (Stages 3-4) or pivot to a fundamentally different substrate?

---

## Session 7 — Evolutionary Search Infrastructure & Literature Synthesis (2026-02-24)

### Context
Session 6 left Stage 3 inconclusive: adaptive eta via delta_rz improves generalization 55% but doesn't beat training performance. Strategic decision: build infrastructure for large-scale evolutionary search of plasticity rules to escape local optima and discover whether the frozen frame floor (6 elements) is fundamental or architectural. Parallelize: literature synthesis on open-ended evolution + prior self-modifying systems to identify theoretical bottlenecks before search begins.

### Infrastructure Built

**Task #1 — Parameterized Experiment Harness (harness.py)**
- Self-contained experiment runner with validated output (515 lines)
- Canonical configuration: training gap = +0.1684, novel gap = +0.0744
- Supports fast/medium/full evaluation modes with configurable K and seeds
- Enables rapid candidate assessment without re-benchmarking canonical

**Task #2 — Plasticity Rule Search Space (search_space.py)**
- 7-parameter plasticity rule space: eta, symmetry_break, amplify, drift, threshold, second_order_mix, nonlinearity_exp
- Mutation operators: Gaussian perturbation (σ = 0.1), periodic boundary wrapping
- Crossover: uniform crossover between parent rules
- Grid sampling: optional exhaustive grid search over parameter ranges

**Task #3 — Constraint Checker (constraint_checker.py)**
- Encodes 9 machine-readable constraints from Sessions 1-6:
  - "Multiplicative self-reference in meta-rate → unstable" (reject eta ~ f(eta))
  - "Global parameters have no local proxy" (don't search local signals for beta/gamma)
  - "Per-cell beta/gamma destroys MI" (don't decompose coupling constants)
  - "resp_z autocorrelation = 0.98" (second-order signals viable)
  - "Self-referential eta causes bang-bang oscillation" (reject self-referential mechanisms)
  - "External meta-rate required for stability" (require externally-specified learning rate)
  - "Training-generalization tradeoff is structural" (do not expect to improve both simultaneously)
  - "No local proxy exists for shared parameters" (focus adaptation on intrinsic signals)
  - "Eta heterogeneity improves generalization" (per-cell parameters may help novel generalization)
- Pre-filters candidates before evaluation, saving compute on known dead ends

**Task #4 — Literature Scan: Open-Ended Evolution & Self-Modifying CAs**
- Surveyed 6 major approaches: Lenia, Neural CAs, POET, Meta-learning CAs, Schmidhuber SRWM, Open-ended evolution
- Key finding: Flow-Lenia's parameter localization is closest published analogue to per-cell alpha
- No published work achieves self-referential modification without external loss functions
- [Detailed findings included in synthesis below]

**Knowledge Base System (.knowledge/)**
- 25 structured entries mapping constraint violations to architectural properties
- compile.py: generates state.md from constraint database (executable constraint codification)
- state.md: Session 7 checkpoint for Session 8 resumption
- Entry schema: constraint name, condition, implication, evidence count, related tasks

**GPU Acceleration (harness_gpu.py)**
- PyTorch-based GPU harness enabling faster candidate evaluation (745 lines)
- 50% GPU ceiling observed (bottleneck in I/O, not compute)
- Numerical validation against CPU harness still pending

### Literature Findings

**Flow-Lenia (2022-2025) — Closest Analogue**
- Embeds CA parameters INTO dynamics (parameter localization) — genuine self-modification, not external optimization
- Uses mass conservation as structural constraint — our response_z signal may be more general
- Achieves Stage 2+ for all parameters simultaneously; we achieved Stage 2 for alpha only
- **Key insight**: Parameter localization (per-cell decomposition) enables intrinsic adaptation — matches why alpha works

**Neural Cellular Automata (Mordvintsev et al.)**
- Pure parameter optimization via gradient descent — no self-referential modification
- Learns target patterns but requires external target (violates Principle I)
- 114 papers built on NCA; none solve learning update rule FORM itself
- Differentiability is powerful but solves morphogenesis, not self-improvement

**POET (Wang et al.) — Co-Evolution Framework**
- Co-evolves environments and agents; escapes local optima via stepping-stone curriculum
- **Fatal flaw**: Adaptation signal is EXTERNAL (agent success/failure) — violates Principle II
- Modifies WHAT it computes (new environments), not HOW it computes
- Relevance: curriculum building may improve search, but core problem remains

**Meta-Learning for CAs (TAPE benchmark)**
- Learning rule weights = parameter optimization (same as NCA)
- Learning adaptation RATE = meta-RL (harder); rule-shift generalization is open problem
- No one solved learning update rule form without external specification
- Insight: Topology matters for rule learning; our coupling (beta/gamma) is precisely a topology frozen frame

**Schmidhuber's Self-Referential Weight Matrices (SRWM)**
- **Only prior work on true self-referential modification** — weight matrix learns its own learning rule
- Recursive meta-learning: learn-to-learn, meta-meta-learn, etc.
- **Critical limitation**: Still assumes EXTERNAL training signal; self-reference is HOW it learns, not WHY
- No empirical validation beyond toy problems (1993) or small experiments (2022)
- Your response_z signal is closer to intrinsic (Principle II) than SRWM's external loss

**Open-Ended Evolution — Novelty vs Direction**
- System continuously generates novel forms; two competing definitions (novelty-first vs complexity-first)
- **Hidden novelty under convergence**: Species achieve identical phenotypes via different mechanisms
- **Gap in literature**: Novelty without IMPROVEMENT on original tasks — this is your core challenge
- Your 5 seeds converging to 5 different (beta,gamma) regions = novelty without improvement

### Evolutionary Search Results

**Configuration:** 15 population, 3 generations completed (8 planned before crash), medium eval (K=[4,6,8], 2 seeds)

**Best Candidate (medium eval):** +0.2032 gap
- eta=0.000518
- symmetry_break=0.531
- amplify=0.449
- drift=0.328
- threshold=0.00109
- clip_lo=0.378
- clip_hi=1.843

### Full Validation Results

**BREAKTHROUGH: Candidate #2 Breaks the Generalization-Training Tradeoff**

Full protocol validation (K=[4,6,8,10], 5 seeds) reveals Candidate #2 improves on BOTH training AND novel:
- **Training**: +0.1813 vs canonical +0.1684 = **+7.6% improvement**
- **Novel**: +0.1024 vs canonical +0.0744 = **+37.6% improvement**
- **Ground truth**: PASS
- Parameters: eta=0.000526, sym_break=0.533, amplify=0.387, drift=0.157, threshold=0.00178, clip_lo=0.358, clip_hi=1.910

**Top 3 Candidates — Full Protocol Validation:**

| Candidate | Medium Eval | Training (full) | Novel (full) | vs Canonical |
|-----------|------------|-----------------|--------------|--------------|
| Canonical | — | +0.1684 | +0.0744 | baseline |
| #1 | +0.3223 | +0.1946 (+15.6%) | +0.0667 (-10.3%) | training only |
| #2 | +0.2162 | +0.1813 (+7.6%) | +0.1024 (+37.6%) | BOTH improve |
| #3 | +0.2032 | +0.1657 (-1.6%) | +0.0854 (+14.8%) | novel only |

**Critical Finding: Medium Eval is Unreliable Proxy**
- Candidate #1 scored highest in medium eval (+0.2032) but performs worst in full validation
- Candidate #2 scored mid-range in medium eval but dominates in full validation
- Medium eval is too noisy for candidate ranking; full protocol essential for final assessment

**Qualitative Difference in Candidate #2 Strategy:**
- Amplification: 0.387 (vs #1's 0.449) = **-14% reduction** — less aggressive plasticity amplification
- Drift: 0.157 (vs #1's 0.328) = **-52% reduction** — more conservative parameter drift
- Threshold: 0.00178 (vs #1's 0.00109) = **+63% increase** — higher activation threshold, more selective plasticity

This suggests the tradeoff is NOT fundamental; a different parameter regime (lower amplify, lower drift, higher threshold) enables simultaneous improvement on both training and generalization.

### Session 7 Verdict

> **RETRACTED in Session 8:** Candidate #2 failed 10-seed statistical validation (training p=0.295, novel p=0.972). The "breakthrough" was seed-specific noise. See Session 8 for full analysis.

**~~STAGE 2 POSITIVE: Evolutionary search discovers escape from training-generalization tradeoff.~~** RETRACTED.

Candidate #2 achieves simultaneous improvement on both training (+7.6%) and novel (+37.6%), disproving the hypothesis that the tradeoff is fundamental to the plasticity architecture. This is the first adaptive rule found that beats canonical on BOTH metrics.

**Key implications:**
1. The tradeoff observed in Sessions 6 and early Session 7 (Candidates #1, #3) was NOT architectural, but a property of the specific parameter regime explored
2. Lower amplification + lower drift + higher threshold enables qualitatively different adaptation strategy
3. Medium-eval fitness is unreliable for candidate ranking; medium eval #1 was worst generalizer, while full-protocol best (#2) was mid-range in medium eval
4. The frozen frame remains at 6 elements (alpha, eta, beta, gamma + 3 hyperparameters), but within this frame, genuine Stage 2 progress has been made

**Infrastructure validation: COMPLETE.** Harness, constraint checker, and .knowledge system all functional and contributed to systematic search.

### Key Insights

1. **Parameter localization enables adaptation:** Per-cell eta works; global beta/gamma doesn't. Flow-Lenia's parameter localization principle applies. Candidate #2 success validates this.

2. **Generalization-training tradeoff was parameter-regime-dependent, NOT architectural:** Sessions 6-7 early results showed consistent tradeoff, suggesting it was fundamental. Candidate #2 disproves this: low amplify + low drift + high threshold enables both improvements simultaneously. The tradeoff reflects a local region of parameter space, not an inescapable property.

3. **Medium eval is dangerously misleading:** Best medium-eval candidate (#1: +0.2032) becomes worst full-protocol performer (-10.2% training). Medium eval ranked #2 mid-range. This invalidates medium eval as fitness proxy; future evolution MUST use full protocol despite 10× slowdown.

4. **Evolutionary search validates constraint system:** Constraint checker pre-filtered known dead ends. No crashes from constraint violations. Knowledge base successfully captured Sessions 1-6 learnings and enabled discovery of previously-inaccessible parameter regime.

5. **Breakthrough is qualitative, not quantitative:** Candidate #2's superiority comes from a different strategy (conservative plasticity, high selectivity), not just parameter tuning within the same regime. This suggests the search space has distinct basins with different adaptation philosophies.

### Open Questions for Session 8

1. **ANSWERED: Can evolutionary search find rules that beat canonical on BOTH training AND novel?** YES. Candidate #2 achieves +7.6% training, +37.6% novel. The tradeoff was parameter-regime-dependent, not architectural.

2. **Why is medium eval so unreliable as fitness proxy?** Best medium-eval candidate was worst full-protocol performer. What is the mechanism of this decoupling? Should evolutionary search switch to stochastic full-protocol eval despite 10× slowdown?

3. **Can the low-amplify/low-drift strategy be pushed further?** Candidate #2 found one point in this regime. Are there even better parameters along this low-plasticity axis? Does further reduction in amplify/drift yield further improvement?

4. **Does frozen frame floor (6 elements) prevent further progress?** Candidate #2 exhausts the 7-parameter space within current structure. To reach Stage 3+, must we make beta/gamma adaptive (which failed) or explore structural changes (topology, rule form)?

### Files Created
- `.knowledge/constraints.json` — 25 entries, 9 executable constraints
- `.knowledge/compile.py` — Knowledge base generator
- `.knowledge/state.md` — Session 7 checkpoint
- `src/harness.py` — Parameterized experiment framework (validated)
- `src/harness_gpu.py` — PyTorch GPU-accelerated harness
- `src/search_space.py` — 7-parameter plasticity rule generation
- `src/constraint_checker.py` — Constraint validator
- `src/evolve.py` — Evolutionary search loop with checkpointing

---

## Session 5 — Local Proxy Search and Finite-Diff MI (2026-02-24)

### Key Results
- **Cleanup**: Deleted 23 stale .py files and 31 bloat .md files from previous sessions
- **Local proxy test (Task #3)**: 4 basic proxies tested. Best: neighbor_correlation r=0.44. None exceed r>0.7.
- **Advanced proxy test (Task #8)**: 3 advanced proxies (mismatch max/p95/std, RDH variance). All performed WORSE than basic proxies (near-zero correlation).
- **Finite-diff MI gradient (Task #2)**: True MI gradient improves MI in 4/5 starting points, but converges to 5 DIFFERENT regions. Multiple local maxima confirmed.
- **Strong thesis verdict**: Holds for per-cell params (alpha), FAILS for global shared params (beta/gamma). Structural impossibility — 2 global scalars have no local decomposition.

### Files Created
- `src/finite_diff_mi.py` — Finite-difference MI gradient experiment
- `src/local_proxy_test.py` — Local proxy correlation test (basic + advanced)
- `src/mi_cost_benchmark.py` — MI computation cost scaling analysis (O(NC²))

### Per-Cell Beta/Gamma Experiment (Task #9)
- Made beta[i,k] and gamma[i,k] per-cell arrays (like alpha), adapted via resp_z
- **Result: NEGATIVE.** ALIVE-all (per-cell bg) MI = 0.092 vs ALIVE-alpha-only MI = 0.210 — **53% MI loss**
- STILL > ALIVE-alpha-only anomaly (0.227 vs 0.210) explained: fork tested K=6,8 only (higher MI regime). Canonical K=4,6,8,10 confirms ALIVE > STILL (+0.074 delta)
- **Conclusion**: Per-cell beta/gamma is not a valid Stage 4 move. Adding per-cell adaptation to structural constants HURTS — the parameters are structurally global and making them local destroys the coupling mechanism

### Session 5 Verdict
All three approaches to shared-parameter adaptation have failed:
1. Analytical proxy gradient — unreliable direction (Session 4)
2. Local proxy statistics — no strong correlate exists (Session 5)
3. Per-cell decomposition — destroys performance (Session 5)

The finite-diff MI gradient WORKS but requires global MI measurement (violates Principle II) and has multiple local maxima (no unique optimum).

### Open Questions
1. Is there a fundamentally different decomposition of beta/gamma that preserves coupling while enabling adaptation?
2. Can the system discover/evolve its own coupling structure (Stage 5 topology) to make beta/gamma unnecessary?
3. Should beta/gamma be accepted as irreducible frozen frame and the search move to Stage 3 (adaptation rate adapts) for alpha?

---

## Session 4 — Gradient Bottleneck Diagnosis (2026-02-23/24)

### Key Results
- **Task #19 DECISIVE**: Gradient descent for beta/gamma produces ~10^-7 updates/step. All 5 randomized starting points stayed at initialization (max shift 0.0035). Stage 2 "validation" was STASIS, not convergence.
- **Gradient bottleneck root causes**: (1) NC×D averaging divides gradient by 72; (2) response×dphi product (~0.03×0.05=0.0015) is 100× weaker than alpha's resp_z signal (~1.0)
- **Z-score cancellation**: For shared parameters, mean(z_i)=0, var(z_i)=1, sum(z_i²)=N are mathematical identities — cannot drive collective adaptation
- **Eta sweep**: eta_bg=0.1 (333× base) is optimal (4/5 success, stable). Parameters move >0.1 but direction may be WRONG (descend instead of ascend)
- **Central question identified**: LOCAL vs GLOBAL adaptation asymmetry. Alpha adapts locally (per-cell, O(1), intrinsic). Beta/gamma need global MI signal (expensive, episodic, extrinsic). Can a local proxy be found?

### Files Created
- `src/stage2_init_test.py` — Task #19 randomized initialization test
- `src/gradient_diagnostic.py` — Gradient magnitude analysis across beta/gamma space
- `src/test_eta_sweep.py` — Eta sweep for beta/gamma learning rates

### Open Questions
1. Does the proxy gradient point in the RIGHT direction? (Need starting vs ending MI comparison)
2. Can finite-difference MI gradient work? (Proposed, not implemented)
3. Do local statistics (activation fraction, response entropy, neighbor correlation) correlate with MI?
4. Is LOCAL adaptation for shared parameters fundamentally impossible (strong thesis fails)?

---

## Session 3 — Stage 2 Gradient Exploration (2026-02-22/23)

### Key Results
- Implemented analytical gradient for beta/gamma (Option D from constitution)
- Multiple gradient fixes attempted: raw gradient, z-scored, response-weighted
- Stage 2 "convergence" initially reported — later proven to be stasis (Task #19)
- Alpha adaptation confirmed working: per-cell plasticity via resp_z signal

### Files Created
- `src/living_seed_stage2_gradient.py` — First gradient implementation
- `src/living_seed_stage2_FIXED.py` — Fixed gradient version
- `src/living_seed_stage2_plasticity.py` — Plasticity-based approach

---

## Session 2 — Living Seed Analysis (2026-02-21/22)

### Key Results
- Living Seed core equation analyzed: `phi[k] = tanh(alpha*x + beta*(x[k+1]+gamma*s)*(x[k-1]+gamma*s))`
- STILL vs ALIVE comparison established
- Principle I (computation without external objectives) — partially satisfied
- Principle II (adaptation inseparable from computation) — satisfied for alpha, NOT for beta/gamma
- SeedGPU ALIVE advantage degrades with dimension: D=12 (+0.1112), D=24 (+0.0054), D=48 (+0.0235)

---

## Session 1 — Project Setup (2026-02-20/21)

### Key Results
- Constitution v2 established with 5 principles and candidate ground truth
- Tempest framework reviewed: 108 reports, all threshold-based CAs (class saturated)
- tempest.rs identified as hardcoded template (381 lines), not general physics engine
- Initial file inventory of src/ directory
