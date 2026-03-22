# Session Blindspots Briefing

**To:** Each Search session on startup
**Purpose:** Prevent re-litigating settled questions. Read before proposing experiments.

---

## 1. Beta/Gamma: Stop Re-Opening This

Beta and gamma **cannot be made adaptive** under the current architecture and Principle II. This is settled. The argument is in `src/beta_gamma_impossibility.md`. Summary:

- **c002**: Analytical gradient is 100-1000x too weak. Boosting learning rate makes parameters move but direction is unreliable (correct in 2/4 configs, wrong in 1/4). Magnitude is not the problem — directionality is.
- **c003**: Seven local statistics tested as proxies. Best: neighbor_correlation r=0.44. Threshold for usability: r>0.7. Three proxies designed specifically for coupling effects performed near zero. The search space is not exhausted, but c007 provides a theoretical reason to expect failure for any purely local measure.
- **c006**: Per-cell decomposition (the exact approach that works for alpha) causes 53% MI loss. It does not fail to help — it destroys the computation.
- **c007**: Beta scales `(x[k+1] + gamma*s) * (x[k-1] + gamma*s)` where `s` is the global mean field. Their function IS uniformity. Making them spatially heterogeneous breaks the coupling mechanism because the mechanism requires them to be global.
- **c004**: Finite-difference MI gradient works and correctly navigates the landscape — but requires external measurement. Violates Principle II by definition.
- **c005**: Even with a working signal (violating Principle II), five starting points converge to five different local maxima. The landscape has no global attractor.

**The wall is architectural, not methodological.** A new local statistic will not fix c007. A better gradient will not fix c006. The escape conditions are: (a) architecture modification introducing a lateral signaling mechanism, (b) a global-but-intrinsic signal (total system entropy, energy), (c) constitutional amendment to Principle II, or (d) Stage 5 topology adaptation making beta/gamma irrelevant. Do not propose beta/gamma experiments that do not address one of these four escapes.

The ANIMA Separation Theorem (`anima/ANIMA_SEPARATION_THEOREM_V2.md`) provides the information-theoretic framing: systems that cannot condition parameter selection on accumulated state face a lower bound on the state required for context-dependent routing. The formal reduction from beta/gamma to that task class has not been written, but the obstruction is the same.

---

## 2. Statistical Methodology: The Specific Errors That Wasted Sessions 6-11

Sessions 6-11 (7 sessions, Stage 3) were spent finding signals that turned out to be noise. The root cause was inadequate validation protocol. The errors, in order of impact:

**Error 1: 3-seed and 5-seed evaluation (c010, c018)**
The system has CV=29% (c014). At 3 seeds, you cannot detect real effects — you select for noise. Evolutionary search with 3-seed eval (Sessions 6-8) found "improvements" that were randomness. The fix: 10+ seeds minimum. Never evaluate a candidate on fewer than 10 seeds before drawing conclusions.

**Error 2: Single-K testing (c027)**
Testing at K=6 only gives CV≈14.8%. Testing across K=[4,6,8,10] and averaging gives CV≈5.1% — a 3x variance reduction. This is free statistical power. Always use multi-K protocol.

**Error 3: Within-seed paired design (c016)**
Pairing seeds within a run increases variance, not decreases it. Do not use within-seed paired designs.

**Error 4: 5-seed false positives get promoted (c032, c034)**
Entry 046 found tau=0.2 had d=+1.202 at 5 seeds (p=0.007). Looked significant. At 10 seeds: d=+0.317, p=0.317 — noise. The same protocol produced eps=0.05 d=-3.092 at 5 seeds; at 10 seeds: d=-0.461, p=0.145 — also noise. Three 5-seed false positives in one entry (tau, eps degradation). Only delta degradation effects survived (d=-2.899 at 10 seeds for delta=0.1).

**The required protocol (c015, c019, c027):** n_perm=8, n_trials=6, K=[4,6,8,10], 10+ seeds. Use this for any claim that something is binding or non-binding. Use Phase 1 theory before running Phase 2 experiments.

---

## 3. What Is Actually Settled vs. Live

**ALL Stage 4 structural parameters are now settled (Sessions 13-16). Stage 4 is CLOSED.**

**Settled (do not re-test):**

| Parameter | Finding | Constraint |
|-----------|---------|------------|
| Beta/gamma | Impossible under current arch + Principle II | c002-c007 |
| Eta (learning rate) | Non-binding. 5 approaches, 7 sessions, Stage 3 vacuously passed | c011, c012, c013 |
| Plasticity threshold | Non-binding across [0.01, 0.1] | c023 |
| Alpha clip bounds | Non-binding under canonical protocol. Entry 042 effect is protocol-specific (short protocol only) | c025, c029 |
| Tau | Non-binding in [0.2, 0.5]. K-dependent direction, no consistent adaptive signal | c031, c032, c033 |
| Eps | Non-binding around canonical [0.05, 0.5]. Entry 046 effect was 5-seed false positive | c034 |
| Delta | Binding (+6% at 1.0) but calibration-only. Adaptive delta is ANTI-SIGNAL (converges to 0.16) | c035, c036 |
| resp_z derivative tower | Collapses at order 1. Higher-order derivatives cannot drive adaptation | c012 |

**Session 16 closure (Entry 054):** Adaptive delta Phase 2 tested whether a Principle-II-compliant mechanism (state-output divergence signal) can discover delta=1.0. Result: DECISIVE FAILURE. Both adaptive conditions converge to ~0.16 regardless of initialization (started at 0.35 or 1.0). The divergence signal is an anti-signal — it drives delta toward a stable wrong attractor. Adaptive delta performs significantly worse than fixed delta=1.0 (d=-2.4, p=0.0000).

**Characterization result ([director], Session 16):** Stage 4 is NOT vacuously passed. Delta IS binding. The Living Seed is a **memoryless signal processor** — pure state replacement is architecturally optimal and is not discoverable by self-generated adaptation.

**Canonical delta updated to 1.0 in `src/the_living_seed.py` (Session 16).**

**What is live:** Stage 5 (topology) and Stage 6 (functional form). The frozen frame floor is 6. Path forward requires topology adaptation or equation-form adaptation.

---

## 4. Frozen Frame: What 6/8 Means

Current frozen frame: **6/8**. History: Session 1 -> 8, Session 4 -> 7 (alpha adaptive), Session 12 -> 6 (eta vacuous-adaptive), Session 16 -> 6 (no change — delta calibrated to 1.0, not made adaptive).

**6/8 is the architectural floor for this implementation.** Nothing in Stage 4 reduced the frozen frame further. The actual performance ceiling did not change — but the canonical implementation improved by +6% from the delta=1.0 calibration.

**Complete Stage 4 characterization:**
- Beta/gamma: impossible (c002-c007) — architectural, not methodological
- Eta, threshold, clip bounds, tau, eps: non-binding (flat performance landscape)
- Delta: binding (+6%) but calibration-only at boundary; adaptive delta is anti-signal (c035, c036)
- Plasticity rule multipliers (symmetry_break_mult, amplify_mult, drift_mult): untested, but same parameter class as eta — likely non-binding

**Performance with canonical delta=1.0:** MI gap ~+0.69 (up from ~+0.65 with old delta=0.35).

**Next stages:** Stage 5 (topology — fixed 1D ring) and Stage 6 (functional form — fixed equation). These are the only remaining paths to frozen frame reduction.

---

## 5. Key Files to Know

| File | What It Contains |
|------|-----------------|
| `src/beta_gamma_impossibility.md` | Full six-constraint evidence chain for beta/gamma impossibility |
| `src/stage4_adaptive_delta.py` | Session 16 adaptive delta experiment (anti-signal convergence to 0.16) |
| `anima/ANIMA_SEPARATION_THEOREM_V2.md` | Formal separation theorem — reactive vs state-conditioned systems |
| `.knowledge/constraints.json` | All 36 active constraints (c036 added Session 16) |
| `.knowledge/state.md` | Current session state, active hypotheses, recent experiments |
| `.knowledge/meta/progress.json` | Stage-by-stage progress, frozen frame history, meta-reviews |
| `CHANGELOG.md` | Per-session narrative — read the last entry before starting work |

---

## 6. Stage Status Summary (as of Session 16)

| Stage | Status | Notes |
|-------|--------|-------|
| 1: Autonomous Computation | PASS | Ground truth holds across all experiments |
| 2: Self-Generated Adaptation | PARTIAL | Alpha adaptive; beta/gamma frozen (impossible) |
| 3: Adaptation Rate Adapts | VACUOUS PASS | Eta non-binding, Session 12 |
| 4: Structural Constants | CHARACTERIZATION RESULT | Living Seed is memoryless signal processor. Delta=1.0 canonical. |
| 5: Topology | NOT STARTED | Fixed 1D ring — active target |
| 6: Functional Form | NOT STARTED | Fixed equation — active target |
| 7-8 | NOT STARTED | — |

**Current frozen frame: 6/8. Canonical performance: MI gap ~+0.69 (delta=1.0).**

---

## 7. Constitutional Amendments

**Amendment 1 (Session 12):** A stage may be declared **vacuously passed** if the frozen element can be made adaptive (mechanism works, non-degenerate) but produces no measurable performance difference, with theoretical explanation. Applied to Stage 3 (eta). Stage 4 was NOT vacuously passed — delta is binding (+6%), characterization result only.

**Amendment 2 (Session 15):** Before any stage is declared passed — vacuously or otherwise — a **forward viability check** must be completed: verify the current substrate can in principle satisfy Stage 7 (self-representation of update rule as modifiable data). If not, halt and declare an **architecture ceiling**, not a stage completion.

**The Living Seed fails the Amendment 2 check.** Stage 7 is categorically impossible: the substrate has no mechanism to represent its own update rule as data. No self-model exists or can be added within the current architecture. The correct result under Amendment 2 is an architecture ceiling declaration at 6/8.

**Consequence:** Stage 4 cannot be closed as complete. The project must either (a) accept the ceiling result and formalize the Living Seed's frozen frame minimum as the scientific contribution, or (b) move to a new substrate capable of passing Stage 7. See Section 8.

---

## 8. Next Substrate: ANIMA

The Living Seed's failure pattern diagnoses what the next substrate must have. Each wall hit is a design requirement:

| Living Seed failure | ANIMA requirement |
|---------------------|------------------|
| c007: beta/gamma coupling requires global signal unavailable locally | W as a real signal source — breaks the local-only wall |
| c003: no local proxy for global MI effects | W expressive enough to carry global coupling information per cell |
| Stage 7 impossible: no self-representation | W expressive enough to model I — self-model exists as a first-class object |
| c008/c009: bang-bang oscillation in self-referential rates | Damping in T adaptation dynamics — prevents instability |
| Beta/gamma impossible under Principle II | I addressable by W-conditioned updates — Principle II satisfied via W |

ANIMA is not a refinement of the Living Seed. It is a substrate designed from the failure analysis. The Living Seed's scientific contribution is the constraints — each one is a specification requirement for ANIMA.

Key files: `anima/ANIMA_SEPARATION_THEOREM_V2.md` (theoretical foundation), `src/stage_viability_audit.md` (ceiling analysis), `src/beta_gamma_impossibility.md` (constraint chain that becomes ANIMA spec).
