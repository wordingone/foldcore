# Research State  - Live Document
*Source of truth for active work. Restructured 2026-03-27.*

---

## The Search Is a Composition Loop

The search finds substrates by composing validated components, testing against developmental stages, identifying the failing stage, finding or building the component for that stage, and composing again.

**Unit of search:** a composition of components, not an individual substrate.
**Architecture:** allosteric  - encoding and action selection share parameters (W).
**Test framework:** stages I3, I1, I4, I5, R3, measured simultaneously on every run.
**Parts bin:** component catalog (C1-C33+), extended as gaps are identified.

---

## Current Composition

```
raw pixels → avgpool4 → centered encoding (C4, 256 dims)
  → W (shared: encoding projection + action salience)
  → LPL update (Hebbian + predictive, modifies W)
  → adaptive softmax (T = 1/std(salience))
  → action
```

**Stage results (Step 1253, 3 games × 5 draws):**

| Stage | Status | Evidence |
|-------|--------|----------|
| R3 | **PASS** (VC33 4/5) | Jacobian diff 0.055-0.066 vs 0.05 threshold. W modifies with experience. |
| I3 | **PASS** (VC33 5/5, LS20 1/5) | Softmax maintains coverage on click games. KB games weaker. |
| I4 | **PARTIAL** (LS20 2/5) | Temporal structure exists but measurement conflated with selection behavior. |
| I1 | **FAIL** (0/15) | Encoding doesn't distinguish analytically-distinct game states. Single-layer W lacks capacity. |
| I5 | **NULL** | No reliable L1 → cross-level transfer unmeasurable. |

**Current bottleneck:** R3 and SAL both pass simultaneously (Step 1258, first time in 1258 experiments). C25 post-recurrent on ext_enc provides R3 (Jacobian diff 0.049-0.055). Viability attention on enc-dims provides SAL (rho 0.21-0.49). LS20: both pass. VC33/SP80: SAL below threshold (sparse visits on 4103-action space make viability alpha flat). I3=0 on GPR (coverage broken). I1=0 (states not distinguished). L1=0 on all games with GPR.

**Remaining gaps:**
1. Bootstrap: viability alpha needs diverse exploration to build spread, needs spread to guide exploration. Proven: encoding CAN distinguish states (ARI=1.000 on VC33 control, Step 1259). The composition can't reach those states.
2. SAL on click games: sparse visits (2.4/action at 10K steps) prevent viability alpha from building spread
3. I3: GPR breaks systematic coverage. Argmin works but ignores encoding.
4. L1: no composition with GPR reaches L1.

**Step 1261 composition (current best):** Argmin warm-up → GPR transition when conc > 0.1. Combined attention (C25 * viability) post-recurrent on ext_enc.
- LS20: R3=0.053, SAL=0.36 (both pass), transition at step 38 (fast)
- VC33: L1 recovered (5/5 at step 2118), transition at step 6220, SAL=-0.16 (insufficient GPR phase)
- SP80: never transitions (conc=0.055 at 10K, uniform action effects)

**Game-type split:** Small action spaces (LS20, 7 actions): viability works, transition fast, R3+SAL pass. Large click spaces (VC33, 4103): L1 recovered via argmin warm-up, but GPR phase too short for SAL. Uniform-response games (SP80): viability can't differentiate actions, stays in argmin permanently.

**Step 1263 (viability-weighted argmin):** SAL=1.00 (15/15, LS20 rho=0.85, VC33 0.49, SP80 0.54). First time SAL passes on ALL games. The encoding perfectly tracks per-action state-change differentiation. L1 unchanged (VC33 5/5 same as control, LS20/SP80 0/5). Gap: encoding knows what each action DOES individually but not what SEQUENCES do collectively. Navigation requires temporal composition.

**Step 1263 FINAL (viability-weighted argmin):** SAL=1.00 all games (LS20 rho=0.85). R3=0.67. L1 unchanged. Encoding perfectly tracks action effects but can't compose sequences.

**Step 1264 (no-selector LPL dynamics):** FAILED. Runaway positive feedback — Hebbian locks onto one action. VC33 regressed 5/5→0/5. SAL=-0.35. LPL creates instability without separate selector.

**Step 1265 (Physarum no-selector):** I4=0.73 (temporal structure passes). But argmax kills coverage — VC33 regressed 5/5→0/5. Same failure as LPL no-selector.

**Step 1266 (Physarum + argmin):** LS20 L1=1/5 (FIRST L1 from encoding-informed composition, outperforms pure argmin 0/5). VC33 L1=5/5 (coverage recovered). SAL=0.829 on LS20. I4=7% on LS20 (passes). Tube-weighted argmin: Physarum dynamics for encoding, argmin for coverage, tube thickness modulates visit priority.

**Step 1267 (anomaly flow + frozen-mean control):** COMPLETE (45 runs, 3 conditions). Mean drift is negligible: live SAL=0.800 vs frozen SAL=0.764 (nearly identical). Frozen I4=9.5% vs live I4=4.8% — slight improvement. Anomaly flow signal is real, not an artifact. VC33 5/5 L1 both conditions. LS20 L1=0/5 both (regression from 1266 raw-flow 1/5, different seeds).

**DIRECTION CHANGE (2026-03-27):** Stop testing at L1. L1 selects for coverage, not understanding. Higher levels require game understanding. Test: substrate observes solver playing all levels on seed A, then acts on seed B. Did the observation transfer?

**Step 1268 (per-level L_max instrumentation, SP80/FT09/TR87):** COMPLETE (30 runs). L_max=0 on ALL 30 draws — substrate never escapes L0 within 10K steps. Confirms direction change: solver handoff mandatory. FT09 physarum reached L1 (4/5 draws, ~7K steps), never L2. R3 passes on SP80/FT09 (0.083), dead on TR87 (0.003). SAL near-zero (rho<0.1 everywhere). I4=0 everywhere. lmax_i4 metric unmeasurable without solver transport. Awaiting Leo spec-gate resolution (missing control baseline + wiring spec) before building solver handoff infrastructure.

**Step 1272 (IncSFA for I1, SP80/FT09/TR87):** COMPLETE (30 runs, 785.6s). I1=null EVERYWHERE — structural measurement issue: repr_log collects only steps 0-1000, but FT09 L1 reached at ~6000-9000 steps → all repr entries are L0, no between-level pairs for I1. IncSFA does NOT improve L1 vs nSFA: SFA FT09 3/5 vs nSFA 4/5 (nSFA slightly better, within noise). FT09 nSFA draw4 reached Lmax=2 (FIRST L2 for Physarum+argmin composition). R3 stable: SP80/FT09 ~0.083-0.085 both conditions (pass), TR87 0.0035 both (fail). I3: FT09 ρ=0.96 both (pass). SP80 ρ=-0.46 (fail). TR87 varies. Conclusion: I1 is unmeasurable with current repr_log design, not because encoding fails. Measurement fix required before I1 can gate future specs.

**Step 1269 (solver handoff, SP80):** COMPLETE (10 runs). All 10 draws Lmax=FAIL(r=0.0). Solver transports substrate to L5 but substrate can't act within L5 — game timeout at step 120. Confirmed: solver handoff insufficient, substrate needs game-understanding not just placement.

**Step 1270 (observe-then-act, SP80):** CANCELLED after 1 partial draw. Observed all 6 levels on seed A (max_level_observed=6), R3=0.083 passes, but no L1 solve on act phase (seed B). Direction cancelled by Leo mail 3508.

**Step 1271 (ARC score vs solver baseline, SP80/FT09/TR87):** COMPLETE (30 runs). ARC=0.0 everywhere — efficiency gap ~1900x on FT09 (substrate L1 at ~7500 steps, solver at 4 steps). FT09 PHY: L1=4/5 vs CtrlC 0/5 (composition provides real L1 benefit). SP80/TR87: Lmax=0 all draws. R3 passes PHY on SP80/FT09 (5/5), fails TR87 (0/5). I3: FT09 both conditions 5/5, TR87 CtrlC 5/5 / PHY 0/5, SP80 0/5 (large action space). ARC score metric correct but provides no gradient at current performance — substrate is 1000-2000x less efficient than solver.

---

## Component Catalog (validated)

| # | Component | Cross-family? | Stage it serves | Status in current composition |
|---|-----------|--------------|-----------------|-------------------------------|
| C4 | Centered encoding | 3 families | preprocessing | ACTIVE |
| C1/C16/C20 | Novelty-triggered growth | 3 families | U3/U17 | AVAILABLE (not in current allosteric) |
| C15/C20 | Transition detection | 3 families | I1 signal | AVAILABLE |
| C14 | Argmin selection | 2 families | I3 coverage | REPLACED by allosteric softmax |
| C25 | Prediction-error attention | 2 families | R3 encoding | SUBSUMED by LPL (same principle, tighter coupling) |
| C21/C26 | Recurrent state | 3 families | I4 temporal | AVAILABLE (not in current allosteric) |
| C22 | Self-observation | 2 families | R3 meta | AVAILABLE |
| NEW | LPL (Hebbian + predictive) |  - | I1, R3 | ACTIVE (single-layer, insufficient for I1) |
| NEW | Adaptive softmax (T=1/std) |  - | I3 explore/exploit | ACTIVE |
| NEEDED | IncSFA / multi-layer LPL |  - | I1 state distinction | NOT YET BUILT |

---

## Solved (locked findings)

1. **R3 solved by composition** (Step 1251). 7 cross-family components composed. Jacobian change 100/100 across 10 games, both wirings. Component-level, not wiring-dependent.

2. **Allosteric principle confirmed** (Steps 1252-1253). Shared W for encoding and action selection. When W changes, both change. No separate bridge needed. First I4 signal (1252). R3+I3 coexist (1253, VC33).

3. **Hierarchy inverted** (Step 1251). R3 passes first, not last. The assumed order I3→I1→I4→I5→R3 is wrong. R3 and I3 are already solved. I1 is the current wall. I4 partially solved. I5 blocked by I1.

4. **Argmin dead in R3 compositions** (Step 1251). I3 identical for composed substrate and argmin-alone. Argmin ignores modified representation. Replaced by allosteric softmax.

5. **Simplicity load-bearing** (PB30, n=6+). Adding complexity degrades. The allosteric substrate has 3 frozen parameters (k, eta_h, eta_p). Fewer than any previous substrate.

---

## Negative Map (what doesn't work as R3→action bridge)

From 200+ debate experiments: alpha attention, forward models, directional attention, state-conditioned rankings, MI+attention, pixel scanning, EMA recency, empowerment, anti-correlated pairs, softmax concentration. All tested, all failed. The allosteric principle (shared W) is the one architecture that isn't on this list.

---

## Phase History (fuel, not structure)

- **Phase 1 (Steps 1-416):** 416 experiments. process() = LVQ (1988). Constraint map extracted. R1-R6 formalized.
- **Phase 2 (Steps 417-1081):** 665 experiments. 16 families, 8 killed. 800b/916 as reference substrates. Graph ban, codebook ban (both lifted). Constraint map expanded. R3 remained at 0.
- **Debate v3 (Steps 1082-1250):** ~170 experiments. ℓ₁ vs ℓ_π. Both ≈ random at L1. Draw variance dominates. 200+ killed bridge mechanisms. PB26 (parity), PB30 (simplicity), PB31 (encoding resolution) confirmed.
- **Composition era (Steps 1251+):** Components composed. R3 solved. Allosteric principle. Stage instrumentation. Composition loop begins.

---

## 10 Fully-Solved Games (analytical solver baselines)

ft09 (6L, 75 clicks), ls20 (7L, 311 moves), vc33 (7L, 176 clicks), tr87 (6L, 123 actions), sp80 (6L, 107 actions), sb26 (8L, 124 actions), tu93 (9L, 185 actions), cn04 (5L), cd82 (6L), lp85 (8L).

---

## Next Step

**Next spec needed:** I1 measurement fix. Extend repr_log to full run (not just steps 0-1000) so between-level pairs exist. I1 cannot gate future decisions until this is resolved.
