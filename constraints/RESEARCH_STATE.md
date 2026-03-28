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

**Current bottleneck (revised Step 1274):** Composition works on FT09 only (4/5 L1, PHY advantage over CTL). Three distinct failure classes across 10 games: (1) R3 failure → Physarum dynamics don't engage (tr87, tu93, borderline ls20); (2) Physarum tube reinforcement kills I3 coverage on large click spaces (cd82, cn04); (3) uniform action effects block all exploration signals (sp80). Encoding and I1 are NOT the wall — when L1 is reached (ft09), I1_enc and I1_act both pass (3/5). Next: fix the Physarum-kills-I3 failure mode on click games.

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

**Step 1273 (repr_log fix + dual I1, SP80/FT09/TR87):** COMPLETE (30 runs, 439.4s). **I1 is NOT the wall.** FT09 PHY: I1_enc=3/5 pass (within=0.855, between=1.113) AND I1_act=3/5 pass — both encoding and policy distinguish states when L1 is reached. I1 passes on the same draws, same rate: no bridge problem. SP80/TR87: I1=null (Lmax=0, only 1 level). ControlC FT09: Lmax=0, I1=null. R3: PHY passes SP80/FT09 (5/5), fails TR87 (0.045 < 0.05). I3: FT09 ρ=0.96 both conditions. TR87 CTL ρ=0.75 (pass), PHY ρ=-0.46 to -0.57 (fail). **Wall = what prevents SP80/TR87 from reaching L1.** SP80: uniform response, viability flat. TR87: R3 fails, no self-modification.

**Paradigm shift:** I1 was called "the wall" from Step 1253 to 1272 (20 steps). It was a broken measurement. The encoding DOES distinguish states (I1_enc passes 3/5 on FT09). The allosteric principle works (I1_enc and I1_act pass together). The real wall is L1 reachability on games where the composition can't build Physarum structure.

**Step 1274 (full 10-game PRISM chain):** COMPLETE (100 runs, 26.3 min). Chain ARC=0.0000010. Full failure mode map:

| Game | PHY L1 | CTL L1 | PHY R3 | PHY I3 | CTL I3 | PHY I1enc | Failure mode |
|------|--------|--------|--------|--------|--------|-----------|--------------|
| ft09 | 4/5 | 0/5 | 5/5 | 5/5 | 5/5 | 3/5 | WORKING — PHY provides real L1 advantage |
| vc33 | 5/5 | 5/5 | 5/5 | 5/5 | 5/5 | 5/5 | Easy L1 — CTL argmin also solves |
| lp85 | 5/5 | 5/5 | 5/5 | null | null | 5/5 | Easy L1 — both conditions, identical l1_step=1870 (game seed-independent) |
| ls20 | 1/5 | 0/5 | 0/5 | 0/5 | 5/5 | null | R3 borderline fail (0.044 vs 0.05 threshold), PHY I3 negative — Physarum hurts LS20 coverage |
| sb26 | 0/5 | 0/5 | 5/5 | 5/5 | 5/5 | null | R3+I3 pass but L1=0 — requires specific click sequence, neither argmin nor Physarum discovers it |
| sp80 | 0/5 | 0/5 | 5/5 | 0/5 | 0/5 | null | Uniform response — I3 fails both conditions |
| cd82 | 0/5 | 0/5 | 5/5 | 0/5 | 0/5 | null | PHY I3 fails — Physarum tube reinforcement kills coverage on large click space |
| cn04 | 0/5 | 0/5 | 5/5 | 0/5 | 0/5 | null | PHY I3 fails — same as cd82 |
| tr87 | 0/5 | 0/5 | 0/5 | 0/5 | 5/5 | null | R3 fails — no self-modification (Jacobian diff ≈0.04 < 0.05 threshold) |
| tu93 | 0/5 | 0/5 | 0/5 | 0/5 | 5/5 | null | R3 fails — same as tr87 |

**Failure mode clusters:**
1. **R3 failure** (tr87, tu93, borderline ls20): W doesn't modify meaningfully. Physarum dynamics don't engage. Jacobian diff 0.04 vs threshold 0.05.
2. **I3 failure — Physarum kills coverage** (cd82, cn04): PHY I3 NEGATIVE where CTL I3 passes. Tube dynamics create action loops that hurt exploration on large click spaces.
3. **Uniform response** (sp80): Both conditions I3 fail. Action effects indistinguishable → no coverage signal.
4. **Sequence-locked** (sb26): All stages pass (R3+I3) but game requires specific click sequence not discoverable by exploration.
5. **Easy L1** (vc33, lp85): Physarum composition provides no unique advantage — argmin alone sufficient.
6. **PHY advantage** (ft09 only): R3+I3+I1 all pass, PHY L1=4/5 vs CTL=0/5. Composition provides real benefit on exactly ONE game.

**Key finding:** Composition works on FT09 only. The wall is not I1 (encoding works when L1 is reached), but stage prerequisites (R3 failing → no Physarum dynamics; I3 failing → Physarum reinforcement hurts coverage). LP85/VC33 L1 is free — CTL argmin also solves.

**Step 1275 (novelty-gated tube flow):** KILL. Novelty gate zero effect on I3. Root cause: Physarum conflates "changes encoding" with "should explore." Responsive actions produce the most novel deltas, so novelty gating still reinforces them. Physarum dynamics component KILLED after 5 consecutive failures (1271-1275).

**Step 1276 (LPL prediction error replaces Physarum):** KILL *(retroactively PASS under I3_cv — see Step 1280)*. 100 runs, 8.1 min.

| Game | LPE L1 | CTL L1 | LPE I3 | CTL I3 | LPE R3 |
|------|---------|---------|---------|---------|---------|
| ft09 | 4/5 | 0/5 | 0.96 (5/5) | 0.96 (5/5) | 0.947 (5/5) |
| ls20 | 0/5 | 0/5 | 0.31 (2/5) | 0.64 (5/5) | 0.047 (0/5) |
| vc33 | 5/5 | 5/5 | 0.96 (5/5) | 0.96 (5/5) | 0.924 (5/5) |
| tr87 | 0/5 | 0/5 | 0.42 (3/5) | 0.75 (5/5) | 0.046 (0/5) |
| sp80 | 0/5 | 0/5 | -0.46 (0/5) | -0.46 (0/5) | 0.909 (5/5) |
| sb26 | 0/5 | 0/5 | 0.89 (5/5) | 0.89 (5/5) | 0.915 (5/5) |
| tu93 | 0/5 | 0/5 | -0.00 (0/5) | 0.86 (5/5) | 0.046 (0/5) |
| cn04 | 0/5 | 0/5 | -0.68 (0/5) | -0.68 (0/5) | 0.925 (5/5) |
| cd82 | 0/5 | 0/5 | 0.39 (0/5) | 0.39 (0/5) | 0.944 (5/5) |
| lp85 | 5/5 | 5/5 | null | null | 0.944 (5/5) |

Root cause: W_pred (256×256 linear) cannot discriminate loop vs game-advancing actions. Loop actions at different game states produce different encoding deltas → W_pred can't learn stable predictions → pe stays high for loop actions. PE diagnostic confirms no differentiation: LS20 top7 pe@5000 all similar (0.30-0.43), TR87 top7 all similar (0.085-0.103). New regression on TU93 (I3 -0.00 vs CTL 0.86). I3 regressed further on LS20/TR87 (worse than Physarum). FT09 L1 maintained (pe≈0 on click games, mechanism neutral). KILL: per-action prediction error signal is too noisy to differentiate action types in high-dimensional encoding space.

**Critical realization from 1276 PE diagnostic:** FT09 pe≈0 for all actions (mean 0.001). LPE ≈ pure argmin on FT09. Yet FT09 L1=4/5 vs CTL 0/5. The ONLY difference is LPL encoding modification (R3). The FT09 advantage was never from dynamics (Physarum or LPE). It's from R3. 12 steps investigating dynamics (1264-1276) when the encoding was doing all the work.

**Step 1277 (strip dynamics, bare LPL+argmin):** COMPLETE (100 runs, 7.9 min). Hypothesis overturned — and more precise.

FT09 L1=0/5 (same as CTL). R3 encoding alone does NOT explain FT09 advantage. Pe signal was contributing even at mean 0.001 — tiny absolute values create selective bias across 4103 actions when most have visit_count=0-2.

I3 FULLY RECOVERED: LS20=0.64(5/5), TR87=0.75(5/5), TU93=0.86(5/5) — all match CTL exactly. Pe dynamics WERE the regression source. Strip them → CTL I3 behavior restored.

**Key finding — pe scale sensitivity:** Pe signal has opposite effects by action space size:
- Large (FT09, 4103 actions): pe_ema mean=0.001, max=0.055. Tiny pe sufficient to differentiate game-advancing from loop clicks at scale. HELPS.
- Small (LS20, 7 actions): pe_ema mean=0.35. Pe values dominate argmin, kill coverage. HURTS.
- SELECTION_ALPHA=0.1 appropriate for FT09 but 100x too strong for LS20.

**Gap: pe normalization.** score = action_counts - alpha * (pe_ema / (max(pe_ema) + eps)) would bound signal to [0,1] regardless of action space. Preserves relative ordering on all games.

**Step 1278 (normalized pe):** KILL. 100 runs, 519.8s.

| Game  | NPE I3        | CTL I3        | NPE L1 | CTL L1 |
|-------|---------------|---------------|--------|--------|
| ft09  | 0.964 (5/5)   | 0.964 (5/5)   | 3/5    | 0/5    |
| ls20  | 0.307 (2/5)   | 0.643 (5/5)   | 0/5    | 0/5    |
| tr87  | 0.421 (3/5)   | 0.750 (5/5)   | 0/5    | 0/5    |
| tu93  | -0.000 (0/5)  | 0.857 (5/5)   | 0/5    | 0/5    |
| vc33  | 0.964 (5/5)   | 0.964 (5/5)   | 5/5    | 5/5    |
| lp85  | 0.000 (0/5)   | 0.000 (0/5)   | 5/5    | 5/5    |
| sb26  | 0.893 (5/5)   | 0.893 (5/5)   | 0/5    | 0/5    |
| cd82  | 0.393 (0/5)   | 0.393 (0/5)   | 0/5    | 0/5    |
| cn04  | -0.679 (0/5)  | -0.679 (0/5)  | 0/5    | 0/5    |
| sp80  | -0.464 (0/5)  | -0.464 (0/5)  | 0/5    | 0/5    |

FT09 L1 restored (3/5 vs CTL 0/5). Kill criterion triggered by I3 regressions on LS20 (2/5 vs CTL 5/5), TR87 (3/5 vs CTL 5/5), TU93 (0/5 vs CTL 5/5).

Root cause: normalization bounds pe to [0,1] but does NOT make pe invisible on KB games. EMA ramps up from first visit (~14 visits/action by step 100 on 7-action games). At step 100: LS20 pe_ema mean=0.276 (max/mean=1.06x), TR87 mean=0.042 (max/mean=1.38x), TU93 mean=0.124 (max/mean=1.36x). After normalization, top7 pe_norm values differ by ~0.15 → score difference ~0.015 against visit_count ~14. Early EMA bootstrap creates order-dependent bias before I3 measurement at step 200. FT09 pe@100 mean=0.000005 (4103 actions, only 7 visited) — normalization is essentially inactive on click games.

**Consolidated finding — pe signal on KB games:** Any pe-based action selection has early-step bootstrap problems on small action spaces (7 actions). Pe EMA ramps up faster than I3 can measure coverage (I3 measured at step 200, pe active from step 1). Scale-based fix (normalization) does not help because the signal is already structured by step 100. Temporal-based fix needed (delayed activation past step 200) OR a different signal that is zero at uniform visitation.

**Step 1279 (pe tiebreaker):** KILL. 100 runs, 519.5s.

| Game  | PTB I3        | CTL I3        | PTB L1 | CTL L1 |
|-------|---------------|---------------|--------|--------|
| ft09  | 0.964 (5/5)   | 0.964 (5/5)   | 3/5    | 0/5    |
| ls20  | 0.307 (2/5)   | 0.643 (5/5)   | 0/5    | 0/5    |
| tr87  | 0.421 (3/5)   | 0.750 (5/5)   | 0/5    | 0/5    |
| tu93  | -0.000 (0/5)  | 0.857 (5/5)   | 0/5    | 0/5    |
| vc33  | 0.964 (5/5)   | 0.964 (5/5)   | 5/5    | 5/5    |
| (others unchanged vs CTL)         |        |        |

Kill criterion: I3 regressions on LS20/TR87/TU93. FT09 L1=3/5 (maintained).

**I3 ARTIFACT CONFIRMED — LS20 I3 is broken metric (Step 1279 permutation diagnostic):**

CTL I3=0.64 on LS20 is entirely deterministic:
- CTL counts at step 200: [29,29,29,29,28,28,28] — IDENTICAL across all 5 draws/seeds
- LS20 kb_delta = [376,16,232,232,0,0,0] — actions 0-3 are responsive, 4-6 = 0
- Argmin always visits actions 0-3 first (200/7=28r4 → first 4 get extra visit)
- LS20's responsive actions happen to be indices 0-3 → Spearman=0.64 is fully determined by index ordering, not coverage quality

**Permutation test (10 runs, LS20):**
- CTL (index order): I3=0.643 (5/5 pass) — all seeds identical
- CTL (random permutation): I3=-0.279 (1/5 pass) — ranges from -0.857 to +0.64 depending on permutation

**Implication:** I3 "regressions" on LS20 in Steps 1274-1278 were measuring whether pe dynamics changed which action wins the argmin tie at step 200 — NOT whether coverage quality degraded. TR87 and TU93 are likely also artifact games (small action spaces, deterministic tie order correlating with kb_delta). All pe-based "LS20 I3 kills" may be false alarms from a broken metric.

I3 criterion on KB games needs replacement. SAL (action salience vs kb_responsiveness) is the better coverage proxy for small-action-space games. Pending Leo decision on retroactive reanalysis of 1275-1278.

**Step 1280 (fix I3 metric + re-run 1276):** COMPOSITION CONFIRMED. 115 runs (100 main + 15 I3_perm), 8.1 min.

**I3_cv (corrected metric): LPE == CTL on ALL 10 games. Zero regressions.**

| Game | LPE_cv | CTL_cv | LPE_L1 | CTL_L1 | LPE_R3 |
|------|--------|--------|--------|--------|--------|
| ft09 | 4.4176 | 4.4176 | 4/5 ✓ | 0/5 | 0.947 ✓ |
| ls20 | 0.0173 | 0.0173 | 0/5 | 0/5 | 0.047 ~ |
| vc33 | 4.4176 | 4.4176 | 5/5 | 5/5 | 0.924 ✓ |
| tr87 | 0.0173 | 0.0173 | 0/5 | 0/5 | 0.046 ~ |
| sp80 | 4.4176 | 4.4176 | 0/5 | 0/5 | 0.909 ✓ |
| sb26 | 4.4176 | 4.4176 | 0/5 | 0/5 | 0.915 ✓ |
| tu93 | 0.0173 | 0.0173 | 0/5 | 0/5 | 0.046 ~ |
| cn04 | 4.4176 | 4.4176 | 0/5 | 0/5 | 0.925 ✓ |
| cd82 | 4.4176 | 4.4176 | 0/5 | 0/5 | 0.944 ✓ |
| lp85 | 4.4176 | 4.4176 | 5/5 | 5/5 | 0.944 ✓ |

I3_cv pass criterion: LPE_cv ≤ CTL_cv × 1.1. Pass ratio = 1.000 on all games (identical distributions).

**I3_perm artifact universal across all KB games:**
- LS20: CTL=0.643 → CTLp=-0.279
- TR87: CTL=0.750 → CTLp=-0.200
- TU93: CTL=0.857 → CTLp=-0.193

**Retroactive verdict (Steps 1274-1278):** All I3 regressions on KB games were false alarms from broken I3_rho metric. Coverage quality (I3_cv) was never degraded in those steps.

**pe_snapshot insight:** pe_ema bootstraps very differently by action space:
- FT09 (4103 actions): pe_ema mean=5e-6 at step 100 — essentially zero
- LS20 (7 actions): pe_ema mean=0.277 at step 100 — already active
Small-action-space pe bootstraps fast because each action is visited ~14x by step 100. This is why additive pe can influence LS20 I3_rho (changes tie-breaking order) but not FT09 I3_rho (pe too small to override counts).

**Conclusion:** Step 1276 (LPL PE, alpha=0.1, unnormalized) IS the current composition.
- FT09 L1=4/5 vs CTL 0/5 (real advantage, unchanged across 1276/1278/1279/1280)
- No I3_cv regressions on any game
- R3 borderline on 7-action games (0.046-0.049 vs 0.05 threshold) — marginal fail on LS20/TR87/TU93
- Next: Step 1281 (pe_delta = compression progress, replaces pe_ema as signal)

**Step 1281 (pe_delta compression progress):** KILL. 100 runs, 527s.

pe_delta = pe_ema_prev - pe_ema. Actions where pe is dropping fastest get priority.

| Game | LPED_cv | CTL_cv | LPED_L1 | CTL_L1 | LPED_R3 |
|------|---------|--------|---------|--------|---------|
| ft09 | 4.4176 | 4.4176 | 3/5 | 0/5 | 0.947 |
| ls20 | 0.0173 | 0.0173 | 0/5 | 0/5 | 0.044 |
| vc33 | 4.4176 | 4.4176 | 5/5 | 5/5 | 0.924 |
| (others unchanged vs CTL)  |

Kill criterion: FT09 L1 regressed 4/5 → 3/5 vs 1276 (pe_ema). No improvement on any game.

Root cause: pe_delta bootstrap harder than pe_ema. On FT09 (4103 actions, ~0.05 visits/action at step 200), most actions have pe_ema_prev = pe_ema = 0 → pe_delta = 0. Signal only activates on actions visited ≥2x. For LS20 (7 actions), pe_delta more aggressive than pe_ema (I3_rho -0.586 vs -0.307) — deeper artifact, same CV.

**1276 (pe_ema, alpha=0.1, unnormalized) confirmed as composition. FT09 L1=4/5 is the benchmark.**

**Step 1282 (eta_h=0.05 vs 0.01):** CONDITIONAL PASS. 100 runs, 780s.

| Game | E05 R3 | E01 R3 | E05 L1 | E01 L1 |
|------|--------|--------|--------|--------|
| ft09 | 0.947 (5/5) ✓ | 0.931 (5/5) ✓ | 4/5 | 5/5 |
| ls20 | **0.086 (5/5) ✓ FIXED** | 0.046 (0/5) ✗ | 0/5 | 0/5 |
| vc33 | 0.924 (5/5) ✓ | 0.923 (5/5) ✓ | 5/5 | 5/5 |
| tr87 | 0.046 (0/5) ✗ UNCHANGED | 0.045 (0/5) ✗ | 0/5 | 0/5 |
| tu93 | **0.054 (5/5) ✓ FIXED** | 0.046 (0/5) ✗ | 0/5 | 0/5 |
| (others) | 0.91-0.94 (5/5) ✓ | same | — | — |

I3_cv: E05 == E01 on all games. No regressions. FT09 kill criterion not triggered (4/5 ≥ 3/5).

TR87 anomaly: 6 actions, R3 essentially unchanged (0.0455→0.0460). flow signal (||enc_t+1 - enc_t||) likely near-zero on TR87 — obs doesn't change visually → W_action doesn't accumulate regardless of eta_h.

LS20/TU93 L1 still 0/5 despite R3 now passing — R3 is necessary but not sufficient. Pending Leo verdict on whether eta_h=0.05 becomes new composition.
