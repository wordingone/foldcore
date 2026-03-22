# R3 Audit: Frozen Element Inventory

Per Constitution Principle IV: *"Enumerate every frozen element. For each, either (a) demonstrate that the system modifies it, or (b) demonstrate that removing it destroys all capability. Any element that is neither modified nor irreducible is unjustified complexity."*

Classification:
- **M** — Modified by the system's own dynamics
- **I** — Irreducible: removing it destroys all capability
- **U** — Unjustified: could be different; system doesn't choose it

**R3 requires every element to be M or I. Any U element means R3 FAILS.**

---

## SelfRef (substrates/selfref/selfref.py)

| # | Element | Class | Justification |
|---|---------|-------|---------------|
| 1 | V (codebook entries) | M | Modified by attract |
| 2 | V.shape[0] (codebook size) | M | Grows via spawn |
| 3 | F.normalize | **U** | L2 norm. Could use unnormalized. System doesn't choose. |
| 4 | matmul (V @ x) | I | Reading state. Remove -> can't match -> dead. |
| 5 | argmax (winner selection) | I | Hard selection (U8). Remove -> can't select -> dead. |
| 6 | Chain depth = 2 | **U** | Fixed at 2. Could be 1, 3, variable. System doesn't choose. |
| 7 | Self-exclusion (ref[w0] = -inf) | **U** | Design choice. System doesn't choose. |
| 8 | % n_actions | **U** | Arbitrary mapping. System doesn't choose. |
| 9 | lr = 1 - sim (learning rate) | **U** | Specific formula. System doesn't choose. |
| 10 | .clamp(0, 1) | **U** | Bounding. System doesn't choose. |
| 11 | F.normalize after attract | **U** | Re-projects to unit sphere. System doesn't choose. |
| 12 | torch.cat (spawn = append) | **U** | Growth mechanism. System doesn't choose. |
| 13 | Threshold = median(max(G)) | **U** | VALUE is data-derived, FORMULA is frozen. System doesn't choose the formula. |
| 14 | Spawn condition: sim < thresh | **U** | Direction of comparison. System doesn't choose. |
| 15 | Attract direction (x - V[w]) | I | Only direction that reduces matching error. |

**Score: 2 M, 3 I, 10 U. R3: FAIL.**

The frozen frame is 10 unjustified design choices. SelfRef is LVQ (Kohonen 1988) with a self-referential chain. The chain is novel but the frozen frame is larger than baseline LVQ.

---

## TapeMachine (substrates/tape/tape.py)

| # | Element | Class | Justification |
|---|---------|-------|---------------|
| 1 | tape (cell values) | M | Modified by writes |
| 2 | hash() for addressing | **U** | Specific hash. System doesn't choose. Violates U20. |
| 3 | & self.mask (address space) | **U** | Fixed space. System doesn't choose. |
| 4 | _read (lookup) | I | Must read state. Remove -> dead. |
| 5 | _write (store) | I | Must write state. Remove -> dead. |
| 6 | topk(3) (feature extraction) | **U** | Why 3? System doesn't choose. |
| 7 | Chain logic (key -> symbol -> next_addr) | **U** | Specific chain topology. System doesn't choose. |
| 8 | % n_actions | **U** | Arbitrary mapping. System doesn't choose. |
| 9 | Write formula (symbol + key & 0xFF + 1) | **U** | Specific formula. System doesn't choose. |
| 10 | Write formula 2 | **U** | Specific formula. System doesn't choose. |
| 11 | K=256 | **U** | Fixed vocabulary. System doesn't choose. |
| 12 | addr_bits=8 | **U** | Fixed. System doesn't choose. |
| 13 | Init values (i*7+13)%K | **U** | Specific seed. System doesn't choose. |

**Score: 1 M, 2 I, 10 U. R3: FAIL.**

Most of the system is frozen. The tape values are the only modifiable state, and they're modified by frozen formulas.

---

## ExprSubstrate (substrates/expr/expr.py)

| # | Element | Class | Justification |
|---|---------|-------|---------------|
| 1 | Tree structure | M | Modified by mutation |
| 2 | Tree values (thresholds, features) | M | Modified by mutation |
| 3 | Population | M | Modified by evolution (replace worst) |
| 4 | Scores | M | Updated by scoring function |
| 5 | evaluate() interpreter | **U** | Fixed interpretation rules. System doesn't choose how 'if' works. |
| 6 | mutate() operations | **U** | Specific mutations (swap threshold, replace subtree). System doesn't choose. |
| 7 | evolve() selection | **U** | Tournament selection. System doesn't choose. |
| 8 | Scoring formula | **U** | diversity x consistency. DEGENERATE (consistency = 1.0 always). |
| 9 | pop_size=4 | **U** | Fixed. System doesn't choose. |
| 10 | evolve_every=32 | **U** | Fixed. System doesn't choose. |
| 11 | Node vocabulary (if, >, leaf) | **U** | Fixed. System doesn't choose what operations exist. |
| 12 | random.random() in evaluate | **U** | Stochasticity source. System doesn't choose. |

**Score: 4 M, 0 I, 8 U. R3: FAIL.**

ExprSubstrate has the most modified elements (4) but zero irreducible ones — every operation could be replaced. The tree IS the program (good for R3) but the interpreter and evolution mechanism are fully frozen.

---

## TemporalPrediction (substrates/temporal/temporal.py) — NEW

| # | Element | Class | Justification |
|---|---------|-------|---------------|
| 1 | W (prediction matrix) | M | Modified by LMS update every step |
| 2 | prev (previous observation) | M | Overwritten every step |
| 3 | matmul (W @ x) | I | Linear prediction. Remove -> no prediction -> dead. |
| 4 | subtract (x - pred) | I | Error computation. Remove -> no learning signal -> dead. |
| 5 | outer_product (err x prev) | I | Unique least-squares gradient. Remove -> can't update -> dead. |
| 6 | argmax (action selection) | I | Hard selection (U8). Remove -> no action -> dead. |
| 7 | / denom (LMS normalization) | ~~U~~ **I** | **TESTED: raw Hebbian -> W=NaN -> dead.** Normalization is irreducible. |
| 8 | Chain depth = 2 | **U** | **TESTED: depth 1 works identically (87% disc).** Removable. |
| 9 | abs() before argmax | **U** | **TESTED: no-abs works identically (86% disc).** Removable. |
| 10 | % n_actions | ~~U~~ **I** | Remove -> output in {0..d-1}, need {0..n-1} -> invalid -> dead. |
| 11 | .clamp(min=1e-8) | **U** | Defensive guard. Never triggers on real input. |

**Original: 2 M, 5 I, 4 U.** With depth 1 + no abs (tested equivalent): **2 M, 6 I, 1 U.**

### Reduced variant: TemporalPrediction depth-1 (1 U)

Remove chain depth 2 and abs (tested: no degradation). Remaining U = clamp only.

```
err = x - W @ prev        # predict + error
W += outer(err, prev) / (prev.dot(prev) + 1e-8)  # LMS update
action = (W @ x).argmax() % n_actions             # depth-1 chain
```

5 lines. 2 M, 6 I, 1 U (the clamp constant 1e-8).

---

## Summary

| Substrate | Modified | Irreducible | Unjustified | R3 |
|-----------|----------|-------------|-------------|-----|
| SelfRef | 2 | 3 | **10** | FAIL |
| TapeMachine | 1 | 2 | **10** | FAIL |
| ExprSubstrate | 4 | 0 | **8** | FAIL |
| TemporalPrediction (original) | 2 | 5 | **4** | FAIL |
| **TemporalPrediction (reduced)** | **2** | **6** | **1** | **NEAREST** |

The reduced temporal variant has 1 unjustified element (numerical guard constant).
All others have 4-10 unjustified elements.

The trend: SelfRef/Tape (10 U) -> Expr (8 U) -> Temporal (5 U). The frozen frame is shrinking, but not zero.

---

## What R3 = 0U Would Require

Every design choice must be either:
- Discovered by the system (M), or
- Proved irreducible (I: removal destroys all capability)

This means: no hyperparameters, no chosen functional forms, no fixed topologies. Only mathematical necessities remain.

This IS the definition of recursive self-improvement. R3 = 0U is the destination, not an expectation for early Phase 2.

---

## Benchmark Gate

Structural R1-R6 tests are necessary but not sufficient. "6/6 R1-R6" has been narrative, not result.

**Every substrate must pass at least one of:**
1. **P-MNIST 1-task**: >25% accuracy in 5K steps (chance = 10%). Proves discrimination.
2. **LS20**: Level 1 in 50K steps. Proves exploration + navigation.

Until a substrate passes a real benchmark, structural "passes" are claims, not capabilities.

---

## Theoretical Reclassification (UNVERIFIED — pending 6-run empirical confirmation)

*2026-03-17. The constraint map (U1-U22) was run against SelfRef's 10 U elements. For each, every alternative was enumerated and tested against specific constraints. If all alternatives are killed, the element is not a choice — it's forced. Reclassify U → I.*

*This section is THEORETICAL. Each reclassification requires empirical confirmation: substitute the alternative and confirm navigation breaks on LS20. Runs 0-5 are specified, sent for execution.*

### Method

For each U element: list every alternative implementation. Kill each alternative with a specific universal constraint. If no alternative survives, the element is **forced** (the constraint map leaves no choice). If an alternative survives, the element is **genuine U** (a real design choice the system doesn't make).

A new classification is introduced:
- **F** — Forced: every alternative killed by the constraint map. Reclassify to I upon empirical confirmation.

### SelfRef Element-by-Element Analysis

**#3: F.normalize (L2 normalization)**
Current: U. Predicted: **F → I**.
Alternatives killed: no normalization (U7: dominant feature amplification + S2: tested fatal), L1 norm (U7: doesn't equalize — high-variance dims still dominate), per-dim standardization (R2: separate system + U4: adds state), rank normalization (U20: discontinuous), softmax (U7: amplifies largest dim), whitening/PCA (R2: separate evaluator + U4: massive state).
No alternative survives. L2 is the unique normalization providing equalization + continuity + no additional state.
**Tested by: Run 1** (L1 normalization). If Run 1 navigates → OVERTURNED, element is genuine U.
Status: ◻ PENDING

**#6: Chain depth = 2**
Current: U. Predicted: **U (confirmed) — depth 1 forced by U4**.
Depth 1 (standard LVQ) navigates Level 1 at 26K (Phase 1). Depth 3+ killed by U7 (iteration amplifies dominant eigenvector). Depth 2 adds 3 frozen elements (chain step, self-exclusion, second attract) for no navigation benefit. U4 (minimal description) and U13 (additions hurt) force depth 1.
The chain is an addition to LVQ, not a necessity. At depth 1, the minimal substrate IS LVQ.
**Tested by: Run 0** (minimal depth-1 substrate). If Run 0 navigates → depth 1 confirmed sufficient.
Status: ◻ PENDING

**#7: Self-exclusion (ref[w0] = -inf)**
Current: U. Predicted: **absent at depth 1**.
At depth 2: without self-exclusion, V[w0] @ V[w0] = 1.0 (unit sphere), so w1 = w0 always. Chain collapses to depth 1. Self-exclusion is forced IF depth = 2. At depth 1: element doesn't exist.
Coupled to #6. Not independently testable.
**Tested by: Run 0** (depth 1 has no self-exclusion). If Run 0 navigates → element eliminated.
Status: ◻ PENDING

**#8: % n_actions (action mapping)**
Current: U. Predicted: **F → I**.
Alternatives killed: no mapping (domain error — |V| grows past n_actions), floor division (U15: unstable under growth — same entry maps to different actions), hash (U20: destroys local coherence), learned mapping (R1: requires external reward + R2: separate system), content-based/argmax of first n dims (U4: first n dims become special, mixes roles), random (R1: action independent of state).
Modular arithmetic is the unique surjection that is parameter-free, stable under growth, and preserves index locality.
**Tested by: Run 4** (content-based action). If Run 4 navigates → OVERTURNED, element is genuine U.
Status: ◻ PENDING

**#9: lr = 1 - sim (learning rate)**
Current: U. Predicted: **F → I** (weakest argument).
Alternatives killed: fixed lr (U4: hyperparameter), lr = 0 (R2: no adaptation), lr = 1 (U3: destroys stored information), lr = 1 - sim² (U4: extra multiply beyond already-computed sim), lr = √(1-sim) (U4: extra sqrt). The sim value is already computed for matching. 1 - sim requires one subtraction — minimum extra operations. Any other parameter-free formula requires additional operations.
This is the weakest reclassification. "One extra multiply" may not be a meaningful difference under U4.
**Tested by: Run 2** (fixed lr = 0.5) **and Run 5** (lr = 1 - sim²). If either navigates → OVERTURNED.
Status: ◻ PENDING

**#10: .clamp(0, 1)**
Current: U. Predicted: **F → I**.
Without clamp: when sim < 0 (happens in bootstrap — S6 confirms antipodal vectors with small codebook), lr > 1 → attract overshoots → oscillation. U15 (robust to perturbation) requires graceful degradation. Clamp bounds {0, 1} are the natural bounds of a convex combination weight. 0 prevents repulsion. 1 prevents overshoot.
No alternative needed: clamp is a consequence of the convex combination in attract.
**Tested by: Run 0** (clamp is present in minimal substrate — its necessity is proven by S6 bootstrap scenario, not by this run).
Status: ◻ PENDING (confirmed structurally; empirical test is whether bootstrap fails without it)

**#11: F.normalize after attract**
Current: U. Predicted: **F → I** (consequence of #3).
Without re-normalization: entries drift off unit sphere. Magnitude accumulates. High-magnitude entries dominate matmul (U7). The sphere invariant established by #3 must be maintained after every modification.
Not independently testable — if #3 is forced, #11 is forced as a direct consequence.
**Tested by: Run 1** (indirectly — if L2 is forced, so is re-normalization). Structurally: removing re-normalize while keeping input normalize breaks the cosine invariant.
Status: ◻ PENDING

**#12: torch.cat (spawn = append)**
Current: U. Predicted: **F → I**.
Alternatives killed: no growth (U17: fixed capacity exhausts exploration + U22: growth prevents convergence), replace worst (U17: still fixed capacity), split winner (U4: requires split parameters + U13: added mechanism), hierarchical growth (U4 + U13: massive structure).
Append is the simplest growth operation. Growth is forced by U17 + U22. Simplest form forced by U4.
**Tested by: Run 0** (append is present in minimal substrate). Structural: U17 + U22 force growth; U4 forces append as simplest form.
Status: ◻ PENDING

**#13: Threshold = median(max(G))**
Current: U. Predicted: **narrow U** (V-derived forced; aggregation is genuine choice).
V-derived threshold forced by: fixed threshold killed (U4: hyperparameter), running statistics killed (R2: separate system + U4: adds state). The threshold MUST be a pure function of V (no additional state).
Aggregation: min killed (codebook explosion), max killed (U17: growth stops), arbitrary percentile killed (U4: parameter). Survivors: **median** (robust, U15) and **mean** (simpler). Both parameter-free. Both approximately equal for large symmetric distributions (gauge symmetry claim).
**Tested by: Run 3** (mean threshold). If mean navigates identically → gauge symmetry confirmed, element is behavioral 0U. If mean fails → median is forced, reclassify to I.
Status: ◻ PENDING

**#14: Spawn condition: sim < thresh**
Current: U. Predicted: **F → I**.
Alternatives killed: sim > thresh (U22: spawns redundant entries, growth doesn't prevent convergence in unvisited regions), random (U4: probability parameter + doesn't target novelty), periodic (U4: parameter N), never spawn (U17 + U22).
"sim < thresh" is the unique direction that makes growth target novel regions. U22 requires growth to prevent convergence, which requires entries where the codebook is sparse.
**Tested by: Run 0** (condition is present in minimal substrate). Structural: U22 forces novelty-directed growth; "sim < thresh" is the only direction that achieves this.
Status: ◻ PENDING

### Predicted Revised Score

| Element | Current | Predicted | Confidence | Tested By |
|---------|---------|-----------|------------|-----------|
| #3 F.normalize | U | I (forced) | High | Run 1 |
| #6 Chain depth | U | U → depth 1 | High | Run 0 |
| #7 Self-exclusion | U | Absent | High | Run 0 |
| #8 % n_actions | U | I (forced) | Medium | Run 4 |
| #9 lr = 1 - sim | U | I (forced) | **Low** | Run 2, 5 |
| #10 .clamp(0,1) | U | I (forced) | High | Structural |
| #11 F.normalize post | U | I (forced) | High | Consequence of #3 |
| #12 torch.cat | U | I (forced) | High | Structural |
| #13 thresh formula | U | narrow U | Medium | Run 3 |
| #14 sim < thresh | U | I (forced) | High | Structural |

**Current score: 2 M, 3 I, 10 U. R3: FAIL.**
**Predicted score (depth 1 minimal): 2 M, 10-11 I, 0-2 U. R3: NEAR-PASS or PASS.**

Worst case (Runs 2, 4, 5 all navigate → 3 elements stay U): 2 M, 8 I, 3 U.
Best case (all runs confirm kills + gauge symmetry): 2 M, 11 I, 0 U. R3: PASS.

### What each run determines

| Run | Substrate | If navigates | If fails |
|-----|-----------|-------------|----------|
| 0 | Minimal depth-1 (forced elements only) | Forced set is sufficient. Depth 1 confirmed. | A forced element is wrong — analysis has an error. |
| 1 | L1 normalization | #3 overturned → U. #11 overturned → U. | #3 confirmed → I. #11 confirmed → I. |
| 2 | Fixed lr = 0.5 | #9 overturned → U. | #9 confirmed → I. |
| 3 | Mean threshold | Gauge symmetry confirmed. #13 is behavioral 0U. | Median forced. #13 → I. |
| 4 | Content-based action | #8 overturned → U. | #8 confirmed → I. |
| 5 | lr = 1 - sim² | #9 weakened (formula family is U, not specific formula). | #9 strengthened (1-sim uniquely forced). |

---

## Empirical Results (Rounds A-B, 2026-03-17)

### CRITICAL CORRECTION: Wrong Baseline

The theoretical reclassification above was built on a false identification. The substrate that navigated LS20 Level 1 at step 26218 (Step 414) is **process_novelty()** from `experiments/run_step353_pure_novelty.py` — NOT SelfRef, NOT MinimalLVQ.

process_novelty() uses: argmin class scoring, fixed spawn_thresh, fixed lr=0.015, labels, top-K scoring, centered_enc. It is a DIFFERENT LVQ variant with MORE frozen elements than SelfRef, not fewer.

The theoretical constraint-killing logic (L1 killed by U7, growth forced by U22, etc.) still holds. But the conclusion "frozen frame near zero" was wrong — it was analyzing the wrong substrate.

### Round A: MinimalLVQ (depth 1) baseline — ALL FAIL

| Run | Variant | unique | levels | cb | dom |
|-----|---------|--------|--------|----|-----|
| 0 | MinimalLVQ baseline | 254 | 0 | 30 | 54% |
| 1 | L1 norm | 94 | 0 | 4 | 100% |
| 2 | fixed lr=0.5 | 211 | 0 | 4 | 98% |
| 3 | mean threshold | 254 | 0 | 30 | 54% |
| 4 | content action | 137 | 0 | 32 | 98% |
| 5 | lr=1-sim² | 317 | 0 | 54 | 29% |

**Run 0 fails.** MinimalLVQ at depth 1 does NOT navigate. The chain is load-bearing.
**Run 3 = Run 0 exactly.** Gauge symmetry confirmed: mean ≡ median in behavior.
**Run 5 is BEST.** lr=1-sim² gives strictly better dynamics than 1-sim (unique +25%, cb +80%, dom 29% vs 54%).
**Runs 1, 2, 4 catastrophic.** L1, fixed lr, content action confirmed kills.

### Round B: SelfRef (depth 2) baseline — ALL FAIL

| Run | Variant | unique | levels | cb | dom |
|-----|---------|--------|--------|----|-----|
| 0b | SelfRef baseline | 1125 | 0 | 164 | 40% |
| 1b | L1 norm depth 2 | 1385 | 0 | 185 | 95% |
| 2b | fixed lr=0.5 depth 2 | 209 | 0 | 5 | 67% |
| 4b | content action depth 2 | 136 | 0 | 268 | 98% |
| 5b | lr=1-sim² depth 2 | 1791 | 0 | 653 | 33% |
| 6b | depth 1 + random attract | 3115 | 0 | 3538 | 32% |

**Run 0b fails.** SelfRef does NOT navigate LS20. Neither depth 1 nor depth 2 navigates with SelfRef's dynamics.
**Run 1b surprise:** L1 at depth 2 has MORE unique states than L2 baseline (1385 vs 1125). The chain rescued L1. L2 normalization may NOT be forced at depth 2.
**Run 5b strictly best again.** lr=1-sim² dominates at depth 2 (unique 1791 vs 1125, cb 653 vs 164).
**Run 6b surprise:** Random attract gives 3x more exploration than chain-directed (3115 vs 1125) but 20x larger codebook (3538 vs 164).

### What Rounds A-B proved

1. **Adaptive lr is forced** — fixed lr=0.5 collapses to cb=4-5 at both depths. Confirmed.
2. **Content-based action mapping fails** — dom=98% at both depths. Confirmed.
3. **1-sim² > 1-sim** — strictly better at both depths. The U4 "minimal operations" argument for 1-sim is WRONG.
4. **Gauge symmetry holds** — mean = median in behavior. Confirmed.
5. **L2 normalization may NOT be forced at depth 2** — L1 at depth 2 gives more unique states. OVERTURNS theoretical prediction.
6. **Neither SelfRef nor MinimalLVQ navigates** — the constraint analysis requires process_novelty() as baseline.

### Round C: process_novelty() baseline — PENDING

6 runs replacing each frozen hyperparameter in process_novelty() with a V-derived approximation. This is the experiment that actually tests whether the feasible region is empty. Sent for execution, awaiting results.

### Updated Status of Theoretical Reclassifications

| Element | Theoretical | Round A | Round B | Status |
|---------|-------------|---------|---------|--------|
| #3 F.normalize | F → I | L1 killed (cb=4) | L1 SURVIVED (1385>1125) | **INCONCLUSIVE** — depth matters |
| #6 Chain depth | depth 1 forced | depth 1 FAILS | depth 2 also fails | **OVERTURNED** — neither navigates |
| #8 % n_actions | F → I | content kills (98% dom) | content kills (98% dom) | **CONFIRMED** (both depths) |
| #9 lr = 1-sim | F → I | 1-sim² BETTER | 1-sim² BETTER | **OVERTURNED** — 1-sim not uniquely forced |
| #13 thresh formula | narrow U | gauge confirmed | — | **CONFIRMED** (mean = median) |
| Fixed lr kill | N/A | cb=4 | cb=5 | **CONFIRMED** (both depths) |

### Round C: Step 417 Constraint Validation — NEVER COMPLETED

Script: `experiments/run_step417_constraint_validation.py` (7 variants of process_novelty() with incremental Gram optimization). Script exists but results never recorded in RESEARCH_STATE.md or returned to R3_AUDIT.md. **Status: LOST or never run.**

**This is the decisive test.** Rounds A-B proved MinimalLVQ and SelfRef don't navigate. Round C tests whether process_novelty()'s frozen hyperparameters can be replaced with V-derived approximations. If all 7 variants fail → the frozen frame is genuinely irreducible. If some navigate → specific U elements are reclassifiable to I.

**Blocker (2026-03-21):** process_novelty() is a codebook substrate (cosine similarity, attract update, LVQ dynamics). The codebook ban (2026-03-18) may prevent running this. However, the R3 audit is ABOUT the codebook — it characterizes the frozen frame of the substrate that actually navigated. Running Round C doesn't violate the ban's intent (no new codebook experiments) because it tests whether existing frozen elements are forced, not whether the codebook can be extended.

**Critical reframing:** Step 353 found Level 1 through stochastic coverage at ~26K steps. 63 experiments after it tried to make navigation purposeful. All failed or degraded. The measurement is STEP COUNT to Level 1, not binary pass/fail. Variant < 26K = added directional signal. Variant = 26K = neutral. Variant > 26K = degraded.

---

## Encoding Compilation (Session 2026-03-17c)

The R3 audit counted frozen elements in the SUBSTRATE. But the most load-bearing frozen element is the ENCODING pipeline — it determines 300x difference in step count.

Evidence:
- LS20 + 16x16 avgpool: Level 1 at ~26K steps (random walk)
- FT09 + 69-class click-space: Level 1 at step 82 (purposeful)
- VC33 + 3-zone encoding: Level 1 at step 283 (purposeful)
- Same substrate (process_novelty), same argmin, same codebook. 300x difference from encoding alone.

### Encoding Frozen Elements

| # | Element | Class | How Determined |
|---|---------|-------|----------------|
| E1 | Resolution (16×16 vs others) | **M** | Step 414: sequential dedication 64→32→16→8. Game selects 16x16 by navigation success. Discoverable through interaction. |
| E2 | Flattening (2D → 1D) | **I** | Forced by substrate's matmul (V @ x requires 1D vectors). |
| E3 | F.normalize | **I** | Same as substrate #3. Forced by U7 + U20. |
| E4 | Centering (subtract V.mean()) | **narrow U** | Step 419: NOT load-bearing at 16x16 (cb=473 without vs 499 with, 5.5% diff). Load-bearing at 64x64 (Step 385b: cb=8 without). Resolution-dependent. Raw encoding gives unique=3106 vs 3312 with centering — marginal. |
| E5 | Pooling type (mean vs max) | **I** | Step 420: mean=3386 unique, max=521. 85% diff. FORCED = mean. Max pooling preserves brightest pixel per block (timer/score region), destroying spatial structure. |
| E6 | Action representation | **M** | Step 361: click-space discovered for FT09. Step 375: zone-mapping for VC33. Game-specific but discoverable through sequential elimination of action encodings. |

### Encoding Compilation Summary

**Score: 3 I, 2 M, 1 narrow U** *(updated with Steps 419-420 empirical results)*

| Element | Pre-empirical | Post-empirical | Evidence |
|---------|--------------|----------------|----------|
| Resolution | M | **M** | Step 414: discoverable |
| Flattening | I | **I** | Forced by matmul |
| F.normalize | I | **I** | Forced by U7+U20 |
| Centering | M/U | **narrow U** | Step 419: 5.5% diff at 16x16, not load-bearing |
| Pooling type | narrow U | **I** | Step 420: 85% diff, mean forced |
| Action repr | M | **M** | Discoverable per game |

### The Meta-Protocol

After compiling both substrate and encoding, the irreducible frozen frame is:

**Substrate:** 0-3 genuine U (depending on Round C results). Most elements forced by constraint map.
**Encoding:** 1 narrow U (pooling type). Rest discoverable.
**Meta-protocol:** "Try encodings, monitor codebook health (V-derived), keep what works." This protocol is frozen — one loop, one health metric, one selection rule.

The meta-protocol IS the frozen frame floor. The substrate and encoding can both reach near-zero U through constraint forcing and self-monitoring discovery. What remains frozen is the decision procedure for encoding selection itself.

### Key Insight

The feasible region question was asking about the substrate. The actual frontier is the encoding. The substrate is nearly compiled (most elements forced). The encoding is nearly compilable (most elements discoverable). The remaining frozen element is the meta-protocol for discovery — which is small (monitor your own state, try alternatives, keep what works) and might itself be derivable from the constitution's principles.

---

## Session 2026-03-17c Corrections Log

Three critical corrections during this session, each sharpening the analysis:

1. **Wrong baseline (SelfRef → process_novelty()):** SelfRef doesn't navigate. process_novelty() does. The substrates share LVQ family but differ in argmin/labels/centering/top-K.

2. **Wrong understanding of process_novelty():** Initially thought it used fixed hyperparameters (spawn_thresh=0.95, lr=0.015). Code review revealed it uses V-derived threshold (Gram median) and 1-sim adaptive lr — same as SelfRef. The difference is argmin class scoring, labels, centering, top-K, seeding.

3. **Wrong location of frozen frame:** R3 audit counted substrate elements. The 300x speedup lives in the encoding, not the substrate. Step 414 proved encoding elements are discoverable through interaction. The frozen frame floor is the meta-protocol, not any individual element.

---

## process_novelty() R3 Audit (the substrate that navigates)

*Added 2026-03-18, autonomous loop iteration 1. This audit covers the ACTUAL navigating substrate from experiments/run_step353_pure_novelty.py — the substrate that found LS20 Level 1 at step 26218.*

### Substrate Elements

| # | Element | Class | Justification |
|---|---------|-------|---------------|
| P1 | V (codebook entries) | **M** | Modified by attract (alpha * (x - V[w])) every non-spawn step |
| P2 | V.shape[0] (codebook size) | **M** | Grows via spawn. ~80% spawn rate on LS20. |
| P3 | labels (per-entry action class) | **M** | New entries inherit the predicted class label. Labels are part of state. |
| P4 | F.normalize (input) | **I** | Forced by U7 (dominant feature amplification) + U20 (local continuity). All alternatives killed. See SelfRef #3 analysis. |
| P5 | matmul (V @ x) | **I** | Reading state. Remove → can't match → dead. |
| P6 | Top-K class scoring (k=3) | **U** | k=3 is a hyperparameter. k=1 untested on process_novelty(). k=all is soft attention (tested in Step 418, different dynamics). The VALUE of k is U; the MECHANISM (per-class similarity aggregation) may be forced. |
| P7 | argmin (least familiar class) | **U** | Novelty-seeking action selection. Alternative: argmax (exploit, not explore — kills navigation by repeating familiar actions). Alternative: random (no signal). argmin is the only selection rule that systematically explores. But: is SYSTEMATIC exploration necessary, or does stochastic coverage suffice? If stochastic coverage suffices, argmin is replaceable by any balanced random selection. 23 experiments (354-376) show argmin ≈ random walk at same speed. Element is U but may be non-load-bearing. |
| P8 | Class-restricted spawn check | **U** | `sims[target_mask].max() < self.thresh` — spawn only checked within PREDICTED class, not globally. This means inputs that are globally similar but class-novel still spawn. Drives the 80% spawn rate. Alternative: global spawn check (sims.max() < thresh) — SelfRef uses this, grows much slower (cb=164 at 50K vs process_novelty ~20K). The class restriction is what makes process_novelty() different from SelfRef. U — system doesn't choose. |
| P9 | Class-restricted attract | **U** | Winner is argmax within target class, not global argmax. This means a closer entry in another class is ignored. Alternative: global attract (SelfRef). The class restriction couples attract to the action label. U — system doesn't choose. |
| P10 | thresh = median(max(G)) | **M** | V-derived. Gram median of max inter-entry similarities. Computed from V every step. VALUE is M. FORMULA is narrow U (median vs mean — gauge symmetry confirmed in Round A: identical behavior). |
| P11 | lr = 1 - sim | **M** | V-derived. Learning rate computed from the similarity of the winner to the input. Adaptive. VALUE is M. FORMULA: 1-sim is the simplest parameter-free formula using already-computed sim (U4). Round A showed 1-sim² is strictly better — formula is narrow U within the power family. |
| P12 | F.normalize after attract | **I** | Consequence of P4. Maintains sphere invariant. See SelfRef #11 analysis. |
| P13 | torch.cat (spawn = append) | **I** | Forced by U17 (fixed capacity exhausts) + U22 (growth prevents convergence) + U4 (simplest growth). See SelfRef #12 analysis. |
| P14 | Attract direction (x - V[w]) | **I** | Only direction that reduces matching error. See SelfRef #15. |
| P15 | Seeding protocol (4 force_add) | **U** | Forces one entry per action class before novelty mode. Cold-start workaround. Without it: first few steps have no class diversity, argmin is degenerate. Alternative: random initial actions. The VALUE (n_actions entries) is game-derived. The DECISION to seed is frozen. |
| P16 | label = prediction (self-labeling) | **M/U** | When no external label, the substrate labels new entries with its own prediction. The VALUE is M (derived from argmin scoring). The DECISION to self-label is frozen but natural — only alternative is random labeling (adds noise) or no labeling (breaks class structure). |

### Encoding Elements (included per anti-inflation rule 7)

| # | Element | Class | Justification |
|---|---------|-------|---------------|
| E1 | avgpool16 (16×16) | **M** | Resolution discoverable via sequential dedication (Step 414). |
| E2 | centered_enc (subtract V.mean()) | **M/U** | Value V-derived. Decision discoverable via codebook health monitoring. |
| E3 | F.normalize (encoding) | **I** | Same as P4. |
| E4 | Flattening (2D → 1D) | **I** | Forced by matmul. |

### I-Element Alternative Elimination (post-audit Finding 4)

*External audit (2026-03-18) critiqued I-classifications as "too generous" — requiring demonstration that ALL alternatives fail, not just that removing the current implementation kills the system. This section enumerates alternatives for each I element and cites the constraints or experiments that rule them out.*

**P4: F.normalize (L2 normalization) — I (forced)**
- No normalization: U7 (dominant feature amplification destroys minority features). Step 412: 0/3 without it.
- L1 normalization: Round A Run 1 (cb=4, killed at depth 1). Round B Run 1b: survived at depth 2 (1385>1125 unique). **Depth-dependent — I at depth 1, inconclusive at depth 2.**
- Per-dim standardization: R2 (requires separate statistics system) + U4 (adds running mean/var state).
- Rank normalization: U20 (discontinuous — nearby inputs can rank differently).
- Softmax: U7 (amplifies largest dimension, same failure mode as no normalization).
- Whitening/PCA: R2 (separate evaluator) + U4 (massive covariance state).
- **Conclusion: L2 is forced at depth 1 (all alternatives killed). At depth 2, L1 survives — reclassify to I-provisional.**

**P5: matmul (V @ x) — I (forced)**
- L1 distance: U20 holds (L1 preserves local continuity). But: cosine saturation IS the Goldilocks zone (Steps 377-412). L1 doesn't saturate the same way → dynamics differ fundamentally. Round B Run 1b tested L1 at depth 2: different dynamics, didn't navigate.
- RBF kernel: Step 386 (sigma collapses to step function). Killed.
- Hamming distance: U20 violation (binary thresholding is discontinuous).
- Dot product (unnormalized): Step 388 (discrimination works, 0 levels at 200K). Killed for navigation.
- **Conclusion: matmul with L2-normalized vectors (cosine similarity) is the unique combination providing local continuity (U20) + Goldilocks dynamics. Alternatives tested and killed. I confirmed.**

**P12: F.normalize after attract — I (consequence of P4)**
- If P4 is forced (L2 normalization on input), then entries must stay on the unit sphere after attract modifies them. Without re-normalization, entries drift off-sphere → magnitude accumulates → high-magnitude entries dominate matmul (U7).
- **Conclusion: logically forced by P4. Not independently testable.**

**P13: torch.cat (spawn = append) — I (forced by U17 + U22 + U4)**
- No growth: U17 (fixed capacity exhausts exploration) + U22 (growth prevents convergence — TemporalPrediction family killed by convergence).
- Replace worst: Still fixed capacity → U17.
- Split winner: U4 (requires split parameters) + U13 (added mechanism).
- Hierarchical growth: U4 + U13 (massive added structure).
- **Conclusion: growth is forced (U17+U22). Append is the simplest growth operation (U4). I confirmed.**

**P14: Attract direction (x - V[w]) — I (unique)**
- (V[w] - x): Moves entry AWAY from input → increases matching error. Anti-learning.
- Random direction: No systematic error reduction.
- Gradient of other loss: R1 (requires external objective).
- **Conclusion: (x - V[w]) is the unique direction reducing ||x - V[w]||². I confirmed.**

### Score

**Substrate: 3-4 M, 5 I (4 confirmed, 1 provisional), 5-6 U.**
**Encoding: 1-2 M, 2 I, 0-1 U.**
**Total system: 4-6 M, 7 I, 5-7 U.**

### Comparison to SelfRef

| | SelfRef | process_novelty() |
|---|---|---|
| Original U count | 10 | 5-7 |
| Navigates LS20? | No | Yes (26K steps) |
| Class structure | No (index-based action) | Yes (argmin class scoring) |
| Thresh | V-derived (same) | V-derived (same) |
| lr | 1-sim (same) | 1-sim (same) |
| Growth rate | cb=164 at 50K | cb=~20K at 26K |

**Key difference:** process_novelty() has class structure (P6-P9, P15-P16) which SelfRef doesn't. The class structure drives higher growth (class-restricted spawn check ≈ 80% spawn) and novelty-seeking action selection (argmin). These are the 4-5 U elements that SelfRef lacks — and they're what makes navigation work.

**The tension:** The class elements (P6-P9) are what enable navigation but are also the largest source of U. Removing them gives SelfRef, which doesn't navigate. The frozen frame and navigation capability are coupled through the class structure.
