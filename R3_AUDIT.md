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

*This section is THEORETICAL. Each reclassification requires empirical confirmation: substitute the alternative and confirm navigation breaks on LS20. Runs 0-5 are specified, sent to Eli for execution.*

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
