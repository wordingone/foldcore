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
