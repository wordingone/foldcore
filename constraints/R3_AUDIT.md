# R3 Audit: Frozen Element Inventory

Per Constitution Principle IV: *"Enumerate every frozen element. For each, either (a) demonstrate that the system modifies it, or (b) demonstrate that removing it destroys all capability. Any element that is neither modified nor irreducible is unjustified complexity."*

Classification:
- **M** — Modified by the system's own dynamics
- **I** — Irreducible: removing it destroys all capability
- **U** — Unjustified: could be different; system doesn't choose it
- **F** — Forced: every alternative killed by constraint map (reclassify to I upon empirical confirmation)

**R3 requires every element to be M or I. Any U element means R3 FAILS.**

---

## Phase 1 Substrate Audits (Summary)

All Phase 1 substrates are codebook-family (BANNED post Step 416). Audits retained for constraint extraction only.

| Substrate | M | I | U | Navigates? | Key Finding |
|-----------|---|---|---|------------|-------------|
| SelfRef | 2 | 3 | **10** | No | LVQ with self-referential chain. Chain is load-bearing but adds frozen elements. |
| TapeMachine | 1 | 2 | **10** | No | Tape values are only M state. All operations frozen. |
| ExprSubstrate | 4 | 0 | **8** | No | Most M elements (tree + population) but zero I — everything replaceable. |
| TemporalPrediction (reduced) | 2 | 6 | **1** | No | Nearest to R3 PASS (1 U = numerical guard). 5 lines of code. |

**Trend:** SelfRef (10U) → Expr (8U) → Temporal (1U). Frozen frame shrinks as architectures simplify.

---

## Theoretical Reclassification (2026-03-17)

Constraint-killed alternatives for SelfRef's 10 U elements. Method: enumerate all alternatives per element, kill each with a specific constraint. If no alternative survives → element is Forced (F → I).

**Predicted:** 2M, 10-11I, 0-2U → near-PASS or PASS.

### Empirical Rounds A-B: ALL FAIL

| Run | Variant | Result | Verdict |
|-----|---------|--------|---------|
| 0 | MinimalLVQ depth-1 | 0 levels, cb=30 | Depth 1 insufficient |
| 1 | L1 norm | cb=4, killed | L2 norm confirmed forced (depth 1) |
| 2 | Fixed lr=0.5 | cb=4, killed | Adaptive lr confirmed forced |
| 3 | Mean threshold | Identical to Run 0 | Gauge symmetry confirmed (mean ≡ median) |
| 4 | Content action | dom=98%, killed | Modular arithmetic confirmed forced |
| 5 | lr=1-sim² | **BEST** (unique +25%, cb +80%) | 1-sim NOT uniquely forced — formula family is U |
| 0b-6b | SelfRef depth-2 variants | All 0 levels | SelfRef doesn't navigate regardless |

**Key findings:** (1) Adaptive lr forced. (2) 1-sim² > 1-sim. (3) L2 norm depth-dependent. (4) Neither SelfRef nor MinimalLVQ navigates — correct baseline is process_novelty().

### Round C: NEVER COMPLETED

process_novelty() constraint validation (7 variants). Script exists, results never recorded. **Decisive test for frozen frame reducibility.** Blocked by codebook ban — though intent is characterization, not extension.

---

## Encoding Compilation

The 300x speedup lives in encoding, not substrate:
- LS20 + avgpool16: L1 at ~26K steps (random walk coverage)
- FT09 + 69-class click-space: L1 at 82 steps (purposeful)
- VC33 + 3-zone encoding: L1 at 283 steps (purposeful)

| # | Element | Class | Evidence |
|---|---------|-------|----------|
| E1 | Resolution (16×16) | **M** | Step 414: discoverable via sequential dedication |
| E2 | Centering (subtract mean) | **narrow U** | Step 419: 5.5% diff at 16x16, not load-bearing |
| E3 | F.normalize | **I** | Forced by U7 + U20 |
| E4 | Flattening (2D→1D) | **I** | Forced by matmul |
| E5 | Pooling type (mean) | **I** | Step 420: mean=3386 unique, max=521. 85% diff. |
| E6 | Action representation | **M** | Step 361/375: discoverable per game |

**Score: 2M, 3I, 1 narrow U.** The meta-protocol ("try encodings, monitor health, keep what works") is the frozen frame floor.

---

## process_novelty() R3 Audit (THE navigating substrate)

*The substrate from experiments/run_step353_pure_novelty.py — LS20 L1 at 26218 steps.*

| # | Element | Class | Justification |
|---|---------|-------|---------------|
| P1 | V (codebook entries) | **M** | Modified by attract every step |
| P2 | V.shape[0] | **M** | Grows via spawn (~80% spawn rate) |
| P3 | labels | **M** | Inherit predicted class |
| P4 | F.normalize | **I** | Forced: all alternatives killed (U7+U20). Step 412: 0/3 without. |
| P5 | matmul (V @ x) | **I** | Remove → can't match → dead |
| P6 | Top-K class scoring (k=3) | **U** | k=3 is hyperparameter |
| P7 | argmin (least familiar class) | **U** | Novelty-seeking. 23 experiments show argmin ≈ random walk speed. May be non-load-bearing. |
| P8 | Class-restricted spawn | **U** | Drives 80% spawn rate. Global check (SelfRef) grows much slower. |
| P9 | Class-restricted attract | **U** | Couples attract to action label. SelfRef uses global. |
| P10 | thresh = median(max(G)) | **M** | V-derived. Formula is narrow U (gauge symmetry confirmed). |
| P11 | lr = 1 - sim | **M** | V-derived. 1-sim² strictly better — formula is narrow U. |
| P12 | F.normalize after attract | **I** | Consequence of P4 |
| P13 | torch.cat (spawn) | **I** | Forced by U17+U22+U4 |
| P14 | Attract direction | **I** | Unique error-reducing direction |
| P15 | Seeding (4 force_add) | **U** | Cold-start workaround |
| P16 | Self-labeling | **M/U** | Value M, decision frozen but natural |

**Score: 4-6M, 5I, 5-7U. R3: FAIL.**

**Key tension:** Class elements (P6-P9) enable navigation but are the largest U source. Removing them gives SelfRef (doesn't navigate). Frozen frame and navigation capability coupled through class structure (→ Proposition 11).

### I-Element Alternative Elimination

| Element | Alternatives tested | Verdict |
|---------|-------------------|---------|
| P4 (L2 norm) | None, L1, per-dim, rank, softmax, PCA | **I confirmed** (depth 1). L1 survives depth 2. |
| P5 (matmul) | L1 dist, RBF, Hamming, dot product | **I confirmed.** Cosine = unique combo of U20 + Goldilocks. |
| P12 (re-normalize) | — | **I** (consequence of P4) |
| P13 (append) | No growth, replace, split, hierarchical | **I** (U17+U22+U4) |
| P14 (attract dir) | Away, random, gradient | **I** (unique error-reducer) |

---

## TransitionTriggered674 (THE frozen bootloader)

*20/20 LS20, 20/20 FT09 with running-mean centering. LSH dual-hash + graph + edge-count argmin. Wrapped as BaseSubstrate.*

**Key insight:** 674 has TWO encoding layers. Layer 2 IS ℓ_π — refinement hyperplanes derived from frame differences. The substrate already self-modifies part of its encoding.

| # | Element | Class | Justification |
|---|---------|-------|---------------|
| T1 | avgpool16 | **U** | Could be other resolutions. Step 574: raw 64×64 works. |
| T2 | channel_0_only | **U** | Uses only first channel |
| T3 | mean_centering | **U** | Step 712: 75% of L1 gain. Load-bearing but formula frozen. |
| T4 | H_nav (k=12 random LSH) | **U** | Random hyperplanes, designer-chosen |
| T5 | H_fine (k=20 random LSH) | **U** | Random hyperplanes, designer-chosen |
| T6 | binary_hash | **I** | Remove → no cell identity → dead |
| T7 | argmin_edge_count | **I** | Steps 477-482: 6 alternatives all worse. Remove → graph harmful. |
| T8 | fine_graph_priority | **U** | System doesn't choose when to switch |
| T9 | min_visits=3 | **U** | Could be V-derived |
| T10 | h_split_threshold=0.05 | **U** | Could be V-derived |
| T11 | multi_successor_criterion | **I** | Remove → no aliasing signal → mechanism disabled |
| T12 | refine_every=5000 | **U** | Could be triggered by aliased count |
| T13 | edge_count_update | **M** | Graph IS the memory |
| T14 | aliased_set | **M** | Grows from dynamics |
| T15 | ref_hyperplanes | **M** | ℓ_π: encoding self-modification from transition statistics |

**Score: 3M, 3I, 9U. R3: FAIL.** Encoding (T1-T3) = 33% of frozen frame. Meta-parameters = 33%. Hashing = 22%.

### R3 Reduction Roadmap

| Step | Element | Change | Signal |
|------|---------|--------|--------|
| 1 | T2 (channel) | Weights from transition stats | Channel weights ≠ at t=0 vs t=N |
| 2 | T1 (avgpool) | Adaptive pooling | Pooling weights change |
| 3-5 | T9,T10,T12 | V-derived thresholds | Thresholds adapt |
| 6 | T4-T5 (hash) | Learned from transitions | Planes change direction |
| 7 | T8 (priority) | Self-determined switching | Criterion adapts |

**Potential: 9U → 0-1U.** Steps 1-5 use existing mechanisms. Step 6 needs clean Recode test.

---

## Cross-Substrate Comparison

| Substrate | M | I | U | Navigates? | ℓ level |
|-----------|---|---|---|------------|---------|
| SelfRef | 2 | 3 | 10 | No | ℓ₁ |
| process_novelty() | 4-6 | 7 | 5-7 | Yes (26K) | ℓ₁ |
| TemporalPrediction | 2 | 6 | 1 | No | ℓ₁ |
| **TransitionTriggered674** | **3** | **3** | **9** | **Yes (20/20)** | **ℓ_π** |

674 is the only substrate that (a) navigates reliably AND (b) achieves ℓ_π. The R3 path is reducing 674's 9U through self-derived alternatives.

---

## Benchmark Gate

Structural R1-R6 tests are necessary but not sufficient. Every substrate must pass:
1. **P-MNIST 1-task**: >25% accuracy in 5K steps (chance=10%)
2. **LS20**: Level 1 in 50K steps

Until a substrate passes a real benchmark, structural "passes" are claims, not capabilities.
