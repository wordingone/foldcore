# The Search — Index

*416 experiments searching for the atomic substrate. Everything is here, organized by what you need.*

---

## North Stars

*The 8 most significant files. Each is a different angle on the same problem. The system that unifies them IS the substrate.*

| File | What it represents | Status |
|------|-------------------|--------|
| `substrates/living-seed/the_living_seed.py` | Purest Principle II (adaptation = computation) | Vacuous — zero effect |
| `substrates/foldcore/foldcore_manytofew.py` | Jun's thesis (spawn/grow/merge = birth/scale/compression) | Baseline, never best |
| `tempest/tempest.rs` | Physics-first computation | Dead by cycle 50 |
| `substrates/topk-fold/self_improving_substrate.py` | S1 unified process() | Close — feature loop still external |
| `substrates/topk-fold/substrate_v2.py` | LOO signal that works (+20pp parity) | Effective but unprincipled |
| **`experiments/run_step181_iterated_knn.py`** | **Self-feeding computation — the atomic operation** | **100% on Rule 110** |
| `experiments/run_step190_fibonacci_e2e.py` | End-to-end: discover + iterate | Strongest result (100%, no backprop) |
| `experiments/run_step266_auto_discovery.py` | Emergent decomposition | Killed at Lipschitz boundary |
| **`experiments/run_step296_dist_matching.py`** | **Per-class distribution matching — breaks Lipschitz ceiling** | **86.8% on a%b (vs 5% 1-NN)** |
| **`experiments/run_step333_cl_filter.py`** | **CL filter discovery — Stage 6: substrate beats designer** | **92.0% on a%b (vs 86.8% prescribed)** |
| `experiments/run_step300_reflection_spawn.py` | Reflection spawn — OOD via detected periodicity | 95.2% OOD on a%b |
| `experiments/run_step305_periodic_encoding.py` | Periodic physics + absorption = 100% | Substrate confirmed |
| `experiments/tempest_fold.py` | Tempest Fold — State(t+1) = f(State(t), D) | Two paths converge |

**The gap (updated):** Stages 1-7 confirmed ON THE COMPRESSED SUBSTRATE. External review (Step 338) forced compression: the 500-line decorated system failed S2. The fix: delete the `def` boundary between `step()` and `eval_batch()` — one function `process()`, ~22 lines. Step 339: 93.10% P-MNIST (beats original TopKFold). Step 341: state-derived threshold → 93.82% P-MNIST, Stage 7 confirmed. Self-referential loop: thresh reads from codebook, codebook shaped by attract, attract gated by thresh. S2 passes (attract load-bearing via feedback). The substrate IS process().

---

## Quick Start

**Want to see the strongest result?**
```bash
python experiments/run_step250_complete_substrate.py
```
Runs in ~30 seconds. Demonstrates: OOD addition (137+200=337), feature discovery, circuit synthesis, program execution — all from one 8-entry truth table.

**Want continual learning without backprop?**
```bash
python experiments/run_step99_topk_vote.py
```
95.4% on Permuted-MNIST, 0.0pp forgetting, 30 lines of core code.

---

## By Topic

### Continual Learning
| What | File | Result |
|------|------|--------|
| Top-K class vote (headline result) | `experiments/run_step99_topk_vote.py` | 95.4% P-MNIST, 0pp forgetting |
| All readout experiments (Steps 97-105) | `experiments/foldcore-steps/run_step97_*` through `run_step105_*` | Differential, coherence, gates — all tested |
| Feature discovery for classification | `substrates/topk-fold/self_improving_substrate.py` | Parity +20pp, XOR +13pp, analogy +43pp |

### Computation from Primitives
| What | File | Result |
|------|------|--------|
| OOD addition (255+255=510) | `experiments/run_step235_decomposed_addition.py` | 100% from 8-entry truth table |
| Classical algorithms (GCD, primality, Fibonacci, isqrt) | `experiments/run_step273_278_algorithms.py` | All 100% (manually decomposed) |
| CA Rule 110 simulation (Turing-complete) | `experiments/run_step182_ca_all_rules.py` | 100% at 10+ steps, 9/10 rules |
| FSM simulation | `experiments/run_step192_fsm_simulator.py` | 100% on 8/16/32/64-state machines |
| Fibonacci iteration | `experiments/run_step190_fibonacci_e2e.py` | 50 steps from raw 2D input |
| Complete demo (all capabilities) | `experiments/run_step250_complete_substrate.py` | Everything in one script |

### Non-Lipschitz Classification (Steps 291-307)
| What | File | Result |
|------|------|--------|
| Per-class distribution matching | `experiments/run_step296_dist_matching.py` | 86.8% LOO on a%b (sort-not-sum readout) |
| OOD via reflection spawn | `experiments/run_step300_reflection_spawn.py` | 95.2% OOD (detect period → extend codebook) |
| Periodic encoding + absorption | `experiments/run_step305_periodic_encoding.py` | 100% in-dist + OOD (prescribed physics) |
| Tempest Fold (f = absorb) | `experiments/tempest_fold.py` | State(t+1) = f(State(t), D) |
| Tempest + phi observer | `experiments/tempest_fold_phi.py` | 86.8% — substrate was working since Step 296 |
| Learned phi weights (308b) | `experiments/run_step308_phi_weighted.py` | 91.2% in-dist — substrate finds k=0 importance. OOD: 17% (memorization). |
| OOD test for learned w (309) | `experiments/run_step309_ood_test.py` | **KILLED** — learned w is memorization (17.3% OOD = chance) |
| Raw distances + w (310) | `experiments/run_step310_raw_dist_w.py` | **KILLED** (14.5%) — substrate finds b-grouping (R²=0.858) but not class |
| Recursive absorption (311) | `experiments/run_step311_recursive_absorb.py` | **KILLED** — residuals diverge, depth hurts |
| Self-scoped partition (312) | `experiments/run_step312_scoped_match.py` | **KILLED** — 56.5% b-match too noisy for phi |
| Loop turn 2: prescribed weights (313) | `experiments/run_step313_loop_turn2.py` | **+0.5pp** — substrate's k=0 discovery prescribed as exp(-k) → 87.2% |
| Per-b weight specialization (314) | — | Stage 4: 107/190 b-pairs diverse. k=0 weight increases with b. |
| **Grow + refine loop (315)** | — | **94.4% LOO + 48.5% OOD** — one automated turn exceeds all manual results |

### ARC-AGI Evaluation (Steps 320-335)
| What | File | Result |
|------|------|--------|
| Flat encoding baseline | `experiments/run_step320_arc_baseline.py` | 45% pixel, 4 solved. Top-K hurts (-4.2pp). |
| Taxonomy cross-reference | `experiments/run_step321_taxonomy_xref.py` | Changed-cell 24%. 32.6pp background inflation. |
| **Local patch (5x5)** | `experiments/run_step322_local_patch.py` | **39.6% changed-cell, 12 solved. 1-NN ceiling.** |
| Feature expansions (323-325) | `experiments/run_step323_*.py` through `325` | All KILLED. 5x5 patch is feature ceiling. |
| Rule extraction (326) | `experiments/run_step326_rule_extraction.py` | KILLED. Only 5/1000 tasks have extractable rules. |
| Substrate on ARC (phi+loop) | `experiments/run_step327_arc_loop.py` | **Substrate contributes ~0.** Phi hurts, loop nothing. |
| Recursive phi ARC (328) | `experiments/run_step328_recursive_phi.py` | KILLED. Identical patches = identical phi at all levels. |
| Spatial phi (329a) | `experiments/run_step329a_spatial_phi.py` | KILLED 1/5. 240-dim kills k-NN at ARC scale. |
| **ARC constraint map** | `experiments/run_step334_arc_constraint_map.py` | 418 conditional, 293 size, 123 symmetry, 99 object. |
| CL filter on ARC objects | `experiments/run_step335_cl_filter_arc.py` | KILLED. Object identity needs graphs, not clustering. |

### Constitution Stage Tests (Steps 330-333)
| What | File | Result |
|------|------|--------|
| Loop on P-MNIST (330) | `experiments/run_step330_pmnist_loop.py` | KILLED. 0pp. Loop is a%b-specific. |
| **Stage 5: clustering in phi** | `experiments/run_step331_local_metric.py` | **R²=0.997 — substrate discovers b-groups. Stage 5 confirmed.** |
| Recursive phi on a%b (332) | `experiments/run_step332_recursive_phi_ab.py` | KILLED. phi_2 amplifies b-grouping, destroys target. |
| **Stage 6: CL filter discovery** | `experiments/run_step333_cl_filter.py` | **92.0% vs 86.8% prescribed. Stage 6 PASSES.** |
| CL+phi compound (336) | `experiments/run_step336_cl_embedded_weights.py` | 96.0% — new best on a%b (CL grouping × phi readout) |
| **Stage 7: per-entry K on mixed func** | `experiments/run_step337_mixed_function.py` | **95.75% beats oracle (95.0%). Stage 7 PASSES.** |
| Spawn as data (338) | `experiments/run_step338_spawn_as_data.py` | Meta-codebook: 0%. Per-group: tie. S2 review triggers compression. |
| **Compressed substrate (339)** | `experiments/run_step339_compressed_substrate.py` | **93.10% P-MNIST. One function. S1+S2 pass.** |
| State-derived thresh+K (340) | `experiments/run_step340_full_substrate.py` | KILLED (-36pp). Per-class K breaks vote. S2: -52.88pp (feedback real). |
| **State-derived thresh only (341)** | `experiments/run_step341_thresh_only.py` | **93.82% P-MNIST. Stage 7 confirmed. Self-referential loop.** |
| **ALL 7 stages (342)** | `experiments/run_step342_stage3_stage5.py` | **91.20% P-MNIST. alpha=1-sim. target=prediction. 3 seeds.** |

### ARC-AGI-3 Interactive (Steps 343-347, Stage 8 diagnostic)
| What | File | Result |
|------|------|--------|
| First contact: avg-pool (343) | `experiments/run_step343_arc3.py` | KILLED. Cosine sim 0.999 — codebook stuck at 1. |
| Diff encoding (344) | `experiments/run_step344_arc3_diff.py` | Codebook grows (9). Action collapses. Self-reinforcing loop. |
| Pure exploration + neg signal (345) | `experiments/run_step345_explore_negative.py` | 0 levels. Neg stamps can't enter codebook (Stage 2 tension). |
| Frame visualization (346) | `experiments/run_step346_visualize.py` | Timer dominates all games. LS20 sprite in rows 5-6. FT09/VC33 opaque. |
| Centered cosine (347) | `experiments/run_step347_centered_cosine.py` | Timer removed. ACTION2/3/4 still identical in diff. |
| Centered absolute (348) | `experiments/run_step348_centered_absolute.py` | Cold-start wall. Force-seed needed. |
| Effect filter (349) | `experiments/run_step349_effect_filter.py` | ACTION2=85% effective. Action collapse persists. |
| 16x16 resolution (350) | `experiments/run_step350_resolution.py` | ALL 4 actions visible. 8x8 was hiding game. Game is 2D. |
| Novelty-seeking (351) | `experiments/run_step351_novelty.py` | 1493 unique states. Goal not reached. Timer wall. |
| Progressive deepening (352) | `experiments/run_step352_progressive.py` | Exploit degenerates (argmax = popularity). 0 levels. |
| **Pure novelty 50K (353)** | `experiments/run_step353_pure_novelty.py` | **LEVEL 1 COMPLETED at step 26218. First game level solved.** |

### Program Synthesis
| What | File | Result |
|------|------|--------|
| Circuit discovery from I/O | `experiments/run_step266_auto_discovery.py` | Discovers XOR+AND, composes into adder |
| Analogy reasoning | `experiments/run_step255_analogy.py` | A:B::C:? at +43pp improvement |

### Theoretical Framework
| What | File |
|------|------|
| Constitution (5 principles, 8 stages) | `CONSTITUTION.md` |
| Substrate architecture + honest limits | `SUBSTRATE.md` |
| Research methodology (damped oscillation) | `research/RESEARCH_DISCIPLINE.md` |
| Four separations diagnosis | `research/WHAT_THE_FAILURES_TEACH.md` |
| Equation candidates (15 tested) | `research/EQUATION_CANDIDATES.md` |
| Full experiment log (Steps 1-96) | `research/EXPERIMENT_LOG.md` |

### Constraint System
| What | File | Count |
|------|------|-------|
| All constraints (JSON) | `knowledge/constraints.json` | 66 |
| Knowledge entries | `knowledge/entries/*.json` | 78 |
| Knowledge compiler | `knowledge/compile.py` | Generates state from entries |
| Paper compiler | `paper/compile_paper.py` | Generates HTML paper |

### Substrate Implementations
| Substrate | Directory | Status |
|-----------|-----------|--------|
| Living Seed (cellular automaton) | `substrates/living-seed/` | Closed — ceiling 6/8 |
| ANIMA (W+I dynamics) | `substrates/anima/` | Closed — Stage 2 vacuous |
| FoldCore (codebook) | `substrates/foldcore/` | Frozen — baseline system |
| Eigenform (matrix dynamics) | `substrates/eigenfold/` | Closed — all applied tests failed |
| TopK Fold (self-improving) | `substrates/topk-fold/` | Phase 1 — LVQ baseline (91.20% P-MNIST) |
| WorldModel (genesis) | `substrates/worldmodel/` | Historical |
| Tempest (Rust, physics) | `tempest/` | Historical — informed the thesis |
| **SelfRef (chain)** | `substrates/selfref/` | **Phase 2 — 6/6 R1-R6. Still cosine. cb saturates.** |
| **TapeMachine (integers)** | `substrates/tape/` | **Phase 2 — 6/6 R1-R6. No vectors. Fails U20 (no local continuity).** |
| **ExprSubstrate (tree)** | `substrates/expr/` | **Phase 2 — 6/6 R1-R6. No vectors. Natural feature selection. Scoring WIP.** |

---

## What Worked (Genuine)

- **k-NN as irreducible core** for non-backprop continual learning (Steps 97-110)
- **Feature discovery via LOO** — parity +20pp, XOR +13pp, analogy +43pp, arithmetic +47pp (Steps 130-170)
- **Iterated k-NN** = computation: CA simulation, FSMs, Fibonacci from raw input (Steps 181-192)
- **Circuit synthesis from I/O** — discovers XOR, AND, carry chains (Steps 243-266)
- **S1 unified process** eliminates CIFAR-100 forgetting (11.7pp → 0.1pp) (Step 119)
- **Constitution + constraint methodology** — 70 constraints from systematic elimination
- **Stage 5: self-discovered topology** — substrate finds b-groups from phi space (R²=0.997, Step 331)
- **Stage 6: CL filter beats designer** — competitive learning discovers filter that outperforms prescribed (+5.25pp, Step 333)

## What Didn't Work (Honest)

- **Eigenform substrate** — algebraically interesting, zero applied capability (Steps 74-96)
- **Representation learning without backprop** — Hebbian PCA, sparse coding, query transform all fail (Steps 112-116)
- **Emergent decomposition for non-Lipschitz functions** — k-NN can't discover a%b or a-b steps (Steps 286-290)
- **OOD without decomposition** — k-NN interpolates, doesn't extrapolate (Steps 225-232)
- **Substrate on ARC-AGI** — phi hurts, loop adds nothing, 12/1000 tasks solved (Steps 320-335)
- **Recursive phi** — amplifies dominant signal (b-grouping), destroys target signal (Steps 328, 332)
- **Loop generality** — weight learning is a%b-specific, 0pp on P-MNIST (Step 330)

## What Was Inflated (Corrected)

- Steps 235-278: "all arithmetic from one truth table" — I wrote the algorithms, k-NN was the calculator
- "The substrate IS a universal computer" — trivially true, but the decomposition was human-designed
- The frozen frame of decomposition strategy GREW, not shrank
- Steps 320-322: "45% pixel accuracy on ARC" — 32.6pp inflated by unchanged background cells

---

## The Open Problem (Phase 2)

**Find a point inside all six walls (R1-R6) that can also DO something.**

Phase 1 proved: process() (LVQ) satisfies R1, R2 (partial), R5, R6 but fails R3 (operations hardcoded) and R4 (no self-testing). The "stage progression" was self-assessed and inflated.

**Honest status: "6/6 R1-R6" was narrative, not result.** An R3 audit (see `R3_AUDIT.md`) reveals ALL Phase 2 substrates have 5-10 unjustified frozen elements. None pass the benchmark gate (P-MNIST >25% or LS20 Level 1).

### Phase 2 Substrates

| Substrate | Data Structure | R3 Audit (U) | Disc (d=32) | LS20 unique | Benchmark |
|-----------|---------------|-------------|-------------|-------------|-----------|
| SelfRef | Codebook (vectors) | **10 U** | 94% | 1125 | 0 levels |
| TapeMachine | Integer tape | **10 U** | 35% | -- | untested |
| ExprSubstrate | Expression tree | **8 U** | degenerate | -- | a%b: 46% = prior |
| TemporalPrediction | Prediction matrix | **4 U** | 88% | **1536** | 0 levels |
| TemporalPred (reduced) | Prediction matrix | **1 U** | 88% | -- | testing |

### Key Findings

**U20 (local continuity):** The substrate must map similar inputs to similar outputs. Cosine provides this for free. Hash violates it. Random splits violate it. Matrix multiplication (temporal prediction) provides it via continuity.

**R3 x U20 tension:** R3 requires self-modifying operations (no hardcoded metric). U20 requires local continuity. Cosine gives U20, fails R3. Trees give R3, fail U20. Prediction matrices partially address both (fewest unjustified frozen elements + natural continuity).

**Benchmark gate:** Structural R1-R6 tests are necessary but not sufficient. Every substrate must pass P-MNIST >25% in 5K steps OR LS20 Level 1 in 50K steps to be taken seriously.

The constraint map (U1-U20 + I1-I9) IS the specification for the next substrate.

---

## Reproducing Results

```bash
# Requirements
pip install torch torchvision numpy

# Quick validation
python experiments/run_step250_complete_substrate.py    # Complete demo
python experiments/run_step99_topk_vote.py              # P-MNIST benchmark
python experiments/run_step266_auto_discovery.py        # Auto-discovery pipeline
python experiments/run_step273_278_algorithms.py        # Classical algorithms

# All require GPU (cuda) for speed, but run on CPU
```

## License

CC BY-NC 4.0
