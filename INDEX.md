# The Search — Index

*315 experiments searching for the atomic substrate. Everything is here, organized by what you need.*

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
| `experiments/run_step300_reflection_spawn.py` | Reflection spawn — OOD via detected periodicity | 95.2% OOD on a%b |
| `experiments/run_step305_periodic_encoding.py` | Periodic physics + absorption = 100% | Substrate confirmed |
| `experiments/tempest_fold.py` | Tempest Fold — State(t+1) = f(State(t), D) | Two paths converge |

**The gap (updated):** The Lipschitz ceiling is broken. The automated grow+refine loop achieves 94.4% LOO + 48.5% OOD in one turn — no human in the loop. The equation is found: f = absorb. Stages 1-4 demonstrated. The frontier: can the loop run more turns without over-growing?

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
| TopK Fold (self-improving) | `substrates/topk-fold/` | Current — the found substrate |
| WorldModel (genesis) | `substrates/worldmodel/` | Historical |
| Tempest (Rust, physics) | `tempest/` | Historical — informed the thesis |

---

## What Worked (Genuine)

- **k-NN as irreducible core** for non-backprop continual learning (Steps 97-110)
- **Feature discovery via LOO** — parity +20pp, XOR +13pp, analogy +43pp, arithmetic +47pp (Steps 130-170)
- **Iterated k-NN** = computation: CA simulation, FSMs, Fibonacci from raw input (Steps 181-192)
- **Circuit synthesis from I/O** — discovers XOR, AND, carry chains (Steps 243-266)
- **S1 unified process** eliminates CIFAR-100 forgetting (11.7pp → 0.1pp) (Step 119)
- **Constitution + constraint methodology** — 66 constraints from systematic elimination

## What Didn't Work (Honest)

- **Eigenform substrate** — algebraically interesting, zero applied capability (Steps 74-96)
- **Representation learning without backprop** — Hebbian PCA, sparse coding, query transform all fail (Steps 112-116)
- **Emergent decomposition for non-Lipschitz functions** — k-NN can't discover a%b or a-b steps (Steps 286-290)
- **OOD without decomposition** — k-NN interpolates, doesn't extrapolate (Steps 225-232)

## What Was Inflated (Corrected)

- Steps 235-278: "all arithmetic from one truth table" — I wrote the algorithms, k-NN was the calculator
- "The substrate IS a universal computer" — trivially true, but the decomposition was human-designed
- The frozen frame of decomposition strategy GREW, not shrank

---

## The Open Problem

**Can the substrate discover its own physics (distance function)?**

The equation is found: State(t+1) = f(State(t), D). f = absorb. With prescribed physics (periodic encoding), the substrate achieves 100%. With task-agnostic physics, phi readout achieves 86.8%. The Lipschitz ceiling is broken.

But the physics (distance function) was human-designed. The substrate discovers codebook geometry, not the metric. Making the metric adaptive is the frontier — the point where the frozen frame approaches zero.

Two paths (The Search: 307 experiments + Tempest: first principles) converged on this equation. The substrate absorbs. The observer reads geometry. The environment provides physics. The next question: can the physics come from within?

307 experiments. The search continues.

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
