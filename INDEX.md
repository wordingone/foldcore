# The Search — Index

*289 experiments searching for the atomic substrate. Everything is here, organized by what you need.*

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

**Can compositional structure emerge from data without backprop?**

k-NN discovers functions where similar inputs → similar outputs (L2-Lipschitz-continuous). For everything else, a human must design the decomposition. Transformers learn decompositions from data via gradient descent through layers. This gap is precisely characterized but unsolved.

289 experiments. The search continues.

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
