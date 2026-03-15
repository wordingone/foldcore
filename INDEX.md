# The Search — Asset Index

*A catalog of everything produced across 289 experiments, 4 substrates, 26+ sessions. Each entry: what it is, what it proved, where the code lives, how to reproduce.*

---

## Substrates (Chronological)

### 1. Living Seed (Sessions 1-17)
**What**: Cellular automaton with per-cell alpha adaptation. `phi[k] = tanh(alpha*x + beta*(x[k+1]+gamma*s)*(x[k-1]+gamma*s))`
**Result**: Stage 1 PASS (autonomous computation). Ceiling at 6/8 frozen frame — code ≠ data, Stage 7 impossible.
**Key finding**: 51 constraints extracted. Beta/gamma are globally coupled (not decomposable). Eta adaptation is vacuous.
**Code**: `substrates/living-seed/the_living_seed.py`, `substrates/living-seed/SeedGPU.py`
**Reproduce**: `python substrates/living-seed/the_living_seed.py`

### 2. ANIMA (Sessions 18-23)
**What**: W+I dynamics — W predicts neighbor interaction, I accumulates prediction error.
**Result**: Stage 2 vacuous (Amendment 1). No Principle-II-compliant adaptation signal found.
**Key finding**: MI-error structural decoupling — per-step signals can't detect sequence-level optimality.
**Code**: `substrates/anima/`
**Reproduce**: `python substrates/anima/anima_organism.py`

### 3. FoldCore / Codebook (Steps 1-96, pre-search)
**What**: Codebook vectors on unit hypersphere. Competitive learning + spawning + many-to-few routing.
**Result**: 33/33 CSI coverage, 0.0pp forgetting P-MNIST (56.7% AA), 33.5% CIFAR-100.
**Key finding**: Structural zero forgetting by construction. Readout (1-NN) is the bottleneck, not storage.
**Code**: `substrates/foldcore/foldcore_manytofew.py`, `substrates/foldcore/foldcore_torch.py`

### 4. Eigenform / Spectral (Steps 74-96) — CLOSED
**What**: Matrix codebook with eigenform dynamics. Spectral Phi, Formula C composition.
**Result**: ALL applied tests FAILED. 22.2% vs 46.2% baseline. Algebraically interesting, computationally trivial.
**Key finding**: Non-commutative pairwise composition exists but doesn't accumulate through chains. DeepSeek confirmed: expensive distance function.
**Code**: `substrates/eigenfold/eigenfold.py`

### 5. Self-Improving Substrate (Steps 97-289) — CURRENT
**What**: k-NN + LOO-scored feature discovery + iterated computation + program synthesis.
**Result**: P-MNIST 95.4%/0pp, parity +20pp, analogy +43pp, arithmetic +47pp. Executes programs. OOD via decomposition (255+255=510).
**Honest limit**: Steps 235-278 were manual compilation. Emergent decomposition KILLED for non-Lipschitz functions (Steps 286-290).
**Key finding**: k-NN discovers L2-locally-consistent functions. Nothing else. The substrate IS a universal computer when given decomposition, but can't discover decomposition for discontinuous/oblique functions.
**Code**: `substrates/topk-fold/self_improving_substrate.py`, `substrates/topk-fold/substrate_v2.py`

---

## Research Frameworks

### Constitution
**What**: 5 principles + 8 stages for recursive self-improvement. Architecture-independent.
**File**: `CONSTITUTION.md`
**Key concepts**: frozen frame reduction, ground truth test (Principle V), vacuous stage amendment, forward viability check.

### Research Discipline
**What**: Kill criteria, damped oscillation (breadth↔depth), Karpathy-inspired autonomous loop, anti-patterns.
**File**: `research/RESEARCH_DISCIPLINE.md`
**Anti-patterns**: characterization loops (eigenform arc), manual compilation (decomposition arc).

### Constraint System
**What**: 66 experimentally-derived constraints (51 Singularity Search + 15 FoldCore).
**File**: `knowledge/constraints.json`
**Format**: `{id, rule, source_entries, tags, active}`

### Knowledge Base
**What**: 78 structured entries (experiments, discoveries, decisions, architecture).
**File**: `knowledge/entries/*.json`
**Compiler**: `knowledge/compile.py` — generates state summary from entries.

### Paper Compiler
**What**: Auto-generates research paper (HTML) from .knowledge/ system.
**File**: `paper/compile_paper.py`
**Run**: `python paper/compile_paper.py` → `paper/paper.html`

---

## Runnable Experiments

| Script | Step | What it demonstrates |
|--------|------|---------------------|
| `experiments/run_step99_topk_vote.py` | 99 | 91.8% P-MNIST, top-k readout |
| `experiments/run_step181_iterated_knn.py` | 181 | Iterated k-NN vs direct (CA computation gap) |
| `experiments/run_step182_ca_all_rules.py` | 182 | 9/10 CA rules iterated 10 steps |
| `experiments/run_step190_fibonacci_e2e.py` | 190 | Fibonacci 50 steps from raw input |
| `experiments/run_step192_fsm_simulator.py` | 192 | Universal FSM simulator |
| `experiments/run_step235_decomposed_addition.py` | 235 | OOD addition 255+255=510 (manual decomposition) |
| `experiments/run_step250_complete_substrate.py` | 250 | Complete substrate demo (all capabilities) |
| `experiments/run_step255_analogy.py` | 255 | Analogy reasoning A:B::C:? (+43pp) |
| `experiments/run_step266_auto_discovery.py` | 266 | Auto pipeline: I/O → circuits → OOD computation |
| `experiments/run_step273_278_algorithms.py` | 273-278 | Primality, GCD, Fibonacci, power, mod_pow, isqrt |

All require: `torch`, `torchvision` (for MNIST scripts). GPU optional (CPU works, slower).

---

## Historical Experiment Scripts

| Directory | Steps | What |
|-----------|-------|------|
| `experiments/foldcore-steps/` | 37-106 | FoldCore codebook experiments |
| `experiments/ss-sessions/` | 1-23 | Singularity Search session scripts |
| `experiments/archive/` | misc | Historical analysis and cleanup |

---

## Other Assets

### Tempest
**What**: Rust implementation of physics-based emergent computation.
**File**: `tempest/tempest.rs`
**Status**: Historical. The thesis (intelligence from physics in constrained environments) informed the search but Tempest itself was not the substrate.

### Genesis World Model
**What**: Tokenizer, dynamics model, quantization experiments.
**Dir**: `substrates/worldmodel/`
**Status**: Historical. Part of the Singularity Search exploration.

### Reflexive Kernel (rk.py)
**What**: Matrix cell dynamics. `Phi(M) = tanh(alpha*M + beta*M^2/k)` with alpha > 1.
**File**: `substrates/living-seed/rk.py`
**Status**: The bridge between Singularity Search and FoldCore. State = transformation = objective. Informed the eigenform arc (which was then closed).

---

## Key Results Summary

| Metric | Value | Step | Honest? |
|--------|-------|------|---------|
| P-MNIST AA | 95.4% | 110 | YES — raw pixels, no feature extractor |
| P-MNIST forgetting | 0.0pp | 110 | YES — structural, by construction |
| CIFAR-100 AA (ResNet) | 39.7% | 119 | YES — with S1 eval anti-forgetting |
| Parity improvement | +20pp | 199 | YES — feature discovery, genuine |
| Analogy improvement | +43pp | 255 | YES — feature discovery, genuine |
| Addition improvement | +47pp | 220 | YES — feature discovery on raw ints |
| OOD addition 255+255 | 510 | 235 | MANUAL — I designed ripple-carry |
| All arithmetic from 1 table | 100% | 237 | MANUAL — I wrote the algorithms |
| Program execution | 100% | 240 | MANUAL — I designed the interpreter |
| Circuit synthesis | 6/6 | 245 | GENUINE — substrate discovers circuits |
| Curriculum carry chain | 100% | 264 | GENUINE — substrate discovers composition |
| Emergent GCD | 12.6% | 283 | HONEST FAILURE — k-NN can't discover a%b |
| Emergent Collatz | 0% LOO | 290 | HONEST FAILURE — discontinuous, undiscoverable |

---

## The Frontier

**What works**: k-NN discovers functions where similar inputs → similar outputs (L2-Lipschitz-continuous). Feature discovery via LOO-scored random projections. Circuit synthesis from I/O for small boolean functions. Curriculum-based progressive composition.

**What doesn't**: emergent decomposition for functions with discontinuous/oblique level sets (modular arithmetic, subtraction, Collatz). The substrate executes algorithms but can't discover them for non-smooth domains.

**The open problem**: how does compositional structure emerge from data without backprop? This is the gap between the substrate and transformers. 289 experiments have characterized this gap precisely.

---

*Every claim maps to a step. Every step maps to code. The honest results and the honest failures are both documented.*
