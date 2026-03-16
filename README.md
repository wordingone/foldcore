# The Search

**Recursive self-improvement through monotonic frozen frame reduction.**

This repository documents an ongoing search for the atomic substrate -- a single operation where memory, learning, inference, and perception are the same thing. It spans 341 experiments across 28+ sessions, four substrate architectures, 70+ constraints, and 78+ knowledge entries. **Stages 1-7 confirmed on a compressed 22-line `process()` function that scores 93.82% on Permuted-MNIST with 0pp forgetting.** Stage 8 is the frontier.

The search follows a [constitution](CONSTITUTION.md): five principles and eight stages that define the path to recursive self-improvement architecture-independently, with empirical tests at each step.

---

## The Thesis

Every technology follows birth, then scale, then compression. Vacuum tubes became transistors became ICs. The current AI stack (transformers, KV cache, frozen weights, bolted-on tools) is late-scale. The compression is coming.

We seek the compressed form: a single equation that collapses the fractured stack into one indivisible operation. Not a better neural network -- the thing that replaces neural networks.

---

## The Results So Far

### Continual Learning (Lipschitz functions)
**91.8% average accuracy, 0.0pp forgetting, 30 lines, no backprop.**

A competitive-learning codebook that solves Permuted-MNIST (10 sequential tasks) in 20 seconds with structural zero forgetting. No gradient descent, no replay buffer, no regularization. 8,597 prototype vectors on the unit hypersphere, classified by top-k cosine vote.

```
Permuted-MNIST, 10 tasks, d=384, 6K train / 10K test per task:

  k=1 (1-NN):   86.8% AA,  0.0pp forgetting
  k=3 (top-k):  91.8% AA,  0.0pp forgetting   <-- +5.0pp from readout alone
  k=5 (top-k):  91.5% AA,  0.0pp forgetting

  Codebook: 8,597 vectors. Runtime: ~20 seconds. No backprop.
```

The complete system is `experiments/foldcore-steps/run_step99_topk_vote.py`. The core class is 30 lines:

```python
class TopKFold:
    def __init__(self, d, lr=0.01, spawn_thresh=0.7):
        self.V = torch.empty(0, d, device=DEVICE)
        self.labels = torch.empty(0, dtype=torch.long, device=DEVICE)
        self.lr, self.spawn_thresh, self.d = lr, spawn_thresh, d

    def step(self, r, label):
        r = F.normalize(r, dim=0)
        if self.V.shape[0] == 0 or (self.V @ r).max().item() < self.spawn_thresh:
            self.V = torch.cat([self.V, r.unsqueeze(0)])
            self.labels = torch.cat([self.labels, torch.tensor([label], device=DEVICE)])
            return
        sims = self.V @ r
        winner = sims.argmax().item()
        self.V[winner] = F.normalize(
            self.V[winner] + self.lr * (r - self.V[winner]), dim=0)

    def eval_batch(self, R, k_vals):
        R = F.normalize(R, dim=1)
        sims = R @ self.V.T
        n, n_cls = len(R), int(self.labels.max().item()) + 1
        results = {}
        for k in k_vals:
            scores = torch.zeros(n, n_cls, device=DEVICE)
            for c in range(n_cls):
                mask = (self.labels == c)
                if mask.sum() == 0: continue
                class_sims = sims[:, mask]
                k_eff = min(k, class_sims.shape[1])
                scores[:, c] = class_sims.topk(k_eff, dim=1).values.sum(dim=1)
            results[k] = scores.argmax(dim=1).cpu()
        return results
```

### Running it

```bash
# Requires: torch, torchvision, numpy
# Downloads MNIST on first run

python experiments/foldcore-steps/run_step99_topk_vote.py        # 10-task P-MNIST
python experiments/foldcore-steps/run_step99_topk_vote.py 5      # 5-task P-MNIST (faster)
```

### Non-Lipschitz Classification (Steps 291-333)
**92.0% on a%b — the substrate discovered a better filter than the human designer.**

Stage 6: competitive learning discovers spatial-proximity grouping that outperforms prescribed same-b filtering by +5.25pp (92.0% vs 86.8%). The discovered groups have 26.3% b-purity — they are NOT b-groups but a genuinely different, better grouping.

The human contribution: per-class sorted distance distributions (phi) break the Lipschitz ceiling. The substrate's contribution: discovered filter (Stage 6) + learned weights (Stage 2) that together exceed all prescribed approaches.

```
a%b classification, (a,b) in 1..20, LOO:

  1-NN:                5.0%   (Lipschitz ceiling)
  Top-K sum:          12.5%
  Phi (sort, K=5):    86.8%   <-- human-designed readout
  Phi + learned w:    91.2%   <-- Stage 2: substrate discovered k=0 importance
  CL filter (Step 333):92.0%  <-- Stage 6: substrate discovered better filter than human
  Phi + exp(-k):      87.2%   <-- substrate's discovery prescribed as fixed physics
  Reflection spawn:   95.2%   (OOD, a in 21..50, with designed mechanism)
  Periodic encoding: 100.0%   (prescribed physics matching function structure)
```

Key insight: same-class inputs have **identical** per-class distance distributions even though they're spatially far apart. The fold already computes these distances — phi just sorts them instead of summing.

**Honest limits:**
- 91.2% is in-distribution only. OOD: 17.3% (the learned weights don't transfer across distribution shift).
- 87.2% (prescribed exp(-k)) generalizes across functions but is smaller (+0.5pp vs +4.4pp).
- Reflection spawn (95.2% OOD) requires three designed mechanisms.
- The substrate discovers b-grouping (R²=0.858) and k=0 importance (+4.4pp). It cannot discover phi itself (Steps 306-312, all killed).
- The encoding IS the physics. The substrate operates within prescribed physics, improving it by +4.4pp.

```
Step 315 automated loop (grow + refine, 1 turn):

  Baseline:           86.8%  LOO,  16.7% OOD
  After grow+refine:  94.4%  LOO,  48.5% OOD   <-- automated, no human in the loop
```

```bash
python experiments/auto_loop.py                     # The automated loop (88.0%)
python experiments/run_step308_phi_weighted.py      # Substrate's 91.2% (learned w)
python experiments/run_step296_dist_matching.py     # Human's 86.8% (phi baseline)
```

---

## The Journey: Three Substrates

### Substrate 1: The Living Seed (Sessions 1-17)

The original architecture. A 1D ring of 6 cells, each with 12 dimensions. Core equation:

```
phi[k] = tanh(alpha*x[k] + beta*(x[k+1] + gamma*s[k+1])*(x[k-1] + gamma*s[k-1]))
```

17 sessions of rigorous stage progression through the constitution's framework:
- **Stage 1 (passed):** Autonomous computation. Ground truth test passes on 3+ seeds.
- **Stage 2 (alpha adaptive):** Per-cell alpha becomes governed by self-generated signal.
- **Stage 3 (vacuous):** 7 sessions, 5 approaches tested. Eta (learning rate) can be made adaptive but produces zero measurable effect. Declared vacuously passed under [Amendment 1](CONSTITUTION.md).
- **Stage 4 (partial):** Delta binding confirmed (+6%). Beta/gamma globally coupled (cannot decompose). All other parameters non-binding.

**Architecture ceiling declared (Session 17):** Frozen frame minimum 6/8. The equation is hardcoded Python -- no mechanism for self-representation (Stage 7). Substrate cannot proceed. This is a scientific result: the Living Seed's ceiling is exactly 6/8 frozen elements.

**Code:** `substrates/living-seed/`

### Substrate 2: ANIMA (Sessions 18-23)

World-model organism designed to satisfy Stage 7's forward viability check: an internal world model W whose update rule can in principle become self-modifiable data.

- **Stage 2 (vacuous):** w_lr has an interior optimum at 0.0003 (3.5x MI gap improvement), but no Principle-II-compliant internal signal can detect it. MI-error structural decoupling confirmed across 100x w_lr range.
- **Stage 3 (architecture ceiling):** All 7 parameters characterized. None is both binding AND adaptable from within. Dual-timescale I, W_velocity, and additive slow I all exhausted.

**Architecture ceiling declared (Session 23):** Complete parameter characterization confirms no adaptive path forward through parameter space. Stage 4 structural sweeps not yet conducted.

**Code:** `substrates/anima/`

### Substrate 3: FoldCore / TopK (Sessions 24-26, Steps 37-106)

The codebook system. Born from the architecture autopsy that stripped the Living Seed down to its load-bearing components: competitive learning on the unit hypersphere.

The readout arc (Steps 97-105) systematically tested 7 readout mechanisms. Each failure extracted a constraint:

| Step | Readout | Result | Constraint Extracted |
|------|---------|--------|---------------------|
| 97 | Differential response | KILLED (15%) | Anti-correlated readout factors fail |
| 98 | Neighborhood coherence | KILLED (85.3%) | Readout must be input-conditional |
| 99 | **Top-k class vote** | **PASSED (91.8%)** | -- |
| 100 | Top-k on CIFAR-100 | Passed readout (38.3%) | Spawn threshold is feature-space dependent |
| 101 | Spawn-only ablation | Passed | Forgetting is class competition, not drift |
| 102 | Self-routing gates | KILLED | Sum-all aggregation drowns signal |
| 103 | Resonance dynamics | KILLED | Iterative dynamics blur, don't sharpen |
| 104 | Centroid accumulation | KILLED (30%) | Sparse storage is load-bearing |
| 105 | LSH counting | Functional | Irrelevant at current codebook sizes |

**Code:** `substrates/foldcore/`, `substrates/topk-fold/`

---

## Benchmark Results

All results use frozen feature extractors (random projection for MNIST, frozen ResNet-18 for CIFAR-100).

### Permuted-MNIST (10 sequential tasks, d=384)

| Method | Avg Accuracy | Forgetting | Notes |
|--------|-------------|------------|-------|
| **TopKFold (k=3, cosine spawn)** | **91.8%** | **0.0pp** | **This work. 30 lines, no backprop.** |
| TopKFold (k=1 / 1-NN, cosine spawn) | 86.8% | 0.0pp | Same codebook, weaker readout |
| FoldCore (attractive-only, 1-NN) | 56.7% | 0.0pp | Original fold, energy-based spawning |
| EWC | ~95.3% | ~2pp | Backprop, quadratic regularization |
| Fine-tune baseline | ~52.5% | ~47pp | |

### Split-CIFAR-100 (20 tasks, d=512, frozen ResNet-18)

| Method | Avg Accuracy | Forgetting | Notes |
|--------|-------------|------------|-------|
| TopKFold (k=10, cosine spawn) | 38.3% | -- | +6.1pp over 1-NN |
| FoldCore (1-NN) | 33.5% | 12.6pp | |
| EWC | ~33% | ~16pp | |
| DER++ | ~51% | ~8pp | Replay buffer |

### What the numbers mean

- **Zero forgetting is structural.** Attractive-only updates and append-only spawning preserve old prototypes by construction.
- **The accuracy gap vs EWC/DER++** comes from nearest-prototype readout vs learned decision boundaries, not from memory failure.
- **Top-k vote closes 5pp of that gap** by aggregating local class evidence instead of relying on a single champion vector.

---

## The Discovery-Prescription Loop (Steps 308-313)

The substrate and the designer improve each other in a loop:
1. Human designed phi (sort instead of sum) → 86.8%
2. Substrate discovered k=0 importance (nearest class-member is most diagnostic) → 91.2% in-dist
3. Human prescribed the discovery as fixed weights (exp(-k)) → 87.2% (generalizes, unlike learned 91.2%)
4. Next turn: substrate operates within new prescribed weights → discovers next level → ...

The substrate's genuine contribution is +0.5pp that generalizes across functions. Small but real — and it's the loop's first turn. The intelligence is in the loop between human and substrate, not in either alone.

## What Failed and Why It Matters

### The eigenform arc (17 experiments) -- CLOSED

Tanh eigenform composition produces genuine algebra (31 eigenforms at k=4, Steiner triple kernel, non-associative non-commutative idempotent magma). But it failed every applied test: 22.2% AA on P-MNIST vs 46.2% baseline. External review (DeepSeek): "Proves it's an expensive distance function."

### The matrix layer -- DEAD

Architecture autopsy: `classify()` never reads matrix state. Removing 8 matrix cells, projection, coupling, autonomy, surprise, and 11 hyperparameters produces identical classification. Two disconnected systems pretending to be one.

### The four separations

Every failure traces to a separation that should not exist:
1. **Training / Inference** -- the fold trains with hard assignment but classifies differently
2. **Storage / Readout** -- the codebook stores perfectly but reads weakly
3. **Memory / Generation** -- codebook and matrix don't interact
4. **System / State** -- the algorithm is external to the codebook

The atomic equation is what you get when all four separations collapse.

### The Lipschitz wall (Steps 291-295) -- THEOREM

NN chain iteration is provably lossy for non-Lipschitz functions in Euclidean space. Each step crosses class boundaries with probability p. Accuracy degrades as ~(1-p)^K. Five experiments confirmed from five angles: soft blending (291), AMR (293), LVQ (294), basin sculpting (295). One-step NN is strictly better than any chain.

### Autonomous metric discovery (Steps 306-312) -- KILLED, then RECOVERED (Step 333)

The substrate discovers b-grouping (R²=0.858) and k=0 feature importance (+0.5pp). It cannot discover phi from raw features (Steps 306-312, 7 kills). But Step 333 showed it CAN discover a better filter via competitive learning dynamics: 92.0% with CL-discovered spatial-proximity filter vs 86.8% with prescribed same-b filter. The encoding IS the physics — but the FILTER is discoverable.

### ARC-AGI evaluation (Steps 320-335) -- CEILING MAPPED

1000 ARC tasks evaluated. The fold solves exactly 12 (spatial transforms). Substrate mechanisms (phi, loop) contribute ~0 on ARC. Changed-cell accuracy: 39.6% with 5x5 patches (best), 24.0% with flat encoding. Constraint map: 418 conditional, 293 size-change, 123 symmetry, 99 object-identity. The fold is a vector machine; ARC needs graphs (object identity) and programs (conditional logic). Loop weight learning is a%b-specific (0pp on P-MNIST, Step 330). Iteration amplifies dominant structure (Steps 291b, 295, 328, 332).

---

## The Atomic Substrate Tests

The next substrate must pass these structural tests:

**S1 -- Single Function:** One function `process(state, input) -> (output, new_state)` where the same code path handles training and inference. No `if training:` branches.

**S2 -- Deletion Test:** You cannot delete any part of the code without losing ALL capabilities simultaneously.

**S3 -- State Completeness:** The state contains ALL information needed to reproduce behavior. No external algorithm, no hyperparameters, no code.

**S4 -- Generation Test:** The same operation handles learning, inference, AND generation.

A substrate passes if it satisfies S1+S2. S3+S4 are aspirational (full collapse).

---

## Repository Structure

```
the-search/
  CONSTITUTION.md        -- The 5 principles + 8 stages (the theoretical framework)
  RESEARCH_STATE.md      -- Live state: current hypothesis, constraints, candidates
  README.md              -- This file

  knowledge/             -- Unified knowledge base (78 entries, 66 constraints)
    entries/             -- ALL entries: SS sessions 1-23 + FoldCore steps 37-106
    constraints.json     -- ALL constraints merged (SS c001-c051 + FoldCore fc001-fc015)
    frozen_frame.json    -- Frozen frame state (Living Seed)
    compile.py           -- Knowledge base integrity checker

  paper/                 -- Paper compiler (generates from knowledge/)
    compile_paper.py     -- Renders paper.html from all 78 entries

  substrates/            -- All substrate implementations
    living-seed/         -- Stage 1-4 (Sessions 1-17)
    anima/               -- ANIMA organism (Sessions 18-23)
    foldcore/            -- Codebook system (manytofew, torch, rk)
    eigenfold/           -- Matrix codebook [CLOSED]
    topk-fold/           -- The 91.8% system

  experiments/           -- All runnable experiment scripts
    ss-sessions/         -- SS session scripts (Living Seed + ANIMA stages)
    foldcore-steps/      -- FoldCore steps 97-106
    benchmarks/          -- Standalone benchmarks

  research/              -- Framework documentation
    FRAMEWORK.md         -- Research framework (thesis, mechanisms, what works)
    EXPERIMENT_LOG.md    -- Full experiment history (96 steps)
    EQUATION_CANDIDATES.md -- Candidate equations for the atomic foundation
    WHAT_THE_FAILURES_TEACH.md -- Four separations analysis
    JUNS_INTENT.md       -- Founder's intent extracted from source conversations

  tempest/               -- Tempest substrate (Rust, wave dynamics)
  tests/                 -- Unit tests
```

## The Constitution

The [constitution](CONSTITUTION.md) defines five architecture-independent principles:

1. **Computation must exist without external objectives** -- remove all loss functions; does it still compute?
2. **Adaptation must arise from computation, not beside it** -- the signal that drives modification must be a byproduct of the computation itself
3. **Each modification must be tested against what came before** -- improvement on trained tasks with degradation on novel tasks is overfitting
4. **The frozen frame must shrink monotonically** -- at each stage, at least one frozen element becomes adaptive
5. **There must be one ground truth the system cannot modify** -- prevents trivial "improvement" by redefining improvement

Eight stages of frozen frame reduction from full external control (Stage 1) to ground truth as the only frozen element (Stage 8). Two amendments: vacuous stages (Amendment 1, Session 12) and forward viability checks (Amendment 2, Session 15).

## Constraints from 335 Experiments

70+ constraints define what does NOT work. Key categories:

- **Non-binding parameters:** Most parameters don't affect performance (massive degeneracy)
- **Anti-signals:** Some adaptation signals drive in the wrong direction (c020, c022, c036)
- **Timescale mismatch:** Per-step signals cannot capture sequence-level properties (c036, c047, c048)
- **Architectural ceilings:** Living Seed 6/8, ANIMA Stage 3 ceiling
- **Readout constraints:** Must be input-conditional, sparse (top-k), no anti-correlated factors
- **ARC constraints:** Phi needs class-correlated distance structure (C23). k-NN in >40 dims needs >>500 entries (C24). Iteration amplifies dominant structure, destroys subordinate (C27-C28)

Each constraint is a closed door. The pattern of elimination IS the search.

## Open Questions

1. ~~Is there a single operation that subsumes spawn + update + classify?~~ **ANSWERED:** f = absorb. State(t+1) = f(State(t), D). (Step 305)
2. ~~Can top-k readout be derived from the same dynamics that produce the codebook?~~ **ANSWERED:** Phi (per-class distribution matching) IS the readout. 86.8% on a%b. (Step 296)
3. ~~Can the substrate discover its own distance function?~~ **PARTIALLY ANSWERED:** Stage 6 (Step 333): CL discovers filter that beats prescribed (+5.25pp). The substrate can't discover phi from raw features, but CAN discover the filter that makes phi optimal. The encoding IS the physics; the filter is discoverable.
4. Can the substrate bootstrap from empty without prescribed encoding? Steps 306-307: NO (chicken-and-egg). Data provides the bootstrap.
5. ~~Does this generalize beyond periodic non-Lipschitz functions?~~ **ANSWERED:** Phi generalizes to floor(a/b) (Step 302). ARC evaluation (Steps 320-335) maps exact limits: fold handles spatial transforms, fails on objects/conditionals/symmetry.
6. ~~Is the equation State(t+1) = f(State(t), D) the atomic operation?~~ **ANSWERED:** Yes (Step 305 + Tempest convergence). f = absorb.
7. ~~Can the update rule become modifiable data? (Stage 7)~~ **ANSWERED:** Yes (Step 337). Per-entry K stored as codebook data. Beats global (+2.00pp) and oracle (+0.75pp) on mixed-function problem. K distribution {3:3, 5:14, 7:3} — non-trivial, non-uniform. Stage 7 passes.
8. **NEW: Can everything become adaptive except the ground truth? (Stage 8)** Step 338 in progress. Spawn rule as codebook data. Forward viability: YES (iterated k-NN is Turing-complete, Step 181).

## Requirements

- Python 3.8+
- Experiments: `torch`, `torchvision`, `numpy`
- `substrates/foldcore/foldcore_manytofew.py`: no dependencies beyond stdlib
- Tests: `pytest`, `numpy`
- Paper compiler: `markdown` (optional, fallback rendering without it)

## License

CC BY-NC 4.0 -- free for non-commercial research and educational use. See [LICENSE](LICENSE).

## Contributing

The most valuable contribution is blunt analysis. Run the code, read the math, tell us what's wrong or what this actually is. Open an issue or submit a PR.

---

*The destination defines the path. Each step either shrinks the frozen frame or it is not a step.*
