# The Search

**91.8% average accuracy, 0.0pp forgetting, 30 lines, no backprop.**

A competitive-learning codebook that solves Permuted-MNIST (10 sequential tasks) in 20 seconds with structural zero forgetting. No gradient descent, no replay buffer, no regularization. 8,597 prototype vectors on the unit hypersphere, classified by top-k cosine vote.

This is the working result from 105 experiments searching for an atomic substrate where memory, learning, and inference are one operation.

## The Working System

Three mechanisms, each load-bearing:

1. **Competitive learning** -- unit vectors in R^d attract toward inputs
2. **Cosine spawning** -- new prototype created when max similarity < 0.7 (no energy functions)
3. **Top-k class vote** -- per-class sum of top-k cosine similarities (k=3-5 optimal)

```
Permuted-MNIST, 10 tasks, d=384, 6K train / 10K test per task:

  k=1 (1-NN):   86.8% AA,  0.0pp forgetting
  k=3 (top-k):  91.8% AA,  0.0pp forgetting   <-- +5.0pp from readout alone
  k=5 (top-k):  91.5% AA,  0.0pp forgetting

  Codebook: 8,597 vectors. Runtime: ~20 seconds. No backprop.
```

### The code

The complete system is `experiments/run_step99_topk_vote.py`. The core class is 30 lines:

```python
class TopKFold:
    def __init__(self, d, lr=0.01, spawn_thresh=0.7):
        self.V = torch.empty(0, d, device=DEVICE)        # codebook
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

python experiments/run_step99_topk_vote.py        # 10-task P-MNIST
python experiments/run_step99_topk_vote.py 5      # 5-task P-MNIST (faster)
```

## Benchmark Results

All results use frozen feature extractors (random projection for MNIST, frozen ResNet-18 for CIFAR-100). The system operates on pre-extracted embeddings, not raw pixels.

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
| TopKFold (k=10, cosine spawn) | 38.3% | — | +6.1pp over 1-NN |
| FoldCore (1-NN) | 33.5% | 12.6pp | |
| EWC | ~33% | ~16pp | |
| DER++ | ~51% | ~8pp | Replay buffer |

### What the numbers mean

- **Zero forgetting is structural.** Attractive-only updates and append-only spawning preserve old prototypes by construction. This is not a claim about the algorithm's cleverness -- it is a property of never overwriting stored vectors.
- **The accuracy gap vs EWC/DER++** comes from nearest-prototype readout vs learned decision boundaries, not from memory failure. The codebook stores well but reads simply.
- **Top-k vote closes 5pp of that gap** by aggregating local class evidence instead of relying on a single champion vector.
- **Published baselines use 60K samples/task** (MNIST) and full training pipelines. This system uses 6K/task with random projection.

## What failed (closed arcs)

### Arc 2: Eigenform substrate -- CLOSED

17 experiments on tanh eigenform composition. The algebra has genuine mathematical properties (31 eigenforms at k=4, Steiner triple kernel, Z2 anti-symmetry, infinite generation) but failed every applied test:

- **Classification:** 22.2% AA on P-MNIST vs 46.2% vector cosine baseline. Perturbation-stability classification is prototype matching with an expensive metric.
- **Cross-application destroys eigenform structure.** 0% convergence after Psi(M*, R) for true eigenforms.
- **Collective coupling:** -0.2pp accuracy, +0.8pp forgetting, 27x slower.
- **k=8 and k=16:** 0% convergence. The structure is k=4 specific.

External review (DeepSeek): "Proves it's an expensive distance function." Confirmed at Step 76.

The eigenform composition algebra is a genuine mathematical object (non-associative, non-commutative idempotent magma). It may be interesting to algebraists. It does not compute.

### Arc 2b: Spectral eigenform -- CLOSED

Scale-independent convergence (100% at k=4, 8, 16) and non-commutative pairwise composition, but:

- **P-MNIST:** 15.9% vs 46.2% baseline.
- **Temporal order discrimination:** 55.5% vs 64.5% order-blind baseline.
- **Root cause:** Long-chain composition collapses to the same small attractor set regardless of input order. Pairwise non-commutativity does not accumulate through chains.

### Failed readout mechanisms (Arc 3, Steps 97-105)

Each failure extracted a constraint narrowing the search:

- **Differential response:** Attention and displacement anti-correlate. High-attention vectors are already aligned, producing small displacements. The signal is in the product, which reduces to cosine similarity.
- **Neighborhood coherence:** Static vector property that penalizes boundary vectors (the informative ones).
- **Self-routing gates:** Sum-all aggregation drowns per-class signal.
- **Resonance dynamics:** Energy-gradient dynamics over dense codebooks converge to centroid blur, not class attractors. More iterations = worse (-34pp at 20 iterations).
- **Centroid accumulation:** Single d x C matrix: 30% AA. Codebook's sparse storage is load-bearing -- averaging destroys task-specific geometry.

Key external feedback: **"The codebook works. The matrix layer doesn't. Kill the matrix layer, publish the codebook."**

## Repository structure

```
src/
  foldcore_manytofew.py   -- Canonical kernel (codebook + matrix layer, pure Python)
  rk.py                   -- Reflexive kernel (matrix cell dynamics)
  eigenfold.py            -- Matrix codebook with eigenform dynamics [CLOSED]
  atomic_fold.py          -- Hopfield-equivalent kernel [DEPRECATED]
  foldcore_torch.py       -- GPU codebook (PyTorch)

experiments/
  run_step99_topk_vote.py        -- THE RESULT: 91.8% P-MNIST, top-k vote
  run_step101_pmnist.py          -- Spawn-only (lr=0) ablation
  run_step100_cifar100_topk.py   -- CIFAR-100 validation
  run_step97_differential_response.py  -- Failed: differential readout
  run_step98_coherence_readout.py      -- Failed: neighborhood coherence
  run_step102_self_routing.py          -- Failed: self-routing gates
  run_step103_resonance.py             -- Failed: resonance dynamics
  run_step104_accumulated_op.py        -- Failed: centroid accumulation
  run_step105_lsh_counting.py          -- LSH counting readout

tests/
  test_manytofew.py       -- Unit tests for canonical kernel
```

## Requirements

- Python 3.8+
- `experiments/`: torch, torchvision, numpy
- `src/foldcore_manytofew.py`: no dependencies beyond stdlib
- Tests: pytest, numpy

## The search

This is not a continual learning library. It is an ongoing search for the atomic substrate -- a single operation where memory, learning, inference, and perception are the same thing.

The codebook works: competitive learning with cosine spawning and top-k readout achieves 91.8% on P-MNIST with zero forgetting. But the codebook is three mechanisms bolted together, not one atomic operation. The search continues.

### What the atomic substrate must satisfy (S1-S4)

- **S1:** A single operation handles storage, retrieval, and update
- **S2:** The operation is its own inverse (or fixed point)
- **S3:** Adding capacity does not require architectural changes
- **S4:** The system's representation IS its computation (no weight/activation split)

### Constraints from 105 experiments

1. Sparse prototype storage is load-bearing (centroid averaging destroys geometry)
2. Readout must be input-conditional (static vector properties fail)
3. Per-class aggregation must use positive-only monotonic scoring
4. Iterative dynamics over dense codebooks converge to blur, not structure
5. Non-commutative matrix composition does not accumulate through long chains
6. Energy-based spawning is strictly worse than cosine threshold
7. The matrix layer (eigenform dynamics, coupling, generation) adds zero classification value

### Open questions

1. Is there a single operation that subsumes spawn + update + classify? The codebook uses three mechanisms. Can they be unified?
2. Can top-k readout be derived from the same dynamics that produce the codebook? Currently readout is a separate evaluation function.
3. What substrate satisfies S4 (representation IS computation) while matching the codebook's 91.8%?

## License

CC BY-NC 4.0 -- free for non-commercial research and educational use. See [LICENSE](LICENSE).

## Contributing

The most valuable contribution is blunt analysis. Run the code, read the math, tell us what's wrong or what this actually is. Open an issue or submit a PR.
