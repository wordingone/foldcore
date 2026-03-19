# The Search

**Can a system improve itself by criteria it generates?**

This repository documents a systematic search for a substrate — a minimal computational structure — that satisfies six simultaneous rules for recursive self-improvement. 525+ experiments across 9 architecture families. No solution found yet. The constraint map from those failures is the main contribution.

---

## The Problem

Six rules ([CONSTITUTION.md](CONSTITUTION.md)) define the feasible region. A substrate either satisfies all six or it doesn't.

| Rule | Requirement |
|------|------------|
| R1 | Computes without external objectives |
| R2 | Adaptation arises from computation, not beside it |
| R3 | Every modifiable aspect IS modified by the system |
| R4 | Each modification tested against prior state |
| R5 | One fixed ground truth test the system cannot modify |
| R6 | No part deletable without losing all capability |

R3 is the binding constraint. Every substrate tested so far has hardcoded operations the system cannot see or modify.

---

## What 525 Experiments Found

### Architecture families

| Family | Experiments | Best Result | Status |
|--------|------------|-------------|--------|
| Codebook (LVQ) | ~435 | 94.48% P-MNIST (supervised), chain 3/3 ARC games | Mapped — banned for further experiments |
| LSH graph | ~55 | LS20 Level 1, chain WIN@1116 (10x faster than codebook) | Active — best non-codebook family |
| L2 k-means graph | ~25 | LS20 9/10 at 120K, chain negative transfer confirmed | Active |
| Reservoir (ESN) | ~20 | Memory contributes nothing to navigation | Killed |
| Graph (cosine) | ~8 | LS20 Level 1 at 25738 steps | Superseded by LSH/k-means |
| Connected-component | 1 | 23 states, too slow (200 steps/sec) | Killed |
| Bloom filter | 2 | 1/10 (random walk luck) | Killed |
| CA | 3 | Degenerate mapping | Killed |
| LLM agent | 1 | Action collapse (100% ACTION1) | Preliminary |

### The chain benchmark

The real test is not single-benchmark performance. It is the **chain**: heterogeneous benchmarks run in sequence with one continuous state, no reset.

```
CIFAR-100 → Atari (ARC-AGI-3) → CIFAR-100
```

| Finding | Step | Result |
|---------|------|--------|
| Frozen centroids → negative transfer | 506, 515 | Universal across codebook AND k-means families |
| Dynamic growth → domain separation | 507-508 | Chain 3/3 ARC games, zero CIFAR forgetting |
| LSH chain via action-scope isolation | 516 | WIN@1116 — different mechanism than codebook chain |
| Threshold tension | 509-513 | CIFAR needs threshold≥3.0, ARC needs ≤0.5. Incompatible. |
| K-means cross-game transfer | 522 | Degenerate (centroid collapse). Attract update load-bearing. |

### Key findings

**All 3 ARC-AGI-3 games Level 1 solved (Steps 503, 505).** Unifying mechanism: graph + edge-count argmin + correct action decomposition. LS20: 4 actions. FT09: 69 actions (64 click grid + 5 simple). VC33: 3 zones (2 magic pixels). Confirmed across codebook and k-means families.

**Classification is supervised (Step 432).** Self-generated labels: 9.8% (below chance). The chain achieves 1% on CIFAR-100 (chance) — encoding has class signal (NMI=0.42) but the threshold is incompatible across domains.

**Negative transfer is universal (Steps 506, 515).** Frozen centroids from one domain break navigation in another. This holds for both codebook (cosine) and k-means (L2). LSH avoids it entirely — random projections are domain-agnostic.

**Two independent chain mechanisms.** Codebook survives the chain via encoding-space separation (CIFAR and ARC centroids are far apart). LSH survives via action-scope isolation (each game queries only its own actions). Both prevent catastrophic interference.

### The constraint map

Constraints extracted from experimental failure across 9 families. See [CONSTRAINTS.md](CONSTRAINTS.md) for the full map with cross-family validation status.

---

## Current Direction

1. **Non-codebook scale-up** — ~55 non-codebook experiments vs ~435 codebook. Scaling non-codebook families to balance the evidence base. Codebook experiments banned.
2. **New architecture families** — Hebbian learning, Markov transition models, sequence prediction agents. Testing genuinely different mechanisms on the chain.
3. **The paper** — [PAPER.md](PAPER.md) compiles to LaTeX. Formal framework (f, g, F), two theorems, 11 degrees of freedom.

---

## Running It

```bash
pip install torch torchvision numpy

# Phase 1: the LVQ baseline
python experiments/run_step250_complete_substrate.py       # Complete demo (~30s)

# Chain experiments (Phase 2)
python experiments/run_step508_full_chain.py                # Full chain CIFAR→ARC→CIFAR
python experiments/run_step474_kmeans_l2.py                 # K-means graph on LS20
```

---

## Repository Structure

```
the-search/
  CONSTITUTION.md        -- 5 principles + 6 rules (R1-R6)
  CONSTRAINTS.md         -- Constraint map with cross-family validation
  RESEARCH_STATE.md      -- Full experiment log (Steps 1-525+)
  PAPER.md               -- Publication draft (compiles to LaTeX)
  build-paper.py         -- LaTeX compilation pipeline

  substrates/            -- Substrate implementations (Phase 1)
  experiments/           -- 525+ experiment scripts
  research/              -- Research methodology and frameworks
  archive/               -- Archived Phase 1 infrastructure
  audits/                -- External audit reports
  tests/                 -- Unit tests
```

## Requirements

- Python 3.8+
- `torch`, `torchvision`, `numpy`
- Tests: `pytest`

## License

CC BY-NC 4.0. See [LICENSE](LICENSE).

## Contributing

Run the code. Read [CONSTRAINTS.md](CONSTRAINTS.md). Tell us what's wrong. Open an issue or PR.

---

*The constraints define the region. The substrate is inside it or it doesn't exist.*
