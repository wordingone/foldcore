# The Search

**Can a system improve itself by criteria it generates?**

This repository documents a systematic search for a substrate — a minimal computational structure — that satisfies six simultaneous rules for recursive self-improvement. 445 experiments across three architecture families. No solution found yet. The constraint map from those failures is the main contribution.

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

## What 445 Experiments Found

### Three architecture families

| Family | Experiments | Best Result | Status |
|--------|------------|-------------|--------|
| Codebook (LVQ) | 435 | 94.48% P-MNIST (supervised), LS20 Level 1 at ~26K steps | Mapped — 26 constraints extracted |
| Reservoir (ESN) | 7 | Rank-1 collapse solved (sparse: rank=251), no useful computation yet | Characterized |
| Graph | 5 | LS20 Level 1 at 25738 steps (3/10 seeds at 50K) | First non-codebook to navigate |

Four Phase 2a candidates (SelfRef, TapeMachine, ExprSubstrate, TemporalPrediction) were killed. See [CONSTRAINTS.md](CONSTRAINTS.md).

### Key findings

**Classification is supervised (Step 432).** The codebook's 94.48% depends entirely on external labels. Self-generated labels: 9.8% (below chance). The self-labeling mechanism compounds errors through softmax voting (U26).

**Navigation hits a wall (Step 428).** Action-score convergence makes the codebook a random walk after ~5K steps. No scoring modification fixes this — 60+ experiments tried (U25). Navigation speed is determined by encoding quality, not scoring formula (300x between encodings).

**Cross-domain survival (Step 433).** P-MNIST knowledge survives LS20 exposure with 0.0pp contamination. The codebook partitions by domain geometry. One-directional: existing knowledge suppresses new-domain exploration.

**A graph navigates without scores (Step 442b).** Nodes as observation landmarks, edges as transition counts, actions from least-visited edges. First architecture to navigate LS20 without codebook machinery. Systematic exploration by construction — no score convergence possible.

### The constraint map

26 universal constraints extracted from experimental failure define what ANY substrate must satisfy. 7 are provisional (codebook-only evidence). See [CONSTRAINTS.md](CONSTRAINTS.md) for the full map.

Each constraint is a closed door. The pattern of elimination IS the search.

---

## Current Direction (Phase 2b)

1. **Graph without codebook DNA** — The current graph still uses cosine-matched nodes (codebook loophole). Next: quantization or LSH-based matching. The relational structure (edges) is the contribution; the spatial mechanism must be genuinely new.

2. **Reservoir family** — Sparse Hebbian solved rank collapse. Open question: how to get useful computation from a self-modifying recurrent network under R1. 7 experiments done, hundreds needed.

3. **Impossibility direction** — 445 experiments of systematic failure. Can U24 + U25 + U26 be formalized into a proof that the feasible region is empty?

---

## Running It

```bash
pip install torch torchvision numpy

# Phase 1: the LVQ baseline
python experiments/run_step250_complete_substrate.py       # Complete demo (~30s)
python experiments/foldcore-steps/run_step99_topk_vote.py  # P-MNIST benchmark

# Phase 2 experiments
python experiments/run_step442_graph_substrate.py           # Graph substrate
python experiments/run_step441_sparse_reservoir.py          # Sparse reservoir
python experiments/run_step432_labeled_vs_self.py           # Label dependency test
```

---

## Repository Structure

```
the-search/
  CONSTITUTION.md        -- 5 principles + 6 rules (R1-R6)
  CONSTRAINTS.md         -- U1-U26, I1-I9, S1-S21
  RESEARCH_STATE.md      -- Full experiment log
  INDEX.md               -- File-by-file index

  substrates/            -- All substrate implementations
    foldcore/            -- Codebook baseline (LVQ + GNG)
    topk-fold/           -- Phase 1 peak (94.48% P-MNIST)
    living-seed/         -- Phase 1: Sessions 1-17 [closed]
    anima/               -- Phase 1: Sessions 18-23 [closed]
    eigenfold/           -- Phase 1: Matrix codebook [closed]
    selfref/             -- Phase 2a: Self-referential [killed]
    tape/                -- Phase 2a: Integer tape [killed]
    expr/                -- Phase 2a: Expression tree [killed]
    temporal/            -- Phase 2a: Temporal prediction [killed]

  experiments/           -- 445 experiment scripts
  knowledge/             -- Structured knowledge base
  audits/                -- External audit reports
  paper/                 -- Paper compiler
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
