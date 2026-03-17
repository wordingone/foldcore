# R4 Benchmark: Before/After Self-Test

Per Constitution Rule 4: *"After any self-modification, the system compares performance to before the modification. Improvement on trained tasks with degradation on novel tasks is overfitting, not improvement."*

Current state: NO Phase 2 substrate has a real before/after benchmark.
- TemporalPrediction: prediction error is an implicit test (closest to R4)
- SelfRef: no before/after comparison
- ExprSubstrate: evolution replaces worst with mutated best (score comparison, but degenerate)
- TapeMachine: no before/after comparison

---

## The Test

For any substrate S with state that changes over time:

```
1. Run S for N steps on input stream D_train (warmup)
2. Snapshot state: S_before = copy(S)
3. Evaluate S_before on test set D_test: score_before
4. Run S for K more steps on D_train (modification)
5. Evaluate S on D_test: score_after
6. PASS if score_after >= score_before (no degradation)
7. Repeat at 5 checkpoints: PASS if majority improve
```

### What is D_test?

The test set must be:
- Disjoint from D_train (no overlap in exact observations)
- Same distribution (same clusters, same transitions, same structure)
- Large enough for stable measurement (>100 samples)

For discrimination tests: D_test = fresh samples from the same clusters.
For prediction tests: D_test = held-out transitions from the same distribution.

### What is "score"?

Must be R1-compatible (no external labels):
- **Discrimination**: action consistency within clusters (dominance %)
- **Prediction**: prediction error (lower = better)
- **Action diversity**: number of distinct actions used (higher = better, up to n_actions)

### Protocol per substrate

**TemporalPrediction:**
- D_train: cyclic 4-cluster sequence, 200 rounds warmup
- D_test: 50 fresh samples from each cluster (200 total)
- Score: avg dominance across clusters
- K = 100 rounds (more training)
- Checkpoint at warmup steps: 50, 100, 150, 200, 250

**SelfRef:**
- D_train: cyclic 4-cluster sequence, 200 rounds warmup
- D_test: 50 fresh samples from each cluster
- Score: avg dominance
- K = 100 rounds
- Checkpoint at: 50, 100, 150, 200, 250

**ExprSubstrate:**
- D_train: cyclic 4-cluster sequence, 200 rounds warmup
- D_test: 50 fresh samples from each cluster
- Score: avg dominance
- K = 100 rounds
- Checkpoint at: 50, 100, 150, 200, 250

**TapeMachine:**
- Same protocol

### What constitutes passing

- **STRONG PASS**: score_after > score_before at all 5 checkpoints
- **PASS**: score_after > score_before at 3+ of 5 checkpoints
- **MARGINAL**: score_after > score_before at 1-2 of 5 checkpoints
- **FAIL**: score_after <= score_before at 4+ checkpoints (degradation)

### Novel task degradation test

R4 specifically says "degradation on novel tasks is overfitting." After training on
cluster set A, test on cluster set B (different centers, same structure). If
score on B decreased, the substrate overfit to A. This is the forgetting test.

```
1. Train on cluster set A for N steps
2. Evaluate on cluster set B: score_B_before
3. Train on cluster set A for K more steps
4. Evaluate on cluster set B: score_B_after
5. PASS if score_B_after >= score_B_before (no degradation on novel tasks)
```

---

## Why this matters

Without R4, "learning" is indistinguishable from "random drift." The substrate's
state changes, but we don't know if the changes are improvements or noise. R4 is
the difference between adaptation and random walk.
