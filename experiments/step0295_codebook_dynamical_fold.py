#!/usr/bin/env python3
"""
Step 295 -- Dynamical System Fold: basin sculpting on a%b.

Spec. New frame: codebook = dynamical system. Classification = trajectory
to attractor. Training = basin sculpting via stepping-stone spawning.

Phase 1: Diagnostic — raw flow field before sculpting (LOO chain following).
Phase 2: Basin sculpting — 5 epochs of stepping-stone spawning.

Kill criterion: chain accuracy after 5 epochs <= 1-step NN accuracy.
Success: chain accuracy > 1-step NN AND same-class cycle fraction improves.
"""

import math
import time
import numpy as np
from collections import Counter

# ─── Config ────────────────────────────────────────────────────────────────────

TRAIN_MAX   = 20
MAX_CHAIN   = 10       # max NN chain steps
N_EPOCHS    = 5
LR_ATTRACT  = 0.005   # small lr for attractor reinforcement
ANNEAL_EPOCH = 3      # epoch at which stepping-stone annealing kicks in

# ─── Encoding: one-hot (same as Step 294 / Step 286) ─────────────────────────

def encode_onehot(a, b, max_val=TRAIN_MAX):
    v = np.zeros(2 * max_val, dtype=np.float32)
    v[a - 1] = 1.0
    v[max_val + b - 1] = 1.0
    n = np.linalg.norm(v)
    return v / n if n > 1e-15 else v

# ─── Dataset ──────────────────────────────────────────────────────────────────

def build_dataset():
    X, Y = [], []
    for a in range(1, TRAIN_MAX + 1):
        for b in range(1, TRAIN_MAX + 1):
            X.append(encode_onehot(a, b))
            Y.append(a % b)
    return np.array(X, dtype=np.float32), np.array(Y, dtype=np.int32)

# ─── NN Chain ─────────────────────────────────────────────────────────────────

def follow_chain(query, V, labels, exclude_idx=None, max_steps=MAX_CHAIN):
    """
    Follow NN chain from query through codebook V.
    Returns (chain_labels, cycle_labels, outcome) where outcome is
    'same_class_cycle', 'mixed_class_cycle', 'max_iter', or 'single'.
    """
    n = len(V)
    if n == 0:
        return [], [], 'empty'

    visited_idxs = []
    visited_labels = []

    # First step: NN of query
    sims = V @ query
    if exclude_idx is not None:
        sims[exclude_idx] = -2.0
    current_idx = int(np.argmax(sims))
    visited_idxs.append(current_idx)
    visited_labels.append(labels[current_idx])

    for _ in range(max_steps - 1):
        sims = V @ V[current_idx]
        sims[current_idx] = -2.0  # exclude self
        if exclude_idx is not None:
            sims[exclude_idx] = -2.0
        nn_idx = int(np.argmax(sims))

        if nn_idx in visited_idxs:
            # Cycle detected
            cycle_start = visited_idxs.index(nn_idx)
            cycle_labels = visited_labels[cycle_start:]
            unique_in_cycle = set(cycle_labels)
            if len(unique_in_cycle) == 1:
                return visited_labels, cycle_labels, 'same_class_cycle'
            else:
                return visited_labels, cycle_labels, 'mixed_class_cycle'

        visited_idxs.append(nn_idx)
        visited_labels.append(labels[nn_idx])
        current_idx = nn_idx

    # Max iterations
    return visited_labels, [], 'max_iter'


def classify_chain(chain_labels, cycle_labels, outcome):
    """Classify input based on chain result."""
    if outcome == 'same_class_cycle':
        return cycle_labels[0]
    elif outcome == 'mixed_class_cycle':
        return Counter(cycle_labels).most_common(1)[0][0]
    else:  # max_iter or empty
        if chain_labels:
            return Counter(chain_labels).most_common(1)[0][0]
        return -1


# ─── Evaluation ───────────────────────────────────────────────────────────────

def evaluate(V, labels, X, Y, mode='chain'):
    """
    mode='chain': LOO chain-based classification
    mode='1nn': simple 1-NN (LOO)
    """
    V_arr = np.array(V)
    n_cb = len(V)
    correct = 0
    outcomes = Counter()

    for i, (x, y) in enumerate(zip(X, Y)):
        # LOO: exclude index i if x is in codebook
        exclude = i if i < n_cb else None

        if mode == 'chain':
            chain_labels, cycle_labels, outcome = follow_chain(
                x, V_arr, labels, exclude_idx=exclude)
            pred = classify_chain(chain_labels, cycle_labels, outcome)
            outcomes[outcome] += 1
        else:  # 1-NN
            sims = V_arr @ x
            if exclude is not None:
                sims[exclude] = -2.0
            nn_idx = int(np.argmax(sims))
            pred = labels[nn_idx]
            outcome = '1nn'
            outcomes[outcome] += 1

        if pred == y:
            correct += 1

    acc = correct / len(X)
    return acc, outcomes


# ─── Basin sculpting ──────────────────────────────────────────────────────────

def sculpt_epoch(V, labels, X, Y, epoch, n_orig):
    """Run one epoch of basin sculpting."""
    V_arr = np.array(V)
    n_spawn = 0
    n_correct = 0

    for i, (x, y) in enumerate(zip(X, Y)):
        # Run chain from x (x IS in codebook at index i if i < len(V))
        exclude = None  # don't exclude self — x is in its own neighborhood
        chain_labels, cycle_labels, outcome = follow_chain(
            x, V_arr, labels, exclude_idx=None, max_steps=MAX_CHAIN)
        pred = classify_chain(chain_labels, cycle_labels, outcome)

        if pred == y:
            n_correct += 1
            # Reinforce attractor (optional small lr)
            if outcome == 'same_class_cycle' and cycle_labels and cycle_labels[0] == y:
                pass  # do nothing (per spec "or move attractor with lr=0.005")
        else:
            # WRONG: spawn stepping stone
            # Find nearest same-class vector to x
            same_class_mask = np.array(labels) == y
            if not same_class_mask.any():
                continue

            # Annealing: in epochs >= ANNEAL_EPOCH, only spawn if still wrong
            if epoch >= ANNEAL_EPOCH:
                # Already running chain; pred != y, so spawn is warranted
                pass

            sims = V_arr @ x
            sims[~same_class_mask] = -2.0
            v_correct_idx = int(np.argmax(sims))
            v_correct = V_arr[v_correct_idx]

            # Insert midpoint stepping stone
            midpoint = x + v_correct
            n = np.linalg.norm(midpoint)
            if n > 1e-15:
                midpoint = midpoint / n
            V.append(midpoint.tolist())
            labels.append(y)
            V_arr = np.array(V)  # update array
            n_spawn += 1

    return n_spawn, n_correct / len(X)


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    t0 = time.time()
    print("Step 295 -- Dynamical System Fold on a%b", flush=True)
    print(f"Dataset: a,b in 1..{TRAIN_MAX}, {TRAIN_MAX**2} pairs", flush=True)
    print(f"Encoding: one-hot ({2*TRAIN_MAX}d)", flush=True)
    print(f"Chain max steps: {MAX_CHAIN}, Epochs: {N_EPOCHS}", flush=True)
    print(f"Anneal from epoch: {ANNEAL_EPOCH}", flush=True)
    print(flush=True)

    X, Y = build_dataset()
    n_orig = len(X)

    # Initialize codebook with ALL training examples
    V = [x.tolist() for x in X]
    labels = list(Y)
    print(f"Codebook initialized: {len(V)} vectors (all training examples)", flush=True)
    print(flush=True)

    # ─── Phase 1: Diagnostic ─────────────────────────────────────────────────
    print("=== Phase 1: Diagnostic (raw flow field, LOO) ===", flush=True)
    t1 = time.time()

    acc_chain, outcomes_chain = evaluate(V, labels, X, Y, mode='chain')
    acc_1nn,   outcomes_1nn   = evaluate(V, labels, X, Y, mode='1nn')

    total = len(X)
    print(f"Chain-based LOO accuracy: {acc_chain*100:.1f}%", flush=True)
    print(f"1-NN LOO accuracy:        {acc_1nn*100:.1f}%", flush=True)
    print(f"Chain delta over 1-NN:    {(acc_chain - acc_1nn)*100:+.1f}pp", flush=True)
    print(f"Chain outcomes:", flush=True)
    for k, v in sorted(outcomes_chain.items()):
        print(f"  {k}: {v} ({v/total*100:.1f}%)", flush=True)
    print(f"Phase 1 time: {time.time()-t1:.1f}s", flush=True)
    print(flush=True)

    # ─── Phase 2: Basin sculpting ─────────────────────────────────────────────
    print("=== Phase 2: Basin Sculpting (5 epochs) ===", flush=True)
    print(f"{'Epoch':>5} | {'CB size':>8} | {'Spawned':>8} | {'Chain acc':>10} | {'1-NN acc':>9} | {'Delta':>7}",
          flush=True)
    print("-" * 62, flush=True)

    for epoch in range(N_EPOCHS):
        te = time.time()
        n_spawn, train_acc = sculpt_epoch(V, labels, X, Y, epoch, n_orig)

        # Evaluate after sculpting
        acc_c, outcomes_c = evaluate(V, labels, X, Y, mode='chain')
        acc_n, _          = evaluate(V, labels, X, Y, mode='1nn')

        same_class_frac = outcomes_c.get('same_class_cycle', 0) / total
        print(f"  {epoch:>3}  | {len(V):>8} | {n_spawn:>8} | {acc_c*100:>9.1f}% | "
              f"{acc_n*100:>8.1f}% | {(acc_c-acc_n)*100:>+6.1f}pp  "
              f"[same_class_cycle={same_class_frac*100:.0f}%]",
              flush=True)

    # ─── Summary ──────────────────────────────────────────────────────────────
    elapsed = time.time() - t0
    final_acc_chain, final_outcomes = evaluate(V, labels, X, Y, mode='chain')
    final_acc_1nn,   _              = evaluate(V, labels, X, Y, mode='1nn')

    print(flush=True)
    print("=" * 70, flush=True)
    print("STEP 295 SUMMARY", flush=True)
    print("=" * 70, flush=True)
    print(f"Phase 1 baseline:   chain={acc_chain*100:.1f}%, 1-NN={acc_1nn*100:.1f}%", flush=True)
    print(f"Final (5 epochs):   chain={final_acc_chain*100:.1f}%, 1-NN={final_acc_1nn*100:.1f}%",
          flush=True)
    print(f"Codebook size: {len(V)} (started: {n_orig}, added: {len(V)-n_orig})", flush=True)
    print(f"Step 286 baseline: 41.8% (plain thermometer LOO)", flush=True)
    print(flush=True)

    # Cycle fraction trend
    same_final = final_outcomes.get('same_class_cycle', 0) / total
    same_phase1 = outcomes_chain.get('same_class_cycle', 0) / total
    print(f"Same-class cycle fraction: Phase1={same_phase1*100:.1f}% → Final={same_final*100:.1f}%",
          flush=True)
    print(flush=True)

    print("KILL CRITERION (Spec):", flush=True)
    if final_acc_chain <= final_acc_1nn:
        print(f"  KILLED — chain ({final_acc_chain*100:.1f}%) <= 1-NN ({final_acc_1nn*100:.1f}%)",
              flush=True)
        print(f"  Flow field adds no information beyond nearest-neighbor.", flush=True)
    else:
        delta_chain = final_acc_chain - final_acc_1nn
        delta_baseline = final_acc_chain - 0.418
        cycle_improves = same_final > same_phase1
        print(f"  PASSES — chain ({final_acc_chain*100:.1f}%) > 1-NN ({final_acc_1nn*100:.1f}%) "
              f"by {delta_chain*100:+.1f}pp", flush=True)
        print(f"  vs Step 286 baseline: {delta_baseline*100:+.1f}pp", flush=True)
        if cycle_improves:
            print(f"  Same-class cycle fraction improved: {same_phase1*100:.1f}% → {same_final*100:.1f}%",
                  flush=True)
        else:
            print(f"  WARNING: same-class cycle fraction did NOT improve "
                  f"({same_phase1*100:.1f}% → {same_final*100:.1f}%)", flush=True)

    print(f"\nElapsed: {elapsed:.1f}s", flush=True)

if __name__ == '__main__':
    main()
