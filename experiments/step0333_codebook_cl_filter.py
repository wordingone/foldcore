#!/usr/bin/env python3
"""
Step 333 — Competitive learning as filter discovery on a%b.

Step 332: same-b filter is load-bearing for phi. Removing it destroys phi.
Step 333: can competitive learning DISCOVER the same-b filter on its own?

Algorithm:
1. Run competitive learning on [a,b] inputs -> smaller codebook (~100 entries)
2. Record which inputs each codebook entry wins for
3. Assign each entry a "group" = modal b-value among its won inputs
4. For a query: find winning codebook entry -> get its group
5. phi filter: entries assigned to same group (discovered, not prescribed)
6. LOO accuracy vs prescribed same-b (86.8%)

Kill: discovered filter must achieve >= 80% LOO (within 7pp of prescribed).

Key question: competitive learning in [a,b] space groups by proximity in
the a×b grid, not necessarily by b-value. Does the substrate discover
b-grouping as a side effect? Or does proximity-based grouping fail?
"""

import numpy as np
import time

# ─── Config ────────────────────────────────────────────────────────────────────

TRAIN_MAX     = 20
K             = 5
SENTINEL      = TRAIN_MAX * 3
SPAWN_THRESHS = [1.5, 2.0, 3.0, 4.0, 5.0]   # L2 distance thresholds for spawning
TARGET_ACC    = 0.80

# ─── Dataset ───────────────────────────────────────────────────────────────────

def build_dataset():
    A, B, Y = [], [], []
    for a in range(1, TRAIN_MAX + 1):
        for b in range(1, TRAIN_MAX + 1):
            A.append(a); B.append(b); Y.append(a % b)
    X = np.column_stack([A, B]).astype(np.float32)
    return X, np.array(A), np.array(B), np.array(Y)


# ─── Competitive learning ──────────────────────────────────────────────────────

def run_competitive_learning(X, spawn_thresh, lr=0.1, n_epochs=3, seed=42):
    """
    Competitive learning in input space.
    Spawns a new codebook entry when distance to nearest winner > spawn_thresh.
    Returns:
      codebook: (n_cb, d) — learned centroids
      cb_winners: list of lists — for each codebook entry, which input indices it won for
    """
    rng = np.random.RandomState(seed)
    n = len(X)
    order = np.arange(n)

    # Init: first entry
    codebook = [X[0].copy()]
    n_cb = 1

    for epoch in range(n_epochs):
        rng.shuffle(order)
        for i in order:
            x = X[i]
            cb = np.array(codebook)
            dists = np.linalg.norm(cb - x, axis=1)
            winner = np.argmin(dists)
            min_dist = dists[winner]
            if min_dist > spawn_thresh:
                codebook.append(x.copy())
                n_cb += 1
            else:
                codebook[winner] = codebook[winner] + lr * (x - codebook[winner])

    codebook = np.array(codebook)

    # Record winners on final pass
    cb_winners = [[] for _ in range(len(codebook))]
    for i in range(n):
        x = X[i]
        dists = np.linalg.norm(codebook - x, axis=1)
        winner = np.argmin(dists)
        cb_winners[winner].append(i)

    return codebook, cb_winners


def assign_groups(cb_winners, B, n_cb):
    """
    Assign each codebook entry to a group = modal b-value among its wins.
    Returns group_labels (n_cb,) int array.
    """
    group_labels = np.zeros(n_cb, dtype=np.int32)
    for i, winners in enumerate(cb_winners):
        if len(winners) == 0:
            group_labels[i] = -1  # unassigned
        else:
            b_vals = B[winners]
            # Modal b-value
            counts = np.bincount(b_vals - 1, minlength=TRAIN_MAX)  # b in 1..20
            group_labels[i] = np.argmax(counts) + 1  # back to 1..20
    return group_labels


def get_query_group(query_x, codebook, group_labels):
    """Find which group a query belongs to via nearest codebook winner."""
    dists = np.linalg.norm(codebook - query_x, axis=1)
    winner = np.argmin(dists)
    return group_labels[winner]


# ─── Phi computation ───────────────────────────────────────────────────────────

def compute_phi_prescribed(query_a, query_b, A, B, Y, exclude_idx, K, max_class):
    """phi with prescribed same-b filter (baseline, from auto_loop.py)."""
    phi = np.full(max_class * K, float(SENTINEL), dtype=np.float32)
    same_b_mask = (B == query_b)
    for c in range(max_class):
        class_mask = (Y == c) & same_b_mask
        if exclude_idx is not None and class_mask[exclude_idx]:
            if Y[exclude_idx] == c:
                class_mask = class_mask.copy()
                class_mask[exclude_idx] = False
        idxs = np.where(class_mask)[0]
        if len(idxs) == 0:
            continue
        dists = np.abs(A[idxs] - query_a).astype(np.float32)
        dists.sort()
        k_eff = min(K, len(dists))
        phi[c * K: c * K + k_eff] = dists[:k_eff]
    return phi


def compute_phi_discovered(query_a, query_b, query_group, A, B, Y, group_map,
                           exclude_idx, K, max_class):
    """
    phi with discovered filter: use only entries in the same group as query.
    group_map: for each entry index, its discovered group.
    """
    phi = np.full(max_class * K, float(SENTINEL), dtype=np.float32)
    same_group_mask = (group_map == query_group)
    for c in range(max_class):
        class_mask = (Y == c) & same_group_mask
        if exclude_idx is not None and class_mask[exclude_idx]:
            if Y[exclude_idx] == c:
                class_mask = class_mask.copy()
                class_mask[exclude_idx] = False
        idxs = np.where(class_mask)[0]
        if len(idxs) == 0:
            continue
        dists = np.abs(A[idxs] - query_a).astype(np.float32)
        dists.sort()
        k_eff = min(K, len(dists))
        phi[c * K: c * K + k_eff] = dists[:k_eff]
    return phi


# ─── LOO evaluation ────────────────────────────────────────────────────────────

def loo_prescribed(A, B, Y, K, max_class):
    """Baseline: LOO phi with prescribed same-b filter."""
    n = len(A)
    all_phi = np.zeros((n, max_class * K), dtype=np.float32)
    for i in range(n):
        all_phi[i] = compute_phi_prescribed(A[i], B[i], A, B, Y, i, K, max_class)

    correct = 0
    for i in range(n):
        phi_q = all_phi[i]
        diffs = all_phi - phi_q
        dists = (diffs ** 2).sum(axis=1)
        dists[i] = float('inf')
        best_j = np.argmin(dists)
        if Y[best_j] == Y[i]:
            correct += 1
    return correct / n, all_phi


def loo_discovered(X, A, B, Y, codebook, group_labels, K, max_class, n_entries=400):
    """
    LOO phi with discovered filter.
    Each query's group is determined by nearest codebook winner.
    """
    n = len(A)

    # Map each entry to its group (via competitive learning winner)
    entry_groups = np.zeros(n, dtype=np.int32)
    for i in range(n):
        entry_groups[i] = get_query_group(X[i], codebook, group_labels)

    # Build phi for all entries using discovered filter
    all_phi = np.zeros((n, max_class * K), dtype=np.float32)
    for i in range(n):
        all_phi[i] = compute_phi_discovered(
            A[i], B[i], entry_groups[i], A, B, Y, entry_groups, i, K, max_class)

    # LOO accuracy
    correct = 0
    for i in range(n):
        phi_q = all_phi[i]
        diffs = all_phi - phi_q
        dists = (diffs ** 2).sum(axis=1)
        dists[i] = float('inf')
        best_j = np.argmin(dists)
        if Y[best_j] == Y[i]:
            correct += 1

    # Analysis: how aligned is discovered grouping with b-values?
    unique_groups = np.unique(entry_groups)
    b_alignment = []
    for g in unique_groups:
        mask = entry_groups == g
        if mask.sum() < 2:
            continue
        b_in_group = B[mask]
        modal_b = np.bincount(b_in_group - 1, minlength=TRAIN_MAX).argmax() + 1
        purity = (b_in_group == modal_b).mean()
        b_alignment.append(purity)

    return correct / n, entry_groups, all_phi, b_alignment


# ─── Main ──────────────────────────────────────────────────────────────────────

def main():
    t0 = time.time()
    np.random.seed(42)

    print("Step 333 — Competitive learning as filter discovery on a%b", flush=True)
    print(f"K={K}, spawn thresholds: {SPAWN_THRESHS}", flush=True)
    print(f"Kill: discovered filter >= {TARGET_ACC*100:.0f}% LOO", flush=True)
    print(f"Baseline (prescribed same-b): 86.8%", flush=True)
    print(flush=True)

    X, A, B, Y = build_dataset()
    n = len(A)
    max_class = int(Y.max()) + 1
    print(f"Dataset: {n} entries, max_class={max_class}", flush=True)
    print(flush=True)

    # Baseline: prescribed same-b filter
    print("Computing baseline (prescribed same-b)...", flush=True)
    acc_prescribed, _ = loo_prescribed(A, B, Y, K, max_class)
    print(f"Prescribed same-b LOO: {acc_prescribed*100:.2f}%", flush=True)
    print(f"Elapsed: {time.time()-t0:.2f}s", flush=True)
    print(flush=True)

    # Sweep spawn thresholds
    results = {}
    print("=" * 70, flush=True)
    print("Competitive learning sweep", flush=True)
    print("=" * 70, flush=True)

    for sp in SPAWN_THRESHS:
        print(f"\n--- spawn_thresh={sp} ---", flush=True)
        codebook, cb_winners = run_competitive_learning(X, sp)
        n_cb = len(codebook)
        group_labels = assign_groups(cb_winners, B, n_cb)

        # Codebook stats
        winner_sizes = [len(w) for w in cb_winners]
        print(f"  Codebook size: {n_cb}", flush=True)
        print(f"  Winners per entry: min={min(winner_sizes)} max={max(winner_sizes)} "
              f"mean={np.mean(winner_sizes):.1f}", flush=True)
        empty = sum(1 for s in winner_sizes if s == 0)
        print(f"  Entries with 0 wins: {empty}", flush=True)

        # Group composition
        unique_groups, group_counts = np.unique(group_labels[group_labels >= 0],
                                                 return_counts=True)
        print(f"  Unique discovered groups: {len(unique_groups)} "
              f"(expected: {TRAIN_MAX} b-values)", flush=True)

        acc_disc, entry_groups, _, b_alignment = loo_discovered(
            X, A, B, Y, codebook, group_labels, K, max_class)

        # Alignment with b-values
        avg_purity = np.mean(b_alignment) if b_alignment else 0.0
        print(f"  Avg group b-purity: {avg_purity*100:.1f}%  "
              f"(1.0 = groups perfectly align with b-values)", flush=True)

        # Group alignment: what % of same-group pairs share the same b?
        same_b_in_group = 0
        total_pairs = 0
        unique_g = np.unique(entry_groups)
        for g in unique_g:
            mask = entry_groups == g
            b_g = B[mask]
            n_g = mask.sum()
            if n_g > 1:
                for i2 in range(n_g):
                    for j2 in range(i2+1, n_g):
                        total_pairs += 1
                        if b_g[i2] == b_g[j2]:
                            same_b_in_group += 1
        b_group_purity = same_b_in_group / total_pairs if total_pairs > 0 else 0.0
        print(f"  Same-b pair purity in groups: {b_group_purity*100:.1f}%", flush=True)

        delta = acc_disc - acc_prescribed
        print(f"  Discovered filter LOO: {acc_disc*100:.2f}%  "
              f"(delta vs prescribed: {delta*100:+.2f}pp)", flush=True)

        results[sp] = {
            'acc': acc_disc,
            'n_cb': n_cb,
            'avg_purity': avg_purity,
            'b_group_purity': b_group_purity,
        }
        print(f"  Elapsed: {time.time()-t0:.2f}s", flush=True)

    # Summary
    elapsed = time.time() - t0
    print(flush=True)
    print("=" * 70, flush=True)
    print("STEP 333 SUMMARY", flush=True)
    print("=" * 70, flush=True)
    print(f"Baseline (prescribed same-b): {acc_prescribed*100:.2f}%", flush=True)
    print(flush=True)
    print(f"{'thresh':>8} {'n_cb':>6} {'LOO%':>8} {'delta':>8} "
          f"{'b-purity':>10} {'pair-purity':>12}", flush=True)
    print("-" * 60, flush=True)
    for sp, r in results.items():
        delta = (r['acc'] - acc_prescribed) * 100
        print(f"{sp:>8.1f} {r['n_cb']:>6d} {r['acc']*100:>7.2f}% {delta:>+7.2f}pp "
              f"{r['avg_purity']*100:>9.1f}% {r['b_group_purity']*100:>11.1f}%", flush=True)

    best_sp  = max(results, key=lambda k: results[k]['acc'])
    best_acc = results[best_sp]['acc']
    kill     = best_acc < TARGET_ACC
    success  = best_acc >= TARGET_ACC

    print(flush=True)
    print(f"Best: spawn_thresh={best_sp}, LOO={best_acc*100:.2f}%", flush=True)

    print(flush=True)
    print("=" * 70, flush=True)
    print("KILL CHECK", flush=True)
    print("=" * 70, flush=True)
    print(f"Target: >= {TARGET_ACC*100:.0f}%  (within 7pp of prescribed {acc_prescribed*100:.2f}%)",
          flush=True)
    print(f"Best discovered: {best_acc*100:.2f}% (spawn_thresh={best_sp})", flush=True)
    print(f"Kill (< {TARGET_ACC*100:.0f}%): {'TRIGGERED' if kill else 'not triggered'}", flush=True)
    print(f"Success (>= {TARGET_ACC*100:.0f}%): {'YES' if success else 'NO'}", flush=True)

    if kill:
        print("\nKILLED — competitive learning does not discover same-b grouping", flush=True)
    else:
        delta = (best_acc - acc_prescribed) * 100
        print(f"\nSUCCESS — discovered filter achieves {best_acc*100:.2f}% "
              f"({delta:+.2f}pp vs prescribed)", flush=True)

    print(f"\nElapsed: {elapsed:.2f}s", flush=True)


if __name__ == '__main__':
    main()
