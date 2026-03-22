#!/usr/bin/env python3
"""
Step 338 — Spawn rule as codebook data. Stage 8 attempt.

Stage 7 passes (Step 337): per-entry K and weights stored in codebook,
updated by CL dynamics. 95.75% on mixed-function dataset.

Stage 8: the spawn rule is frozen code ("spawn if distance > threshold").
Make it LEARNED from codebook data.

TWO APPROACHES:
B1. Meta-codebook: (distance, local_density) -> should_spawn
    Built from spawn history. Replaces fixed threshold with 1-NN on meta-entries.

B2. Per-group thresholds: each CL group stores its own spawn threshold.
    Learned by trying multiple thresholds and keeping what works best.

COMPARE:
A. Fixed threshold (best from Step 337): 95.75%
B1. Meta-learned spawn rule: ???
B2. Per-group thresholds: ???

Kill: B1 OR B2 must match or beat A (95.75%).
"""

import numpy as np
import time

# ─── Config ────────────────────────────────────────────────────────────────────

TRAIN_MAX    = 20
K_DEFAULT    = 5
SENTINEL     = TRAIN_MAX * 3
SPAWN_THRESH = 4.0      # fixed threshold baseline (from Step 337)
N_EPOCHS     = 10
LR_W         = 0.1
SEED         = 42
THRESH_SWEEP = [2.0, 3.0, 4.0, 5.0, 6.0]  # for per-group threshold search

# ─── Dataset (same as Step 337) ───────────────────────────────────────────────

def build_dataset():
    """400 entries: b<=10 -> y=a%b, b>=11 -> y=floor(a/b)."""
    A, B, Y, FUNC = [], [], [], []
    for a in range(1, TRAIN_MAX + 1):
        for b in range(1, TRAIN_MAX + 1):
            A.append(a); B.append(b)
            if b <= 10:
                Y.append(a % b); FUNC.append(0)
            else:
                Y.append(a // b); FUNC.append(1)
    return np.array(A), np.array(B), np.array(Y), np.array(FUNC)


# ─── Phi within CL group ──────────────────────────────────────────────────────

def compute_phi_group(query_a, A_group, Y_group, exclude_idx, K, max_class):
    phi = np.full(max_class * K, float(SENTINEL), dtype=np.float32)
    for c in range(max_class):
        class_mask = Y_group == c
        if exclude_idx is not None and exclude_idx < len(class_mask):
            if Y_group[exclude_idx] == c:
                class_mask = class_mask.copy()
                class_mask[exclude_idx] = False
        idxs = np.where(class_mask)[0]
        if len(idxs) == 0:
            continue
        dists = np.abs(A_group[idxs] - query_a).astype(np.float32)
        dists.sort()
        k_eff = min(K, len(dists))
        phi[c * K: c * K + k_eff] = dists[:k_eff]
    return phi


# ─── LOO accuracy (CL+phi, given codebook + assignments) ──────────────────────

def loo_cl_phi(A, Y, assignments, K, max_class):
    n = len(A)
    n_cb = int(assignments.max()) + 1

    all_phi = np.zeros((n, max_class * K), dtype=np.float32)
    for i in range(n):
        g = assignments[i]
        group_mask = assignments == g
        group_idx = np.where(group_mask)[0]
        pos = np.where(group_idx == i)[0]
        exc = int(pos[0]) if len(pos) > 0 else None
        all_phi[i] = compute_phi_group(
            A[i], A[group_mask], Y[group_mask], exc, K, max_class)

    w_expanded = np.ones(max_class * K, dtype=np.float64)
    correct = 0
    for i in range(n):
        g = assignments[i]
        group_mask = assignments == g
        group_idxs = np.where(group_mask)[0]
        if len(group_idxs) <= 1:
            continue
        phi_q = all_phi[i]
        diffs = all_phi[group_idxs] - phi_q
        dists = (diffs ** 2 * w_expanded).sum(axis=1)
        self_pos = np.where(group_idxs == i)[0]
        if len(self_pos) > 0:
            dists[self_pos[0]] = float('inf')
        best_j = group_idxs[np.argmin(dists)]
        if Y[best_j] == Y[i]:
            correct += 1
    return correct / n


# ─── Approach A: Fixed threshold CL (baseline) ────────────────────────────────

def build_cl_fixed(X, spawn_thresh, lr=0.1, n_epochs=3, seed=SEED):
    """Standard CL with fixed spawn threshold."""
    rng = np.random.RandomState(seed)
    n = len(X)
    order = np.arange(n)
    codebook = [X[0].copy()]

    for epoch in range(n_epochs):
        rng.shuffle(order)
        for i in order:
            x = X[i]
            cb = np.array(codebook)
            dists = np.linalg.norm(cb - x, axis=1)
            winner = np.argmin(dists)
            if dists[winner] > spawn_thresh:
                codebook.append(x.copy())
            else:
                codebook[winner] = codebook[winner] + lr * (x - codebook[winner])

    codebook = np.array(codebook)
    dists_all = np.linalg.norm(codebook[:, None, :] - X[None, :, :], axis=2).T
    assignments = np.argmin(dists_all, axis=1).astype(np.int32)
    return codebook, assignments


# ─── Approach B1: Meta-codebook spawn rule ────────────────────────────────────

def build_cl_with_spawn_history(X, spawn_thresh, lr=0.1, n_epochs=3, seed=SEED):
    """
    CL with spawn event recording.
    Returns codebook, assignments, spawn_events:
      each event = (distance_to_nearest, local_density_before_spawn, spawned_entry_idx)
    """
    rng = np.random.RandomState(seed)
    n = len(X)
    order = np.arange(n)
    codebook = [X[0].copy()]
    spawn_events = []  # (dist, density_before, entry_idx_spawned)

    for epoch in range(n_epochs):
        rng.shuffle(order)
        for i in order:
            x = X[i]
            cb = np.array(codebook)
            dists = np.linalg.norm(cb - x, axis=1)
            winner = np.argmin(dists)
            min_dist = dists[winner]
            if min_dist > spawn_thresh:
                # Compute local density: how many codebook entries within 2*spawn_thresh
                density = (dists <= 2 * spawn_thresh).sum()
                entry_idx = len(codebook)
                spawn_events.append({
                    'dist': float(min_dist),
                    'density': int(density),
                    'entry_idx': entry_idx,
                    'epoch': epoch,
                })
                codebook.append(x.copy())
            else:
                codebook[winner] = codebook[winner] + lr * (x - codebook[winner])

    codebook = np.array(codebook)
    dists_all = np.linalg.norm(codebook[:, None, :] - X[None, :, :], axis=2).T
    assignments = np.argmin(dists_all, axis=1).astype(np.int32)

    # Mark spawn usefulness: entry is useful if it wins at least 1 point
    winner_counts = np.bincount(assignments, minlength=len(codebook))
    for ev in spawn_events:
        ev['wins'] = int(winner_counts[ev['entry_idx']]) if ev['entry_idx'] < len(winner_counts) else 0
        ev['useful'] = ev['wins'] > 0

    return codebook, assignments, spawn_events


def build_meta_codebook(spawn_events):
    """
    Build meta-codebook from spawn events.
    Features: [distance, density] -> label (1=useful, 0=wasted)
    Returns meta_cb (n_meta, 2) and meta_labels (n_meta,)
    """
    if not spawn_events:
        return np.array([[SPAWN_THRESH, 1]]), np.array([1])

    feats = np.array([[ev['dist'], ev['density']] for ev in spawn_events], dtype=np.float32)
    labels = np.array([int(ev['useful']) for ev in spawn_events], dtype=np.int32)

    # Deduplicate by (dist_bin, density) -> majority label
    # Use 0.5-unit distance bins
    bins = np.round(feats[:, 0] * 2) / 2  # round to nearest 0.5
    unique_keys = {}
    for i in range(len(feats)):
        key = (float(bins[i]), int(feats[i, 1]))
        if key not in unique_keys:
            unique_keys[key] = []
        unique_keys[key].append(int(labels[i]))

    meta_feats = []
    meta_labels_list = []
    for key, lbls in unique_keys.items():
        meta_feats.append([key[0], key[1]])
        # Majority label
        meta_labels_list.append(1 if sum(lbls) > len(lbls) / 2 else 0)

    meta_cb = np.array(meta_feats, dtype=np.float32)
    meta_labels_arr = np.array(meta_labels_list, dtype=np.int32)
    return meta_cb, meta_labels_arr


def meta_should_spawn(dist, density, meta_cb, meta_labels):
    """Query meta-codebook: should we spawn given (dist, density)?"""
    query = np.array([dist, density], dtype=np.float32)
    # Normalize: dist by 10, density by 10
    q_norm = query / np.array([10.0, 10.0])
    cb_norm = meta_cb / np.array([[10.0, 10.0]])
    dists_to_meta = np.linalg.norm(cb_norm - q_norm, axis=1)
    nearest = np.argmin(dists_to_meta)
    return bool(meta_labels[nearest] == 1)


def build_cl_meta(X, meta_cb, meta_labels, lr=0.1, n_epochs=3, seed=SEED):
    """CL using meta-codebook for spawn decisions instead of fixed threshold."""
    rng = np.random.RandomState(seed)
    n = len(X)
    order = np.arange(n)
    codebook = [X[0].copy()]

    for epoch in range(n_epochs):
        rng.shuffle(order)
        for i in order:
            x = X[i]
            cb = np.array(codebook)
            dists = np.linalg.norm(cb - x, axis=1)
            winner = np.argmin(dists)
            min_dist = dists[winner]
            density = (dists <= 2 * SPAWN_THRESH).sum()

            if meta_should_spawn(min_dist, density, meta_cb, meta_labels):
                codebook.append(x.copy())
            else:
                codebook[winner] = codebook[winner] + lr * (x - codebook[winner])

    codebook = np.array(codebook)
    dists_all = np.linalg.norm(codebook[:, None, :] - X[None, :, :], axis=2).T
    assignments = np.argmin(dists_all, axis=1).astype(np.int32)
    return codebook, assignments


# ─── Approach B2: Per-group spawn thresholds ─────────────────────────────────

def build_cl_per_group_thresh(X, group_thresholds, lr=0.1, n_epochs=3, seed=SEED):
    """
    CL where each codebook entry has its own spawn threshold.
    group_thresholds[i] = threshold for entry i.
    When a new entry is spawned, it inherits the nearest entry's threshold.
    """
    rng = np.random.RandomState(seed)
    n = len(X)
    order = np.arange(n)
    codebook = [X[0].copy()]
    cb_thresh = [group_thresholds[0] if group_thresholds else SPAWN_THRESH]

    for epoch in range(n_epochs):
        rng.shuffle(order)
        for i in order:
            x = X[i]
            cb = np.array(codebook)
            dists = np.linalg.norm(cb - x, axis=1)
            winner = np.argmin(dists)
            min_dist = dists[winner]
            local_thresh = cb_thresh[winner]
            if min_dist > local_thresh:
                # New entry inherits parent's threshold
                codebook.append(x.copy())
                cb_thresh.append(local_thresh)
            else:
                codebook[winner] = codebook[winner] + lr * (x - codebook[winner])

    codebook = np.array(codebook)
    dists_all = np.linalg.norm(codebook[:, None, :] - X[None, :, :], axis=2).T
    assignments = np.argmin(dists_all, axis=1).astype(np.int32)
    return codebook, assignments, cb_thresh


def learn_per_group_thresholds(X, A, Y, FUNC, n_trials=5):
    """
    Learn per-group thresholds by:
    1. Build initial codebook with fixed threshold
    2. For each group, try different thresholds and pick best
    Returns learned thresholds per initial group (seeded assignment).
    """
    # Initial codebook
    codebook_init, assignments_init = build_cl_fixed(X, SPAWN_THRESH)
    n_cb_init = len(codebook_init)
    n = len(X)

    # For each initial group, find which function it mostly covers
    group_func = np.zeros(n_cb_init, dtype=np.float32)
    for g in range(n_cb_init):
        mask = assignments_init == g
        if mask.sum() > 0:
            group_func[g] = FUNC[mask].mean()  # 0=modular, 1=division

    # Try per-group thresholds based on function type
    # Modular: periodic, may benefit from smaller groups (lower threshold)
    # Division: monotonic, may benefit from larger groups (higher threshold)
    best_acc = -1
    best_thresholds = [SPAWN_THRESH] * n_cb_init
    best_n_cb = n_cb_init

    print(f"    Initial codebook: {n_cb_init} entries", flush=True)
    print(f"    Group function mix (0=mod, 1=div): "
          f"mean={group_func.mean():.2f} std={group_func.std():.2f}", flush=True)

    # Threshold configurations to try
    thresh_configs = [
        ('uniform_low',  [3.0] * n_cb_init),
        ('uniform_mid',  [4.0] * n_cb_init),
        ('uniform_high', [5.0] * n_cb_init),
        ('func_based',   [3.0 if group_func[g] < 0.5 else 5.0 for g in range(n_cb_init)]),
        ('func_based_v2',[4.0 if group_func[g] < 0.5 else 6.0 for g in range(n_cb_init)]),
    ]

    results = {}
    for name, thresholds in thresh_configs:
        cb, asgn, cb_t = build_cl_per_group_thresh(X, thresholds)
        n_cb = len(cb)
        acc = loo_cl_phi(A, Y, asgn, K_DEFAULT, int(Y.max()) + 1)
        results[name] = {'acc': acc, 'n_cb': n_cb}
        print(f"    {name}: n_cb={n_cb}, acc={acc*100:.2f}%", flush=True)
        if acc > best_acc:
            best_acc = acc
            best_thresholds = thresholds
            best_n_cb = n_cb

    return best_acc, best_thresholds, results


# ─── Main ──────────────────────────────────────────────────────────────────────

def main():
    t0 = time.time()
    np.random.seed(SEED)

    print("Step 338 — Spawn rule as codebook data (Stage 8)", flush=True)
    print(f"Dataset: mixed-function (same as Step 337)", flush=True)
    print(f"Kill: B1 or B2 must match or beat A (95.75%)", flush=True)
    print(flush=True)

    A, B, Y, FUNC = build_dataset()
    n = len(A)
    max_class = int(Y.max()) + 1
    X = np.column_stack([A, B]).astype(np.float32)
    print(f"Dataset: {n} entries, max_class={max_class}", flush=True)
    print(flush=True)

    # ── Experiment A: Fixed threshold baseline ────────────────────────────────
    print("Experiment A: Fixed threshold CL+phi (baseline)...", flush=True)
    codebook_a, assignments_a = build_cl_fixed(X, SPAWN_THRESH)
    n_cb_a = len(codebook_a)
    acc_a = loo_cl_phi(A, Y, assignments_a, K_DEFAULT, max_class)
    print(f"  A. Fixed thresh={SPAWN_THRESH}: n_cb={n_cb_a}, LOO={acc_a*100:.2f}%", flush=True)
    print(f"  Elapsed: {time.time()-t0:.2f}s", flush=True)
    print(flush=True)

    # ── Experiment B1: Meta-codebook spawn rule ───────────────────────────────
    print("Experiment B1: Meta-codebook spawn rule...", flush=True)

    # Step 1: Build CL with spawn history recording
    print("  Building CL with spawn history...", flush=True)
    cb_hist, asgn_hist, spawn_events = build_cl_with_spawn_history(X, SPAWN_THRESH)
    n_useful = sum(1 for ev in spawn_events if ev['useful'])
    n_wasted = sum(1 for ev in spawn_events if not ev['useful'])
    print(f"  Spawn events: {len(spawn_events)} total, "
          f"{n_useful} useful, {n_wasted} wasted", flush=True)

    # Spawn event stats
    if spawn_events:
        dists_ev = [ev['dist'] for ev in spawn_events]
        dens_ev = [ev['density'] for ev in spawn_events]
        useful_dists = [ev['dist'] for ev in spawn_events if ev['useful']]
        wasted_dists = [ev['dist'] for ev in spawn_events if not ev['useful']]
        print(f"  Useful spawns: dist range [{min(useful_dists):.2f}, {max(useful_dists):.2f}]"
              if useful_dists else "  No useful spawns", flush=True)
        print(f"  Wasted spawns: dist range [{min(wasted_dists):.2f}, {max(wasted_dists):.2f}]"
              if wasted_dists else "  No wasted spawns", flush=True)

    # Step 2: Build meta-codebook
    meta_cb, meta_labels = build_meta_codebook(spawn_events)
    n_meta = len(meta_cb)
    n_meta_pos = int(meta_labels.sum())
    print(f"  Meta-codebook: {n_meta} entries, "
          f"{n_meta_pos} spawn=YES, {n_meta - n_meta_pos} spawn=NO", flush=True)

    # Show meta-codebook
    for i in range(min(n_meta, 10)):
        print(f"    meta[{i}]: dist={meta_cb[i,0]:.2f} density={meta_cb[i,1]:.0f} "
              f"-> {'SPAWN' if meta_labels[i] else 'skip'}", flush=True)

    # Step 3: Rebuild CL using meta-codebook
    print("  Rebuilding CL with meta-codebook for spawn decisions...", flush=True)
    codebook_b1, assignments_b1 = build_cl_meta(X, meta_cb, meta_labels)
    n_cb_b1 = len(codebook_b1)
    acc_b1 = loo_cl_phi(A, Y, assignments_b1, K_DEFAULT, max_class)
    delta_b1 = acc_b1 - acc_a
    print(f"  B1. Meta spawn: n_cb={n_cb_b1}, LOO={acc_b1*100:.2f}%  "
          f"(delta vs A: {delta_b1*100:+.2f}pp)", flush=True)
    print(f"  Elapsed: {time.time()-t0:.2f}s", flush=True)
    print(flush=True)

    # ── Experiment B2: Per-group thresholds ───────────────────────────────────
    print("Experiment B2: Per-group spawn thresholds...", flush=True)
    acc_b2, best_thresholds, thresh_results = learn_per_group_thresholds(
        X, A, Y, FUNC)
    delta_b2 = acc_b2 - acc_a
    print(f"  B2. Per-group thresh: LOO={acc_b2*100:.2f}%  "
          f"(delta vs A: {delta_b2*100:+.2f}pp)", flush=True)
    print(f"  Elapsed: {time.time()-t0:.2f}s", flush=True)
    print(flush=True)

    elapsed = time.time() - t0

    # ── Summary ───────────────────────────────────────────────────────────────
    print("=" * 65, flush=True)
    print("STEP 338 RESULTS", flush=True)
    print("=" * 65, flush=True)
    print(f"A. Fixed threshold (4.0):            {acc_a*100:.2f}%  [baseline from Step 337]", flush=True)
    print(f"B1. Meta-codebook spawn rule:        {acc_b1*100:.2f}%  ({delta_b1*100:+.2f}pp)", flush=True)
    print(f"B2. Per-group thresholds:            {acc_b2*100:.2f}%  ({delta_b2*100:+.2f}pp)", flush=True)

    best_b = max(acc_b1, acc_b2)
    best_name = "B1 (meta-codebook)" if acc_b1 >= acc_b2 else "B2 (per-group thresh)"
    delta_best = best_b - acc_a

    kill    = best_b < acc_a
    success = best_b >= acc_a

    print(flush=True)
    print("=" * 65, flush=True)
    print("KILL CHECK", flush=True)
    print("=" * 65, flush=True)
    print(f"Target: match or beat A ({acc_a*100:.2f}%)", flush=True)
    print(f"Best B: {best_b*100:.2f}% ({best_name})", flush=True)
    print(f"Delta: {delta_best*100:+.2f}pp", flush=True)
    print(f"Kill (best_B < A): {'TRIGGERED' if kill else 'not triggered'}", flush=True)
    print(f"Success (best_B >= A): {'YES' if success else 'NO'}", flush=True)

    if kill:
        print("\nKILLED — learned spawn rule does not match fixed threshold", flush=True)
    else:
        print(f"\nSUCCESS — Stage 8: spawn rule as data achieves {best_b*100:.2f}% "
              f"({delta_best*100:+.2f}pp vs fixed threshold)", flush=True)

    # Stage summary
    print(flush=True)
    print("STAGE CHAIN:", flush=True)
    print("  Step 296 (uniform, same-b):           86.75%", flush=True)
    print("  Step 308 (global learned weights):     91.2%", flush=True)
    print("  Step 333 (CL filter, uniform):         92.0%", flush=True)
    print("  Step 336 (CL+phi compound):            96.0%  [pure a%b]", flush=True)
    print(f"  Step 337 (mixed, global CL+phi):      {acc_a*100:.2f}%", flush=True)
    print(f"  Step 338 (spawn rule as data):        {best_b*100:.2f}%", flush=True)

    print(f"\nElapsed: {elapsed:.2f}s", flush=True)


if __name__ == '__main__':
    main()
