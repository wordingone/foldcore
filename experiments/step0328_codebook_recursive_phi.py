#!/usr/bin/env python3
"""
Step 328 — Recursive phi (Level 2) on ARC-AGI.

Insight: phi fails on cells with identical patches but different
global context (e.g., interior of large vs small rectangle). Level 2 phi
captures relationships BETWEEN phi_1 vectors.

Algorithm:
  1. Build codebook from training cells (Step 322, 39-dim patch features)
  2. Compute phi_1 for each codebook entry (LOO):
       phi_1[i, c*K:c*K+K] = top-K distances in 39-dim space from cell i
       to class-c codebook entries (excluding self)
  3. For each test cell:
       a. Compute test phi_1: per-class top-K distances to codebook in 39-dim space
       b. Compute phi_2: per-class top-K distances from test_phi_1 to codebook
          phi_1 vectors in 30-dim phi_1 space
       c. class_score[c] = min(phi_2[c*K:c*K+K]) → argmin = prediction

Kill criterion: recover at least 3/5 phi-kill tasks from Step 327.
Phi-kill tasks (1-NN solves, phi L1 kills):
  ce22a75a, a85d4709, a9f96cdd, d4469b4b, dc433765
"""

import json
import numpy as np
import time
import arckit
from scipy.spatial.distance import cdist
from pathlib import Path

MAX_GRID = 30
N_COLORS = 10
PATCH_R = 2   # 5x5
K_PHI = 3
PHI_DIM = N_COLORS * K_PHI  # 30
SENTINEL = 1e6

# Tasks to track specifically
PHI_KILL_TASKS = {'ce22a75a', 'a85d4709', 'a9f96cdd', 'd4469b4b', 'dc433765'}
PHI_WIN_TASKS  = {'2072aba6', '239be575', '694f12f3', 'b60334d2', 'c0f76784'}


# ── Step 322 encoding ─────────────────────────────────────────────────────────

def extract_patch(grid, r, c, radius=PATCH_R, fill=-1):
    h, w = grid.shape
    size = 2 * radius + 1
    patch = np.full((size, size), fill, dtype=np.float32)
    for dr in range(-radius, radius + 1):
        for dc in range(-radius, radius + 1):
            rr, cc = r + dr, c + dc
            if 0 <= rr < h and 0 <= cc < w:
                patch[dr + radius, dc + radius] = grid[rr, cc]
    return patch.flatten()


def color_hist(grid):
    h = np.zeros(N_COLORS, dtype=np.float32)
    for v in grid.flatten():
        if 0 <= int(v) < N_COLORS:
            h[int(v)] += 1
    s = h.sum()
    if s > 0:
        h /= s
    return h


def build_cell_features(inp, out_h, out_w):
    patch_dim = (2 * PATCH_R + 1) ** 2  # 25
    feat_dim = patch_dim + N_COLORS + 4  # 39
    n_cells = out_h * out_w
    feats = np.empty((n_cells, feat_dim), dtype=np.float32)
    hist = color_hist(inp)
    in_h, in_w = inp.shape
    for idx in range(n_cells):
        r, c = divmod(idx, out_w)
        if in_h == out_h and in_w == out_w:
            ir, ic = r, c
        else:
            ir = min(int(r * in_h / out_h), in_h - 1) if out_h > 0 else 0
            ic = min(int(c * in_w / out_w), in_w - 1) if out_w > 0 else 0
        patch = extract_patch(inp, ir, ic)
        feats[idx, :patch_dim] = patch
        feats[idx, patch_dim:patch_dim + N_COLORS] = hist
        feats[idx, -4] = r / max(out_h, 1)
        feats[idx, -3] = c / max(out_w, 1)
        feats[idx, -2] = out_h / MAX_GRID
        feats[idx, -1] = out_w / MAX_GRID
    return feats


def build_codebook(train_examples):
    all_feats, all_labels = [], []
    for inp, out in train_examples:
        feats = build_cell_features(inp, out.shape[0], out.shape[1])
        all_feats.append(feats)
        all_labels.append(out.flatten())
    return np.vstack(all_feats), np.concatenate(all_labels).astype(np.int32)


# ── Phi computation ───────────────────────────────────────────────────────────

def compute_phi1_codebook(cb_feats, cb_labels):
    """
    Compute phi_1 for all codebook entries (LOO excluded).
    phi_1[i, c*K:c*K+K] = top-K distances in 39-dim space from cell i to class-c cells.
    Returns (n_cb, PHI_DIM).
    """
    n = len(cb_feats)
    all_dists = cdist(cb_feats, cb_feats, metric='sqeuclidean')  # (n, n)

    phi1 = np.full((n, PHI_DIM), SENTINEL, dtype=np.float32)
    for c in range(N_COLORS):
        class_idxs = np.where(cb_labels == c)[0]
        if len(class_idxs) == 0:
            continue
        class_dists = all_dists[:, class_idxs]  # (n, n_c)
        for i in range(n):
            row = class_dists[i].copy()
            if cb_labels[i] == c:
                self_pos = np.where(class_idxs == i)[0]
                if len(self_pos) > 0:
                    row[self_pos[0]] = SENTINEL
            sorted_row = np.sort(row)
            valid = sorted_row[sorted_row < SENTINEL]
            k_eff = min(K_PHI, len(valid))
            if k_eff > 0:
                phi1[i, c * K_PHI: c * K_PHI + k_eff] = valid[:k_eff]
    return phi1


def compute_phi1_test(test_feats, cb_feats, cb_labels):
    """
    Compute phi_1 for test cells against codebook (no LOO).
    Returns (n_test, PHI_DIM).
    """
    dists = cdist(test_feats, cb_feats, metric='sqeuclidean')  # (n_test, n_cb)
    n_test = len(test_feats)

    phi1 = np.full((n_test, PHI_DIM), SENTINEL, dtype=np.float32)
    for c in range(N_COLORS):
        class_idxs = np.where(cb_labels == c)[0]
        if len(class_idxs) == 0:
            continue
        class_dists = dists[:, class_idxs]  # (n_test, n_c)
        sorted_class = np.sort(class_dists, axis=1)
        for i in range(n_test):
            valid = sorted_class[i][sorted_class[i] < SENTINEL]
            k_eff = min(K_PHI, len(valid))
            if k_eff > 0:
                phi1[i, c * K_PHI: c * K_PHI + k_eff] = valid[:k_eff]
    return phi1


def compute_phi2_and_predict(test_phi1, cb_phi1, cb_labels):
    """
    Level 2: for each test cell, compute phi_2 in phi_1 space.
    phi_2[c*K:c*K+K] = top-K distances in phi_1 space from test_phi_1 to class-c codebook phi_1.
    Class score = min distance per class. Prediction = argmin class score.
    Returns predicted labels (n_test,).
    """
    n_test = len(test_phi1)
    # Distances in phi_1 space: (n_test, n_cb)
    phi1_dists = cdist(test_phi1, cb_phi1, metric='sqeuclidean')

    preds = np.zeros(n_test, dtype=np.int32)
    for i in range(n_test):
        class_scores = np.full(N_COLORS, SENTINEL, dtype=np.float32)
        for c in range(N_COLORS):
            class_idxs = np.where(cb_labels == c)[0]
            if len(class_idxs) == 0:
                continue
            class_dists = phi1_dists[i, class_idxs]
            class_dists_sorted = np.sort(class_dists)
            k_eff = min(K_PHI, len(class_dists_sorted))
            # Score = sum of top-K distances (lower = more class-similar neighbors)
            class_scores[c] = class_dists_sorted[:k_eff].sum()
        preds[i] = int(np.argmin(class_scores))
    return preds


def infer_output_size(train_examples, test_input):
    out_sizes = [out.shape for _, out in train_examples]
    in_sizes = [inp.shape for inp, _ in train_examples]
    if len(set(out_sizes)) == 1:
        return out_sizes[0]
    if all(o == i for o, i in zip(out_sizes, in_sizes)):
        return test_input.shape
    ratios = set()
    for inp, out in train_examples:
        rh = round(out.shape[0] / inp.shape[0], 4) if inp.shape[0] > 0 else 1
        rw = round(out.shape[1] / inp.shape[1], 4) if inp.shape[1] > 0 else 1
        ratios.add((rh, rw))
    if len(ratios) == 1:
        rh, rw = ratios.pop()
        return (max(1, int(test_input.shape[0] * rh)), max(1, int(test_input.shape[1] * rw)))
    from collections import Counter
    return Counter(out_sizes).most_common(1)[0][0]


def evaluate_task(task):
    """Returns (mode_1nn, pixel_1nn, mode_phi2, pixel_phi2, changed metrics)."""
    cb_feats, cb_labels = build_codebook(task.train)

    # Phi_1 for all codebook entries
    cb_phi1 = compute_phi1_codebook(cb_feats, cb_labels)

    results_1nn, results_phi2 = [], []
    cc_1nn, ct_1nn = 0, 0
    cc_phi2, ct_phi2 = 0, 0

    for test_inp, test_out in task.test:
        out_h, out_w = infer_output_size(task.train, test_inp)
        test_feats = build_cell_features(test_inp, out_h, out_w)

        # Method A: 1-NN in feature space
        feat_dists = cdist(test_feats, cb_feats, metric='sqeuclidean')
        pred_a = cb_labels[np.argmin(feat_dists, axis=1)].reshape(out_h, out_w)

        # Method L2: recursive phi
        test_phi1 = compute_phi1_test(test_feats, cb_feats, cb_labels)
        pred_l2 = compute_phi2_and_predict(test_phi1, cb_phi1, cb_labels).reshape(out_h, out_w)

        def score(pred):
            if pred.shape != test_out.shape:
                return 0.0, False, 0, 0
            match = pred == test_out
            cc, ct = 0, 0
            if test_inp.shape == test_out.shape:
                changed = test_out != test_inp
                ct = int(changed.sum())
                cc = int((match & changed).sum())
            return float(match.mean()), bool(np.array_equal(pred, test_out)), cc, ct

        acc_a, ex_a, cc_a_, ct_a_ = score(pred_a)
        acc_l2, ex_l2, cc_l2_, ct_l2_ = score(pred_l2)

        results_1nn.append((acc_a, ex_a))
        results_phi2.append((acc_l2, ex_l2))
        cc_1nn += cc_a_; ct_1nn += ct_a_
        cc_phi2 += cc_l2_; ct_phi2 += ct_l2_

    def mode(results):
        accs = [r[0] for r in results]
        exacts = [r[1] for r in results]
        if all(exacts): return 'SOLVED'
        avg = np.mean(accs)
        if avg > 0.8: return 'ALMOST'
        if avg > 0.5: return 'PARTIAL'
        if avg > 0.2: return 'WEAK'
        return 'FAIL'

    return {
        'mode_1nn': mode(results_1nn),
        'pixel_1nn': float(np.mean([r[0] for r in results_1nn])),
        'mode_l2': mode(results_phi2),
        'pixel_l2': float(np.mean([r[0] for r in results_phi2])),
        'cc_1nn': cc_1nn, 'ct_1nn': ct_1nn,
        'cc_l2': cc_phi2, 'ct_l2': ct_phi2,
    }


def main():
    t0 = time.time()
    print("Step 328 — Recursive phi (Level 2) on ARC-AGI", flush=True)
    print(f"K_PHI={K_PHI}, PHI_DIM={PHI_DIM}", flush=True)
    print("phi_1: per-class top-K distances in 39-dim patch space", flush=True)
    print("phi_2: per-class top-K distances in 30-dim phi_1 space", flush=True)
    print(f"Kill: recover >=3/5 phi-kill tasks: {PHI_KILL_TASKS}", flush=True)
    print(flush=True)

    tax_path = Path(__file__).parent.parent / 'data' / 'arc_taxonomy.json'
    with open(tax_path) as f:
        taxonomy = json.load(f)
    tax_by_id = {t['id']: t for t in taxonomy}

    train_tasks, _ = arckit.load_data()
    tasks = list(train_tasks)
    print(f"Evaluating {len(tasks)} tasks...", flush=True)

    results = []
    counts_1nn, counts_l2 = {}, {}
    total_cc_1nn, total_ct_1nn = 0, 0
    total_cc_l2, total_ct_l2 = 0, 0

    for i, task in enumerate(tasks):
        try:
            r = evaluate_task(task)
        except Exception as e:
            r = {'mode_1nn': 'ERROR', 'pixel_1nn': 0.0,
                 'mode_l2': 'ERROR', 'pixel_l2': 0.0,
                 'cc_1nn': 0, 'ct_1nn': 0, 'cc_l2': 0, 'ct_l2': 0}
        r['id'] = task.id
        r.update(tax_by_id.get(task.id, {}))
        results.append(r)
        counts_1nn[r['mode_1nn']] = counts_1nn.get(r['mode_1nn'], 0) + 1
        counts_l2[r['mode_l2']] = counts_l2.get(r['mode_l2'], 0) + 1
        total_cc_1nn += r['cc_1nn']; total_ct_1nn += r['ct_1nn']
        total_cc_l2 += r['cc_l2']; total_ct_l2 += r['ct_l2']

        if (i + 1) % 100 == 0:
            t_el = time.time() - t0
            print(f"  [{i+1:4d}/{len(tasks)}]  "
                  f"1nn={counts_1nn.get('SOLVED',0)}s  "
                  f"L2={counts_l2.get('SOLVED',0)}s  "
                  f"[{t_el:.0f}s]", flush=True)

    elapsed = time.time() - t0
    n = len(tasks)

    print(flush=True)
    print("=" * 65, flush=True)
    print("STEP 328 RESULTS", flush=True)
    print("=" * 65, flush=True)

    cc_1nn_pct = total_cc_1nn / max(total_ct_1nn, 1)
    cc_l2_pct = total_cc_l2 / max(total_ct_l2, 1)
    px_1nn = float(np.mean([r['pixel_1nn'] for r in results]))
    px_l2 = float(np.mean([r['pixel_l2'] for r in results]))

    print(f"\n{'Method':<12} {'Pixel%':>7} {'Changed%':>9} {'Solved':>7} {'Almost':>7}", flush=True)
    print("-" * 50, flush=True)
    print(f"{'1-NN (A)':<12} {px_1nn*100:>6.1f}% {cc_1nn_pct*100:>8.1f}% "
          f"{counts_1nn.get('SOLVED',0):>7} {counts_1nn.get('ALMOST',0):>7}", flush=True)
    print(f"{'Phi L2 (C)':<12} {px_l2*100:>6.1f}% {cc_l2_pct*100:>8.1f}% "
          f"{counts_l2.get('SOLVED',0):>7} {counts_l2.get('ALMOST',0):>7}", flush=True)

    print(f"\nDelta L2 vs 1-NN:", flush=True)
    print(f"  Pixel: {(px_l2-px_1nn)*100:+.2f}pp  Changed: {(cc_l2_pct-cc_1nn_pct)*100:+.2f}pp", flush=True)

    # Track the 5 phi-kill tasks
    print(f"\nPhi-kill task tracker (1-NN solves, L1 killed):", flush=True)
    kill_recovered = 0
    for r in results:
        if r['id'] in PHI_KILL_TASKS:
            recovered = (r['mode_l2'] == 'SOLVED')
            if recovered:
                kill_recovered += 1
            print(f"  {r['id']}: 1nn={r['mode_1nn']}  L2={r['mode_l2']}  "
                  f"{'RECOVERED' if recovered else 'still killed'}", flush=True)

    print(f"  Recovered: {kill_recovered}/5", flush=True)

    # Phi-win tasks (phi should keep winning)
    print(f"\nPhi-win task tracker:", flush=True)
    win_kept = 0
    for r in results:
        if r['id'] in PHI_WIN_TASKS:
            kept = (r['mode_l2'] == 'SOLVED')
            if kept:
                win_kept += 1
            print(f"  {r['id']}: 1nn={r['mode_1nn']}  L2={r['mode_l2']}  "
                  f"{'KEPT' if kept else 'LOST'}", flush=True)
    print(f"  Kept: {win_kept}/5", flush=True)

    # All L2 solves
    l2_solved = [r for r in results if r['mode_l2'] == 'SOLVED']
    print(f"\nAll L2 solved ({len(l2_solved)}):", flush=True)
    for r in l2_solved:
        print(f"  {r['id']}  size={r.get('io_size_relation','?')}  "
              f"color={r.get('color_change_type','?')}", flush=True)

    # Net analysis: L2 better/worse/same vs 1-NN per task
    l2_wins = sum(1 for r in results if r['pixel_l2'] > r['pixel_1nn'] + 0.01)
    l2_loses = sum(1 for r in results if r['pixel_1nn'] > r['pixel_l2'] + 0.01)
    l2_ties = n - l2_wins - l2_loses
    print(f"\nPer-task comparison L2 vs 1-NN:", flush=True)
    print(f"  L2 better:  {l2_wins}/{n}", flush=True)
    print(f"  L2 worse:   {l2_loses}/{n}", flush=True)
    print(f"  Tied:       {l2_ties}/{n}", flush=True)

    # Kill check
    kill = kill_recovered < 3
    success = kill_recovered >= 3

    print(flush=True)
    print("=" * 65, flush=True)
    print("KILL CHECK", flush=True)
    print("=" * 65, flush=True)
    print(f"Phi-kill tasks recovered: {kill_recovered}/5 (need >=3)", flush=True)
    print(f"Kill (recovered < 3): {'TRIGGERED' if kill else 'not triggered'}", flush=True)
    print(f"Success (recovered >= 3): {'YES' if success else 'NO'}", flush=True)

    if kill:
        print("\nKILLED", flush=True)
    else:
        print("\nSUCCESS", flush=True)

    print(f"\nElapsed: {elapsed:.1f}s", flush=True)


if __name__ == '__main__':
    main()
