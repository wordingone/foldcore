#!/usr/bin/env python3
"""
Step 327 — Port auto_loop to ARC-AGI.

Question: how many tasks did the SUBSTRATE solve?
Answer from Steps 320-326: zero. Everything was 1-NN or conventional ML.

This step applies the substrate's actual mechanisms per ARC task:
  Codebook = training cells (39-dim local patch features + output color labels)
  Phi = per-class sorted top-K feature distances (transforms 39-dim into phi-space)
  Weight learning = upweight phi dims where cross-class cells differ most
  Loop = learn → prescribe → evaluate within each task

Three-way comparison (all using Step 322 encoding):
  A. 1-NN in feature space (Step 322 baseline)
  B. 1-NN in phi space, uniform weights (does phi transform help?)
  C. 1-NN in phi space, learned weights (does the loop add value?)

Kill: loop (C) must improve over phi-uniform (B) on >100 tasks.
Delta B-A = value of phi transformation
Delta C-B = value of the loop's weight learning
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
K_PHI = 3     # top-3 distances per class in phi
PHI_DIM = N_COLORS * K_PHI  # 30
SENTINEL = 1e6
LR_W = 0.05
N_EPOCHS = 3


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

def compute_phi_codebook(cb_feats, cb_labels):
    """
    Compute phi for all codebook cells (LOO: exclude self when computing own phi).
    phi[i, c*K_PHI : c*K_PHI+K_PHI] = top-K sorted feature distances from cell i
    to class-c cells (excluding cell i itself).
    Returns (n, PHI_DIM) array.
    """
    n = len(cb_feats)
    # All-pairs feature distances
    all_feat_dists = cdist(cb_feats, cb_feats, metric='sqeuclidean')  # (n, n)

    phi = np.full((n, PHI_DIM), SENTINEL, dtype=np.float32)
    for c in range(N_COLORS):
        class_idxs = np.where(cb_labels == c)[0]
        if len(class_idxs) == 0:
            continue
        # Distances from every cell to class-c cells
        class_dists = all_feat_dists[:, class_idxs]  # (n, n_c)
        n_c = len(class_idxs)
        for i in range(n):
            row = class_dists[i].copy()
            # LOO: if cell i is class c, exclude its self-distance
            if cb_labels[i] == c:
                self_pos = np.where(class_idxs == i)[0]
                if len(self_pos) > 0:
                    row[self_pos[0]] = SENTINEL
            row_sorted = np.sort(row)
            # Filter out SENTINEL
            valid = row_sorted[row_sorted < SENTINEL]
            k_eff = min(K_PHI, len(valid))
            if k_eff > 0:
                phi[i, c * K_PHI: c * K_PHI + k_eff] = valid[:k_eff]
    return phi


def compute_phi_query(query_feats, cb_feats, cb_labels):
    """
    Compute phi for query cells against codebook (no LOO exclusion).
    Returns (n_query, PHI_DIM).
    """
    n_q = len(query_feats)
    all_feat_dists = cdist(query_feats, cb_feats, metric='sqeuclidean')  # (n_q, n_cb)

    phi = np.full((n_q, PHI_DIM), SENTINEL, dtype=np.float32)
    for c in range(N_COLORS):
        class_idxs = np.where(cb_labels == c)[0]
        if len(class_idxs) == 0:
            continue
        class_dists = all_feat_dists[:, class_idxs]  # (n_q, n_c)
        class_dists_sorted = np.sort(class_dists, axis=1)
        # Filter SENTINEL
        for i in range(n_q):
            valid = class_dists_sorted[i][class_dists_sorted[i] < SENTINEL]
            k_eff = min(K_PHI, len(valid))
            if k_eff > 0:
                phi[i, c * K_PHI: c * K_PHI + k_eff] = valid[:k_eff]
    return phi


# ── LOO in phi space ──────────────────────────────────────────────────────────

def loo_phi(phi_all, cb_labels, weights):
    """LOO accuracy using weighted phi distance."""
    n = len(phi_all)
    w_expanded = np.tile(weights, N_COLORS).astype(np.float32)

    # Vectorized all-pairs phi distance
    diffs = phi_all[:, np.newaxis, :] - phi_all[np.newaxis, :, :]  # (n, n, PHI_DIM)
    dists = (diffs ** 2 * w_expanded).sum(axis=2)  # (n, n)
    np.fill_diagonal(dists, np.inf)

    nearest = np.argmin(dists, axis=1)
    correct = int((cb_labels[nearest] == cb_labels).sum())
    return correct / n


# ── Weight learning ───────────────────────────────────────────────────────────

def learn_weights_phi(phi_all, cb_labels, weights, lr=LR_W, epochs=N_EPOCHS):
    """
    Upweight phi k-indices where cross-class phi pairs differ most.
    Same mechanism as auto_loop.learn_weights, but in phi space over cells.
    """
    n = len(phi_all)
    w = weights.copy()
    w_expanded = np.tile(w, N_COLORS).astype(np.float32)

    # Pre-compute all-pairs phi distances
    diffs_all = phi_all[:, np.newaxis, :] - phi_all[np.newaxis, :, :]  # (n, n, PHI_DIM)

    for epoch in range(epochs):
        # Recompute distances with current weights
        dists = (diffs_all ** 2 * w_expanded).sum(axis=2)  # (n, n)
        np.fill_diagonal(dists, np.inf)
        nearest = np.argmin(dists, axis=1)  # (n,)

        cross_mask = (cb_labels[nearest] != cb_labels)
        cross_idxs = np.where(cross_mask)[0]

        if len(cross_idxs) == 0:
            break

        for i in cross_idxs:
            j = nearest[i]
            diff_sq = (phi_all[i] - phi_all[j]) ** 2  # (PHI_DIM,)
            per_k_signal = np.zeros(K_PHI, dtype=np.float32)
            for k in range(K_PHI):
                slots = [c * K_PHI + k for c in range(N_COLORS)]
                per_k_signal[k] = diff_sq[slots].mean()
            w += lr * per_k_signal
            w = np.maximum(w, 0.01)

        w = w / w.sum() * K_PHI
        w_expanded = np.tile(w, N_COLORS).astype(np.float32)

    return w


# ── Prediction in phi space ───────────────────────────────────────────────────

def predict_in_phi_space(query_phi, phi_cb, cb_labels, weights):
    """1-NN prediction in phi space with learned weights."""
    w_expanded = np.tile(weights, N_COLORS).astype(np.float32)
    diffs = phi_cb[np.newaxis, :, :] - query_phi[:, np.newaxis, :]  # (n_q, n_cb, PHI_DIM)
    dists = (diffs ** 2 * w_expanded).sum(axis=2)  # (n_q, n_cb)
    nearest = np.argmin(dists, axis=1)
    return cb_labels[nearest]


# ── Evaluation ────────────────────────────────────────────────────────────────

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


def score_prediction(pred, test_inp, test_out):
    """Returns pixel_acc, exact, changed_correct, changed_total."""
    if pred.shape != test_out.shape:
        return 0.0, False, 0, 0
    match = pred == test_out
    pixel_acc = float(match.mean())
    exact = bool(np.array_equal(pred, test_out))
    changed_correct, changed_total = 0, 0
    if test_inp.shape == test_out.shape:
        changed = test_out != test_inp
        changed_total = int(changed.sum())
        changed_correct = int((match & changed).sum())
    return pixel_acc, exact, changed_correct, changed_total


def classify_mode(pixel_accs, exact_flags, size_ok_flags):
    if all(exact_flags): return 'SOLVED'
    if not all(size_ok_flags): return 'SIZE_WRONG'
    avg = np.mean(pixel_accs)
    if avg > 0.8: return 'ALMOST'
    if avg > 0.5: return 'PARTIAL'
    if avg > 0.2: return 'WEAK'
    return 'FAIL'


def evaluate_task(task):
    """Run all three methods on one task. Returns (A_result, B_result, C_result)."""
    cb_feats, cb_labels = build_codebook(task.train)
    n_cb = len(cb_feats)

    # Method A: 1-NN in feature space (Step 322)
    # Method B: 1-NN in phi space, uniform weights
    # Method C: 1-NN in phi space, learned weights

    # Compute phi for codebook (shared for B and C)
    phi_cb = compute_phi_codebook(cb_feats, cb_labels)

    # LOO accuracy in phi space: uniform weights
    weights_uniform = np.ones(K_PHI, dtype=np.float32)
    loo_b = loo_phi(phi_cb, cb_labels, weights_uniform)

    # Learn weights (the loop)
    weights_learned = learn_weights_phi(phi_cb, cb_labels, weights_uniform.copy())
    loo_c = loo_phi(phi_cb, cb_labels, weights_learned)

    # Evaluate on test pairs
    results_a, results_b, results_c = [], [], []

    for test_inp, test_out in task.test:
        out_h, out_w = infer_output_size(task.train, test_inp)
        test_feats = build_cell_features(test_inp, out_h, out_w)
        query_phi = compute_phi_query(test_feats, cb_feats, cb_labels)

        # A: feature-space 1-NN
        feat_dists = cdist(test_feats, cb_feats, metric='sqeuclidean')
        pred_a = cb_labels[np.argmin(feat_dists, axis=1)].reshape(out_h, out_w)

        # B: phi-space 1-NN, uniform
        pred_b = predict_in_phi_space(query_phi, phi_cb, cb_labels, weights_uniform).reshape(out_h, out_w)

        # C: phi-space 1-NN, learned weights
        pred_c = predict_in_phi_space(query_phi, phi_cb, cb_labels, weights_learned).reshape(out_h, out_w)

        results_a.append(score_prediction(pred_a, test_inp, test_out))
        results_b.append(score_prediction(pred_b, test_inp, test_out))
        results_c.append(score_prediction(pred_c, test_inp, test_out))

    def aggregate(scores):
        accs = [s[0] for s in scores]
        exacts = [s[1] for s in scores]
        size_oks = [s[0] > 0 or s[1] for s in scores]  # size_ok if acc>0 or exact
        # Recompute size_ok properly
        size_oks = [True] * len(scores)  # simplification — pred shape != test triggers 0 acc
        cc = sum(s[2] for s in scores)
        ct = sum(s[3] for s in scores)
        return {
            'pixel_acc': float(np.mean(accs)),
            'mode': classify_mode(accs, exacts, size_oks),
            'changed_correct': cc, 'changed_total': ct,
        }

    r_a = aggregate(results_a)
    r_b = aggregate(results_b)
    r_c = aggregate(results_c)
    r_a['loo'] = None  # no phi LOO for A
    r_b['loo'] = loo_b
    r_c['loo'] = loo_c
    r_c['weights'] = weights_learned.tolist()

    return r_a, r_b, r_c


def main():
    t0 = time.time()
    print("Step 327 — Auto-loop applied per-task to ARC-AGI", flush=True)
    print("Question: what does the SUBSTRATE contribute?", flush=True)
    print(f"K_PHI={K_PHI}, PHI_DIM={PHI_DIM}, N_EPOCHS={N_EPOCHS}, LR={LR_W}", flush=True)
    print(flush=True)
    print("Three-way comparison:", flush=True)
    print("  A: 1-NN in feature space (Step 322 baseline)", flush=True)
    print("  B: 1-NN in phi space, uniform weights (does phi transform help?)", flush=True)
    print("  C: 1-NN in phi space, learned weights  (does the loop add value?)", flush=True)
    print(flush=True)
    print(f"Kill: loop (C) must improve over phi-uniform (B) on >100 tasks", flush=True)
    print(flush=True)

    tax_path = Path(__file__).parent.parent / 'data' / 'arc_taxonomy.json'
    with open(tax_path) as f:
        taxonomy = json.load(f)
    tax_by_id = {t['id']: t for t in taxonomy}

    train_tasks, _ = arckit.load_data()
    tasks = list(train_tasks)
    print(f"Evaluating {len(tasks)} tasks...", flush=True)

    all_a, all_b, all_c = [], [], []
    counts_a, counts_b, counts_c = {}, {}, {}
    cc_a, ct_a = 0, 0
    cc_b, ct_b = 0, 0
    cc_c, ct_c = 0, 0
    n_loop_improved = 0  # C better than B
    n_phi_improved = 0   # B better than A

    for i, task in enumerate(tasks):
        try:
            r_a, r_b, r_c = evaluate_task(task)
        except Exception as e:
            r_a = r_b = r_c = {'pixel_acc': 0.0, 'mode': 'ERROR', 'changed_correct': 0, 'changed_total': 0, 'loo': None}
        tax = tax_by_id.get(task.id, {})
        for r in (r_a, r_b, r_c):
            r['id'] = task.id
            r.update(tax)

        all_a.append(r_a); all_b.append(r_b); all_c.append(r_c)
        counts_a[r_a['mode']] = counts_a.get(r_a['mode'], 0) + 1
        counts_b[r_b['mode']] = counts_b.get(r_b['mode'], 0) + 1
        counts_c[r_c['mode']] = counts_c.get(r_c['mode'], 0) + 1
        cc_a += r_a['changed_correct']; ct_a += r_a['changed_total']
        cc_b += r_b['changed_correct']; ct_b += r_b['changed_total']
        cc_c += r_c['changed_correct']; ct_c += r_c['changed_total']

        if r_c['pixel_acc'] > r_b['pixel_acc'] + 0.01:
            n_loop_improved += 1
        if r_b['pixel_acc'] > r_a['pixel_acc'] + 0.01:
            n_phi_improved += 1

        if (i + 1) % 100 == 0:
            t_el = time.time() - t0
            print(f"  [{i+1:4d}/{len(tasks)}]  "
                  f"A={counts_a.get('SOLVED',0)}s  "
                  f"B={counts_b.get('SOLVED',0)}s  "
                  f"C={counts_c.get('SOLVED',0)}s  "
                  f"loop_improved={n_loop_improved}  [{t_el:.0f}s]", flush=True)

    elapsed = time.time() - t0
    n = len(tasks)

    def cc_acc(cc, ct):
        return cc / max(ct, 1)

    print(flush=True)
    print("=" * 65, flush=True)
    print("STEP 327 RESULTS", flush=True)
    print("=" * 65, flush=True)
    print(f"\n{'Method':<10} {'Pixel%':>7} {'Changed%':>9} {'Solved':>7} {'Almost':>7}", flush=True)
    print("-" * 45, flush=True)
    for label, results, counts, cc, ct in [
        ('A: 1-NN', all_a, counts_a, cc_a, ct_a),
        ('B: phi-u', all_b, counts_b, cc_b, ct_b),
        ('C: phi-w', all_c, counts_c, cc_c, ct_c),
    ]:
        avg_px = float(np.mean([r['pixel_acc'] for r in results]))
        s = counts.get('SOLVED', 0)
        a = counts.get('ALMOST', 0)
        print(f"{label:<10} {avg_px*100:>6.1f}% {cc_acc(cc,ct)*100:>8.1f}% {s:>7} {a:>7}", flush=True)

    print(flush=True)
    print(f"Delta B vs A (phi transform):", flush=True)
    avg_a = np.mean([r['pixel_acc'] for r in all_a])
    avg_b = np.mean([r['pixel_acc'] for r in all_b])
    avg_c = np.mean([r['pixel_acc'] for r in all_c])
    print(f"  Pixel: {(avg_b-avg_a)*100:+.2f}pp  Changed: {(cc_acc(cc_b,ct_b)-cc_acc(cc_a,ct_a))*100:+.2f}pp", flush=True)
    print(f"  Tasks where B > A: {n_phi_improved}/{n}", flush=True)
    print(flush=True)
    print(f"Delta C vs B (loop weight learning):", flush=True)
    print(f"  Pixel: {(avg_c-avg_b)*100:+.2f}pp  Changed: {(cc_acc(cc_c,ct_c)-cc_acc(cc_b,ct_b))*100:+.2f}pp", flush=True)
    print(f"  Tasks where C > B (loop improved): {n_loop_improved}/{n}", flush=True)

    # By category
    print(flush=True)
    print("By I/O size (Method C — phi+loop):", flush=True)
    for sv in ['same', 'output_smaller', 'output_larger']:
        group = [r for r in all_c if r.get('io_size_relation') == sv]
        if group:
            accs = [r['pixel_acc'] for r in group]
            s = sum(1 for r in group if r['mode'] == 'SOLVED')
            print(f"  {sv:18s} n={len(group):4d}  avg={np.mean(accs)*100:5.1f}%  solved={s}", flush=True)

    # Solved tasks for C
    solved_c = [r for r in all_c if r['mode'] == 'SOLVED']
    print(f"\nC solved ({len(solved_c)}):", flush=True)
    for r in solved_c:
        print(f"  {r['id']}  size={r.get('io_size_relation','?')}  color={r.get('color_change_type','?')}", flush=True)

    # Kill check
    kill = n_loop_improved <= 100
    success = n_loop_improved > 100
    print(flush=True)
    print("=" * 65, flush=True)
    print("KILL CHECK", flush=True)
    print("=" * 65, flush=True)
    print(f"Loop improved on: {n_loop_improved}/{n} tasks", flush=True)
    print(f"Kill (<=100 tasks improved): {'TRIGGERED' if kill else 'not triggered'}", flush=True)
    print(f"Success (>100 tasks improved): {'YES' if success else 'NO'}", flush=True)
    if kill:
        print("\nKILLED", flush=True)
    else:
        print("\nSUCCESS", flush=True)

    print(f"\nElapsed: {elapsed:.1f}s", flush=True)


if __name__ == '__main__':
    main()
