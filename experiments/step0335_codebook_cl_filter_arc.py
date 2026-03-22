#!/usr/bin/env python3
"""
Step 335 — CL filter discovery on ARC OBJECT_IDENTITY tasks.

Step 333: CL filter on a%b -> +5.25pp by discovering proximity grouping.
Step 335: same mechanism on 99 OBJECT_IDENTITY ARC tasks.

Hypothesis: cells in the same object have similar 5x5 patches -> CL groups
them together -> restricting 1-NN to same CL group approximates same-object
restriction -> helps predict per-object output colors.

Algorithm per task:
1. Build 39-dim cell features (5x5 patch + hist + pos, as Step 322)
2. Run CL on training cells -> smaller codebook
3. For each training cell, assign to nearest CL winner -> CL group
4. For each test cell, find CL group (nearest codebook entry)
5. 1-NN restricted to same-group training cells (fallback: global 1-NN)

Sweep spawn thresholds: [1.0, 2.0, 5.0, 10.0, 20.0] in L2 distance space.

Baseline: Step 322 (plain 1-NN on all training cells).
Kill: CL filter must beat Step 322 on 99 object-identity tasks by > 5pp changed-cell.
"""

import json
import numpy as np
import time
import arckit
from scipy.spatial.distance import cdist
from pathlib import Path
from collections import deque

MAX_GRID  = 30
N_COLORS  = 10
PATCH_R   = 2  # 5x5
SPAWN_THRESHS = [1.0, 2.0, 5.0, 10.0, 20.0]

OBJ_TASKS_PATH = Path(__file__).parent.parent / 'data' / 'arc_constraint_map.json'


# ─── Cell features (Step 322) ──────────────────────────────────────────────────

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
    """39-dim feature: [5x5_patch | global_hist | r/H | c/W | H/30 | W/30]"""
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


# ─── Competitive learning (L2, no normalization) ──────────────────────────────

def run_cl(feats, spawn_thresh, lr=0.1, n_epochs=3, seed=42):
    """
    CL on feats (n, d) in L2 space.
    Returns: codebook (n_cb, d), assignments (n,) int array.
    """
    rng = np.random.RandomState(seed)
    n = len(feats)
    if n == 0:
        return np.empty((0, feats.shape[1])), np.empty(0, dtype=np.int32)

    order = np.arange(n)
    codebook = [feats[0].copy()]

    for epoch in range(n_epochs):
        rng.shuffle(order)
        for i in order:
            x = feats[i]
            cb = np.array(codebook)
            dists = np.linalg.norm(cb - x, axis=1)
            winner = np.argmin(dists)
            if dists[winner] > spawn_thresh:
                codebook.append(x.copy())
            else:
                codebook[winner] = codebook[winner] + lr * (x - codebook[winner])

    codebook = np.array(codebook)
    # Assign each cell to nearest codebook entry
    if len(codebook) == 1:
        assignments = np.zeros(n, dtype=np.int32)
    else:
        dists_all = cdist(feats, codebook, metric='euclidean')
        assignments = np.argmin(dists_all, axis=1).astype(np.int32)
    return codebook, assignments


# ─── Output size inference ─────────────────────────────────────────────────────

def infer_output_size(train_examples, test_input):
    out_sizes = [out.shape for _, out in train_examples]
    in_sizes  = [inp.shape for inp, _ in train_examples]
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


# ─── Evaluation ────────────────────────────────────────────────────────────────

def evaluate_task_baseline(task):
    """Step 322: plain 1-NN on all training cells."""
    all_feats, all_labels = [], []
    for inp, out in task.train:
        f = build_cell_features(inp, out.shape[0], out.shape[1])
        all_feats.append(f)
        all_labels.append(out.flatten())
    cb_feats  = np.vstack(all_feats)
    cb_labels = np.concatenate(all_labels).astype(np.int32)

    pixel_accs, exact_flags, size_ok_flags = [], [], []
    changed_correct, changed_total = 0, 0

    for test_inp, test_out in task.test:
        out_h, out_w = infer_output_size(task.train, test_inp)
        test_feats = build_cell_features(test_inp, out_h, out_w)
        dists = cdist(test_feats, cb_feats, metric='sqeuclidean')
        pred_flat = cb_labels[np.argmin(dists, axis=1)]
        pred = pred_flat.reshape(out_h, out_w)

        if pred.shape != test_out.shape:
            pixel_accs.append(0.0); exact_flags.append(False); size_ok_flags.append(False)
        else:
            match = pred == test_out
            pixel_accs.append(float(match.mean()))
            exact_flags.append(bool(np.array_equal(pred, test_out)))
            size_ok_flags.append(True)
            if test_inp.shape == test_out.shape:
                changed = test_out != test_inp
                changed_total += int(changed.sum())
                changed_correct += int((match & changed).sum())

    mode = classify_mode(pixel_accs, exact_flags, size_ok_flags)
    return {
        'avg_acc': float(np.mean(pixel_accs)), 'mode': mode,
        'changed_correct': changed_correct, 'changed_total': changed_total,
    }


def evaluate_task_cl(task, spawn_thresh):
    """CL filter: 1-NN restricted to same CL group."""
    # Build training cell features + CL codebook
    all_feats, all_labels = [], []
    for inp, out in task.train:
        f = build_cell_features(inp, out.shape[0], out.shape[1])
        all_feats.append(f)
        all_labels.append(out.flatten())
    cb_feats  = np.vstack(all_feats)
    cb_labels = np.concatenate(all_labels).astype(np.int32)

    # Run CL on training cell features
    codebook, assignments = run_cl(cb_feats, spawn_thresh)
    n_cb = len(codebook)

    pixel_accs, exact_flags, size_ok_flags = [], [], []
    changed_correct, changed_total = 0, 0

    for test_inp, test_out in task.test:
        out_h, out_w = infer_output_size(task.train, test_inp)
        test_feats = build_cell_features(test_inp, out_h, out_w)

        # Assign test cells to CL groups
        if n_cb <= 1:
            test_groups = np.zeros(len(test_feats), dtype=np.int32)
        else:
            test_dists = cdist(test_feats, codebook, metric='euclidean')
            test_groups = np.argmin(test_dists, axis=1).astype(np.int32)

        # Predict each test cell using same-group training cells
        pred_flat = np.zeros(len(test_feats), dtype=np.int32)
        for i in range(len(test_feats)):
            g = test_groups[i]
            group_mask = assignments == g
            if group_mask.sum() == 0:
                # Fallback: global 1-NN
                dists = cdist(test_feats[i:i+1], cb_feats, metric='sqeuclidean')[0]
                pred_flat[i] = cb_labels[np.argmin(dists)]
            else:
                group_feats  = cb_feats[group_mask]
                group_labels = cb_labels[group_mask]
                dists = cdist(test_feats[i:i+1], group_feats, metric='sqeuclidean')[0]
                pred_flat[i] = group_labels[np.argmin(dists)]

        pred = pred_flat.reshape(out_h, out_w)

        if pred.shape != test_out.shape:
            pixel_accs.append(0.0); exact_flags.append(False); size_ok_flags.append(False)
        else:
            match = pred == test_out
            pixel_accs.append(float(match.mean()))
            exact_flags.append(bool(np.array_equal(pred, test_out)))
            size_ok_flags.append(True)
            if test_inp.shape == test_out.shape:
                changed = test_out != test_inp
                changed_total += int(changed.sum())
                changed_correct += int((match & changed).sum())

    mode = classify_mode(pixel_accs, exact_flags, size_ok_flags)
    return {
        'avg_acc': float(np.mean(pixel_accs)), 'mode': mode,
        'changed_correct': changed_correct, 'changed_total': changed_total,
        'n_cl_entries': n_cb,
    }


def classify_mode(pixel_accs, exact_flags, size_ok_flags):
    avg = float(np.mean(pixel_accs))
    if all(exact_flags):
        return 'SOLVED'
    elif not all(size_ok_flags):
        return 'SIZE_WRONG'
    elif avg > 0.8:
        return 'ALMOST'
    elif avg > 0.5:
        return 'PARTIAL'
    elif avg > 0.2:
        return 'WEAK'
    return 'FAIL'


# ─── Main ──────────────────────────────────────────────────────────────────────

def main():
    t0 = time.time()
    print("Step 335 — CL filter on ARC OBJECT_IDENTITY tasks", flush=True)
    print(f"Spawn thresholds (L2): {SPAWN_THRESHS}", flush=True)
    print("Kill: CL filter > Step 322 baseline by > 5pp changed-cell", flush=True)
    print(flush=True)

    # Load constraint map to get object-identity task IDs
    with open(OBJ_TASKS_PATH) as f:
        constraint_map = json.load(f)
    obj_task_ids = {tid for tid, v in constraint_map.items()
                    if v['category'] == 'OBJECT_IDENTITY'}
    print(f"Object-identity tasks: {len(obj_task_ids)}", flush=True)

    train_tasks, _ = arckit.load_data()
    tasks_all = list(train_tasks)
    tasks = [t for t in tasks_all if t.id in obj_task_ids]
    print(f"Loaded {len(tasks)} object-identity tasks", flush=True)
    print(flush=True)

    # Baseline: Step 322 on these 99 tasks
    print("Computing Step 322 baseline...", flush=True)
    baseline_results = {}
    for task in tasks:
        try:
            r = evaluate_task_baseline(task)
        except Exception as e:
            r = {'avg_acc': 0.0, 'mode': 'ERROR', 'changed_correct': 0, 'changed_total': 0}
        baseline_results[task.id] = r

    base_cc = sum(r['changed_correct'] for r in baseline_results.values())
    base_ct = sum(r['changed_total'] for r in baseline_results.values())
    base_changed = base_cc / max(base_ct, 1)
    base_solved  = sum(1 for r in baseline_results.values() if r['mode'] == 'SOLVED')
    base_avg_acc = float(np.mean([r['avg_acc'] for r in baseline_results.values()]))

    print(f"Baseline (Step 322):", flush=True)
    print(f"  Changed-cell: {base_changed*100:.2f}%", flush=True)
    print(f"  Solved: {base_solved}/{len(tasks)}", flush=True)
    print(f"  Avg pixel acc: {base_avg_acc*100:.1f}%", flush=True)
    print(f"  Elapsed: {time.time()-t0:.1f}s", flush=True)
    print(flush=True)

    # CL filter sweep
    results_by_thresh = {}
    print("=" * 65, flush=True)
    print("CL filter sweep", flush=True)
    print("=" * 65, flush=True)

    for sp in SPAWN_THRESHS:
        print(f"\n--- spawn_thresh={sp} ---", flush=True)
        cl_results = {}
        for task in tasks:
            try:
                r = evaluate_task_cl(task, sp)
            except Exception as e:
                r = {'avg_acc': 0.0, 'mode': 'ERROR', 'changed_correct': 0,
                     'changed_total': 0, 'n_cl_entries': 0}
            cl_results[task.id] = r

        cl_cc = sum(r['changed_correct'] for r in cl_results.values())
        cl_ct = sum(r['changed_total'] for r in cl_results.values())
        cl_changed = cl_cc / max(cl_ct, 1)
        cl_solved  = sum(1 for r in cl_results.values() if r['mode'] == 'SOLVED')
        cl_avg_acc = float(np.mean([r['avg_acc'] for r in cl_results.values()]))

        avg_cl_entries = float(np.mean([r.get('n_cl_entries', 0)
                                         for r in cl_results.values()]))
        delta_cc = cl_changed - base_changed

        print(f"  Avg CL entries per task: {avg_cl_entries:.1f}", flush=True)
        print(f"  Changed-cell: {cl_changed*100:.2f}%  (delta: {delta_cc*100:+.2f}pp)", flush=True)
        print(f"  Solved: {cl_solved}/{len(tasks)}", flush=True)
        print(f"  Avg pixel acc: {cl_avg_acc*100:.1f}%", flush=True)
        print(f"  Elapsed: {time.time()-t0:.1f}s", flush=True)

        results_by_thresh[sp] = {
            'changed': cl_changed,
            'solved': cl_solved,
            'avg_acc': cl_avg_acc,
            'avg_cl_entries': avg_cl_entries,
            'delta_cc': delta_cc,
        }

    # Summary
    elapsed = time.time() - t0
    print(flush=True)
    print("=" * 65, flush=True)
    print("STEP 335 SUMMARY — CL filter on OBJECT_IDENTITY tasks", flush=True)
    print("=" * 65, flush=True)
    print(f"Baseline (Step 322):  changed={base_changed*100:.2f}%  "
          f"solved={base_solved}/{len(tasks)}", flush=True)
    print(flush=True)
    print(f"{'thresh':>8} {'n_cb':>6} {'changed%':>10} {'delta':>8} {'solved':>8}",
          flush=True)
    print("-" * 50, flush=True)
    for sp, r in results_by_thresh.items():
        print(f"{sp:>8.1f} {r['avg_cl_entries']:>6.0f} {r['changed']*100:>9.2f}% "
              f"{r['delta_cc']*100:>+7.2f}pp {r['solved']:>4}/{len(tasks)}", flush=True)

    best_sp = max(results_by_thresh, key=lambda k: results_by_thresh[k]['changed'])
    best    = results_by_thresh[best_sp]
    kill    = best['delta_cc'] <= 0.05
    success = best['delta_cc'] > 0.05

    print(flush=True)
    print(f"Best: spawn_thresh={best_sp}, changed={best['changed']*100:.2f}%, "
          f"delta={best['delta_cc']*100:+.2f}pp", flush=True)

    print(flush=True)
    print("=" * 65, flush=True)
    print("KILL CHECK", flush=True)
    print("=" * 65, flush=True)
    print(f"Baseline changed-cell: {base_changed*100:.2f}%", flush=True)
    print(f"Best CL filter: {best['changed']*100:.2f}% (spawn_thresh={best_sp})", flush=True)
    print(f"Delta: {best['delta_cc']*100:+.2f}pp  (need > +5pp)", flush=True)
    print(f"Kill (delta <= 5pp): {'TRIGGERED' if kill else 'not triggered'}", flush=True)
    print(f"Success (delta > 5pp): {'YES' if success else 'NO'}", flush=True)

    if kill:
        print("\nKILLED — CL groups don't approximate object membership on ARC", flush=True)
    else:
        print(f"\nSUCCESS — CL filter improves object-identity tasks by "
              f"{best['delta_cc']*100:.2f}pp", flush=True)

    print(f"\nElapsed: {elapsed:.1f}s", flush=True)


if __name__ == '__main__':
    main()
