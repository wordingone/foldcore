#!/usr/bin/env python3
"""
Step 324 — Example-level retrieval before cell-level 1-NN.

Step 323 killed (7x7+row/col hurt). Step 322 (5x5, 39 dims) is best so far:
changed-cell 39.6%, 12 solved, 246 ALMOST.

Hypothesis: ALMOST->SOLVED gap is from cross-example codebook contamination.
Training pair A has (r,c) as class 0; pair B has similar patch as class 1.
1-NN flips a coin at those ambiguous cells.

Fix: before cell-level 1-NN, find the SINGLE most similar training input to
the test input (by global color histogram distance). Use ONLY that example's
cells as the codebook.

For same-size tasks (94.8% of ALMOST), this eliminates cross-example noise
entirely. The best-match training pair's cell mapping is applied directly.

Kill: solved >= 20 OR changed-cell >= 45%.
Step 322 comparison: changed-cell 39.6%, 12 solved, 246 ALMOST.
"""

import json
import numpy as np
import time
import arckit
from scipy.spatial.distance import cdist
from pathlib import Path

MAX_GRID = 30
N_COLORS = 10
PATCH_R = 2  # 5x5 (same as Step 322)


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


def grid_descriptor(inp):
    """Global descriptor for input similarity: color hist + normalized size."""
    hist = color_hist(inp)
    size_feat = np.array([inp.shape[0] / MAX_GRID, inp.shape[1] / MAX_GRID], dtype=np.float32)
    return np.concatenate([hist, size_feat])  # 12 dims


def build_cell_features(inp, out_h, out_w):
    """Same 39-dim encoding as Step 322."""
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


def predict_single_example(test_inp, best_inp, best_out, out_h, out_w):
    """
    Use only the best-match training pair's cells as the codebook.
    Falls back to full 1-NN only if codebook is empty.
    """
    test_feats = build_cell_features(test_inp, out_h, out_w)
    cb_feats = build_cell_features(best_inp, best_out.shape[0], best_out.shape[1])
    cb_labels = best_out.flatten().astype(np.int32)

    dists = cdist(test_feats, cb_feats, metric='sqeuclidean')
    pred_flat = cb_labels[np.argmin(dists, axis=1)]
    return pred_flat.reshape(out_h, out_w)


def evaluate_task(task):
    train = task.train
    train_descs = np.array([grid_descriptor(inp) for inp, _ in train], dtype=np.float32)

    pixel_accs, exact_flags, size_ok_flags = [], [], []
    changed_correct, changed_total = 0, 0

    for test_inp, test_out in task.test:
        out_h, out_w = infer_output_size(train, test_inp)

        # Find best matching training input
        test_desc = grid_descriptor(test_inp).reshape(1, -1)
        dists_global = cdist(test_desc, train_descs, metric='sqeuclidean')[0]
        best_idx = int(np.argmin(dists_global))
        best_inp, best_out_train = train[best_idx]

        pred = predict_single_example(test_inp, best_inp, best_out_train, out_h, out_w)

        if pred.shape != test_out.shape:
            pixel_accs.append(0.0)
            exact_flags.append(False)
            size_ok_flags.append(False)
        else:
            match = pred == test_out
            pixel_accs.append(float(match.mean()))
            exact_flags.append(bool(np.array_equal(pred, test_out)))
            size_ok_flags.append(True)
            if test_inp.shape == test_out.shape:
                changed = test_out != test_inp
                changed_total += int(changed.sum())
                changed_correct += int((match & changed).sum())

    avg_acc = float(np.mean(pixel_accs))
    if all(exact_flags):
        mode = 'SOLVED'
    elif not all(size_ok_flags):
        mode = 'SIZE_WRONG'
    elif avg_acc > 0.8:
        mode = 'ALMOST'
    elif avg_acc > 0.5:
        mode = 'PARTIAL'
    elif avg_acc > 0.2:
        mode = 'WEAK'
    else:
        mode = 'FAIL'

    return {
        'avg_acc': avg_acc, 'mode': mode,
        'changed_correct': changed_correct, 'changed_total': changed_total,
    }


def main():
    t0 = time.time()
    print("Step 324 — Example retrieval + cell 1-NN", flush=True)
    print("Best-match training input -> single-example codebook", flush=True)
    print("Encoding: 5x5 patch + hist + pos = 39 dims (Step 322 encoding)", flush=True)
    print("Kill: solved>=20 OR changed-cell>=45%", flush=True)
    print(flush=True)

    # Load taxonomy
    tax_path = Path(__file__).parent.parent / 'data' / 'arc_taxonomy.json'
    with open(tax_path) as f:
        taxonomy = json.load(f)
    tax_by_id = {t['id']: t for t in taxonomy}

    train_tasks, _ = arckit.load_data()
    tasks = list(train_tasks)
    print(f"Evaluating {len(tasks)} tasks...", flush=True)

    results = []
    total_changed_correct, total_changed = 0, 0
    counts = {}

    for i, task in enumerate(tasks):
        try:
            r = evaluate_task(task)
        except Exception as e:
            r = {'avg_acc': 0.0, 'mode': 'ERROR', 'changed_correct': 0, 'changed_total': 0}
        r['id'] = task.id
        r.update(tax_by_id.get(task.id, {}))
        results.append(r)
        total_changed_correct += r['changed_correct']
        total_changed += r['changed_total']
        counts[r['mode']] = counts.get(r['mode'], 0) + 1

        if (i + 1) % 200 == 0:
            ch = total_changed_correct / max(total_changed, 1)
            print(f"  {i+1}/{len(tasks)}  changed={ch*100:.1f}%  "
                  f"solved={counts.get('SOLVED',0)}  [{time.time()-t0:.0f}s]", flush=True)

    elapsed = time.time() - t0
    changed_acc = total_changed_correct / max(total_changed, 1)
    avg_pixel = float(np.mean([r['avg_acc'] for r in results]))

    print(flush=True)
    print("=" * 65, flush=True)
    print("STEP 324 RESULTS", flush=True)
    print("=" * 65, flush=True)
    print(f"Avg pixel accuracy:    {avg_pixel*100:.1f}%  (Step 322: ~52%)", flush=True)
    print(f"Changed-cell accuracy: {changed_acc*100:.1f}%  (Step 322: 39.6%)", flush=True)
    print(flush=True)

    print("Failure modes:", flush=True)
    for m in ['SOLVED', 'ALMOST', 'PARTIAL', 'WEAK', 'SIZE_WRONG', 'FAIL']:
        n = counts.get(m, 0)
        if n:
            print(f"  {m:12s}: {n:4d} ({n/len(tasks)*100:.1f}%)", flush=True)

    solved = [r for r in results if r['mode'] == 'SOLVED']
    print(f"\nSolved ({len(solved)}):", flush=True)
    for r in solved:
        print(f"  {r['id']}  size={r.get('io_size_relation','?')}  "
              f"color={r.get('color_change_type','?')}", flush=True)

    # Category breakdown for ALMOST
    almost = [r for r in results if r['mode'] == 'ALMOST']
    if almost:
        print(f"\nALMOST ({len(almost)}) — category:", flush=True)
        for field in ['io_size_relation', 'color_change_type']:
            counts2 = {}
            for r in almost:
                v = r.get(field, '?')
                counts2[v] = counts2.get(v, 0) + 1
            vals = sorted(counts2.items(), key=lambda x: -x[1])
            print(f"  {field}: " + ", ".join(f"{v}={c}" for v, c in vals), flush=True)

    # Comparison with Step 322 by category
    print(f"\nBy I/O size:", flush=True)
    for sv in ['same', 'output_smaller', 'output_larger']:
        group = [r for r in results if r.get('io_size_relation') == sv]
        if group:
            accs = [r['avg_acc'] for r in group]
            s = sum(1 for r in group if r['mode'] == 'SOLVED')
            a = sum(1 for r in group if r['mode'] == 'ALMOST')
            print(f"  {sv:18s} n={len(group):4d}  avg={np.mean(accs)*100:5.1f}%  "
                  f"solved={s}  almost={a}", flush=True)

    # Kill check
    success_count = counts.get('SOLVED', 0)
    kill = changed_acc <= 0.45 and success_count < 20
    success = changed_acc > 0.45 or success_count >= 20

    delta_cc = changed_acc - 0.396
    delta_solved = success_count - 12

    print(flush=True)
    print("=" * 65, flush=True)
    print("KILL CHECK", flush=True)
    print("=" * 65, flush=True)
    print(f"Changed-cell: {changed_acc*100:.1f}% (Step 322: 39.6%, delta={delta_cc*100:+.1f}pp)", flush=True)
    print(f"Solved: {success_count} (Step 322: 12, delta={delta_solved:+d})", flush=True)
    print(f"Kill (cc<=45% AND solved<20): {'TRIGGERED' if kill else 'not triggered'}", flush=True)
    print(f"Success (cc>45% OR solved>=20): {'YES' if success else 'NO'}", flush=True)

    if kill:
        print("\nKILLED", flush=True)
    elif success:
        print("\nSUCCESS", flush=True)
    else:
        print("\nPARTIAL", flush=True)

    print(f"\nElapsed: {elapsed:.1f}s", flush=True)


if __name__ == '__main__':
    main()
