#!/usr/bin/env python3
"""
Step 323 — Row/column histogram encoding.

Step 322 (5x5 patch, 39 dims) got changed-cell 39.6%, 12 solved, 246 ALMOST.
The gap: 5x5 patch misses long-range line structure (many ARC tasks are
row/column invariant, use line symmetry, or tile along axes).

Step 323 encoding:
  [7x7_patch | row_color_hist | col_color_hist | global_hist | r/H | c/W | H/30 | W/30]
  = [49       | 10             | 10             | 10          | 4  ]
  = 83 dims

Row/col histograms: for cell (r,c), the 10-bin color histogram of row r
and column c of the input. This captures:
  - Line patterns (same row → same row hist regardless of where in row)
  - Column structure (same column → same col hist)
  - Long-range dependencies that 5x5 can't see

Kill: changed-cell acc must beat Step 322 (39.6%) by >5pp (>44.6%).
Success: >=20 solved tasks OR changed-cell >=50%.

Also outputs per-task breakdown joined with taxonomy for Q3 analysis.
"""

import json
import numpy as np
import time
import sys
import arckit
from scipy.spatial.distance import cdist
from pathlib import Path

MAX_GRID = 30
N_COLORS = 10
PATCH_R = 3  # 7x7


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


def color_hist(values, n_colors=N_COLORS):
    """Normalized color histogram from a 1D array of color values."""
    h = np.zeros(n_colors, dtype=np.float32)
    for v in values:
        if 0 <= int(v) < n_colors:
            h[int(v)] += 1
    s = h.sum()
    if s > 0:
        h /= s
    return h


def build_cell_features(inp, out_h, out_w):
    patch_dim = (2 * PATCH_R + 1) ** 2  # 49
    feat_dim = patch_dim + N_COLORS * 3 + 4  # 49+10+10+10+4 = 83
    n_cells = out_h * out_w
    feats = np.empty((n_cells, feat_dim), dtype=np.float32)

    in_h, in_w = inp.shape
    global_h = color_hist(inp.flatten())

    # Precompute per-row and per-col histograms
    row_hists = np.array([color_hist(inp[r, :]) for r in range(in_h)], dtype=np.float32)
    col_hists = np.array([color_hist(inp[:, c]) for c in range(in_w)], dtype=np.float32)

    for idx in range(n_cells):
        r, c = divmod(idx, out_w)
        # Map output cell to input position
        if in_h == out_h and in_w == out_w:
            ir, ic = r, c
        else:
            ir = min(int(r * in_h / out_h), in_h - 1) if out_h > 0 else 0
            ic = min(int(c * in_w / out_w), in_w - 1) if out_w > 0 else 0

        patch = extract_patch(inp, ir, ic)
        offset = 0
        feats[idx, offset:offset + patch_dim] = patch; offset += patch_dim
        feats[idx, offset:offset + N_COLORS] = row_hists[ir]; offset += N_COLORS
        feats[idx, offset:offset + N_COLORS] = col_hists[ic]; offset += N_COLORS
        feats[idx, offset:offset + N_COLORS] = global_h; offset += N_COLORS
        feats[idx, offset] = r / max(out_h, 1)
        feats[idx, offset + 1] = c / max(out_w, 1)
        feats[idx, offset + 2] = out_h / MAX_GRID
        feats[idx, offset + 3] = out_w / MAX_GRID

    return feats


def build_codebook(train_examples):
    all_feats, all_labels = [], []
    for inp, out in train_examples:
        feats = build_cell_features(inp, out.shape[0], out.shape[1])
        all_feats.append(feats)
        all_labels.append(out.flatten())
    return np.vstack(all_feats), np.concatenate(all_labels).astype(np.int32)


def predict_1nn(test_feats, cb_feats, cb_labels):
    dists = cdist(test_feats, cb_feats, metric='sqeuclidean')
    return cb_labels[np.argmin(dists, axis=1)]


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
    cb_feats, cb_labels = build_codebook(task.train)
    pixel_accs, exact_flags, size_ok_flags = [], [], []
    changed_correct, changed_total = 0, 0

    for test_inp, test_out in task.test:
        out_h, out_w = infer_output_size(task.train, test_inp)
        test_feats = build_cell_features(test_inp, out_h, out_w)
        pred_flat = predict_1nn(test_feats, cb_feats, cb_labels)
        pred = pred_flat.reshape(out_h, out_w)

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
        'avg_acc': avg_acc,
        'mode': mode,
        'changed_correct': changed_correct,
        'changed_total': changed_total,
    }


def main():
    t0 = time.time()
    print("Step 323 — Row/col histogram encoding", flush=True)
    print("Encoding: 7x7 patch + row_hist + col_hist + global_hist + pos = 83 dims", flush=True)
    print("Step 322 baseline: changed-cell 39.6%, 12 solved, 246 ALMOST", flush=True)
    print("Kill: changed-cell must beat 39.6% by >5pp (>44.6%)", flush=True)
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
    print("STEP 323 RESULTS", flush=True)
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
              f"color={r.get('color_change_type','?')}  "
              f"sym={r.get('has_symmetry','?')}", flush=True)

    # Category breakdown for ALMOST
    almost = [r for r in results if r['mode'] == 'ALMOST']
    print(f"\nALMOST ({len(almost)}) — category breakdown:", flush=True)
    for field in ['io_size_relation', 'color_change_type']:
        counts2 = {}
        for r in almost:
            v = r.get(field, '?')
            counts2[v] = counts2.get(v, 0) + 1
        vals = sorted(counts2.items(), key=lambda x: -x[1])
        print(f"  {field}: " + ", ".join(f"{v}={c}" for v, c in vals), flush=True)

    sym_count = sum(1 for r in almost if r.get('has_symmetry'))
    print(f"  has_symmetry: {sym_count}/{len(almost)}", flush=True)

    # Taxonomy cross-reference
    print(f"\nBy I/O size (all tasks):", flush=True)
    for size_val in ['same', 'output_smaller', 'output_larger', 'variable']:
        group = [r for r in results if r.get('io_size_relation') == size_val]
        if group:
            accs = [r['avg_acc'] for r in group]
            s = sum(1 for r in group if r['mode'] == 'SOLVED')
            a = sum(1 for r in group if r['mode'] == 'ALMOST')
            print(f"  {size_val:18s} n={len(group):4d}  avg={np.mean(accs)*100:5.1f}%  "
                  f"solved={s}  almost={a}", flush=True)

    print(f"\nBy color change (all tasks):", flush=True)
    for ct in ['no_change', 'new_colors', 'palette_subset', 'mixed', 'substitution']:
        group = [r for r in results if r.get('color_change_type') == ct]
        if group:
            accs = [r['avg_acc'] for r in group]
            s = sum(1 for r in group if r['mode'] == 'SOLVED')
            a = sum(1 for r in group if r['mode'] == 'ALMOST')
            print(f"  {ct:18s} n={len(group):4d}  avg={np.mean(accs)*100:5.1f}%  "
                  f"solved={s}  almost={a}", flush=True)

    # Kill check
    delta = changed_acc - 0.396
    success_count = counts.get('SOLVED', 0)
    print(flush=True)
    print("=" * 65, flush=True)
    print("KILL CHECK", flush=True)
    print("=" * 65, flush=True)
    print(f"Changed-cell: {changed_acc*100:.1f}% (Step 322: 39.6%, delta={delta*100:+.1f}pp)", flush=True)
    print(f"Solved: {success_count} (Step 322: 12)", flush=True)
    kill = delta <= 0.05 and success_count < 20
    success = delta > 0.05 or success_count >= 20
    print(f"Kill (delta<=5pp AND solved<20): {'TRIGGERED' if kill else 'not triggered'}", flush=True)
    print(f"Success (delta>5pp OR solved>=20): {'YES' if success else 'NO'}", flush=True)
    if kill:
        print("\nKILLED", flush=True)
    elif success:
        print("\nSUCCESS", flush=True)
    else:
        print("\nPARTIAL", flush=True)

    print(f"\nElapsed: {elapsed:.1f}s", flush=True)


if __name__ == '__main__':
    main()
