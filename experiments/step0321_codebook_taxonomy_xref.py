#!/usr/bin/env python3
"""
Step 321 — Cross-reference failure map with task taxonomy.

Three questions:
  1. Which task categories does flat 1-NN accidentally solve vs fail on?
  2. Changed-cell accuracy: for same-size tasks, % of (output!=input) cells correct?
  3. Diagnosis: what encoding would make phi actually help?

Data sources:
  - Arc tasks via arckit
  - Taxonomy at data/arc_taxonomy.json
  - Evaluation re-run (43s, same method as Step 320)
"""

import json
import numpy as np
import time
import arckit
from scipy.spatial.distance import cdist
from pathlib import Path

MAX_GRID = 30
N_COLORS = 10


# ── evaluation (same as Step 320) ────────────────────────────────────────────

def pad_grid(grid, size=MAX_GRID, fill=-1):
    h, w = grid.shape
    padded = np.full((size, size), fill, dtype=np.float32)
    padded[:h, :w] = grid
    return padded


def build_cell_features(inp, out_h, out_w):
    inp_flat = pad_grid(inp).flatten()
    n_cells = out_h * out_w
    feats = np.empty((n_cells, 904), dtype=np.float32)
    feats[:, :900] = inp_flat
    rows, cols = np.divmod(np.arange(n_cells), out_w)
    feats[:, 900] = rows / MAX_GRID
    feats[:, 901] = cols / MAX_GRID
    feats[:, 902] = out_h / MAX_GRID
    feats[:, 903] = out_w / MAX_GRID
    return feats


def build_codebook(train_examples):
    all_feats, all_labels = [], []
    for inp, out in train_examples:
        out_h, out_w = out.shape
        feats = build_cell_features(inp, out_h, out_w)
        all_feats.append(feats)
        all_labels.append(out.flatten())
    return np.vstack(all_feats), np.concatenate(all_labels).astype(np.int32)


def predict_1nn(test_feats, cb_feats, cb_labels):
    dists = cdist(test_feats, cb_feats, metric='sqeuclidean')
    best_idx = np.argmin(dists, axis=1)
    return cb_labels[best_idx]


def infer_output_size(train_examples, test_input):
    out_sizes = [out.shape for _, out in train_examples]
    in_sizes = [inp.shape for inp, _ in train_examples]
    if len(set(out_sizes)) == 1:
        return out_sizes[0]
    if all(os == is_ for os, is_ in zip(out_sizes, in_sizes)):
        return test_input.shape
    ratios = set()
    for (inp, out) in train_examples:
        rh = out.shape[0] / inp.shape[0] if inp.shape[0] > 0 else 1
        rw = out.shape[1] / inp.shape[1] if inp.shape[1] > 0 else 1
        ratios.add((round(rh, 4), round(rw, 4)))
    if len(ratios) == 1:
        rh, rw = ratios.pop()
        return (max(1, int(test_input.shape[0] * rh)),
                max(1, int(test_input.shape[1] * rw)))
    from collections import Counter
    return Counter(out_sizes).most_common(1)[0][0]


def evaluate_task_detailed(task):
    """Returns per-task metrics including changed-cell accuracy."""
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
            acc = float(np.mean(pred == test_out))
            pixel_accs.append(acc)
            exact_flags.append(bool(np.array_equal(pred, test_out)))
            size_ok_flags.append(True)

            # Changed-cell accuracy (only for same-size I/O in training)
            if test_inp.shape == test_out.shape:
                changed_mask = (test_inp != test_out)
                n_changed = int(changed_mask.sum())
                if n_changed > 0:
                    changed_total += n_changed
                    changed_correct += int((pred[changed_mask] == test_out[changed_mask]).sum())

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
        'id': task.id,
        'mode': mode,
        'avg_acc': avg_acc,
        'changed_correct': changed_correct,
        'changed_total': changed_total,
    }


# ── analysis ──────────────────────────────────────────────────────────────────

def group_stats(values):
    if not values:
        return {'n': 0, 'mean': None}
    return {'n': len(values), 'mean': float(np.mean(values))}


def mode_distribution(modes):
    order = ['SOLVED', 'ALMOST', 'PARTIAL', 'WEAK', 'SIZE_WRONG', 'FAIL']
    total = len(modes)
    out = {}
    for m in order:
        c = modes.count(m)
        out[m] = {'n': c, 'pct': c / total * 100 if total else 0}
    return out


def main():
    t0 = time.time()
    print("Step 321 — Taxonomy cross-reference + changed-cell analysis", flush=True)
    print(flush=True)

    # Load taxonomy
    tax_path = Path(__file__).parent.parent / 'data' / 'arc_taxonomy.json'
    with open(tax_path) as f:
        taxonomy = json.load(f)
    tax_by_id = {t['id']: t for t in taxonomy}
    print(f"Taxonomy: {len(taxonomy)} tasks loaded", flush=True)

    # Load tasks
    train_tasks, eval_tasks = arckit.load_data()
    all_tasks = list(train_tasks)  # match Step 320 (1000 train only)
    print(f"Evaluating {len(all_tasks)} tasks (train split)...", flush=True)

    # Run evaluation
    results = []
    for i, task in enumerate(all_tasks):
        try:
            r = evaluate_task_detailed(task)
        except Exception as e:
            r = {'id': task.id, 'mode': 'ERROR', 'avg_acc': 0.0,
                 'changed_correct': 0, 'changed_total': 0}
        # Merge taxonomy
        tax = tax_by_id.get(task.id, {})
        r.update(tax)
        results.append(r)
        if (i + 1) % 200 == 0:
            print(f"  {i+1}/{len(all_tasks)}  [{time.time()-t0:.0f}s]", flush=True)

    print(f"  done [{time.time()-t0:.1f}s]", flush=True)
    print(flush=True)

    # ── Q1: Category breakdown ──────────────────────────────────────────────
    print("=" * 65, flush=True)
    print("Q1: FAILURE MAP BY CATEGORY", flush=True)
    print("=" * 65, flush=True)

    for field, label in [
        ('io_size_relation', 'I/O Size Relation'),
        ('color_change_type', 'Color Change Type'),
    ]:
        print(f"\n{label}:", flush=True)
        groups = {}
        for r in results:
            v = r.get(field, 'unknown')
            groups.setdefault(v, []).append(r)

        for v, group in sorted(groups.items(), key=lambda x: -len(x[1])):
            accs = [g['avg_acc'] for g in group]
            modes = [g['mode'] for g in group]
            solved = modes.count('SOLVED')
            almost = modes.count('ALMOST')
            fail = modes.count('FAIL') + modes.count('SIZE_WRONG')
            print(f"  {v:20s} n={len(group):4d}  "
                  f"avg={np.mean(accs)*100:5.1f}%  "
                  f"solved={solved}  almost={almost}  fail={fail}", flush=True)

    for field, label in [
        ('position_dependent', 'Position-dependent'),
        ('has_symmetry', 'Has symmetry'),
        ('has_tiling', 'Has tiling'),
    ]:
        print(f"\n{label}:", flush=True)
        true_group = [r for r in results if r.get(field) is True]
        false_group = [r for r in results if r.get(field) is False]
        for name, group in [('True', true_group), ('False', false_group)]:
            if not group:
                continue
            accs = [g['avg_acc'] for g in group]
            modes = [g['mode'] for g in group]
            solved = modes.count('SOLVED')
            print(f"  {name:5s}  n={len(group):4d}  "
                  f"avg={np.mean(accs)*100:5.1f}%  solved={solved}", flush=True)

    # ── Q2: Changed-cell accuracy ───────────────────────────────────────────
    print(flush=True)
    print("=" * 65, flush=True)
    print("Q2: CHANGED-CELL ACCURACY", flush=True)
    print("=" * 65, flush=True)

    total_changed = sum(r['changed_total'] for r in results)
    total_correct = sum(r['changed_correct'] for r in results)

    print(f"\nOverall changed-cell accuracy: ", end='', flush=True)
    if total_changed > 0:
        print(f"{total_correct/total_changed*100:.1f}%  "
              f"({total_correct}/{total_changed} cells)", flush=True)
    else:
        print("N/A", flush=True)

    # Break down by mode
    print("\nChanged-cell accuracy by mode:", flush=True)
    for mode in ['SOLVED', 'ALMOST', 'PARTIAL', 'WEAK', 'FAIL']:
        group = [r for r in results if r['mode'] == mode]
        tc = sum(r['changed_total'] for r in group)
        cc = sum(r['changed_correct'] for r in group)
        if tc > 0:
            print(f"  {mode:12s}: {cc/tc*100:5.1f}%  ({tc} changed cells, {len(group)} tasks)", flush=True)

    # Background vs changed
    print("\nBackground (unchanged) vs changed cells:", flush=True)
    same_size = [r for r in results if r.get('io_size_relation') == 'same'
                 and r['mode'] not in ['SIZE_WRONG', 'ERROR']]
    if same_size:
        # Avg pixel acc for same-size tasks
        avg_same = np.mean([r['avg_acc'] for r in same_size])
        # Changed-cell acc for same-size tasks
        tc = sum(r['changed_total'] for r in same_size)
        cc = sum(r['changed_correct'] for r in same_size)
        print(f"  Same-size tasks (n={len(same_size)}):", flush=True)
        print(f"    Pixel accuracy (all cells):    {avg_same*100:.1f}%", flush=True)
        if tc > 0:
            print(f"    Changed-cell accuracy:         {cc/tc*100:.1f}%  ({tc} cells)", flush=True)
        print(f"    Background inflation estimate: {(avg_same - cc/tc if tc else avg_same)*100:.1f}pp", flush=True)

    # ── Q3: Diagnosis ───────────────────────────────────────────────────────
    print(flush=True)
    print("=" * 65, flush=True)
    print("Q3: DIAGNOSIS — WHAT ENCODING WOULD MAKE PHI HELP?", flush=True)
    print("=" * 65, flush=True)

    # Show the 4 solved tasks + their taxonomy
    solved = [r for r in results if r['mode'] == 'SOLVED']
    print(f"\nSolved tasks ({len(solved)}):", flush=True)
    for r in solved:
        print(f"  {r['id']}  size={r.get('io_size_relation','?')}  "
              f"color={r.get('color_change_type','?')}  "
              f"pos_dep={r.get('position_dependent','?')}", flush=True)

    # Almost tasks — what do they have in common?
    almost = [r for r in results if r['mode'] == 'ALMOST']
    print(f"\nALMOST tasks ({len(almost)}) — category breakdown:", flush=True)
    for field in ['io_size_relation', 'color_change_type']:
        counts = {}
        for r in almost:
            v = r.get(field, 'unknown')
            counts[v] = counts.get(v, 0) + 1
        vals = sorted(counts.items(), key=lambda x: -x[1])
        print(f"  {field}: " + ", ".join(f"{v}={c}" for v, c in vals), flush=True)

    pos_false = [r for r in almost if r.get('position_dependent') is False]
    print(f"  Not position-dependent: {len(pos_false)}/{len(almost)}", flush=True)

    print(flush=True)
    print(f"Elapsed: {time.time()-t0:.1f}s", flush=True)


if __name__ == '__main__':
    main()
