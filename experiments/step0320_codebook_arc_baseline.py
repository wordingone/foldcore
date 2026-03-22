#!/usr/bin/env python3
"""
Step 320: ARC-AGI Baseline with Fold

Apply the fold + phi BLINDLY to ARC-AGI tasks.
Flat vector, dumb encoding. The failure map IS the research.

Encoding: pad each grid to 30x30, flatten. Per-cell prediction:
  feature = [input_flat, row, col, out_h, out_w]
  label = output_color at (row, col)
  prediction = 1-NN over all training cells (vectorized)

Kill criterion: if the fold can't beat random (10%) on pixel accuracy
across ALL tasks, the encoding is wrong — not the fold.
"""

import numpy as np
import time
import sys
import arckit
from scipy.spatial.distance import cdist


MAX_GRID = 30
N_COLORS = 10


def pad_grid(grid, size=MAX_GRID, fill=-1):
    h, w = grid.shape
    padded = np.full((size, size), fill, dtype=np.float32)
    padded[:h, :w] = grid
    return padded


def build_cell_features(inp, out_h, out_w):
    """Build feature matrix for all cells in one (input, output_size) pair.
    Returns (out_h * out_w, 904) matrix."""
    inp_flat = pad_grid(inp).flatten()  # 900-dim
    n_cells = out_h * out_w
    # Tile input across all cells
    feats = np.empty((n_cells, 904), dtype=np.float32)
    feats[:, :900] = inp_flat  # same for all cells
    # Position features
    rows, cols = np.divmod(np.arange(n_cells), out_w)
    feats[:, 900] = rows / MAX_GRID
    feats[:, 901] = cols / MAX_GRID
    feats[:, 902] = out_h / MAX_GRID
    feats[:, 903] = out_w / MAX_GRID
    return feats


def build_codebook(train_examples):
    """Build codebook from all training cells. Returns (features, labels)."""
    all_feats = []
    all_labels = []
    for inp, out in train_examples:
        out_h, out_w = out.shape
        feats = build_cell_features(inp, out_h, out_w)
        all_feats.append(feats)
        all_labels.append(out.flatten())
    return np.vstack(all_feats), np.concatenate(all_labels).astype(np.int32)


def predict_1nn(test_feats, cb_feats, cb_labels):
    """Vectorized 1-NN prediction. Returns predicted labels."""
    # cdist computes all pairwise distances at once
    dists = cdist(test_feats, cb_feats, metric='sqeuclidean')
    best_idx = np.argmin(dists, axis=1)
    return cb_labels[best_idx]


def predict_topk_class(test_feats, cb_feats, cb_labels, K=3):
    """Top-K class vote (the phi mechanism). For each test cell,
    score each class by sum of top-K nearest distances."""
    dists = cdist(test_feats, cb_feats, metric='sqeuclidean')
    n_test = test_feats.shape[0]
    preds = np.zeros(n_test, dtype=np.int32)

    # Unique classes present
    classes = np.unique(cb_labels)

    for i in range(n_test):
        best_score = float('inf')
        best_cls = 0
        for cls in classes:
            cls_dists = np.sort(dists[i, cb_labels == cls])
            k_eff = min(K, len(cls_dists))
            score = cls_dists[:k_eff].sum()
            if score < best_score:
                best_score = score
                best_cls = cls
        preds[i] = best_cls

    return preds


def infer_output_size(train_examples, test_input):
    """Infer test output size from training patterns."""
    out_sizes = [out.shape for _, out in train_examples]
    in_sizes = [inp.shape for inp, _ in train_examples]

    # All outputs same size → fixed output
    if len(set(out_sizes)) == 1:
        return out_sizes[0]

    # Output = input for all → same as test input
    if all(os == is_ for os, is_ in zip(out_sizes, in_sizes)):
        return test_input.shape

    # Consistent ratio
    ratios = set()
    for (inp, out) in train_examples:
        rh = out.shape[0] / inp.shape[0] if inp.shape[0] > 0 else 1
        rw = out.shape[1] / inp.shape[1] if inp.shape[1] > 0 else 1
        ratios.add((round(rh, 4), round(rw, 4)))
    if len(ratios) == 1:
        rh, rw = ratios.pop()
        return (max(1, int(test_input.shape[0] * rh)),
                max(1, int(test_input.shape[1] * rw)))

    # Fallback: most common
    from collections import Counter
    return Counter(out_sizes).most_common(1)[0][0]


def evaluate_task(task, method='1nn', K=3):
    """Evaluate one ARC task."""
    cb_feats, cb_labels = build_codebook(task.train)

    results = []
    for test_inp, test_out in task.test:
        out_h, out_w = infer_output_size(task.train, test_inp)
        test_feats = build_cell_features(test_inp, out_h, out_w)

        if method == 'topk':
            pred_flat = predict_topk_class(test_feats, cb_feats, cb_labels, K)
        else:
            pred_flat = predict_1nn(test_feats, cb_feats, cb_labels)

        pred = pred_flat.reshape(out_h, out_w)

        if pred.shape != test_out.shape:
            results.append({'pixel_acc': 0.0, 'exact': False, 'size_ok': False})
        else:
            acc = float(np.mean(pred == test_out))
            results.append({
                'pixel_acc': acc,
                'exact': bool(np.array_equal(pred, test_out)),
                'size_ok': True,
            })

    # Task metadata
    in_sizes = [inp.shape for inp, _ in task.train]
    out_sizes = [out.shape for _, out in task.train]
    same_size = all(i == o for i, o in zip(in_sizes, out_sizes))
    colors_in = len(set(c for inp, _ in task.train for c in inp.flatten()))
    colors_out = len(set(c for _, out in task.train for c in out.flatten()))

    info = {
        'id': task.id,
        'n_train': len(task.train),
        'same_io_size': same_size,
        'colors_in': colors_in,
        'colors_out': colors_out,
        'cb_size': len(cb_labels),
    }
    return results, info


def classify(results):
    if all(r['exact'] for r in results):
        return 'SOLVED'
    if not all(r['size_ok'] for r in results):
        return 'SIZE_WRONG'
    avg = np.mean([r['pixel_acc'] for r in results])
    if avg > 0.8: return 'ALMOST'
    if avg > 0.5: return 'PARTIAL'
    if avg > 0.2: return 'WEAK'
    return 'FAIL'


def run():
    t0 = time.time()
    train_tasks, _ = arckit.load_data()
    tasks = train_tasks

    print("=" * 70)
    print("STEP 320: ARC-AGI BASELINE WITH FOLD")
    print("=" * 70)
    print(f"Tasks: {len(tasks)}")
    print(f"Encoding: pad-{MAX_GRID}, flatten, per-cell [inp_flat|r|c|h|w]")
    print(f"Method: 1-NN (vectorized via cdist)")
    print(f"Kill: must beat random ({100/N_COLORS:.0f}%) on avg pixel acc")
    print()
    sys.stdout.flush()

    # === Phase 1: 1-NN baseline ===
    all_results = []
    counts = {}

    for i, task in enumerate(tasks):
        try:
            res, info = evaluate_task(task, method='1nn')
            mode = classify(res)
            avg_acc = float(np.mean([r['pixel_acc'] for r in res]))
        except Exception as e:
            res, info, mode, avg_acc = None, {'id': task.id}, 'ERROR', 0.0

        counts[mode] = counts.get(mode, 0) + 1
        all_results.append((info, res, mode, avg_acc))

        if (i + 1) % 100 == 0:
            elapsed = time.time() - t0
            avg_all = np.mean([r[3] for r in all_results])
            print(f"  [{i+1:4d}/{len(tasks)}] avg={avg_all*100:.1f}%  "
                  f"solved={counts.get('SOLVED',0)}  {elapsed:.0f}s")
            sys.stdout.flush()

    elapsed = time.time() - t0

    # === Results ===
    print()
    print("=" * 70)
    print("1-NN RESULTS")
    print("=" * 70)

    accs = [r[3] for r in all_results]
    avg_pixel = np.mean(accs)
    print(f"Avg pixel accuracy: {avg_pixel*100:.1f}%")
    print(f"Random baseline:    {100/N_COLORS:.1f}%")
    print(f"Delta:              {(avg_pixel - 1/N_COLORS)*100:+.1f}pp")
    print()

    print("Failure modes:")
    for m in ['SOLVED', 'ALMOST', 'PARTIAL', 'WEAK', 'SIZE_WRONG', 'FAIL', 'ERROR']:
        n = counts.get(m, 0)
        if n: print(f"  {m:12s}: {n:4d} ({n/len(tasks)*100:.1f}%)")
    print()

    # Solved tasks
    solved = [r[0]['id'] for r in all_results if r[2] == 'SOLVED']
    if solved:
        print(f"Solved ({len(solved)}):")
        for tid in solved[:30]:
            print(f"  {tid}")
        if len(solved) > 30:
            print(f"  ...+{len(solved)-30}")
    print()

    # Accuracy distribution
    print("Accuracy distribution:")
    for t in [0.0, 0.1, 0.2, 0.3, 0.5, 0.8, 0.9, 1.0]:
        n = sum(1 for a in accs if a >= t)
        print(f"  >={t*100:5.1f}%: {n:4d} ({n/len(tasks)*100:.1f}%)")
    print()

    # By properties
    print("By properties:")
    same = [r[3] for r in all_results if r[0].get('same_io_size')]
    diff = [r[3] for r in all_results if r[0] and not r[0].get('same_io_size')]
    if same: print(f"  Same I/O size: {np.mean(same)*100:.1f}% ({len(same)} tasks)")
    if diff: print(f"  Diff I/O size: {np.mean(diff)*100:.1f}% ({len(diff)} tasks)")

    for nt in [2, 3, 4, 5]:
        sub = [r[3] for r in all_results if r[0].get('n_train') == nt]
        if sub: print(f"  {nt} train ex:   {np.mean(sub)*100:.1f}% ({len(sub)} tasks)")

    lo = [r[3] for r in all_results if r[0].get('colors_out', 99) <= 3]
    hi = [r[3] for r in all_results if r[0].get('colors_out', 0) > 3]
    if lo: print(f"  <=3 colors:   {np.mean(lo)*100:.1f}% ({len(lo)} tasks)")
    if hi: print(f"  >3 colors:    {np.mean(hi)*100:.1f}% ({len(hi)} tasks)")
    print()

    # === Phase 2: Top-K on promising tasks ===
    promising = [(i, r) for i, r in enumerate(all_results) if r[3] > 0.15 and r[1]]
    if promising:
        print("=" * 70)
        print(f"TOP-K CLASS VOTE ON {len(promising)} TASKS (>15% 1-NN)")
        print("=" * 70)
        t1 = time.time()
        improvements = []
        for idx, (info, _, mode, acc1) in promising:
            try:
                res2, _ = evaluate_task(tasks[idx], method='topk', K=3)
                acc2 = float(np.mean([r['pixel_acc'] for r in res2]))
                improvements.append((info['id'], acc1, acc2, acc2 - acc1))
            except:
                pass

        if improvements:
            avg1 = np.mean([p[1] for p in improvements])
            avg2 = np.mean([p[2] for p in improvements])
            n_up = sum(1 for p in improvements if p[3] > 0.01)
            n_dn = sum(1 for p in improvements if p[3] < -0.01)

            # Show biggest changes
            improvements.sort(key=lambda p: -abs(p[3]))
            for tid, a1, a2, d in improvements[:10]:
                if abs(d) > 0.01:
                    print(f"  {tid}: {a1*100:.0f}% -> {a2*100:.0f}% ({d*100:+.1f}pp)")

            print(f"\n  Avg 1-NN: {avg1*100:.1f}%  Avg top-K: {avg2*100:.1f}%")
            print(f"  Improved: {n_up}/{len(improvements)}  Hurt: {n_dn}/{len(improvements)}")
            print(f"  Elapsed: {time.time()-t1:.0f}s")
    print()

    # === The Map ===
    print("=" * 70)
    print("THE FAILURE MAP")
    print("=" * 70)
    ns = counts.get('SOLVED', 0)
    na = counts.get('ALMOST', 0)
    np_ = counts.get('PARTIAL', 0)
    nf = len(tasks) - ns - na - np_
    print(f"  SOLVED  {ns:4d}/{len(tasks)} — pixel similarity IS the rule")
    print(f"  ALMOST  {na:4d}/{len(tasks)} — structure partially captured")
    print(f"  PARTIAL {np_:4d}/{len(tasks)} — some signal, structure mostly lost")
    print(f"  FAILED  {nf:4d}/{len(tasks)} — flat encoding destroys relevant structure")
    print()
    pct_beyond = (len(tasks) - ns) / len(tasks) * 100
    print(f"  {pct_beyond:.0f}% of ARC requires structure beyond pixel similarity.")
    print(f"  That {pct_beyond:.0f}% is the frozen frame the fold can't see yet.")
    print(f"\n  Elapsed total: {time.time()-t0:.0f}s")


if __name__ == '__main__':
    np.random.seed(42)
    run()
