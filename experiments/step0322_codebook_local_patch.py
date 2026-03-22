#!/usr/bin/env python3
"""
Step 322: Local Patch Encoding for ARC-AGI

Spec: replace 900-dim global flat input with 5x5 local patch.
Feature: [patch_5x5_flat | global_color_hist | r/H | c/W | H/30 | W/30] = 39 dims

Kill criterion: changed-cell accuracy must improve >10pp over Step 320 baseline (24%).
"""

import numpy as np
import time
import sys
import arckit
from scipy.spatial.distance import cdist

MAX_GRID = 30
N_COLORS = 10
PATCH_R = 2  # 5x5 = radius 2


def extract_patch(grid, r, c, radius=PATCH_R, fill=-1):
    """Extract (2*radius+1)^2 patch centered at (r,c), pad with fill."""
    h, w = grid.shape
    size = 2 * radius + 1
    patch = np.full((size, size), fill, dtype=np.float32)
    for dr in range(-radius, radius + 1):
        for dc in range(-radius, radius + 1):
            rr, cc = r + dr, c + dc
            if 0 <= rr < h and 0 <= cc < w:
                patch[dr + radius, dc + radius] = grid[rr, cc]
    return patch.flatten()  # 25-dim


def color_histogram(grid):
    """10-bin color histogram, normalized."""
    hist = np.zeros(N_COLORS, dtype=np.float32)
    for c in grid.flatten():
        if 0 <= c < N_COLORS:
            hist[int(c)] += 1
    total = hist.sum()
    if total > 0:
        hist /= total
    return hist


def build_cell_features(inp, out_h, out_w):
    """Feature matrix for all cells: [patch | color_hist | r/H | c/W | H/30 | W/30]."""
    patch_dim = (2 * PATCH_R + 1) ** 2  # 25
    feat_dim = patch_dim + N_COLORS + 4  # 39
    n_cells = out_h * out_w
    feats = np.empty((n_cells, feat_dim), dtype=np.float32)

    hist = color_histogram(inp)  # same for all cells in this input
    in_h, in_w = inp.shape

    for idx in range(n_cells):
        r, c = divmod(idx, out_w)
        # Map output cell to input position (handle different sizes)
        if in_h == out_h and in_w == out_w:
            ir, ic = r, c
        else:
            ir = int(r * in_h / out_h) if out_h > 0 else 0
            ic = int(c * in_w / out_w) if out_w > 0 else 0
            ir = min(ir, in_h - 1)
            ic = min(ic, in_w - 1)

        patch = extract_patch(inp, ir, ic)
        feats[idx, :patch_dim] = patch
        feats[idx, patch_dim:patch_dim + N_COLORS] = hist
        feats[idx, -4] = r / max(out_h, 1)
        feats[idx, -3] = c / max(out_w, 1)
        feats[idx, -2] = out_h / MAX_GRID
        feats[idx, -1] = out_w / MAX_GRID

    return feats


def build_codebook(train_examples):
    all_feats = []
    all_labels = []
    for inp, out in train_examples:
        feats = build_cell_features(inp, out.shape[0], out.shape[1])
        all_feats.append(feats)
        all_labels.append(out.flatten())
    return np.vstack(all_feats), np.concatenate(all_labels).astype(np.int32)


def predict_1nn(test_feats, cb_feats, cb_labels):
    dists = cdist(test_feats, cb_feats, metric='sqeuclidean')
    return cb_labels[np.argmin(dists, axis=1)]


def predict_topk(test_feats, cb_feats, cb_labels, K=3):
    dists = cdist(test_feats, cb_feats, metric='sqeuclidean')
    n = test_feats.shape[0]
    preds = np.zeros(n, dtype=np.int32)
    classes = np.unique(cb_labels)
    for i in range(n):
        best_score, best_cls = float('inf'), 0
        for cls in classes:
            cls_d = np.sort(dists[i, cb_labels == cls])
            k = min(K, len(cls_d))
            s = cls_d[:k].sum()
            if s < best_score:
                best_score, best_cls = s, cls
        preds[i] = best_cls
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


def evaluate_task(task, method='1nn', K=3):
    cb_feats, cb_labels = build_codebook(task.train)
    results = []

    for test_inp, test_out in task.test:
        out_h, out_w = infer_output_size(task.train, test_inp)
        test_feats = build_cell_features(test_inp, out_h, out_w)

        if method == 'topk':
            pred_flat = predict_topk(test_feats, cb_feats, cb_labels, K)
        else:
            pred_flat = predict_1nn(test_feats, cb_feats, cb_labels)

        pred = pred_flat.reshape(out_h, out_w)

        if pred.shape != test_out.shape:
            results.append({'pixel_acc': 0.0, 'exact': False, 'size_ok': False,
                            'changed_correct': 0, 'changed_total': 0})
        else:
            same_size = True
            match = pred == test_out
            # Changed cells: where test_out differs from test_inp (for same-size only)
            if test_inp.shape == test_out.shape:
                changed = test_out != test_inp
                changed_total = int(changed.sum())
                changed_correct = int((match & changed).sum())
            else:
                changed_total = test_out.size
                changed_correct = int(match.sum())

            results.append({
                'pixel_acc': float(match.mean()),
                'exact': bool(np.array_equal(pred, test_out)),
                'size_ok': True,
                'changed_correct': changed_correct,
                'changed_total': changed_total,
            })

    return results


def run():
    t0 = time.time()
    train_tasks, _ = arckit.load_data()
    tasks = train_tasks

    print("=" * 70)
    print("STEP 322: LOCAL PATCH ENCODING FOR ARC-AGI")
    print("=" * 70)
    print(f"Tasks: {len(tasks)}")
    print(f"Encoding: 5x5 local patch + color_hist + position = 39 dims")
    print(f"Kill: changed-cell acc must beat Step 320 baseline (24%) by >10pp")
    print()
    sys.stdout.flush()

    # === 1-NN ===
    all_1nn = []
    counts = {}
    total_changed_correct = 0
    total_changed = 0

    for i, task in enumerate(tasks):
        try:
            res = evaluate_task(task, method='1nn')
            avg_acc = float(np.mean([r['pixel_acc'] for r in res]))
            cc = sum(r['changed_correct'] for r in res)
            ct = sum(r['changed_total'] for r in res)
            total_changed_correct += cc
            total_changed += ct
        except Exception as e:
            res = [{'pixel_acc': 0.0, 'exact': False, 'size_ok': False,
                    'changed_correct': 0, 'changed_total': 0}]
            avg_acc = 0.0

        exact = all(r['exact'] for r in res)
        size_ok = all(r['size_ok'] for r in res)
        if exact:
            mode = 'SOLVED'
        elif not size_ok:
            mode = 'SIZE_WRONG'
        elif avg_acc > 0.8:
            mode = 'ALMOST'
        elif avg_acc > 0.5:
            mode = 'PARTIAL'
        elif avg_acc > 0.2:
            mode = 'WEAK'
        else:
            mode = 'FAIL'

        counts[mode] = counts.get(mode, 0) + 1
        all_1nn.append((task.id, res, mode, avg_acc))

        if (i + 1) % 100 == 0:
            el = time.time() - t0
            avg = np.mean([r[3] for r in all_1nn])
            ch_acc = total_changed_correct / max(total_changed, 1)
            print(f"  [{i+1:4d}/{len(tasks)}] pixel={avg*100:.1f}%  "
                  f"changed={ch_acc*100:.1f}%  solved={counts.get('SOLVED',0)}  {el:.0f}s")
            sys.stdout.flush()

    elapsed = time.time() - t0
    accs = [r[3] for r in all_1nn]
    avg_pixel = np.mean(accs)
    changed_acc = total_changed_correct / max(total_changed, 1)

    print()
    print("=" * 70)
    print("1-NN LOCAL PATCH RESULTS")
    print("=" * 70)
    print(f"Avg pixel accuracy:   {avg_pixel*100:.1f}%  (Step 320: 45.0%)")
    print(f"Changed-cell accuracy: {changed_acc*100:.1f}%  (Step 320: 24.0%)")
    print(f"Random baseline:       {100/N_COLORS:.1f}%")
    print()

    print("Failure modes:")
    for m in ['SOLVED', 'ALMOST', 'PARTIAL', 'WEAK', 'SIZE_WRONG', 'FAIL']:
        n = counts.get(m, 0)
        if n: print(f"  {m:12s}: {n:4d} ({n/len(tasks)*100:.1f}%)")
    print()

    solved = [r[0] for r in all_1nn if r[2] == 'SOLVED']
    if solved:
        print(f"Solved ({len(solved)}):")
        for tid in solved[:20]:
            print(f"  {tid}")
    print()

    print(f"Elapsed: {elapsed:.0f}s")

    # === Top-K on promising ===
    promising = [(i, r) for i, r in enumerate(all_1nn) if r[3] > 0.15]
    if promising:
        print()
        print("=" * 70)
        print(f"TOP-K (K=3) ON {len(promising)} TASKS (>15% 1-NN)")
        print("=" * 70)
        t1 = time.time()
        topk_changed_correct = 0
        topk_changed_total = 0
        improvements = []

        for idx, (tid, _, mode, acc1) in promising:
            try:
                res2 = evaluate_task(tasks[idx], method='topk', K=3)
                acc2 = float(np.mean([r['pixel_acc'] for r in res2]))
                cc = sum(r['changed_correct'] for r in res2)
                ct = sum(r['changed_total'] for r in res2)
                topk_changed_correct += cc
                topk_changed_total += ct
                improvements.append((tid, acc1, acc2, acc2 - acc1))
            except:
                pass

        if improvements:
            avg1 = np.mean([p[1] for p in improvements])
            avg2 = np.mean([p[2] for p in improvements])
            n_up = sum(1 for p in improvements if p[3] > 0.01)
            n_dn = sum(1 for p in improvements if p[3] < -0.01)
            topk_ch = topk_changed_correct / max(topk_changed_total, 1)

            improvements.sort(key=lambda p: -p[3])
            for tid, a1, a2, d in improvements[:10]:
                if d > 0.02:
                    print(f"  {tid}: {a1*100:.0f}% -> {a2*100:.0f}% ({d*100:+.1f}pp)")

            print(f"\n  Avg 1-NN: {avg1*100:.1f}%  Avg top-K: {avg2*100:.1f}%")
            print(f"  Changed-cell top-K: {topk_ch*100:.1f}%")
            print(f"  Improved: {n_up}/{len(improvements)}  Hurt: {n_dn}/{len(improvements)}")
            print(f"  Elapsed: {time.time()-t1:.0f}s")

    # === Kill check ===
    print()
    print("=" * 70)
    print("KILL CHECK")
    print("=" * 70)
    delta = changed_acc - 0.24  # baseline
    print(f"  Changed-cell acc: {changed_acc*100:.1f}% (baseline: 24.0%)")
    print(f"  Delta: {delta*100:+.1f}pp")
    if delta > 0.10:
        print(f"  PASSES kill criterion (>10pp improvement)")
    elif delta > 0:
        print(f"  Improved but below kill threshold (need >10pp)")
    else:
        print(f"  NO IMPROVEMENT or regression")

    print(f"\n  Total elapsed: {time.time()-t0:.0f}s")


if __name__ == '__main__':
    np.random.seed(42)
    run()
