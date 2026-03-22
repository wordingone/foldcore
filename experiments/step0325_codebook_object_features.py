#!/usr/bin/env python3
"""
Step 325 — Object-level features (connected components).

Steps 323 and 324 both killed. Step 322 (5x5 patch, 39 dims) remains best.

Root cause of ALMOST->SOLVED gap: ALMOST tasks are 94.8% same-size.
Interior cells of monochromatic objects all have the same 5x5 patch.
1-NN can't distinguish "change this object" vs "keep this object" from
interior patch alone. Need to know WHICH OBJECT the cell belongs to.

Fix: add connected-component object features per cell.
  - my_color: color of my component
  - my_size: component size, normalized
  - my_cx, my_cy: component centroid, normalized to grid dims
  - my_dx, my_dy: cell's offset from component centroid (relative position within object)
  - n_components: number of distinct components in input

Feature vector: [5x5_patch | global_hist | obj_features | r/H | c/W | H/30 | W/30]
              = [25        | 10          | 7             | 4  ] = 46 dims

Kill: solved >= 20 OR changed-cell >= 45%.
Step 322: changed-cell 39.6%, 12 solved, 246 ALMOST.
"""

import json
import numpy as np
import time
import arckit
from scipy.spatial.distance import cdist
from pathlib import Path
from collections import deque

MAX_GRID = 30
N_COLORS = 10
PATCH_R = 2  # 5x5


def connected_components(grid):
    """
    BFS connected-component labeling (4-connectivity).
    Returns: labels array (same shape as grid), list of component dicts.
    Component dict: {color, cells, size, cx, cy}
    Background (color 0) is treated as its own component(s).
    """
    h, w = grid.shape
    labels = np.full((h, w), -1, dtype=np.int32)
    components = []
    comp_id = 0

    for r0 in range(h):
        for c0 in range(w):
            if labels[r0, c0] != -1:
                continue
            color = int(grid[r0, c0])
            q = deque([(r0, c0)])
            labels[r0, c0] = comp_id
            cells = [(r0, c0)]
            while q:
                r, c = q.popleft()
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    rr, cc = r + dr, c + dc
                    if 0 <= rr < h and 0 <= cc < w and labels[rr, cc] == -1 and int(grid[rr, cc]) == color:
                        labels[rr, cc] = comp_id
                        q.append((rr, cc))
                        cells.append((rr, cc))
            rows = [c[0] for c in cells]
            cols = [c[1] for c in cells]
            components.append({
                'color': color,
                'size': len(cells),
                'cy': float(np.mean(rows)),
                'cx': float(np.mean(cols)),
            })
            comp_id += 1

    return labels, components


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
    """
    46-dim feature: [5x5_patch | global_hist | obj_features | r/H | c/W | H/30 | W/30]
    obj_features: [my_color/9 | my_size/(H*W) | my_cy/H | my_cx/W | my_dy/H | my_dx/W | n_comps/20]
    """
    patch_dim = (2 * PATCH_R + 1) ** 2  # 25
    obj_dim = 7
    feat_dim = patch_dim + N_COLORS + obj_dim + 4  # 46
    n_cells = out_h * out_w
    feats = np.empty((n_cells, feat_dim), dtype=np.float32)

    in_h, in_w = inp.shape
    global_h = color_hist(inp)
    labels, components = connected_components(inp)
    n_comps = len(components)
    total_cells = in_h * in_w

    for idx in range(n_cells):
        r, c = divmod(idx, out_w)
        # Map output cell to input position
        if in_h == out_h and in_w == out_w:
            ir, ic = r, c
        else:
            ir = min(int(r * in_h / out_h), in_h - 1) if out_h > 0 else 0
            ic = min(int(c * in_w / out_w), in_w - 1) if out_w > 0 else 0

        patch = extract_patch(inp, ir, ic)

        # Object features for cell (ir, ic)
        comp_id = int(labels[ir, ic])
        comp = components[comp_id]
        my_color = comp['color'] / max(N_COLORS - 1, 1)
        my_size = comp['size'] / max(total_cells, 1)
        my_cy = comp['cy'] / max(in_h - 1, 1)
        my_cx = comp['cx'] / max(in_w - 1, 1)
        my_dy = (ir - comp['cy']) / max(in_h - 1, 1)
        my_dx = (ic - comp['cx']) / max(in_w - 1, 1)
        n_comp_feat = n_comps / 20.0

        offset = 0
        feats[idx, offset:offset + patch_dim] = patch; offset += patch_dim
        feats[idx, offset:offset + N_COLORS] = global_h; offset += N_COLORS
        feats[idx, offset] = my_color; offset += 1
        feats[idx, offset] = my_size; offset += 1
        feats[idx, offset] = my_cy; offset += 1
        feats[idx, offset] = my_cx; offset += 1
        feats[idx, offset] = my_dy; offset += 1
        feats[idx, offset] = my_dx; offset += 1
        feats[idx, offset] = n_comp_feat; offset += 1
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
        'avg_acc': avg_acc, 'mode': mode,
        'changed_correct': changed_correct, 'changed_total': changed_total,
    }


def main():
    t0 = time.time()
    print("Step 325 — Object features (connected components)", flush=True)
    print("Feature: 5x5_patch + global_hist + obj_features + pos = 46 dims", flush=True)
    print("Steps 323+324 killed. Step 322 baseline: cc=39.6%, 12 solved.", flush=True)
    print("Kill: solved>=20 OR changed-cell>=45%", flush=True)
    print(flush=True)

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
    print("STEP 325 RESULTS", flush=True)
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

    almost = [r for r in results if r['mode'] == 'ALMOST']
    if almost:
        print(f"\nALMOST ({len(almost)}) — categories:", flush=True)
        for field in ['io_size_relation', 'color_change_type']:
            counts2 = {}
            for r in almost:
                v = r.get(field, '?')
                counts2[v] = counts2.get(v, 0) + 1
            vals = sorted(counts2.items(), key=lambda x: -x[1])
            print(f"  {field}: " + ", ".join(f"{v}={c}" for v, c in vals), flush=True)

    print(f"\nBy I/O size:", flush=True)
    for sv in ['same', 'output_smaller', 'output_larger']:
        group = [r for r in results if r.get('io_size_relation') == sv]
        if group:
            accs = [r['avg_acc'] for r in group]
            s = sum(1 for r in group if r['mode'] == 'SOLVED')
            a = sum(1 for r in group if r['mode'] == 'ALMOST')
            print(f"  {sv:18s} n={len(group):4d}  avg={np.mean(accs)*100:5.1f}%  "
                  f"solved={s}  almost={a}", flush=True)

    print(f"\nBy color change:", flush=True)
    for ct in ['no_change', 'new_colors', 'palette_subset', 'mixed']:
        group = [r for r in results if r.get('color_change_type') == ct]
        if group:
            accs = [r['avg_acc'] for r in group]
            s = sum(1 for r in group if r['mode'] == 'SOLVED')
            a = sum(1 for r in group if r['mode'] == 'ALMOST')
            print(f"  {ct:18s} n={len(group):4d}  avg={np.mean(accs)*100:5.1f}%  "
                  f"solved={s}  almost={a}", flush=True)

    success_count = counts.get('SOLVED', 0)
    kill = changed_acc <= 0.45 and success_count < 20
    success = changed_acc > 0.45 or success_count >= 20
    delta_cc = changed_acc - 0.396

    print(flush=True)
    print("=" * 65, flush=True)
    print("KILL CHECK", flush=True)
    print("=" * 65, flush=True)
    print(f"Changed-cell: {changed_acc*100:.1f}% (Step 322: 39.6%, delta={delta_cc*100:+.1f}pp)", flush=True)
    print(f"Solved: {success_count} (Step 322: 12, delta={success_count-12:+d})", flush=True)
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
