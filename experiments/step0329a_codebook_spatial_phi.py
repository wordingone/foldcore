#!/usr/bin/env python3
"""
Step 329a — Spatial phi_2.

Steps 324-328 all killed. Step 322 (5x5 patch, 39 dims) remains best:
changed-cell 39.6%, 12 solved, 246 ALMOST.

Root cause of global phi failure (Step 328): identical patches -> identical phi_1.
Recursive phi propagates same lack of structure regardless of depth.

Fix: spatial phi. For each cell (r,c):
1. Compute phi_1 for ALL cells in the image using training codebook
2. Aggregate 8 spatial neighbors' phi_1 vectors:
   spatial_desc(r,c) = [phi_1(r-1,c-1), phi_1(r-1,c), ..., phi_1(r+1,c+1)]
                     = 8 * 30 = 240-dim descriptor
3. Match spatial_desc against training spatial_descs (same construction)
4. Predict from nearest match

Key insight: two cells with identical patches in different spatial contexts
(e.g., interior of red object vs. interior of blue object) will have different
spatial_desc because their neighbors have different phi_1 vectors.

Kill: recover >= 3/5 phi-kill tasks.
Phi-kill tasks (1-NN solves but phi fails): ce22a75a, a85d4709, a9f96cdd, d4469b4b, dc433765
"""

import json
import numpy as np
import time
import arckit
from scipy.spatial.distance import cdist
from pathlib import Path

MAX_GRID = 30
N_COLORS = 10
PATCH_R = 2    # 5x5
K_PHI = 3
PHI_DIM = N_COLORS * K_PHI   # 30
SPATIAL_DIM = 8 * PHI_DIM    # 240

PHI_KILL_TASKS = {'ce22a75a', 'a85d4709', 'a9f96cdd', 'd4469b4b', 'dc433765'}

NEIGHBOR_OFFSETS = [(-1, -1), (-1, 0), (-1, 1),
                    (0, -1),           (0, 1),
                    (1, -1),  (1, 0),  (1, 1)]


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


def compute_phi1(query_feats, cb_feats, cb_labels):
    """
    Compute phi_1 for query cells against codebook.
    phi_1[i, c*K:(c+1)*K] = sorted top-K distances from query_i to class-c codebook cells.
    Returns (n_query, PHI_DIM) array.
    """
    n = len(query_feats)
    phi = np.zeros((n, PHI_DIM), dtype=np.float32)
    all_dists = cdist(query_feats, cb_feats, metric='sqeuclidean')

    for c in range(N_COLORS):
        class_idxs = np.where(cb_labels == c)[0]
        if len(class_idxs) == 0:
            continue
        class_dists = all_dists[:, class_idxs]
        k = min(K_PHI, class_dists.shape[1])
        sorted_dists = np.sort(class_dists, axis=1)[:, :k]
        phi[:, c * K_PHI:c * K_PHI + k] = sorted_dists
        # remaining K_PHI-k slots stay 0 if fewer than K_PHI class cells

    return phi


def phi1_to_spatial_desc(phi1_flat, h, w):
    """
    phi1_flat: (h*w, PHI_DIM) — phi_1 for each cell in a h×w grid.
    For each cell (r,c), concat the phi_1 of its 8 spatial neighbors.
    Missing neighbors (border) get zero vector.
    Returns (h*w, SPATIAL_DIM) spatial descriptors.
    """
    zero_vec = np.zeros(PHI_DIM, dtype=np.float32)
    spatial = np.zeros((h * w, SPATIAL_DIM), dtype=np.float32)

    for idx in range(h * w):
        r, c = divmod(idx, w)
        slot = 0
        for dr, dc in NEIGHBOR_OFFSETS:
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w:
                vec = phi1_flat[nr * w + nc]
            else:
                vec = zero_vec
            spatial[idx, slot:slot + PHI_DIM] = vec
            slot += PHI_DIM

    return spatial


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
    train = task.train

    # Step 1: build global feature codebook from all training pairs
    all_feats, all_labels = [], []
    for inp, out in train:
        feats = build_cell_features(inp, out.shape[0], out.shape[1])
        all_feats.append(feats)
        all_labels.append(out.flatten())
    cb_feats = np.vstack(all_feats)
    cb_labels = np.concatenate(all_labels).astype(np.int32)

    # Step 2: compute phi_1 for ALL training cells using the full codebook
    cb_phi1 = compute_phi1(cb_feats, cb_feats, cb_labels)

    # Step 3: build spatial_desc codebook for training cells
    all_spatial, all_spatial_labels = [], []
    feat_offset = 0
    for inp, out in train:
        n_cells = out.shape[0] * out.shape[1]
        img_phi1 = cb_phi1[feat_offset:feat_offset + n_cells]
        feat_offset += n_cells

        h, w = out.shape
        spatial_desc = phi1_to_spatial_desc(img_phi1, h, w)
        all_spatial.append(spatial_desc)
        all_spatial_labels.append(out.flatten().astype(np.int32))

    spatial_cb = np.vstack(all_spatial)
    spatial_labels = np.concatenate(all_spatial_labels)

    # Step 4: evaluate test examples
    pixel_accs, exact_flags, size_ok_flags = [], [], []
    changed_correct, changed_total = 0, 0

    for test_inp, test_out in task.test:
        out_h, out_w = infer_output_size(train, test_inp)

        # phi_1 for test cells against training codebook
        test_feats = build_cell_features(test_inp, out_h, out_w)
        test_phi1 = compute_phi1(test_feats, cb_feats, cb_labels)

        # spatial_desc for test cells
        test_spatial = phi1_to_spatial_desc(test_phi1, out_h, out_w)

        # 1-NN in spatial descriptor space
        dists = cdist(test_spatial, spatial_cb, metric='sqeuclidean')
        pred_flat = spatial_labels[np.argmin(dists, axis=1)]
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
    print("Step 329a — Spatial phi_2", flush=True)
    print(f"Encoding: 8-neighbor phi_1 aggregation = {SPATIAL_DIM}-dim spatial desc", flush=True)
    print(f"PHI_DIM={PHI_DIM} (N_COLORS={N_COLORS} x K_PHI={K_PHI})", flush=True)
    print("Kill: recover >= 3/5 phi-kill tasks", flush=True)
    print(f"Phi-kill: {sorted(PHI_KILL_TASKS)}", flush=True)
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
    phi_kill_results = {}

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

        if task.id in PHI_KILL_TASKS:
            phi_kill_results[task.id] = r['mode']

        if (i + 1) % 100 == 0:
            ch = total_changed_correct / max(total_changed, 1)
            print(f"  {i+1}/{len(tasks)}  changed={ch*100:.1f}%  "
                  f"solved={counts.get('SOLVED',0)}  [{time.time()-t0:.0f}s]", flush=True)

    elapsed = time.time() - t0
    changed_acc = total_changed_correct / max(total_changed, 1)
    avg_pixel = float(np.mean([r['avg_acc'] for r in results]))

    print(flush=True)
    print("=" * 65, flush=True)
    print("STEP 329a RESULTS", flush=True)
    print("=" * 65, flush=True)
    print(f"Avg pixel accuracy:    {avg_pixel*100:.1f}%  (Step 322: ~52%)", flush=True)
    print(f"Changed-cell accuracy: {changed_acc*100:.1f}%  (Step 322: 39.6%)", flush=True)
    print(flush=True)

    print("Failure modes:", flush=True)
    for m in ['SOLVED', 'ALMOST', 'PARTIAL', 'WEAK', 'SIZE_WRONG', 'FAIL', 'ERROR']:
        n = counts.get(m, 0)
        if n:
            print(f"  {m:12s}: {n:4d} ({n/len(tasks)*100:.1f}%)", flush=True)

    solved = [r for r in results if r['mode'] == 'SOLVED']
    print(f"\nSolved ({len(solved)}):", flush=True)
    for r in solved:
        print(f"  {r['id']}  size={r.get('io_size_relation','?')}  "
              f"color={r.get('color_change_type','?')}", flush=True)

    print(flush=True)
    print("PHI-KILL TASK RECOVERY:", flush=True)
    recovered = 0
    for tid in sorted(PHI_KILL_TASKS):
        mode = phi_kill_results.get(tid, 'NOT_FOUND')
        rec = (mode == 'SOLVED')
        if rec:
            recovered += 1
        print(f"  {tid}: {mode} {'<- RECOVERED' if rec else ''}", flush=True)
    print(f"  Total recovered: {recovered}/5 (need >= 3)", flush=True)

    print(flush=True)
    print("By I/O size:", flush=True)
    for sv in ['same', 'output_smaller', 'output_larger']:
        group = [r for r in results if r.get('io_size_relation') == sv]
        if group:
            accs = [r['avg_acc'] for r in group]
            s = sum(1 for r in group if r['mode'] == 'SOLVED')
            a = sum(1 for r in group if r['mode'] == 'ALMOST')
            print(f"  {sv:18s} n={len(group):4d}  avg={np.mean(accs)*100:5.1f}%  "
                  f"solved={s}  almost={a}", flush=True)

    kill = recovered < 3
    success = recovered >= 3
    delta_cc = changed_acc - 0.396

    print(flush=True)
    print("=" * 65, flush=True)
    print("KILL CHECK", flush=True)
    print("=" * 65, flush=True)
    print(f"Changed-cell: {changed_acc*100:.1f}% (Step 322: 39.6%, delta={delta_cc*100:+.1f}pp)", flush=True)
    print(f"Solved: {counts.get('SOLVED',0)} (Step 322: 12)", flush=True)
    print(f"Phi-kill recovered: {recovered}/5 (need >= 3)", flush=True)
    print(f"Kill (recovered < 3): {'TRIGGERED' if kill else 'not triggered'}", flush=True)
    print(f"Success (recovered >= 3): {'YES' if success else 'NO'}", flush=True)

    if kill:
        print("\nKILLED", flush=True)
    else:
        print("\nSUCCESS", flush=True)

    print(f"\nElapsed: {elapsed:.1f}s", flush=True)


if __name__ == '__main__':
    main()
