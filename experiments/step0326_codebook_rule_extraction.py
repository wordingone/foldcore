#!/usr/bin/env python3
"""
Step 326 — Rule extraction for same-size ARC tasks.

The framing:
  Level 0: flat encoding (45% pixel) — thawed by local patches
  Level 1: local patches (52% pixel, 12 solved) — ceiling of pattern matching
  Level 2: rule extraction (this step) — ceiling of deterministic color rules
  Level 3: object reasoning (TBD)
  Level 4: abstract reasoning (counting, symmetry, conditional logic)

This step identifies Level 2 tasks: those solvable by extracting
a consistent color mapping from training pairs and applying it.

Two rule types attempted (in priority order):
  1. GLOBAL: every cell of input_color C maps to output_color D,
     consistently across ALL training pairs. Pure color substitution/recoloring.
  2. POSITIONAL: for each position (r,c), the mapping (r,c,input_color) ->
     output_color is consistent across all training pairs. Handles position-
     dependent transforms (e.g., "top half changes, bottom half stays").
  3. FALLBACK: 5x5 patch 1-NN (Step 322 encoding).

Tracks: which rule type solved each task, and what ALMOST tasks still
require beyond rule extraction (Level 3 candidates).

Kill: solved >= 30 (Step 322 baseline: 12).
"""

import json
import numpy as np
import time
import arckit
from scipy.spatial.distance import cdist
from pathlib import Path

MAX_GRID = 30
N_COLORS = 10
PATCH_R = 2  # 5x5


# ── Rule extraction ───────────────────────────────────────────────────────────

def try_global_color_rule(train_pairs):
    """
    Extract a consistent global color→color mapping from all training pairs.
    Returns dict {in_color: out_color} if consistent, else None.
    Only applicable when all pairs have same I/O size.
    """
    mapping = {}
    for inp, out in train_pairs:
        if inp.shape != out.shape:
            return None
        for ic, oc in zip(inp.flatten().tolist(), out.flatten().tolist()):
            if ic in mapping:
                if mapping[ic] != oc:
                    return None  # contradiction
            else:
                mapping[ic] = oc
    return mapping if mapping else None


def try_positional_rule(train_pairs):
    """
    Extract a consistent positional (r,c,color)->color mapping.
    Only works for same-size I/O where same positions have same color across pairs.
    Returns dict {(r,c,in_color): out_color} if at least one consistent entry exists.
    """
    if any(inp.shape != out.shape for inp, out in train_pairs):
        return None

    mapping = {}
    contradictions = set()

    for inp, out in train_pairs:
        h, w = inp.shape
        for r in range(h):
            for c in range(w):
                key = (r, c, int(inp[r, c]))
                val = int(out[r, c])
                if key in contradictions:
                    continue
                if key in mapping:
                    if mapping[key] != val:
                        contradictions.add(key)
                        del mapping[key]
                else:
                    mapping[key] = val

    return mapping if mapping else None


def apply_global_rule(mapping, test_inp, out_h, out_w):
    """Apply global color rule. For same-size: direct mapping. For different-size: None."""
    if test_inp.shape != (out_h, out_w):
        return None
    pred = np.zeros((out_h, out_w), dtype=np.int32)
    for r in range(out_h):
        for c in range(out_w):
            ic = int(test_inp[r, c])
            pred[r, c] = mapping.get(ic, ic)  # default: identity
    return pred


def apply_positional_rule(mapping, test_inp, out_h, out_w, fallback_pred):
    """
    Apply positional rule where we have entries, use fallback elsewhere.
    Only for same-size.
    """
    if test_inp.shape != (out_h, out_w):
        return None
    pred = fallback_pred.copy() if fallback_pred is not None else np.zeros((out_h, out_w), dtype=np.int32)
    coverage = 0
    for r in range(out_h):
        for c in range(out_w):
            key = (r, c, int(test_inp[r, c]))
            if key in mapping:
                pred[r, c] = mapping[key]
                coverage += 1
    return pred, coverage


# ── 1-NN fallback (Step 322 encoding) ────────────────────────────────────────

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


def build_cell_features_322(inp, out_h, out_w):
    patch_dim = (2 * PATCH_R + 1) ** 2
    feat_dim = patch_dim + N_COLORS + 4
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
        feats = build_cell_features_322(inp, out.shape[0], out.shape[1])
        all_feats.append(feats)
        all_labels.append(out.flatten())
    return np.vstack(all_feats), np.concatenate(all_labels).astype(np.int32)


def predict_1nn_322(task, out_h, out_w, test_inp):
    cb_feats, cb_labels = build_codebook(task.train)
    test_feats = build_cell_features_322(test_inp, out_h, out_w)
    dists = cdist(test_feats, cb_feats, metric='sqeuclidean')
    return cb_labels[np.argmin(dists, axis=1)].reshape(out_h, out_w)


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


# ── Main evaluation ───────────────────────────────────────────────────────────

def evaluate_task(task):
    train = task.train

    # Try to extract rules from training pairs
    global_rule = try_global_color_rule(train)
    pos_rule = try_positional_rule(train)

    pixel_accs, exact_flags, size_ok_flags = [], [], []
    changed_correct, changed_total = 0, 0
    rule_types_used = []

    for test_inp, test_out in task.test:
        out_h, out_w = infer_output_size(train, test_inp)
        pred = None
        rule_used = '1nn'

        # Try global rule first
        if global_rule is not None:
            pred = apply_global_rule(global_rule, test_inp, out_h, out_w)
            if pred is not None:
                rule_used = 'global'

        # Try positional rule if global failed or inapplicable
        if pred is None and pos_rule is not None:
            # Need 1-NN as base for uncovered cells
            base_pred = predict_1nn_322(task, out_h, out_w, test_inp)
            result = apply_positional_rule(pos_rule, test_inp, out_h, out_w, base_pred)
            if result is not None:
                pred, coverage = result
                rule_used = f'positional({coverage}cells)'

        # Fallback to 1-NN
        if pred is None:
            pred = predict_1nn_322(task, out_h, out_w, test_inp)
            rule_used = '1nn'

        rule_types_used.append(rule_used)

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

    primary_rule = rule_types_used[0] if rule_types_used else '1nn'
    if primary_rule.startswith('positional'):
        primary_rule = 'positional'

    return {
        'avg_acc': avg_acc, 'mode': mode,
        'rule_used': primary_rule,
        'has_global_rule': global_rule is not None,
        'changed_correct': changed_correct, 'changed_total': changed_total,
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    t0 = time.time()
    print("Step 326 — Rule Extraction (Level 2 frozen frame)", flush=True)
    print("Global color rule -> positional rule -> 5x5 1-NN fallback", flush=True)
    print("Kill: solved >= 30 (Step 322 baseline: 12)", flush=True)
    print(flush=True)
    print("Frozen frame hierarchy:", flush=True)
    print("  Level 0: flat encoding — thawed by local patches (+15.6pp cc)", flush=True)
    print("  Level 1: local patches — ceiling ~12 solved (pattern matching)", flush=True)
    print("  Level 2: rule extraction — THIS STEP", flush=True)
    print("  Level 3: object reasoning — TBD", flush=True)
    print("  Level 4: abstract reasoning — TBD", flush=True)
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
            r = {'avg_acc': 0.0, 'mode': 'ERROR', 'rule_used': 'error',
                 'has_global_rule': False, 'changed_correct': 0, 'changed_total': 0}
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
    n_tasks = len(results)

    print(flush=True)
    print("=" * 65, flush=True)
    print("STEP 326 RESULTS", flush=True)
    print("=" * 65, flush=True)
    print(f"Avg pixel accuracy:    {avg_pixel*100:.1f}%", flush=True)
    print(f"Changed-cell accuracy: {changed_acc*100:.1f}%  (Step 322: 39.6%)", flush=True)
    print(flush=True)

    print("Failure modes:", flush=True)
    for m in ['SOLVED', 'ALMOST', 'PARTIAL', 'WEAK', 'SIZE_WRONG', 'FAIL']:
        n = counts.get(m, 0)
        if n:
            print(f"  {m:12s}: {n:4d} ({n/n_tasks*100:.1f}%)", flush=True)
    print(flush=True)

    # Rule type breakdown
    print("Rule type usage:", flush=True)
    for rule in ['global', 'positional', '1nn', 'error']:
        group = [r for r in results if r['rule_used'] == rule]
        if not group:
            continue
        solved_n = sum(1 for r in group if r['mode'] == 'SOLVED')
        almost_n = sum(1 for r in group if r['mode'] == 'ALMOST')
        accs = [r['avg_acc'] for r in group]
        print(f"  {rule:12s}: n={len(group):4d}  solved={solved_n:3d}  "
              f"almost={almost_n:3d}  avg={np.mean(accs)*100:.1f}%", flush=True)
    print(flush=True)

    # Tasks that rule extraction solves (global rule)
    global_solved = [r for r in results if r['rule_used'] == 'global' and r['mode'] == 'SOLVED']
    print(f"Global rule solved ({len(global_solved)}):", flush=True)
    for r in global_solved[:20]:
        print(f"  {r['id']}  size={r.get('io_size_relation','?')}  "
              f"color={r.get('color_change_type','?')}", flush=True)
    if len(global_solved) > 20:
        print(f"  ...+{len(global_solved)-20}", flush=True)
    print(flush=True)

    # ALL solved
    all_solved = [r for r in results if r['mode'] == 'SOLVED']
    print(f"Total solved ({len(all_solved)}):", flush=True)
    for r in all_solved:
        print(f"  {r['id']}  rule={r['rule_used']}  size={r.get('io_size_relation','?')}  "
              f"color={r.get('color_change_type','?')}", flush=True)
    print(flush=True)

    # ALMOST tasks not solved by rule extraction — Level 3 candidates
    almost = [r for r in results if r['mode'] == 'ALMOST']
    print(f"ALMOST ({len(almost)}) — Level 3 candidates:", flush=True)
    for field in ['io_size_relation', 'color_change_type']:
        counts2 = {}
        for r in almost:
            v = r.get(field, '?')
            counts2[v] = counts2.get(v, 0) + 1
        vals = sorted(counts2.items(), key=lambda x: -x[1])
        print(f"  {field}: " + ", ".join(f"{v}={c}" for v, c in vals), flush=True)
    # What rules do ALMOST tasks have?
    almost_rules = {}
    for r in almost:
        almost_rules[r['rule_used']] = almost_rules.get(r['rule_used'], 0) + 1
    print(f"  rule_used: " + ", ".join(f"{k}={v}" for k, v in sorted(almost_rules.items(), key=lambda x: -x[1])), flush=True)
    print(flush=True)

    # Compare with Step 322 by category
    print("By color change (all tasks):", flush=True)
    for ct in ['no_change', 'new_colors', 'palette_subset', 'mixed']:
        group = [r for r in results if r.get('color_change_type') == ct]
        if group:
            accs = [r['avg_acc'] for r in group]
            s = sum(1 for r in group if r['mode'] == 'SOLVED')
            a = sum(1 for r in group if r['mode'] == 'ALMOST')
            print(f"  {ct:18s} n={len(group):4d}  avg={np.mean(accs)*100:5.1f}%  "
                  f"solved={s}  almost={a}", flush=True)

    # Has global rule stats
    has_rule = [r for r in results if r.get('has_global_rule')]
    print(f"\nTasks with extractable global color rule: {len(has_rule)}/{n_tasks}", flush=True)
    if has_rule:
        s = sum(1 for r in has_rule if r['mode'] == 'SOLVED')
        a = sum(1 for r in has_rule if r['mode'] == 'ALMOST')
        accs = [r['avg_acc'] for r in has_rule]
        print(f"  Of those: solved={s}  almost={a}  avg={np.mean(accs)*100:.1f}%", flush=True)

    # Kill check
    success_count = counts.get('SOLVED', 0)
    kill = success_count < 30
    success = success_count >= 30

    print(flush=True)
    print("=" * 65, flush=True)
    print("FROZEN FRAME LEVELS — UPDATED", flush=True)
    print("=" * 65, flush=True)
    print(f"  Level 0: flat encoding       — 4 solved  (thawed by patches)", flush=True)
    print(f"  Level 1: local patches (322) — 12 solved (thawed by rule extraction?)", flush=True)
    print(f"  Level 2: rule extraction     — {success_count} solved  {'(THIS STEP)' if not success else '(THAWED BY LEVEL 3)'}", flush=True)
    print(f"  Level 3: object reasoning    — TBD", flush=True)
    print(flush=True)
    print(f"  Level 2 delta over Level 1: +{success_count - 12} solved tasks", flush=True)
    print(f"  Frozen frame cost: {success_count - 12} tasks require rule extraction beyond pattern matching", flush=True)
    print(flush=True)

    print("=" * 65, flush=True)
    print("KILL CHECK", flush=True)
    print("=" * 65, flush=True)
    print(f"Kill (solved < 30): {'TRIGGERED' if kill else 'not triggered'}", flush=True)
    print(f"Success (solved >= 30): {'YES' if success else 'NO'}", flush=True)
    if kill:
        print("\nKILLED", flush=True)
    else:
        print("\nSUCCESS", flush=True)

    print(f"\nElapsed: {elapsed:.1f}s", flush=True)


if __name__ == '__main__':
    main()
