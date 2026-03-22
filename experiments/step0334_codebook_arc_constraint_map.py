#!/usr/bin/env python3
"""
Step 334 — Classify every unsolved ARC task by capability gap.

For each task, determine the PRIMARY failure reason from these categories:
  COLOR_MAP        - consistent color substitution (X→Y across all cells)
  SPATIAL_TRANSFORM- output = rotation/flip/translate of input
  OBJECT_IDENTITY  - output depends on which connected component cell belongs to
  SYMMETRY         - output has symmetry not present in input
  PATTERN_COMPLETE - output continues/completes a repeating tiling pattern
  COUNTING         - output depends on counting objects/cells/colors
  CONDITIONAL      - different rules for different regions (if-then logic)
  SIZE_CHANGE      - output has different dimensions than input
  UNKNOWN          - can't determine from heuristic analysis

Cross-referenced with existing arc_taxonomy.json.
Output: data/arc_constraint_map.json + printed histogram.
"""

import json
import numpy as np
import arckit
from pathlib import Path
from collections import deque, Counter

# ─── Taxonomy ──────────────────────────────────────────────────────────────────

TAX_PATH = Path(__file__).parent.parent / 'data' / 'arc_taxonomy.json'
OUT_PATH  = Path(__file__).parent.parent / 'data' / 'arc_constraint_map.json'

# ─── Connected components ───────────────────────────────────────────────────────

def connected_components(grid):
    h, w = grid.shape
    labels = np.full((h, w), -1, dtype=np.int32)
    comp_id = 0
    for r0 in range(h):
        for c0 in range(w):
            if labels[r0, c0] != -1:
                continue
            color = int(grid[r0, c0])
            q = deque([(r0, c0)])
            labels[r0, c0] = comp_id
            while q:
                r, c = q.popleft()
                for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                    rr, cc = r+dr, c+dc
                    if 0 <= rr < h and 0 <= cc < w and labels[rr,cc] == -1 \
                            and int(grid[rr,cc]) == color:
                        labels[rr,cc] = comp_id
                        q.append((rr, cc))
            comp_id += 1
    return labels


# ─── Heuristic checks ──────────────────────────────────────────────────────────

def check_color_map(train):
    """
    True if there's a consistent global color substitution (f: color→color)
    consistent across all training pairs.
    A cell's output color depends only on its input color.
    """
    global_map = {}
    for inp, out in train:
        if inp.shape != out.shape:
            return False
        for r in range(inp.shape[0]):
            for c in range(inp.shape[1]):
                ic = int(inp[r, c])
                oc = int(out[r, c])
                if ic in global_map:
                    if global_map[ic] != oc:
                        return False
                else:
                    global_map[ic] = oc
    # Must actually change something
    return any(k != v for k, v in global_map.items())


def check_no_change(train):
    """True if output = input for all training pairs."""
    for inp, out in train:
        if inp.shape != out.shape:
            return False
        if not np.array_equal(inp, out):
            return False
    return True


def get_transforms(grid):
    """Return all 8 D4 transforms of a grid."""
    transforms = {}
    transforms['identity'] = grid
    transforms['rot90']    = np.rot90(grid, 1)
    transforms['rot180']   = np.rot90(grid, 2)
    transforms['rot270']   = np.rot90(grid, 3)
    transforms['flip_h']   = np.fliplr(grid)
    transforms['flip_v']   = np.flipud(grid)
    transforms['flip_d']   = np.transpose(grid)
    transforms['flip_ad']  = np.fliplr(np.transpose(grid))
    return transforms


def check_spatial_transform(train):
    """
    True if output = D4 transform of input (possibly with color substitution).
    Requires same-size I/O.
    """
    for name in ['rot90', 'rot180', 'rot270', 'flip_h', 'flip_v', 'flip_d', 'flip_ad']:
        consistent = True
        for inp, out in train:
            if inp.shape != out.shape:
                consistent = False
                break
            t = get_transforms(inp).get(name)
            if t is None or t.shape != out.shape:
                consistent = False
                break
            # Allow color substitution alongside transform
            cm = {}
            fail = False
            for r in range(t.shape[0]):
                for c in range(t.shape[1]):
                    tc = int(t[r, c])
                    oc = int(out[r, c])
                    if tc in cm:
                        if cm[tc] != oc:
                            fail = True
                            break
                    else:
                        cm[tc] = oc
                if fail:
                    break
            if fail:
                consistent = False
                break
        if consistent:
            return True, name
    return False, None


def check_object_identity(train):
    """
    True if: cells in the same connected component get the same output color,
    AND that output color depends on the component (not just the cell's input color alone).
    Indicates object-level processing.
    """
    # First check: any cell with same input color but different output color?
    # (necessary condition for object-dependency over simple color map)
    has_object_dep = False
    for inp, out in train:
        if inp.shape != out.shape:
            continue
        color_to_out = {}
        for r in range(inp.shape[0]):
            for c in range(inp.shape[1]):
                ic = int(inp[r, c])
                oc = int(out[r, c])
                if ic in color_to_out:
                    if color_to_out[ic] != oc:
                        has_object_dep = True
                        break
                else:
                    color_to_out[ic] = oc
            if has_object_dep:
                break
        if has_object_dep:
            break

    if not has_object_dep:
        return False

    # Second check: cells in same connected component have same output color?
    comp_uniform = True
    for inp, out in train:
        if inp.shape != out.shape:
            continue
        labels = connected_components(inp)
        max_label = labels.max() + 1
        for comp in range(max_label):
            mask = labels == comp
            out_colors = np.unique(out[mask])
            if len(out_colors) > 1:
                comp_uniform = False
                break
        if not comp_uniform:
            break

    return comp_uniform


def check_symmetry(train):
    """
    True if output has axis-aligned or rotational symmetry that input doesn't.
    """
    def has_mirror_sym(grid):
        return (np.array_equal(grid, np.fliplr(grid)) or
                np.array_equal(grid, np.flipud(grid)))

    def has_rot_sym(grid):
        return (np.array_equal(grid, np.rot90(grid, 2)) or
                (grid.shape[0] == grid.shape[1] and
                 np.array_equal(grid, np.rot90(grid, 1))))

    for inp, out in train:
        if inp.shape == out.shape:
            inp_sym = has_mirror_sym(inp) or has_rot_sym(inp)
            out_sym = has_mirror_sym(out) or has_rot_sym(out)
            if out_sym and not inp_sym:
                return True
    return False


def check_tiling(train):
    """
    True if output can be explained as tiling or repetition of a pattern in input.
    Checks: does output contain a repeating tile that matches input or part of it?
    """
    for inp, out in train:
        oh, ow = out.shape
        ih, iw = inp.shape
        # Check if output is a tiling of the input
        if oh >= ih and ow >= iw and oh % ih == 0 and ow % iw == 0:
            reps_h = oh // ih
            reps_w = ow // iw
            tiled = np.tile(inp, (reps_h, reps_w))
            if np.array_equal(tiled, out):
                return True
            # Also check with color map
            cm = {}
            fail = False
            for r in range(oh):
                for c in range(ow):
                    tc = int(tiled[r, c])
                    oc = int(out[r, c])
                    if tc in cm:
                        if cm[tc] != oc:
                            fail = True
                            break
                    else:
                        cm[tc] = oc
                if fail:
                    break
            if not fail:
                return True

        # Check if input contains a subgrid that tiles to output
        # (for smaller inputs)
        for tile_h in range(1, ih + 1):
            for tile_w in range(1, iw + 1):
                if oh % tile_h == 0 and ow % tile_w == 0:
                    tile = inp[:tile_h, :tile_w]
                    rh = oh // tile_h
                    rw = ow // tile_w
                    tiled = np.tile(tile, (rh, rw))
                    if tiled.shape == out.shape and np.array_equal(tiled, out):
                        return True
    return False


def check_counting(train, tax=None):
    """
    True if there's evidence of counting: output size or content depends on
    count of objects, colors, or cells in input.
    Falls back to taxonomy hint.
    """
    if tax and tax.get('involves_counting'):
        return True
    # Heuristic: output is a single cell or small grid where value = count of something
    for inp, out in train:
        oh, ow = out.shape
        ih, iw = inp.shape
        if oh <= 3 and ow <= 3 and (oh < ih or ow < iw):
            return True
        # Output has fewer unique colors than input
        n_in_colors = len(np.unique(inp))
        n_out_colors = len(np.unique(out))
        if n_out_colors <= 3 and n_in_colors > 5:
            return True
    return False


def check_size_change(train):
    """True if any training pair has different I/O size."""
    for inp, out in train:
        if inp.shape != out.shape:
            return True
    return False


# ─── Primary category assignment ───────────────────────────────────────────────

def classify_task(task, tax=None):
    """
    Assign primary category to a task.
    Returns (category, confidence, notes).
    """
    train = task.train

    # 0. Trivial: no change
    if check_no_change(train):
        return 'NO_CHANGE', 'high', 'output == input'

    # 1. Size change (structural — must handle separately)
    if check_size_change(train):
        # Check if it's also tiling
        if check_tiling(train):
            return 'PATTERN_COMPLETE', 'high', 'size_change+tiling'
        return 'SIZE_CHANGE', 'high', 'output size differs from input'

    # (Same-size tasks below this point)

    # 2. Spatial transform (before color map — transforms may include color maps)
    transform_ok, transform_name = check_spatial_transform(train)
    if transform_ok:
        return 'SPATIAL_TRANSFORM', 'high', f'transform={transform_name}'

    # 3. Simple color map
    if check_color_map(train):
        return 'COLOR_MAP', 'high', 'consistent global color substitution'

    # 4. Object identity (same color -> different output depending on component)
    if check_object_identity(train):
        return 'OBJECT_IDENTITY', 'high', 'per-component output color'

    # 5. Symmetry completion
    if check_symmetry(train):
        return 'SYMMETRY', 'medium', 'output gains symmetry'

    # 6. Tiling/pattern completion (same-size)
    if check_tiling(train):
        return 'PATTERN_COMPLETE', 'medium', 'tiling pattern'

    # 7. Counting
    if check_counting(train, tax):
        return 'COUNTING', 'medium', 'counting-dependent output'

    # 8. Taxonomy hints for remaining
    if tax:
        if tax.get('has_tiling'):
            return 'PATTERN_COMPLETE', 'low', 'taxonomy:has_tiling'
        if tax.get('has_symmetry'):
            return 'SYMMETRY', 'low', 'taxonomy:has_symmetry'

    return 'CONDITIONAL', 'low', 'no clear category — likely conditional/compositional'


# ─── Main ──────────────────────────────────────────────────────────────────────

def main():
    import time
    t0 = time.time()
    print("Step 334 — ARC constraint map (capability gap classification)", flush=True)
    print("Categories: COLOR_MAP, SPATIAL_TRANSFORM, OBJECT_IDENTITY,", flush=True)
    print("  SYMMETRY, PATTERN_COMPLETE, COUNTING, CONDITIONAL, SIZE_CHANGE, NO_CHANGE, UNKNOWN", flush=True)
    print(flush=True)

    # Load taxonomy
    with open(TAX_PATH) as f:
        taxonomy = json.load(f)
    tax_by_id = {t['id']: t for t in taxonomy}

    train_tasks, _ = arckit.load_data()
    tasks = list(train_tasks)
    print(f"Classifying {len(tasks)} tasks...", flush=True)

    results = {}
    for i, task in enumerate(tasks):
        tax = tax_by_id.get(task.id, {})
        try:
            cat, conf, notes = classify_task(task, tax)
        except Exception as e:
            cat, conf, notes = 'UNKNOWN', 'none', f'error: {e}'

        results[task.id] = {
            'category': cat,
            'confidence': conf,
            'notes': notes,
            # Cross-reference with taxonomy
            'io_size_relation': tax.get('io_size_relation', '?'),
            'color_change_type': tax.get('color_change_type', '?'),
            'has_symmetry': tax.get('has_symmetry', False),
            'has_tiling': tax.get('has_tiling', False),
            'involves_counting': tax.get('involves_counting', False),
        }

        if (i + 1) % 200 == 0:
            print(f"  {i+1}/{len(tasks)}  [{time.time()-t0:.1f}s]", flush=True)

    # Save
    with open(OUT_PATH, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {OUT_PATH}", flush=True)

    # Histogram
    cats = [v['category'] for v in results.values()]
    conf_counts = Counter(v['confidence'] for v in results.values())
    cat_counts = Counter(cats)

    print(flush=True)
    print("=" * 65, flush=True)
    print("STEP 334 — ARC CONSTRAINT MAP", flush=True)
    print("=" * 65, flush=True)
    print(f"\nTotal tasks: {len(tasks)}", flush=True)
    print(flush=True)

    ordered = ['COLOR_MAP', 'SPATIAL_TRANSFORM', 'OBJECT_IDENTITY', 'SYMMETRY',
               'PATTERN_COMPLETE', 'COUNTING', 'CONDITIONAL', 'SIZE_CHANGE',
               'NO_CHANGE', 'UNKNOWN']

    print(f"{'Category':22s} {'Count':>6} {'%':>6}  Confidence distribution", flush=True)
    print("-" * 70, flush=True)
    for cat in ordered:
        n = cat_counts.get(cat, 0)
        if n == 0:
            continue
        cat_tasks = [tid for tid, v in results.items() if v['category'] == cat]
        conf_dist = Counter(results[tid]['confidence'] for tid in cat_tasks)
        conf_str = '  '.join(f"{k}:{v}" for k, v in sorted(conf_dist.items()))
        print(f"  {cat:20s} {n:6d} {n/len(tasks)*100:5.1f}%  [{conf_str}]", flush=True)

    # By io_size_relation
    print(flush=True)
    print("Cross-reference: category by I/O size relation", flush=True)
    print("-" * 70, flush=True)
    for sv in ['same', 'output_smaller', 'output_larger']:
        group = {tid: v for tid, v in results.items() if v.get('io_size_relation') == sv}
        if group:
            gc = Counter(v['category'] for v in group.values())
            top = gc.most_common(4)
            top_str = '  '.join(f"{cat}:{n}" for cat, n in top)
            print(f"  {sv:18s} n={len(group):4d}  top: {top_str}", flush=True)

    # CONDITIONAL breakdown — what's in there?
    cond = {tid: v for tid, v in results.items() if v['category'] == 'CONDITIONAL'}
    if cond:
        print(flush=True)
        print(f"CONDITIONAL ({len(cond)} tasks) — taxonomy breakdown:", flush=True)
        print(f"  has_symmetry:     {sum(1 for v in cond.values() if v['has_symmetry'])}", flush=True)
        print(f"  has_tiling:       {sum(1 for v in cond.values() if v['has_tiling'])}", flush=True)
        print(f"  involves_counting:{sum(1 for v in cond.values() if v['involves_counting'])}", flush=True)
        cct = Counter(v['color_change_type'] for v in cond.values())
        print(f"  color_change:     {dict(cct)}", flush=True)

    # SIZE_CHANGE breakdown
    size_ch = {tid: v for tid, v in results.items() if v['category'] == 'SIZE_CHANGE'}
    if size_ch:
        print(flush=True)
        print(f"SIZE_CHANGE ({len(size_ch)} tasks) — by direction:", flush=True)
        szc = Counter(v['io_size_relation'] for v in size_ch.values())
        for sv, n in szc.most_common():
            print(f"  {sv}: {n}", flush=True)

    # Priority ordering for thawing
    print(flush=True)
    print("=" * 65, flush=True)
    print("CONSTRAINT MAP — frozen frames by frequency", flush=True)
    print("(highest count = thaw first for maximum task coverage)", flush=True)
    print("=" * 65, flush=True)
    for cat, n in cat_counts.most_common():
        bar = '#' * (n // 10)
        print(f"  {cat:22s} {n:4d}  {bar}", flush=True)

    elapsed = time.time() - t0
    print(f"\nElapsed: {elapsed:.1f}s", flush=True)


if __name__ == '__main__':
    main()
