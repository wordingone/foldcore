#!/usr/bin/env python3
"""
ARC task taxonomy — structural classification of ARC-AGI tasks.

Classifies each task by heuristic analysis of training pairs:
  1. io_size_relation: same / output_larger / output_smaller / variable
  2. color_change_type: no_change / substitution / palette_subset / new_colors / mixed
  3. position_dependent: bool — spatial structure changes beyond recoloring
  4. has_symmetry: output has horizontal/vertical/diagonal reflection symmetry
  5. involves_counting: output size or content correlates with count in input
  6. has_tiling: output looks like a tiled/repeated version of a subgrid
  7. n_colors_in / n_colors_out: mean per training pair

Output: CSV + JSON summary.
"""

import json
import numpy as np
import arckit
from pathlib import Path


# ── helpers ───────────────────────────────────────────────────────────────────

def grid_shape(g):
    return (g.shape[0], g.shape[1])


def color_sets(inp, out):
    return set(inp.flatten().tolist()), set(out.flatten().tolist())


def classify_color_change(pairs):
    """
    no_change     — output colors == input colors, exact same multiset structure
    substitution  — output is a 1:1 color remapping of input (bijection)
    palette_subset— output colors ⊆ input colors (filtering/masking)
    new_colors    — output introduces colors not in any input
    mixed         — inconsistent across pairs
    """
    verdicts = []
    for inp, out in pairs:
        in_colors, out_colors = color_sets(inp, out)
        new = out_colors - in_colors
        if new:
            verdicts.append('new_colors')
        elif in_colors == out_colors:
            # Same palette — check if it's a bijection (color substitution)
            in_flat, out_flat = inp.flatten(), out.flatten()
            if in_flat.shape == out_flat.shape:
                mapping = {}
                bijection = True
                for a, b in zip(in_flat.tolist(), out_flat.tolist()):
                    if a in mapping:
                        if mapping[a] != b:
                            bijection = False; break
                    else:
                        mapping[a] = b
                verdicts.append('substitution' if bijection else 'no_change')
            else:
                verdicts.append('no_change')
        else:
            verdicts.append('palette_subset')

    if len(set(verdicts)) == 1:
        return verdicts[0]
    return 'mixed'


def classify_size(pairs):
    """
    Returns: same / output_larger / output_smaller / variable
    """
    relations = []
    for inp, out in pairs:
        ih, iw = grid_shape(inp)
        oh, ow = grid_shape(out)
        in_cells = ih * iw
        out_cells = oh * ow
        if (ih, iw) == (oh, ow):
            relations.append('same')
        elif out_cells > in_cells:
            relations.append('output_larger')
        else:
            relations.append('output_smaller')
    if len(set(relations)) == 1:
        return relations[0]
    return 'variable'


def is_position_dependent(pairs):
    """
    Heuristic: if the transformation is purely a color mapping (same shape,
    each input color maps to a fixed output color), it's NOT position-dependent.
    If shapes differ or colors shift based on location, it IS position-dependent.
    """
    for inp, out in pairs:
        if grid_shape(inp) != grid_shape(out):
            return True
        # Check if a global color bijection explains output
        mapping = {}
        dependent = False
        for a, b in zip(inp.flatten().tolist(), out.flatten().tolist()):
            if a in mapping:
                if mapping[a] != b:
                    dependent = True; break
            else:
                mapping[a] = b
        if dependent:
            return True
    return False


def has_reflection_symmetry(grid):
    """Check H/V/diagonal symmetry in a single grid."""
    if grid.shape[0] == grid.shape[1]:
        if np.array_equal(grid, grid.T):
            return True
        if np.array_equal(grid, np.fliplr(grid).T):
            return True
    if np.array_equal(grid, np.flipud(grid)):
        return True
    if np.array_equal(grid, np.fliplr(grid)):
        return True
    return False


def check_symmetry(pairs):
    """True if any training output has reflection symmetry."""
    return any(has_reflection_symmetry(out) for _, out in pairs)


def check_tiling(pairs):
    """
    True if output looks like a tiled repetition of a subgrid.
    Heuristic: output is NxM larger than input and contains tiled copies.
    """
    for inp, out in pairs:
        ih, iw = grid_shape(inp)
        oh, ow = grid_shape(out)
        if oh < ih or ow < iw:
            continue
        if oh % ih == 0 and ow % iw == 0:
            rh, rw = oh // ih, ow // iw
            # Check if output is tiled
            tile_match = True
            for r in range(rh):
                for c in range(rw):
                    block = out[r*ih:(r+1)*ih, c*iw:(c+1)*iw]
                    candidate = inp if not np.array_equal(block, np.zeros_like(block)) else None
                    if candidate is not None and not (
                        np.array_equal(block, inp) or
                        np.array_equal(block, np.flipud(inp)) or
                        np.array_equal(block, np.fliplr(inp)) or
                        np.array_equal(block, np.flipud(np.fliplr(inp)))
                    ):
                        tile_match = False; break
                if not tile_match:
                    break
            if tile_match:
                return True
    return False


def check_counting(pairs):
    """
    Heuristic: output size is a small integer multiple/fraction of
    count of a specific color in input, or output is a 1D/scalar result.
    """
    for inp, out in pairs:
        oh, ow = grid_shape(out)
        # Single-cell output is often a count/arithmetic result
        if oh == 1 and ow == 1:
            return True
        if oh == 1 or ow == 1:
            # 1D strip — often counting-based
            size = oh * ow
            in_colors = set(inp.flatten().tolist()) - {0}
            for c in in_colors:
                cnt = int((inp == c).sum())
                if cnt == size:
                    return True
    return False


# ── main classification ───────────────────────────────────────────────────────

def classify_task(task):
    pairs = task.train  # list of (inp_array, out_array)

    size_rel = classify_size(pairs)
    color_type = classify_color_change(pairs)
    pos_dep = is_position_dependent(pairs)
    symmetry = check_symmetry(pairs)
    tiling = check_tiling(pairs)
    counting = check_counting(pairs)

    mean_in_colors = float(np.mean([len(color_sets(i, o)[0]) for i, o in pairs]))
    mean_out_colors = float(np.mean([len(color_sets(i, o)[1]) for i, o in pairs]))
    mean_in_h = float(np.mean([i.shape[0] for i, _ in pairs]))
    mean_in_w = float(np.mean([i.shape[1] for i, _ in pairs]))
    mean_out_h = float(np.mean([o.shape[0] for _, o in pairs]))
    mean_out_w = float(np.mean([o.shape[1] for _, o in pairs]))

    return {
        'id': task.id,
        'io_size_relation': size_rel,
        'color_change_type': color_type,
        'position_dependent': pos_dep,
        'has_symmetry': symmetry,
        'has_tiling': tiling,
        'involves_counting': counting,
        'n_train_pairs': len(pairs),
        'mean_in_colors': round(mean_in_colors, 2),
        'mean_out_colors': round(mean_out_colors, 2),
        'mean_in_h': round(mean_in_h, 1),
        'mean_in_w': round(mean_in_w, 1),
        'mean_out_h': round(mean_out_h, 1),
        'mean_out_w': round(mean_out_w, 1),
    }


# ── entry point ───────────────────────────────────────────────────────────────

def main():
    print("Loading ARC tasks...", flush=True)
    train_set, eval_set = arckit.load_data()
    all_tasks = list(train_set) + list(eval_set)
    print(f"  {len(train_set)} train + {len(eval_set)} eval = {len(all_tasks)} total", flush=True)

    results = []
    for task in all_tasks:
        try:
            r = classify_task(task)
            r['split'] = 'train' if task in set(train_set) else 'eval'
            results.append(r)
        except Exception as e:
            print(f"  WARN: {task.id} failed: {e}", flush=True)

    # Save JSON
    out_dir = Path(__file__).parent.parent / 'data'
    out_dir.mkdir(exist_ok=True)
    json_path = out_dir / 'arc_taxonomy.json'
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved {len(results)} task records -> {json_path}", flush=True)

    # Print summary table
    print("\n=== Taxonomy Summary ===", flush=True)
    n = len(results)

    for field, label in [
        ('io_size_relation', 'I/O size'),
        ('color_change_type', 'Color change'),
    ]:
        counts = {}
        for r in results:
            v = r[field]
            counts[v] = counts.get(v, 0) + 1
        print(f"\n{label}:", flush=True)
        for v, c in sorted(counts.items(), key=lambda x: -x[1]):
            print(f"  {v:20s} {c:4d}  ({c/n*100:.1f}%)", flush=True)

    for field, label in [
        ('position_dependent', 'Position-dependent'),
        ('has_symmetry', 'Has output symmetry'),
        ('has_tiling', 'Has tiling pattern'),
        ('involves_counting', 'Involves counting'),
    ]:
        true_count = sum(1 for r in results if r[field])
        print(f"\n{label}: {true_count}/{n} ({true_count/n*100:.1f}%)", flush=True)

    print(f"\nMean train pairs per task: {np.mean([r['n_train_pairs'] for r in results]):.1f}", flush=True)
    print(f"Mean input colors: {np.mean([r['mean_in_colors'] for r in results]):.1f}", flush=True)
    print(f"Mean output colors: {np.mean([r['mean_out_colors'] for r in results]):.1f}", flush=True)
    print(f"Mean input size: {np.mean([r['mean_in_h'] for r in results]):.1f}x{np.mean([r['mean_in_w'] for r in results]):.1f}", flush=True)
    print(f"Mean output size: {np.mean([r['mean_out_h'] for r in results]):.1f}x{np.mean([r['mean_out_w'] for r in results]):.1f}", flush=True)


if __name__ == '__main__':
    main()
