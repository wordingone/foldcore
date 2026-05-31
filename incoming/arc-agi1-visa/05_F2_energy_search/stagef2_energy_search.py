"""Stage F2 — Energy-Ranked Best-First vs BFS Reach

ONE VARIABLE: candidate ordering (energy-ranked within batch vs random BFS).
Grammar, budget, time wall, held set, per-task seed all frozen from Stage 1k/F1.

Pre-reg: 05_F2_energy_search/pre_reg/stagef2_energy_search_preregistration.md
Issue:   wordingone/the-search#17

Usage:
  python stagef2_energy_search.py             # canonical (200 tasks, both arms)
  python stagef2_energy_search.py --smoke     # 3 tasks, TEMP path
  python stagef2_energy_search.py --test-gate # gate injection tests only (no run)
"""

import argparse
import collections
import gc
import hashlib
import json
import math
import os
import random
import sys
import time
from collections import deque

import numpy as np

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DATA_PATH = "B:/M/the-search/incoming/arc-agi1-visa/ARC-AGI/data"
S1D_PATH = (
    "B:/M/the-search/incoming/arc-agi1-visa/03_R4_transfer_wall/"
    "stage1d_premise_truth_b30000_minimality_result.json"
)
OUT_DIR = "B:/M/the-search/incoming/arc-agi1-visa/05_F2_energy_search"
RESULT_PATH = os.path.join(OUT_DIR, "stagef2_energy_search_result.json")
TEMP_PATH = os.path.join(OUT_DIR, "stagef2_energy_search_TEMP.json")
MEASURE_TEMP_PATH = os.path.join(OUT_DIR, "stagef2_measurement_smoke_TEMP.json")

# Stage 1k armseed7 canonical solved task IDs (D=10/200); used for --measure-smoke targeting
S1K_SOLVED_IDS = [
    "1cf80156", "3618c87e", "3c9b0459", "67a3c6ac", "68b16354",
    "74dd1130", "a1570a43", "a740d043", "a79310a0", "f25fbde4",
]

F1_RESULT_PATH = (
    "B:/M/the-search/incoming/arc-agi1-visa/04_F1_energy_probe/"
    "stagef1_energy_probe_result.json"
)

# ---------------------------------------------------------------------------
# Frozen constants (pre-registered)
# ---------------------------------------------------------------------------
FROZEN_ENERGY_HASH = "2cd1ed64b357292b"
EVAL_SPLIT_HASH = "08be6f980cec510c"
PREV_SPACE_HASH = "4978b6739bf55beb"   # Stage 1k space hash (grammar unchanged)
ARM_D_SEED = 7
BUDGET_B = 300_000
TIME_PER_TASK_D = 1800.0              # raised from Stage 1k's 600s
MAX_DEPTH = 3
N_HELD = 200
BFS_EXPECTED_N_SOLVED = 10            # Stage 1k armseed7 canonical result
BATCH_SIZE_ENERGY = 500               # programs per energy-ranking batch
FROZEN_LINK_TOL = 1e-4               # max tolerated |computed - f1_energy| for behavioral link

# E_theta training constants (deterministic; must match F1 exactly)
TRAIN_N_EPOCHS = 100
TRAIN_LR = 1e-3
TRAIN_BATCH_SIZE = 64
TRAIN_RNG_SEED = 0
K_FRAC_LADDER = [0.05, 0.10, 0.20, 0.30, 0.40, 0.60]
N_NEAR_MISS_PER_K = 2
N_RANDOM_FAR = 5

# ---------------------------------------------------------------------------
# Grammar — copied verbatim from stage1k_time_wall.py
# ---------------------------------------------------------------------------

_OBJ_CACHE = {}


def get_bg(grid):
    vals, counts = np.unique(grid, return_counts=True)
    return int(vals[np.argmax(counts)])


def extract_objects(grid):
    h, w = grid.shape
    visited = np.zeros((h, w), dtype=bool)
    objects = []
    for r in range(h):
        for c in range(w):
            if not visited[r, c]:
                color = int(grid[r, c])
                cells = []
                q = deque([(r, c)])
                visited[r, c] = True
                while q:
                    rr, cc = q.popleft()
                    cells.append((rr, cc))
                    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        nr, nc = rr + dr, cc + dc
                        if (
                            0 <= nr < h and 0 <= nc < w
                            and not visited[nr, nc]
                            and int(grid[nr, nc]) == color
                        ):
                            visited[nr, nc] = True
                            q.append((nr, nc))
                objects.append({"color": color, "cells": cells, "area": len(cells)})
    return objects


def _extract_cached(grid):
    key = grid.tobytes()
    if key not in _OBJ_CACHE:
        _OBJ_CACHE[key] = extract_objects(grid)
    return _OBJ_CACHE[key]


def _pred_largest(objs):
    if not objs:
        return []
    mx = max(o["area"] for o in objs)
    return [o for o in objs if o["area"] == mx]


def _pred_smallest(objs):
    if not objs:
        return []
    mn = min(o["area"] for o in objs)
    return [o for o in objs if o["area"] == mn]


PREDICATES = {
    "largest": lambda g, objs: _pred_largest(objs),
    "smallest": lambda g, objs: _pred_smallest(objs),
    "non_bg": lambda g, objs: [o for o in objs if o["color"] != get_bg(g)],
    "bg_obj": lambda g, objs: [o for o in objs if o["color"] == get_bg(g)],
    "unique_color": lambda g, objs: [
        o for o in objs
        if collections.Counter(x["color"] for x in objs)[o["color"]] == 1
    ],
    "most_common_c": lambda g, objs: (
        [
            o for o in objs
            if o["color"] == collections.Counter(x["color"] for x in objs).most_common(1)[0][0]
        ]
        if objs else []
    ),
    "all": lambda g, objs: objs,
}
for _c in range(1, 10):
    PREDICATES[f"color_{_c}"] = (lambda c: lambda g, objs: [o for o in objs if o["color"] == c])(_c)
PRED_NAMES = sorted(PREDICATES.keys())


def _recolor(new_c):
    def fn(grid, sel):
        g = grid.copy()
        for obj in sel:
            for r, c in obj["cells"]:
                g[r, c] = new_c
        return g
    return fn


def _delete(grid, sel):
    g = grid.copy()
    bg = get_bg(grid)
    for obj in sel:
        for r, c in obj["cells"]:
            g[r, c] = bg
    return g


def _keep_only(grid, sel):
    g = np.full_like(grid, get_bg(grid))
    for obj in sel:
        for r, c in obj["cells"]:
            g[r, c] = obj["color"]
    return g


def _translate(dy, dx):
    def fn(grid, sel):
        h, w = grid.shape
        g = grid.copy()
        bg = get_bg(grid)
        for obj in sel:
            for r, c in obj["cells"]:
                g[r, c] = bg
        for obj in sel:
            for r, c in obj["cells"]:
                nr, nc = r + dy, c + dx
                if 0 <= nr < h and 0 <= nc < w:
                    g[nr, nc] = obj["color"]
        return g
    return fn


def _geom_flip_h(grid, sel):
    g = grid.copy()
    bg = get_bg(grid)
    for obj in sel:
        cells = obj["cells"]
        if not cells:
            continue
        cols = [c for r, c in cells]
        c_min, c_max = min(cols), max(cols)
        for r, c in cells:
            g[r, c] = bg
        for r, c in cells:
            nc = c_min + (c_max - c)
            if 0 <= r < g.shape[0] and 0 <= nc < g.shape[1]:
                g[r, nc] = obj["color"]
    return g


def _geom_flip_v(grid, sel):
    g = grid.copy()
    bg = get_bg(grid)
    for obj in sel:
        cells = obj["cells"]
        if not cells:
            continue
        rows = [r for r, c in cells]
        r_min, r_max = min(rows), max(rows)
        for r, c in cells:
            g[r, c] = bg
        for r, c in cells:
            nr = r_min + (r_max - r)
            if 0 <= nr < g.shape[0] and 0 <= c < g.shape[1]:
                g[nr, c] = obj["color"]
    return g


def _geom_rot_90(grid, sel):
    g = grid.copy()
    bg = get_bg(grid)
    for obj in sel:
        cells = obj["cells"]
        if not cells:
            continue
        rows = [r for r, c in cells]
        cols = [c for r, c in cells]
        r_min, c_min = min(rows), min(cols)
        c_max = max(cols)
        w = c_max - c_min + 1
        for r, c in cells:
            g[r, c] = bg
        for r, c in cells:
            r_rel = r - r_min
            c_rel = c - c_min
            nr = r_min + (w - 1 - c_rel)
            nc = c_min + r_rel
            if 0 <= nr < g.shape[0] and 0 <= nc < g.shape[1]:
                g[nr, nc] = obj["color"]
    return g


def _geom_rot_180(grid, sel):
    g = grid.copy()
    bg = get_bg(grid)
    for obj in sel:
        cells = obj["cells"]
        if not cells:
            continue
        rows = [r for r, c in cells]
        cols = [c for r, c in cells]
        r_min, r_max = min(rows), max(rows)
        c_min, c_max = min(cols), max(cols)
        for r, c in cells:
            g[r, c] = bg
        for r, c in cells:
            nr = r_min + (r_max - r)
            nc = c_min + (c_max - c)
            if 0 <= nr < g.shape[0] and 0 <= nc < g.shape[1]:
                g[nr, nc] = obj["color"]
    return g


def _geom_rot_270(grid, sel):
    g = grid.copy()
    bg = get_bg(grid)
    for obj in sel:
        cells = obj["cells"]
        if not cells:
            continue
        rows = [r for r, c in cells]
        cols = [c for r, c in cells]
        r_min, r_max = min(rows), max(rows)
        c_min = min(cols)
        h = r_max - r_min + 1
        for r, c in cells:
            g[r, c] = bg
        for r, c in cells:
            r_rel = r - r_min
            c_rel = c - c_min
            nr = r_min + c_rel
            nc = c_min + (h - 1 - r_rel)
            if 0 <= nr < g.shape[0] and 0 <= nc < g.shape[1]:
                g[nr, nc] = obj["color"]
    return g


RELATE_MODES = ["adjacent_down", "adjacent_left", "adjacent_right", "adjacent_up"]


def _largest_cc(selected_objects):
    if not selected_objects:
        return None
    return sorted(
        selected_objects,
        key=lambda o: (
            -o["area"],
            min(r for r, c in o["cells"]),
            min(c for r, c in o["cells"]),
        ),
    )[0]


def _map_relate(grid, src_sel, anchor_sel, mode):
    anchor_obj = _largest_cc(anchor_sel)
    if anchor_obj is None:
        return grid.copy()
    anc_cells = anchor_obj["cells"]
    anc_r0 = min(r for r, c in anc_cells)
    anc_r1 = max(r for r, c in anc_cells)
    anc_c0 = min(c for r, c in anc_cells)
    anc_c1 = max(c for r, c in anc_cells)
    h, w = grid.shape
    g = grid.copy()
    bg = get_bg(grid)
    for obj in src_sel:
        for r, c in obj["cells"]:
            g[r, c] = bg
    for obj in src_sel:
        cells = obj["cells"]
        if not cells:
            continue
        obj_r0 = min(r for r, c in cells)
        obj_r1 = max(r for r, c in cells)
        obj_c0 = min(c for r, c in cells)
        obj_c1 = max(c for r, c in cells)
        if mode == "adjacent_left":
            dr = anc_r0 - obj_r0
            dc = (anc_c0 - 1) - obj_c1
        elif mode == "adjacent_right":
            dr = anc_r0 - obj_r0
            dc = (anc_c1 + 1) - obj_c0
        elif mode == "adjacent_up":
            dr = (anc_r0 - 1) - obj_r1
            dc = anc_c0 - obj_c0
        elif mode == "adjacent_down":
            dr = (anc_r1 + 1) - obj_r0
            dc = anc_c0 - obj_c0
        else:
            dr, dc = 0, 0
        for r, c in cells:
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w:
                g[nr, nc] = obj["color"]
    return g


TRANSFORMS = {
    "delete": _delete,
    "keep_only": _keep_only,
    "geom_flip_h": _geom_flip_h,
    "geom_flip_v": _geom_flip_v,
    "geom_rot_90": _geom_rot_90,
    "geom_rot_180": _geom_rot_180,
    "geom_rot_270": _geom_rot_270,
}
for _c in range(10):
    TRANSFORMS[f"recolor_{_c}"] = (lambda c: lambda g, sel: _recolor(c)(g, sel))(_c)
for _dy, _dx in [(-1, 0), (1, 0), (0, -1), (0, 1), (1, 1), (-1, -1), (1, -1), (-1, 1)]:
    TRANSFORMS[f"translate_{_dy:+d}_{_dx:+d}"] = (
        lambda dy, dx: lambda g, sel: _translate(dy, dx)(g, sel)
    )(_dy, _dx)
TRANSFORM_NAMES = sorted(TRANSFORMS.keys())

GRID_PRIMS = {
    "flip_h": lambda g: g[:, ::-1],
    "flip_v": lambda g: g[::-1],
    "rot_90": lambda g: np.rot90(g, 1),
    "rot_180": lambda g: np.rot90(g, 2),
    "rot_270": lambda g: np.rot90(g, 3),
    "tr": lambda g: g.T,
    "up2": lambda g: np.repeat(np.repeat(g, 2, 0), 2, 1),
    "down2": lambda g: g[::2, ::2],
    "id": lambda g: g,
    "crop": lambda g: (
        lambda bg: (
            lambda nz: g[nz[:, 0].min():nz[:, 0].max() + 1, nz[:, 1].min():nz[:, 1].max() + 1]
            if len(nz) > 0 else g
        )(np.argwhere(g != bg))
    )(get_bg(g)),
}
PRIM_NAMES = sorted(GRID_PRIMS.keys())

_RECOLOR_TRANSFORMS = sorted(t for t in TRANSFORM_NAMES if t.startswith("recolor_"))
_TRANSLATE_TRANSFORMS = sorted(t for t in TRANSFORM_NAMES if t.startswith("translate_"))
_DELETE_TRANSFORMS = ["delete"]
_KEEPONLY_TRANSFORMS = ["keep_only"]
_GEOM_TRANSFORMS = sorted(t for t in TRANSFORM_NAMES if t.startswith("geom_"))

MAP_TRANSFORM_FAMILIES = {
    "MAP_RECOLOR": _RECOLOR_TRANSFORMS,
    "MAP_DELETE": _DELETE_TRANSFORMS,
    "MAP_KEEPONLY": _KEEPONLY_TRANSFORMS,
    "MAP_TRANSLATE": _TRANSLATE_TRANSFORMS,
    "MAP_GEOM": _GEOM_TRANSFORMS,
}
PRIM_SKELETON_NAMES = [f"PRIM_{n.upper()}" for n in PRIM_NAMES]
SKELETON_ORDER = (
    PRIM_SKELETON_NAMES
    + ["MAP_DELETE", "MAP_KEEPONLY", "MAP_TRANSLATE", "MAP_GEOM", "MAP_RELATE", "MAP_RECOLOR"]
)


def skeleton_fills(sk_name):
    if sk_name.startswith("PRIM_"):
        prim_n = sk_name[5:].lower()
        return [("prim", prim_n)] if prim_n in GRID_PRIMS else []
    if sk_name == "MAP_RELATE":
        return [
            ("map_relate", src_pred, anchor_pred, mode)
            for src_pred in PRED_NAMES
            for anchor_pred in PRED_NAMES
            for mode in RELATE_MODES
            if src_pred != anchor_pred
        ]
    if sk_name in MAP_TRANSFORM_FAMILIES:
        trs = MAP_TRANSFORM_FAMILIES[sk_name]
        return [("map_apply", pred, tr) for pred in PRED_NAMES for tr in trs]
    return []


def all_leaves_expanded():
    result = []
    for sk in SKELETON_ORDER:
        result.extend(skeleton_fills(sk))
    return result


def tuple_to_prog(tup, leaves):
    if len(tup) == 1:
        return leaves[tup[0]]
    return ("compose", leaves[tup[0]], tuple_to_prog(tup[1:], leaves))


def eval_program(prog, grid):
    try:
        return _eval_prog(prog, np.array(grid, dtype=np.int64))
    except Exception:
        return None


def _eval_prog(prog, g):
    t = prog[0]
    if t == "prim":
        return GRID_PRIMS[prog[1]](g)
    if t == "map_apply":
        objs = _extract_cached(g)
        selected = PREDICATES[prog[1]](g, objs)
        if not selected:
            return g.copy()
        return TRANSFORMS[prog[2]](g, selected)
    if t == "map_relate":
        objs = _extract_cached(g)
        src_sel = PREDICATES[prog[1]](g, objs)
        anchor_sel = PREDICATES[prog[2]](g, objs)
        if not src_sel or not anchor_sel:
            return g.copy()
        return _map_relate(g, src_sel, anchor_sel, prog[3])
    if t == "compose":
        r = _eval_prog(prog[2], g)
        if r is None:
            return None
        return _eval_prog(prog[1], r)
    return None


def check_prog(prog, task_io):
    for pair in task_io:
        inp = np.array(pair["input"], dtype=np.int64)
        out = np.array(pair["output"], dtype=np.int64)
        r = eval_program(prog, inp)
        if r is None or r.shape != out.shape or not np.array_equal(r, out):
            return False
    return True


def compute_space_hash(leaves):
    leaf_strs = [str(l) for l in leaves]
    space_descriptor = json.dumps(
        {
            "n_leaves": len(leaves),
            "max_depth": MAX_DEPTH,
            "canonical_form": "left_linear",
            "leaf_ordering": "SKELETON_ORDER",
            "leaf_repr_sample": leaf_strs[:5] + ["..."] + leaf_strs[-3:],
        },
        sort_keys=True,
    )
    return hashlib.sha256(space_descriptor.encode()).hexdigest()[:16]


# ---------------------------------------------------------------------------
# E_theta (MLP) — copied from stagef1_energy_probe.py; must stay byte-identical
# ---------------------------------------------------------------------------

def relu(x):
    return np.maximum(0, x)


class MLP:
    def __init__(self, rng_seed=0):
        np.random.seed(rng_seed)
        self.W1 = np.random.randn(36, 64) * 0.1
        self.b1 = np.zeros(64)
        self.W2 = np.random.randn(64, 32) * 0.1
        self.b2 = np.zeros(32)
        self.W3 = np.random.randn(32, 1) * 0.1
        self.b3 = np.zeros(1)

    def forward(self, X):
        h1 = relu(X @ self.W1 + self.b1)
        h2 = relu(h1 @ self.W2 + self.b2)
        return (h2 @ self.W3 + self.b3).squeeze(-1)

    def backward(self, X, y):
        N = X.shape[0]
        h1 = relu(X @ self.W1 + self.b1)
        h2 = relu(h1 @ self.W2 + self.b2)
        out = (h2 @ self.W3 + self.b3).squeeze(-1)
        err = out - y
        loss = (err ** 2).mean()
        dout = (2.0 / N) * err.reshape(-1, 1)
        dW3 = h2.T @ dout
        db3 = dout.sum(0)
        dh2 = dout @ self.W3.T
        dh2 *= (h2 > 0).astype(float)
        dW2 = h1.T @ dh2
        db2 = dh2.sum(0)
        dh1 = dh2 @ self.W2.T
        dh1 *= (h1 > 0).astype(float)
        dW1 = X.T @ dh1
        db1 = dh1.sum(0)
        return loss, {"W1": dW1, "b1": db1, "W2": dW2, "b2": db2, "W3": dW3, "b3": db3}

    def step(self, grads, m, v, t, lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8):
        t += 1
        for name, g in grads.items():
            if name not in m:
                m[name] = np.zeros_like(g)
                v[name] = np.zeros_like(g)
            m[name] = beta1 * m[name] + (1 - beta1) * g
            v[name] = beta2 * v[name] + (1 - beta2) * g ** 2
            m_hat = m[name] / (1 - beta1 ** t)
            v_hat = v[name] / (1 - beta2 ** t)
            p = getattr(self, name)
            p -= lr * m_hat / (np.sqrt(v_hat) + eps)
        return t

    def score(self, feat_list):
        X = np.array(feat_list, dtype=np.float32)
        return self.forward(X).tolist()

    def score_one(self, feat):
        X = np.array([feat], dtype=np.float32)
        return float(self.forward(X)[0])


def train_mlp(X_train, y_train, n_epochs=100, batch_size=64, lr=1e-3, rng_seed=0):
    model = MLP(rng_seed=rng_seed)
    m, v, t = {}, {}, 0
    N = len(X_train)
    idx = np.arange(N)
    for epoch in range(n_epochs):
        np.random.shuffle(idx)
        epoch_loss = 0.0
        n_batches = 0
        for start in range(0, N, batch_size):
            batch = idx[start:start + batch_size]
            Xb = X_train[batch]
            yb = y_train[batch]
            loss, grads = model.backward(Xb, yb)
            t = model.step(grads, m, v, t, lr=lr)
            epoch_loss += loss
            n_batches += 1
        if (epoch + 1) % 20 == 0:
            print(f"  epoch {epoch+1}/{n_epochs}  loss={epoch_loss/n_batches:.6f}")
    return model


def compute_frozen_hash(model):
    """sha256[:16] of concatenated float64 weight bytes — W1,b1,W2,b2,W3,b3."""
    wb = b""
    for w in [model.W1, model.b1, model.W2, model.b2, model.W3, model.b3]:
        wb += w.astype(np.float64).tobytes()
    return hashlib.sha256(wb).hexdigest()[:16]


def verify_behavioral_frozen_link(model, held_ids_set):
    """Re-score F1's per_candidate_records with model; return max|computed - f1_energy|.

    Caller asserts return value < FROZEN_LINK_TOL (1e-4). Returning the diff (rather than
    asserting internally) lets gate injection tests detect that a perturbed model fails.
    """
    with open(F1_RESULT_PATH) as f:
        f1_result = json.load(f)
    records = f1_result["per_candidate_records"]

    # Load only held tasks (need train examples for build_feature)
    train_dir = os.path.join(DATA_PATH, "training")
    task_lookup = {}
    for fn in sorted(os.listdir(train_dir)):
        if fn.endswith(".json"):
            tid = fn[:-5]
            if tid not in held_ids_set:
                continue
            with open(os.path.join(train_dir, fn)) as f_t:
                dat = json.load(f_t)
            task_lookup[tid] = {"id": tid, "train": dat.get("train", []),
                                "test": dat.get("test", [])}

    max_diff = 0.0
    n_checked = 0
    for rec in records:
        tid = rec["task_id"]
        task = task_lookup.get(tid)
        if task is None or not task["test"]:
            continue
        cands = _generate_f1_candidates(task)
        cidx = rec["candidate_idx"]
        if cidx >= len(cands):
            continue
        cand_grid = cands[cidx]["grid"]
        computed = model.score_one(build_feature(cand_grid, task["train"]))
        diff = abs(computed - rec["energy"])
        if diff > max_diff:
            max_diff = diff
        n_checked += 1

    print(f"  Frozen-link: {n_checked}/{len(records)} records, max_diff={max_diff:.2e}")
    return max_diff


# ---------------------------------------------------------------------------
# Feature extraction (must match F1 exactly)
# ---------------------------------------------------------------------------

def color_hist(grid):
    h = [0] * 10
    total = 0
    for row in grid:
        for v in row:
            if 0 <= v <= 9:
                h[v] += 1
            total += 1
    return [v / total for v in h] if total > 0 else h


def mean_color_hist(examples):
    hists = [color_hist(ex["output"]) for ex in examples]
    if not hists:
        return [0.0] * 10
    return [sum(h[i] for h in hists) / len(hists) for i in range(10)]


def grid_shape_norm(grid):
    H = len(grid)
    W = len(grid[0]) if grid else 0
    return [H / 30.0, W / 30.0]


def pass_fraction(cand_grid, examples):
    if not examples:
        return 0.0
    return sum(1 for ex in examples if ex["output"] == cand_grid) / len(examples)


def build_feature(cand_grid, task_train_examples):
    """36-dim feature vector. cand_grid is a list-of-lists (int values)."""
    cand_hist = color_hist(cand_grid)
    mean_out_hist = mean_color_hist(task_train_examples)
    hist_diff = [abs(a - b) for a, b in zip(cand_hist, mean_out_hist)]
    cand_shape = grid_shape_norm(cand_grid)
    out_shapes = [grid_shape_norm(ex["output"]) for ex in task_train_examples]
    mean_out_shape = (
        [sum(s[i] for s in out_shapes) / len(out_shapes) for i in range(2)]
        if out_shapes else [0.0, 0.0]
    )
    cand_H = len(cand_grid)
    cand_W = len(cand_grid[0]) if cand_grid else 0
    shape_match = 0.0
    if out_shapes:
        avg_H = round(sum(s[0] * 30 for s in out_shapes) / len(out_shapes))
        avg_W = round(sum(s[1] * 30 for s in out_shapes) / len(out_shapes))
        shape_match = 1.0 if (cand_H == avg_H and cand_W == avg_W) else 0.0
    pf = pass_fraction(cand_grid, task_train_examples)
    feat = (
        cand_hist + mean_out_hist + hist_diff
        + cand_shape + mean_out_shape
        + [shape_match, pf]
    )
    assert len(feat) == 36
    return feat


def np_grid_to_list(arr):
    """numpy int64 grid → list-of-lists of Python ints."""
    return arr.tolist()


def score_output_arr(out_arr, task_io, model):
    """Score a pre-evaluated output grid with E_theta. Returns float energy."""
    if out_arr is None:
        return 1.0
    try:
        out_list = np_grid_to_list(out_arr)
        feat = build_feature(out_list, task_io)
        return model.score_one(feat)
    except Exception:
        return 1.0


# ---------------------------------------------------------------------------
# Load E_theta (train deterministically; assert frozen hash)
# ---------------------------------------------------------------------------

def load_and_verify_energy_model(held_ids_set, rng_seed=TRAIN_RNG_SEED, n_epochs=TRAIN_N_EPOCHS):
    """Train E_theta identically to F1; assert frozen_energy_hash. Returns (model, hash_str)."""
    # Load all ARC training tasks
    train_dir = os.path.join(DATA_PATH, "training")
    train_tasks = []
    for fn in sorted(os.listdir(train_dir)):
        if fn.endswith(".json"):
            with open(os.path.join(train_dir, fn)) as f:
                dat = json.load(f)
            task_id = fn[:-5]
            train_tasks.append({"id": task_id, "train": dat.get("train", []),
                                 "test": dat.get("test", [])})

    # Training corpus = ARC-train minus held
    corpus = [t for t in train_tasks if t["id"] not in held_ids_set]

    # Build training data
    X_list, y_list = [], []
    for task in corpus:
        if not task["test"]:
            continue
        target = task["test"][0]["output"]
        examples = task["train"]
        # Generate 18 candidates per task (same as F1)
        cands = _generate_f1_candidates(task)
        for cand in cands:
            dist = _hamming_distance(cand["grid"], target)
            feat = build_feature(cand["grid"], examples)
            X_list.append(feat)
            y_list.append(dist)

    X_train = np.array(X_list, dtype=np.float32)
    y_train = np.array(y_list, dtype=np.float32)
    print(f"  E_theta training corpus: {len(corpus)} tasks, {len(X_train)} samples")
    print(f"  Training {n_epochs} epochs (seed={rng_seed})...")
    model = train_mlp(X_train, y_train, n_epochs=n_epochs, lr=TRAIN_LR,
                      batch_size=TRAIN_BATCH_SIZE, rng_seed=rng_seed)
    wh = compute_frozen_hash(model)
    print(f"  frozen_energy_hash computed: {wh}")

    # Behavioral frozen-link: assert model reproduces F1's recorded energies (canonical only)
    if n_epochs == TRAIN_N_EPOCHS:
        max_diff = verify_behavioral_frozen_link(model, held_ids_set)
        if max_diff >= FROZEN_LINK_TOL:
            raise AssertionError(
                f"FATAL: frozen-link broken — max|computed-f1| = {max_diff:.2e} >= "
                f"{FROZEN_LINK_TOL:.2e}. Model is NOT behaviorally identical to F1's E_theta."
            )
        print(f"  Frozen-link PASSED (max_diff={max_diff:.2e} < {FROZEN_LINK_TOL:.2e})")

    return model, wh


def _hamming_distance(a, b):
    if len(a) != len(b) or (a and len(a[0]) != len(b[0])):
        return 1.0
    H, W = len(a), len(a[0])
    if H == 0 or W == 0:
        return 0.0
    return sum(a[r][c] != b[r][c] for r in range(H) for c in range(W)) / (H * W)


def _generate_f1_candidates(task):
    """Generate the same 18 candidates as F1 (for E_theta training)."""
    target = task["test"][0]["output"]
    task_id = task["id"]
    candidates = [{"grid": [list(row) for row in target]}]
    for k_frac in K_FRAC_LADDER:
        for sample_i in range(N_NEAR_MISS_PER_K):
            rng = random.Random((task_id, k_frac, sample_i))
            H, W = len(target), len(target[0]) if target else 0
            n_replace = max(1, math.ceil(k_frac * H * W))
            cand = [list(row) for row in target]
            cells = [(r, c) for r in range(H) for c in range(W)]
            chosen = rng.sample(cells, min(n_replace, len(cells)))
            for r, c in chosen:
                orig = target[r][c]
                alt = rng.choice([v for v in range(10) if v != orig])
                cand[r][c] = alt
            candidates.append({"grid": cand})
    for far_i in range(N_RANDOM_FAR):
        rng = random.Random((task_id, "far", far_i))
        H, W = len(target), len(target[0]) if target else 0
        candidates.append({"grid": [[rng.randint(0, 9) for _ in range(W)] for _ in range(H)]})
    return candidates


# ---------------------------------------------------------------------------
# BFS arm — exact Stage 1k armseed7 replay
# ---------------------------------------------------------------------------

def run_arm_bfs(held_tasks, leaves, n_leaves, n_tasks):
    """Random program search, same as Stage 1k arm_seed=7. Returns per_task dict."""
    arm_bfs_per_task = {}
    t_start = time.time()

    for task_num, task in enumerate(held_tasks[:n_tasks]):
        rng = random.Random((ARM_D_SEED, task["id"]))
        t_task = time.time()
        n_evals = 0
        solved = False

        while time.time() - t_task < TIME_PER_TASK_D:
            i = rng.randrange(n_leaves)
            j = rng.randrange(n_leaves)
            k = rng.randrange(n_leaves)
            l = rng.randrange(n_leaves)
            n_evals += 1
            tup = (i, j, k, l)
            prog = tuple_to_prog(tup, leaves)
            if check_prog(prog, task["io"]):
                arm_bfs_per_task[task["id"]] = {
                    "solved": True,
                    "n_evals": n_evals,
                    "time_limit_hit": False,
                    "budget_exhausted": False,
                }
                solved = True
                break
            if n_evals >= BUDGET_B:
                break

        if not solved:
            time_hit = (time.time() - t_task) >= TIME_PER_TASK_D
            arm_bfs_per_task[task["id"]] = {
                "solved": False,
                "n_evals": n_evals,
                "time_limit_hit": time_hit,
                "budget_exhausted": n_evals >= BUDGET_B,
            }

        _OBJ_CACHE.clear()
        gc.collect()

        if (task_num + 1) % 20 == 0 or n_tasks <= 5:
            elapsed = time.time() - t_start
            n_so_far = sum(1 for r in arm_bfs_per_task.values() if r.get("solved"))
            print(f"  BFS {task_num+1}/{n_tasks}: {n_so_far} solved, {elapsed:.0f}s elapsed")

    return arm_bfs_per_task


# ---------------------------------------------------------------------------
# Energy arm — same programs, energy-sorted within batch
# ---------------------------------------------------------------------------

def run_arm_energy(held_tasks, leaves, n_leaves, model, n_tasks):
    """Energy-ranked best-first: same rng as BFS, batch-sort by E_theta.

    Pair_0 result cached: evaluated once for scoring, reused for check_prog's first
    pair comparison — same grammar-eval cost as BFS, plus small feature+MLP overhead.
    """
    arm_energy_per_task = {}
    t_start = time.time()

    for task_num, task in enumerate(held_tasks[:n_tasks]):
        rng = random.Random((ARM_D_SEED, task["id"]))
        t_task = time.time()
        n_evals = 0
        solved = False

        # Pre-cache pair_0 input/output arrays (constant per task)
        if not task["io"]:
            arm_energy_per_task[task["id"]] = {"solved": False, "n_evals": 0,
                                                "time_limit_hit": False, "budget_exhausted": True}
            continue
        inp0 = np.array(task["io"][0]["input"], dtype=np.int64)
        expected0 = np.array(task["io"][0]["output"], dtype=np.int64)

        while time.time() - t_task < TIME_PER_TASK_D and n_evals < BUDGET_B:
            # Sample a batch (up to BATCH_SIZE_ENERGY, respecting budget)
            remaining = min(BATCH_SIZE_ENERGY, BUDGET_B - n_evals)
            if remaining <= 0:
                break
            tuples = []
            for _ in range(remaining):
                i = rng.randrange(n_leaves)
                j = rng.randrange(n_leaves)
                k = rng.randrange(n_leaves)
                l = rng.randrange(n_leaves)
                tuples.append((i, j, k, l))

            # Build programs, evaluate pair_0, score with E_theta
            progs = [tuple_to_prog(tup, leaves) for tup in tuples]
            out0_cache = [eval_program(prog, inp0) for prog in progs]
            energies = [score_output_arr(o, task["io"], model) for o in out0_cache]
            n_evals += len(tuples)  # count at scoring phase — symmetric with BFS (1 per program)

            # Sort by energy ascending
            order = sorted(range(len(progs)), key=lambda idx: energies[idx])

            # Check in energy order, reusing cached pair_0 result
            for pos in order:
                prog = progs[pos]
                out0 = out0_cache[pos]

                # Pair_0 check via cached result (no re-eval)
                if (out0 is None
                        or out0.shape != expected0.shape
                        or not np.array_equal(out0, expected0)):
                    continue  # pair_0 fails

                # Remaining pairs check
                ok = True
                for pair in task["io"][1:]:
                    inp = np.array(pair["input"], dtype=np.int64)
                    out = np.array(pair["output"], dtype=np.int64)
                    r = eval_program(prog, inp)
                    if r is None or r.shape != out.shape or not np.array_equal(r, out):
                        ok = False
                        break
                if ok:
                    arm_energy_per_task[task["id"]] = {
                        "solved": True,
                        "n_evals": n_evals,
                        "time_limit_hit": False,
                        "budget_exhausted": False,
                        "batch_energy_rank": pos,
                        "batch_size": len(progs),
                    }
                    solved = True
                    break
            if solved:
                break

        if not solved:
            time_hit = (time.time() - t_task) >= TIME_PER_TASK_D
            arm_energy_per_task[task["id"]] = {
                "solved": False,
                "n_evals": n_evals,
                "time_limit_hit": time_hit,
                "budget_exhausted": n_evals >= BUDGET_B,
            }

        _OBJ_CACHE.clear()
        gc.collect()

        if (task_num + 1) % 20 == 0 or n_tasks <= 5:
            elapsed = time.time() - t_start
            n_so_far = sum(1 for r in arm_energy_per_task.values() if r.get("solved"))
            print(f"  Energy {task_num+1}/{n_tasks}: {n_so_far} solved, {elapsed:.0f}s elapsed")

    return arm_energy_per_task


def run_arm_energy_measured(held_tasks, leaves, n_leaves, model):
    """Measurement variant of the energy arm for --measure-smoke.

    Collects per-batch pair0-pass rate and solution energy-rank vs stream-position.
    Returns (arm_energy_per_task, measurements) where measurements = {
      task_id: {
        "pair0_pass_count": int,
        "pair0_total": int,
        "pair0_pass_rate": float,
        "solved": bool,
        "solution": {   # only if solved
          "energy_rank_in_batch": int,   # 0-indexed; 0 = lowest energy (best rank)
          "stream_position_in_batch": int,  # 0-indexed; position in original draw order
          "batch_num": int,
          "batch_size": int,
        } | None,
      }
    }
    """
    arm_energy_per_task = {}
    measurements = {}
    t_start = time.time()
    n_tasks = len(held_tasks)

    for task_num, task in enumerate(held_tasks):
        rng = random.Random((ARM_D_SEED, task["id"]))
        t_task = time.time()
        n_evals = 0
        solved = False
        pair0_pass = 0
        pair0_total = 0
        solution_info = None

        if not task["io"]:
            arm_energy_per_task[task["id"]] = {"solved": False, "n_evals": 0,
                                                "time_limit_hit": False, "budget_exhausted": True}
            measurements[task["id"]] = {"pair0_pass_count": 0, "pair0_total": 0,
                                         "pair0_pass_rate": 0.0, "solved": False, "solution": None}
            continue
        inp0 = np.array(task["io"][0]["input"], dtype=np.int64)
        expected0 = np.array(task["io"][0]["output"], dtype=np.int64)

        batch_num = 0
        while time.time() - t_task < TIME_PER_TASK_D and n_evals < BUDGET_B:
            remaining = min(BATCH_SIZE_ENERGY, BUDGET_B - n_evals)
            if remaining <= 0:
                break
            tuples = []
            for _ in range(remaining):
                i = rng.randrange(n_leaves)
                j = rng.randrange(n_leaves)
                k = rng.randrange(n_leaves)
                l = rng.randrange(n_leaves)
                tuples.append((i, j, k, l))

            progs = [tuple_to_prog(tup, leaves) for tup in tuples]
            out0_cache = [eval_program(prog, inp0) for prog in progs]
            energies = [score_output_arr(o, task["io"], model) for o in out0_cache]
            n_evals += len(tuples)

            # Measure pair0-pass rate for this batch
            batch_pair0 = sum(
                1 for o in out0_cache
                if o is not None and o.shape == expected0.shape and np.array_equal(o, expected0)
            )
            pair0_pass += batch_pair0
            pair0_total += len(tuples)

            order = sorted(range(len(progs)), key=lambda idx: energies[idx])

            for energy_rank, pos in enumerate(order):
                prog = progs[pos]
                out0 = out0_cache[pos]

                if (out0 is None
                        or out0.shape != expected0.shape
                        or not np.array_equal(out0, expected0)):
                    continue

                ok = True
                for pair in task["io"][1:]:
                    inp = np.array(pair["input"], dtype=np.int64)
                    out = np.array(pair["output"], dtype=np.int64)
                    r = eval_program(prog, inp)
                    if r is None or r.shape != out.shape or not np.array_equal(r, out):
                        ok = False
                        break
                if ok:
                    # pos = index in progs (= stream position within batch, 0-indexed)
                    solution_info = {
                        "energy_rank_in_batch": energy_rank,
                        "stream_position_in_batch": pos,
                        "batch_num": batch_num,
                        "batch_size": len(tuples),
                    }
                    arm_energy_per_task[task["id"]] = {
                        "solved": True,
                        "n_evals": n_evals,
                        "time_limit_hit": False,
                        "budget_exhausted": False,
                    }
                    solved = True
                    break
            if solved:
                break
            batch_num += 1

        if not solved:
            time_hit = (time.time() - t_task) >= TIME_PER_TASK_D
            arm_energy_per_task[task["id"]] = {
                "solved": False,
                "n_evals": n_evals,
                "time_limit_hit": time_hit,
                "budget_exhausted": n_evals >= BUDGET_B,
            }

        pair0_rate = pair0_pass / pair0_total if pair0_total > 0 else 0.0
        measurements[task["id"]] = {
            "pair0_pass_count": pair0_pass,
            "pair0_total": pair0_total,
            "pair0_pass_rate": pair0_rate,
            "solved": solved,
            "solution": solution_info,
        }

        _OBJ_CACHE.clear()
        gc.collect()

        elapsed = time.time() - t_start
        n_so_far = sum(1 for r in arm_energy_per_task.values() if r.get("solved"))
        print(f"  Measure {task_num+1}/{n_tasks} ({task['id']}): "
              f"solved={solved}, pair0_rate={pair0_rate:.4f} ({pair0_pass}/{pair0_total}), "
              f"rank={solution_info['energy_rank_in_batch'] if solution_info else 'N/A'}/"
              f"{solution_info['batch_size'] if solution_info else '?'}, "
              f"stream={solution_info['stream_position_in_batch'] if solution_info else 'N/A'}, "
              f"{elapsed:.0f}s")

    return arm_energy_per_task, measurements


# ---------------------------------------------------------------------------
# Gate
# ---------------------------------------------------------------------------

GATE_VIOLATIONS = []


def gate_assert(condition, message):
    if not condition:
        GATE_VIOLATIONS.append(message)
        print(f"GATE VIOLATION: {message}", file=sys.stderr)


def check_gate(artifact, stage1d_held_id_set):
    gate_assert(
        artifact.get("frozen_energy_hash") == FROZEN_ENERGY_HASH,
        f"frozen_energy_hash={artifact.get('frozen_energy_hash')!r} != {FROZEN_ENERGY_HASH!r}",
    )
    gate_assert(
        artifact.get("eval_split_hash") == EVAL_SPLIT_HASH,
        f"eval_split_hash={artifact.get('eval_split_hash')!r} != {EVAL_SPLIT_HASH!r}",
    )
    gate_assert(
        artifact.get("space_hash") == PREV_SPACE_HASH,
        f"space_hash={artifact.get('space_hash')!r} != {PREV_SPACE_HASH!r}",
    )
    gate_assert(
        artifact.get("arm_bfs_n_solved") == BFS_EXPECTED_N_SOLVED,
        f"arm_bfs_n_solved={artifact.get('arm_bfs_n_solved')} != {BFS_EXPECTED_N_SOLVED} "
        f"(must reproduce Stage 1k armseed7)",
    )
    # Budget enforcement — both arms
    for arm_name in ["arm_bfs_per_task", "arm_energy_per_task"]:
        for tid, rec in artifact.get(arm_name, {}).items():
            if rec.get("n_evals", 0) > BUDGET_B:
                gate_assert(
                    False,
                    f"{arm_name}[{tid}]: n_evals={rec['n_evals']} > BUDGET_B={BUDGET_B}",
                )
                break
    return list(GATE_VIOLATIONS)


def run_gate_injection_tests(stage1d_held_id_set, leaves, n_leaves, space_hash, held_task_ids):
    """Inject known violations; confirm all caught. Returns True if all caught."""
    print("\n--- Gate injection tests ---")
    caught_all = True
    dummy_per_task = {tid: {"solved": False, "n_evals": BUDGET_B, "time_limit_hit": False,
                            "budget_exhausted": True}
                      for tid in held_task_ids}

    good = {
        "frozen_energy_hash": FROZEN_ENERGY_HASH,
        "eval_split_hash": EVAL_SPLIT_HASH,
        "space_hash": space_hash,
        "arm_bfs_n_solved": BFS_EXPECTED_N_SOLVED,
        "arm_energy_n_solved": BFS_EXPECTED_N_SOLVED,
        "arm_bfs_per_task": dummy_per_task,
        "arm_energy_per_task": dummy_per_task,
    }

    # Verify good artifact passes
    saved = list(GATE_VIOLATIONS)
    check_gate(good, stage1d_held_id_set)
    new_viols = GATE_VIOLATIONS[len(saved):]
    if new_viols:
        print(f"  SMOKE FAIL: good artifact has violations: {new_viols}")
        caught_all = False
    else:
        print("  [OK] Good artifact passes gate")

    def _inject_and_check(label, bad_artifact):
        nonlocal caught_all
        before = len(GATE_VIOLATIONS)
        check_gate(bad_artifact, stage1d_held_id_set)
        caught = len(GATE_VIOLATIONS) > before
        print(f"  [{'OK' if caught else 'FAIL'}] '{label}' -> {'CAUGHT' if caught else 'MISSED'}")
        caught_all &= caught

    # Test 1: wrong frozen_energy_hash (simulates retrained-energy or wrong-hash constant)
    _inject_and_check(
        "wrong frozen_energy_hash (retrained / wrong constant)",
        {**good, "frozen_energy_hash": "0000000000000000"},
    )

    # Test 2: budget overrun in arm_bfs
    first_tid = list(held_task_ids)[0]
    over_budget = dict(dummy_per_task)
    over_budget[first_tid] = {"solved": False, "n_evals": BUDGET_B + 1, "time_limit_hit": False}
    _inject_and_check(
        "arm_bfs budget overrun (n_evals > 300K)",
        {**good, "arm_bfs_per_task": over_budget},
    )

    # Test 3: budget overrun in arm_energy
    over_e = dict(dummy_per_task)
    over_e[first_tid] = {"solved": False, "n_evals": BUDGET_B + 1, "time_limit_hit": False}
    _inject_and_check(
        "arm_energy budget overrun (n_evals > 300K)",
        {**good, "arm_energy_per_task": over_e},
    )

    # Test 4: BFS arm not reproducing D=10
    _inject_and_check(
        "arm_bfs_n_solved != 10 (baseline not reproduced)",
        {**good, "arm_bfs_n_solved": 9},
    )

    # Test 5: eval_split_hash mismatch
    _inject_and_check(
        "eval_split_hash mismatch",
        {**good, "eval_split_hash": "0000000000000000"},
    )

    # Test 6: space_hash mismatch
    _inject_and_check(
        "space_hash mismatch",
        {**good, "space_hash": "0000000000000000"},
    )

    # Test 7: F1 frozen-link — untrained perturbed model (rng_seed=1) fails behavioral link
    print("  [running] 'F1 frozen-link: perturbed model' ...")
    perturbed = MLP(rng_seed=1)  # untrained, random weights — energies will diverge from F1's
    perturbed_diff = verify_behavioral_frozen_link(perturbed, stage1d_held_id_set)
    caught_7 = perturbed_diff >= FROZEN_LINK_TOL
    print(f"  [{'OK' if caught_7 else 'FAIL'}] 'F1 frozen-link: perturbed model' -> "
          f"{'CAUGHT' if caught_7 else 'MISSED'} (max_diff={perturbed_diff:.2e})")
    caught_all &= caught_7

    return caught_all


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true", help="3 tasks, TEMP path")
    parser.add_argument("--test-gate", action="store_true", help="Gate injection tests only")
    parser.add_argument("--measure-smoke", action="store_true",
                        help="Measurement smoke: run on Stage1k solved tasks, collect pair0-pass + rank")
    args = parser.parse_args()

    n_tasks = 3 if args.smoke else N_HELD
    result_path = TEMP_PATH if args.smoke else RESULT_PATH
    print(f"Stage F2 — Energy-Ranked Best-First vs BFS Reach")
    print(f"  {'SMOKE MODE' if args.smoke else 'CANONICAL MODE'}: {n_tasks} tasks -> {result_path}")

    # Load Stage 1d held IDs
    with open(S1D_PATH) as f:
        s1d = json.load(f)
    held_ids = s1d["held_task_ids"]
    held_ids_set = set(held_ids)

    eval_split_hash = hashlib.sha256(
        ",".join(sorted(held_ids)).encode("utf-8")
    ).hexdigest()[:16]
    print(f"  eval_split_hash: {eval_split_hash}")

    # Load all ARC tasks (training split only — held set is from there)
    all_tasks = []
    train_dir = os.path.join(DATA_PATH, "training")
    for fn in sorted(os.listdir(train_dir)):
        if fn.endswith(".json"):
            with open(os.path.join(train_dir, fn)) as f:
                dat = json.load(f)
            task_id = fn[:-5]
            all_tasks.append({
                "id": task_id,
                "io": dat.get("train", []),   # training pairs for check_prog
            })
    held_tasks = [t for t in all_tasks if t["id"] in held_ids_set]
    assert len(held_tasks) == N_HELD, f"Expected {N_HELD} held tasks, got {len(held_tasks)}"
    print(f"  Loaded {len(held_tasks)} held tasks")

    # Grammar
    leaves = all_leaves_expanded()
    n_leaves = len(leaves)
    space_hash = compute_space_hash(leaves)
    print(f"  Grammar: {n_leaves} leaves, space_hash={space_hash}")
    assert space_hash == PREV_SPACE_HASH, \
        f"FATAL: space_hash {space_hash} != expected {PREV_SPACE_HASH}"

    if args.test_gate:
        ok = run_gate_injection_tests(held_ids_set, leaves, n_leaves, space_hash, held_ids)
        print(f"\nGate injection tests: {'ALL PASSED' if ok else 'SOME MISSED'}")
        sys.exit(0 if ok else 1)

    if args.measure_smoke:
        # Target: first 3 Stage 1k armseed7 solved task IDs (known to solve within B=300K)
        target_ids = S1K_SOLVED_IDS[:3]
        measure_tasks = [t for t in held_tasks if t["id"] in set(target_ids)]
        assert len(measure_tasks) == 3, f"Expected 3 measure tasks, got {len(measure_tasks)}"
        print(f"\nMEASURE-SMOKE: targeting {len(measure_tasks)} Stage1k-solved tasks")
        print(f"  Tasks: {[t['id'] for t in measure_tasks]}")

        print(f"\nLoading frozen E_theta (5 epochs for measurement smoke)...")
        model, wh = load_and_verify_energy_model(held_ids_set, n_epochs=5)
        print(f"  Hash (5-epoch smoke, not canonical): {wh}")

        print(f"\n=== Measurement smoke: pair0-pass rate + solution rank ===")
        _, measurements = run_arm_energy_measured(measure_tasks, leaves, n_leaves, model)

        print(f"\n--- Pair0-pass rate summary ---")
        for tid, m in measurements.items():
            print(f"  {tid}: pair0_rate={m['pair0_pass_rate']:.6f} "
                  f"({m['pair0_pass_count']}/{m['pair0_total']}), solved={m['solved']}")

        print(f"\n--- Solution rank summary ---")
        for tid, m in measurements.items():
            if m["solution"]:
                s = m["solution"]
                print(f"  {tid}: energy_rank={s['energy_rank_in_batch']}/{s['batch_size']} "
                      f"(stream_pos={s['stream_position_in_batch']}, batch={s['batch_num']})")
            else:
                print(f"  {tid}: not solved in measurement smoke")

        artifact = {
            "stage": "F2_measurement_smoke",
            "target_task_ids": [t["id"] for t in measure_tasks],
            "n_tasks": len(measure_tasks),
            "measurements": measurements,
        }
        os.makedirs(OUT_DIR, exist_ok=True)
        with open(MEASURE_TEMP_PATH, "w") as f:
            json.dump(artifact, f, indent=2)
        print(f"\nArtifact written: {MEASURE_TEMP_PATH}")
        sys.exit(0)

    # Load and verify E_theta
    print(f"\nLoading frozen E_theta (F1 deterministic re-train)...")
    n_train_epochs = 5 if args.smoke else TRAIN_N_EPOCHS
    model, wh = load_and_verify_energy_model(held_ids_set, n_epochs=n_train_epochs)
    print(f"  Computed hash: {wh}")
    if not args.smoke:
        assert wh == FROZEN_ENERGY_HASH, (
            f"FATAL: frozen_energy_hash {wh} != expected {FROZEN_ENERGY_HASH}\n"
            "Training is not deterministic or weights are not the same as F1."
        )
        print(f"  Hash matches pre-registered value")
    else:
        print(f"  Smoke mode: hash check skipped (epochs={n_train_epochs})")

    # BFS arm
    print(f"\n=== Arm BFS: random search (Stage 1k armseed7 replay) ===")
    t_bfs_start = time.time()
    arm_bfs_per_task = run_arm_bfs(held_tasks, leaves, n_leaves, n_tasks)
    arm_bfs_elapsed = time.time() - t_bfs_start
    arm_bfs_n_solved = sum(1 for r in arm_bfs_per_task.values() if r.get("solved"))
    arm_bfs_n_time = sum(1 for r in arm_bfs_per_task.values() if r.get("time_limit_hit"))
    arm_bfs_n_budget = sum(1 for r in arm_bfs_per_task.values() if r.get("budget_exhausted"))
    print(f"  BFS done: {arm_bfs_n_solved}/{n_tasks} solved in {arm_bfs_elapsed:.0f}s")
    print(f"  time_limit_hit={arm_bfs_n_time} budget_exhausted={arm_bfs_n_budget}")
    if not args.smoke:
        print(f"  Gate check: arm_bfs_n_solved={arm_bfs_n_solved} (expected {BFS_EXPECTED_N_SOLVED})")

    # Energy arm
    print(f"\n=== Arm Energy: energy-ranked best-first (batch={BATCH_SIZE_ENERGY}) ===")
    t_energy_start = time.time()
    arm_energy_per_task = run_arm_energy(held_tasks, leaves, n_leaves, model, n_tasks)
    arm_energy_elapsed = time.time() - t_energy_start
    arm_energy_n_solved = sum(1 for r in arm_energy_per_task.values() if r.get("solved"))
    arm_energy_n_time = sum(1 for r in arm_energy_per_task.values() if r.get("time_limit_hit"))
    arm_energy_n_budget = sum(1 for r in arm_energy_per_task.values() if r.get("budget_exhausted"))
    print(f"  Energy done: {arm_energy_n_solved}/{n_tasks} solved in {arm_energy_elapsed:.0f}s")
    print(f"  time_limit_hit={arm_energy_n_time} budget_exhausted={arm_energy_n_budget}")

    # Verdict
    if arm_energy_n_solved > arm_bfs_n_solved:
        verdict = "SIGNAL"
    elif arm_energy_n_solved == arm_bfs_n_solved:
        verdict = "NULL"
    else:
        verdict = "REGRESS"
    print(f"\nVerdict: {verdict} "
          f"(energy={arm_energy_n_solved} vs bfs={arm_bfs_n_solved})")

    # Build artifact
    artifact = {
        "stage": "F2_energy_search",
        "spec": "05_F2_energy_search/pre_reg/stagef2_energy_search_preregistration.md",
        "tracking_issue": "#17 (wordingone/the-search)",
        "smoke": args.smoke,
        "n_tasks": n_tasks,
        "frozen_energy_hash": wh if not args.smoke else f"{wh}_smoke_{n_train_epochs}ep",
        "eval_split_hash": eval_split_hash,
        "space_hash": space_hash,
        "arm_d_seed": ARM_D_SEED,
        "budget_b": BUDGET_B,
        "time_per_task_s": TIME_PER_TASK_D,
        "batch_size_energy": BATCH_SIZE_ENERGY,
        "arm_bfs_n_solved": arm_bfs_n_solved,
        "arm_bfs_time_limit_hit": arm_bfs_n_time,
        "arm_bfs_budget_exhausted": arm_bfs_n_budget,
        "arm_bfs_elapsed_s": round(arm_bfs_elapsed, 1),
        "arm_energy_n_solved": arm_energy_n_solved,
        "arm_energy_time_limit_hit": arm_energy_n_time,
        "arm_energy_budget_exhausted": arm_energy_n_budget,
        "arm_energy_elapsed_s": round(arm_energy_elapsed, 1),
        "verdict": verdict,
        "gate_violations": [],
        "arm_bfs_per_task": arm_bfs_per_task,
        "arm_energy_per_task": arm_energy_per_task,
    }

    # Gate check (canonical only)
    if not args.smoke:
        viols = check_gate(artifact, held_ids_set)
        if viols:
            print(f"\nFAIL-CLOSED: {len(viols)} violation(s) — artifact NOT written", file=sys.stderr)
            for v in viols:
                print(f"  {v}", file=sys.stderr)
            sys.exit(1)
        artifact["gate_violations"] = []
        print("Fail-closed gate: PASSED")

    os.makedirs(OUT_DIR, exist_ok=True)
    with open(result_path, "w") as f:
        json.dump(artifact, f, indent=2)
    print(f"\nArtifact written: {result_path}")
    print(f"Summary: BFS={arm_bfs_n_solved}/{n_tasks}  Energy={arm_energy_n_solved}/{n_tasks}  "
          f"Verdict={verdict}")


if __name__ == "__main__":
    main()
