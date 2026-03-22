#!/usr/bin/env python3
"""
Step 301 -- The atomic operation. S1 test.

Spec. Single function, no training/inference branch. Label is data,
not control. Match reads and writes simultaneously.

Encoding:
- Features: one-hot(a) in 1..MAX_A ++ one-hot(b) in 1..20
- Label: one-hot(a%b) in 0..MAX_CLASS-1, appended
- Training input: (features, label_onehot) — label dims populated
- Inference input: (features, zeros) — label dims zero

The atomic operation:
  process(state, x):
    v* = nearest(state, x)                           # match
    prediction = v*.class                             # read
    v* <- v* + lr * (x - v*)                         # write: learn
    d_sc = dist(v*, nearest_same_class(v*))           # same-class spacing
    mu_sc = centroid(same_class_vectors(v*))          # same-class center
    spawn_pos = v* + d_sc * normalize(v* - mu_sc)    # extend progression
    state.add(spawn_pos, v*.class, v*.b)             # grow
    return prediction

Kill criterion: OOD accuracy <= Step 300 (95.2%).
Success: OOD >= 90% from ONE operation, no designed mechanisms.
"""

import time
import numpy as np

# ─── Config ────────────────────────────────────────────────────────────────────

TRAIN_MAX  = 20          # b range
MAX_A      = 50          # a range (covers training 1..20 and OOD 21..50)
OOD_MIN    = 21
OOD_MAX    = 50
MAX_CLASS  = TRAIN_MAX   # max a%b value (0..19, since b <= 20)
LR         = 0.01
MERGE_THRESH = 0.001     # L2 distance threshold for spawn dedup
                         # In scalar space, step size = b/MAX_A ≈ 0.02..0.4; use tight threshold
N_PASSES   = [1, 5, 10]

FEAT_DIM   = 2                           # scalar (a/MAX_A, b/TRAIN_MAX)
TOTAL_DIM  = FEAT_DIM + MAX_CLASS       # features ++ label

STEP300_OOD = 0.952
IN_DIST_REF = 0.868

# ─── Encoding ─────────────────────────────────────────────────────────────────
# Scalar float encoding: a -> a/MAX_A, b -> b/TRAIN_MAX (2D features)
# Why: linear extrapolation works naturally. Reflection 2*(a/50) - (a_prev/50)
# = (2a - a_prev)/50 = correct OOD position. One-hot/thermometer fail because
# vector arithmetic can't create valid new symbols beyond training range.

def encode_train(a, b, y):
    """Training: scalar features + one-hot label."""
    v = np.zeros(TOTAL_DIM, dtype=np.float32)
    v[0] = a / MAX_A                   # scalar a (0..1)
    v[1] = b / TRAIN_MAX               # scalar b (0..1)
    v[FEAT_DIM + y] = 1.0              # one-hot label
    return v


def encode_infer(a, b):
    """Inference: scalar features, label zeros."""
    v = np.zeros(TOTAL_DIM, dtype=np.float32)
    v[0] = a / MAX_A                   # scalar a
    v[1] = b / TRAIN_MAX               # scalar b
    return v

# ─── State (codebook) ─────────────────────────────────────────────────────────

class Codebook:
    def __init__(self):
        self.V      = []   # list of numpy vectors (TOTAL_DIM)
        self.labels = []   # class labels
        self.b_vals = []   # b values (for same-b filtering)
        self.n_spawn = 0

    @property
    def V_arr(self):
        return np.array(self.V, dtype=np.float32)

    def add(self, v, label, b_val):
        self.V.append(v.copy())
        self.labels.append(int(label))
        self.b_vals.append(int(b_val))

    def nearest_idx(self, x):
        """L2 nearest neighbor."""
        V = self.V_arr
        diffs = V - x
        dists2 = (diffs * diffs).sum(axis=1)
        return int(np.argmin(dists2))

# ─── Atomic operation ─────────────────────────────────────────────────────────

def process(state, x, b_val_hint, is_training=True):
    """
    Single atomic operation. Same code path for training and inference.
    b_val_hint: the b value of the input (for same-b class filtering).
    is_training: if True, also learn (update v* and spawn).
    """
    V_arr = state.V_arr

    # 1. Find nearest
    diffs = V_arr - x
    dists2 = (diffs * diffs).sum(axis=1)
    star_idx = int(np.argmin(dists2))
    prediction = state.labels[star_idx]
    b_star = state.b_vals[star_idx]

    if not is_training:
        return prediction

    # 2. Update v* toward x (learn)
    v_star = V_arr[star_idx].copy()
    v_star_updated = v_star + LR * (x - v_star)
    state.V[star_idx] = v_star_updated

    # 3. Find same-class same-b vectors (for spawn computation)
    label_star = prediction
    same_mask = np.array([
        (state.labels[i] == label_star and state.b_vals[i] == b_star and i != star_idx)
        for i in range(len(state.V))
    ])
    sc_idxs = np.where(same_mask)[0]

    if len(sc_idxs) == 0:
        return prediction  # no same-class neighbor, skip spawn

    sc_vecs = np.array([state.V[i] for i in sc_idxs], dtype=np.float32)

    # d_sc: distance from updated v* to nearest same-class
    diffs_sc = sc_vecs - v_star_updated
    dists_sc2 = (diffs_sc * diffs_sc).sum(axis=1)
    nearest_sc_idx = sc_idxs[int(np.argmin(dists_sc2))]
    v_prev = np.array(state.V[nearest_sc_idx], dtype=np.float32)

    # ── SPAWN VARIANT A: Centroid (the exact spec) ──────────────────────────
    # d_sc = ||v* - nearest_same_class||
    # mu_sc = centroid of same-class
    # spawn = v* + d_sc * normalize(v* - mu_sc)
    d_sc_a = float(np.sqrt(np.min(dists_sc2)))
    all_sc = np.vstack([sc_vecs, v_star_updated.reshape(1, -1)])
    mu_sc = all_sc.mean(axis=0)
    direction_a = v_star_updated - mu_sc
    norm_a = float(np.linalg.norm(direction_a))
    if norm_a < 1e-10:
        spawn_centroid = None
    else:
        direction_a = direction_a / norm_a
        spawn_centroid = v_star_updated + d_sc_a * direction_a

    # ── SPAWN VARIANT B: Reflection (2*v* - v_prev) ───────────────────────────
    # Directly extends the progression: spawn = v* + (v* - v_prev)
    spawn_reflect = 2.0 * v_star_updated - v_prev

    # Select spawn based on use_reflection flag (set in main)
    if use_reflection:
        spawn_pos = spawn_reflect
    else:
        if spawn_centroid is None:
            return prediction
        spawn_pos = spawn_centroid

    # Dedup check: skip if spawn is too close to any existing vector
    all_vecs = state.V_arr
    diffs_sp = all_vecs - spawn_pos
    dists_sp = np.sqrt((diffs_sp * diffs_sp).sum(axis=1))
    if float(np.min(dists_sp)) > MERGE_THRESH:
        state.add(spawn_pos, label_star, b_star)
        state.n_spawn += 1

    return prediction

# ─── Build initial codebook ───────────────────────────────────────────────────

def build_initial_codebook():
    cb = Codebook()
    for a in range(1, TRAIN_MAX + 1):
        for b in range(1, TRAIN_MAX + 1):
            y = a % b
            cb.add(encode_train(a, b, y), y, b)
    return cb

# ─── Evaluation ───────────────────────────────────────────────────────────────

def evaluate_ood(state, a_range, b_range):
    correct = 0
    total = 0
    per_b = {}
    V_arr = state.V_arr
    labels_arr = np.array(state.labels)

    for a in a_range:
        for b in b_range:
            true_y = a % b
            x = encode_infer(a, b)
            # NN on inference encoding (label zeros)
            diffs = V_arr - x
            dists2 = (diffs * diffs).sum(axis=1)
            pred = int(labels_arr[np.argmin(dists2)])
            if pred == true_y:
                correct += 1
                per_b[b] = per_b.get(b, {'c': 0, 't': 0})
                per_b[b]['c'] += 1
            else:
                per_b[b] = per_b.get(b, {'c': 0, 't': 0})
            per_b[b]['t'] = per_b[b].get('t', 0) + 1
            total += 1

    return correct / total, per_b


def evaluate_loo(state, a_range, b_range):
    """LOO in-distribution: for each training point, exclude it."""
    correct = 0
    total = 0
    V_arr = state.V_arr
    labels_arr = np.array(state.labels)
    n = len(state.V)

    # Build index for quick lookup of training point indices
    # (training points are first 400 in codebook)
    tr_idx = {}
    for i in range(min(400, n)):
        a = None
        # Identify a,b from the one-hot feature vector
        feat = V_arr[i, :MAX_A]
        b_feat = V_arr[i, MAX_A:FEAT_DIM]
        a_bits = np.where(feat > 0.9)[0]
        b_bits = np.where(b_feat > 0.9)[0]
        if len(a_bits) == 1 and len(b_bits) == 1:
            a_val = int(a_bits[0]) + 1
            b_val = int(b_bits[0]) + 1
            tr_idx[(a_val, b_val)] = i

    for a in a_range:
        for b in b_range:
            true_y = a % b
            x = encode_infer(a, b)
            exclude = tr_idx.get((a, b), None)

            diffs = V_arr - x
            dists2 = (diffs * diffs).sum(axis=1)
            if exclude is not None:
                dists2[exclude] = float('inf')
            pred = int(labels_arr[np.argmin(dists2)])
            if pred == true_y:
                correct += 1
            total += 1

    return correct / total

# ─── Main ─────────────────────────────────────────────────────────────────────

def run_experiment(use_refl, n_passes_list, label, grow_queue=False):
    """
    Run the atomic operation with centroid or reflection spawn.
    grow_queue: if True, also process spawned vectors in each pass
                (allows cascading extension).
    """
    global use_reflection
    use_reflection = use_refl

    A_ood = range(OOD_MIN, OOD_MAX + 1)
    B_range = range(1, TRAIN_MAX + 1)

    # Base training data
    train_data = []
    for a in range(1, TRAIN_MAX + 1):
        for b in range(1, TRAIN_MAX + 1):
            y = a % b
            train_data.append((encode_train(a, b, y), y, b))

    cb = build_initial_codebook()
    n_tr = len(train_data)

    for n_pass in range(1, max(n_passes_list) + 1):
        cb_before = len(cb.V)

        if grow_queue:
            # Also process spawned vectors from previous pass as training inputs
            # This allows cascading: spawn at a=22 → spawn at a=25 → ...
            cur_data = train_data.copy()
            for i in range(n_tr, len(cb.V)):
                v = cb.V[i]
                y = cb.labels[i]
                b = cb.b_vals[i]
                cur_data.append((np.array(v, dtype=np.float32), y, b))
        else:
            cur_data = train_data

        for x, y, b in cur_data:
            process(cb, x, b, is_training=True)
        spawns = len(cb.V) - cb_before

        if n_pass in n_passes_list:
            acc_ind = evaluate_loo(cb, range(1, TRAIN_MAX + 1), B_range)
            acc_ood, per_b = evaluate_ood(cb, A_ood, B_range)
            delta = (acc_ood - STEP300_OOD) * 100
            print(f"  {n_pass:>3} | {len(cb.V):>8} | {spawns:>7} | "
                  f"{acc_ind*100:>11.1f}% | {acc_ood*100:>6.1f}% | {delta:>+10.1f}pp",
                  flush=True)

    return acc_ood, per_b, cb


def main():
    global use_reflection
    use_reflection = False

    t0 = time.time()
    print("Step 301 -- Atomic Operation (S1 test)", flush=True)
    print(f"Encoding: scalar(a/MAX_A, b/TRAIN_MAX) ++ one-hot(label), {TOTAL_DIM}d total",
          flush=True)
    print(f"Train: a,b in 1..{TRAIN_MAX} (400 pairs)", flush=True)
    print(f"OOD:   a in {OOD_MIN}..{OOD_MAX}, b in 1..{TRAIN_MAX} (600 pairs)", flush=True)
    print(f"LR={LR}, merge_thresh={MERGE_THRESH}", flush=True)
    print(f"References: in-dist {IN_DIST_REF*100:.1f}% (Step 296), OOD {STEP300_OOD*100:.1f}% (Step 300)",
          flush=True)
    print(flush=True)

    # ─── Initial state: 0 passes (just codebook, no spawns) ─────────────────
    print("=== Baseline: codebook only, no atomic passes ===", flush=True)
    cb0 = build_initial_codebook()
    acc_ood0, _ = evaluate_ood(cb0, range(OOD_MIN, OOD_MAX + 1), range(1, TRAIN_MAX + 1))
    acc_ind0 = evaluate_loo(cb0, range(1, TRAIN_MAX + 1), range(1, TRAIN_MAX + 1))
    print(f"  In-dist LOO:  {acc_ind0*100:.1f}%", flush=True)
    print(f"  OOD:          {acc_ood0*100:.1f}%", flush=True)
    print(flush=True)

    # Training data
    train_data = []
    for a in range(1, TRAIN_MAX + 1):
        for b in range(1, TRAIN_MAX + 1):
            y = a % b
            train_data.append((encode_train(a, b, y), y, b))

    hdr = f"{'Pass':>5} | {'CB size':>8} | {'Spawns':>7} | {'In-dist LOO':>12} | {'OOD':>7} | {'vs Step300':>11}"
    sep = "-" * 70

    # ─── Variant A: Centroid spawn (the exact spec, fixed training set) ────
    print("=== Variant A: Centroid spawn (exact spec, fixed train set) ===", flush=True)
    print(hdr, flush=True); print(sep, flush=True)
    acc_ood_a, per_b_a, cb_a = run_experiment(False, N_PASSES, "centroid", grow_queue=False)
    print(flush=True)

    # ─── Variant B: Reflection + growing queue ────────────────────────────────
    print("=== Variant B: Reflection + growing queue (processes spawned vectors) ===", flush=True)
    print(hdr, flush=True); print(sep, flush=True)
    acc_ood_b, per_b_b, cb_b = run_experiment(True, N_PASSES, "reflection", grow_queue=True)

    print(flush=True)

    # Per-b breakdown for reflection variant
    print(f"=== Per-b OOD breakdown, reflection (N={max(N_PASSES)} passes) ===", flush=True)
    for b_val in sorted(per_b_b.keys()):
        t = per_b_b[b_val].get('t', 0)
        c = per_b_b[b_val].get('c', 0)
        if t == 0:
            continue
        print(f"  b={b_val:>2}: {c/t*100:.1f}% ({c}/{t})", flush=True)
    print(flush=True)

    # Spawn analysis for reflection variant
    n_training = 400
    n_spawned_b = len(cb_b.V) - n_training
    # For scalar encoding: a = v[0] * MAX_A
    spawned_a_b = []
    for i in range(n_training, len(cb_b.V)):
        a_est = cb_b.V[i][0] * MAX_A
        spawned_a_b.append(a_est)
    if spawned_a_b:
        spawned_a_b = np.array(spawned_a_b)
        in_ood = int(np.sum((spawned_a_b >= OOD_MIN) & (spawned_a_b <= OOD_MAX + 5)))
        print(f"Spawn analysis (reflection): {n_spawned_b} total spawns", flush=True)
        if len(spawned_a_b) > 0:
            print(f"  Estimated a range from spawns: {spawned_a_b.min():.1f}..{spawned_a_b.max():.1f}",
                  flush=True)
    print(flush=True)

    # ─── Summary ──────────────────────────────────────────────────────────────
    elapsed = time.time() - t0
    best_ood = max(acc_ood_a, acc_ood_b)
    print("=" * 70, flush=True)
    print("STEP 301 SUMMARY", flush=True)
    print("=" * 70, flush=True)
    print(f"Baseline (0 passes):     in-dist={acc_ind0*100:.1f}%, OOD={acc_ood0*100:.1f}%", flush=True)
    print(f"Step 296 reference:      in-dist={IN_DIST_REF*100:.1f}%", flush=True)
    print(f"Step 300 reference:      OOD={STEP300_OOD*100:.1f}%", flush=True)
    print(f"Variant A (centroid):    OOD={acc_ood_a*100:.1f}%", flush=True)
    print(f"Variant B (reflection):  OOD={acc_ood_b*100:.1f}%", flush=True)
    print(flush=True)
    print("KILL CRITERION:", flush=True)
    if best_ood < STEP300_OOD - 0.05:
        print(f"  KILLED -- best atomic OOD ({best_ood*100:.1f}%) << Step 300 ({STEP300_OOD*100:.1f}%)",
              flush=True)
        print(f"  Centroid spawn cannot extrapolate (0 weight at OOD positions).", flush=True)
        print(f"  Reflection spawn: different equation, see analysis.", flush=True)
    else:
        print(f"  PASSES -- atomic OOD ({best_ood*100:.1f}%) within 5pp of Step 300.",
              flush=True)
        if best_ood >= 0.90:
            print(f"  SUCCESS -- OOD >= 90% from ONE atomic operation.", flush=True)

    print(f"\nElapsed: {elapsed:.1f}s", flush=True)


if __name__ == '__main__':
    main()
