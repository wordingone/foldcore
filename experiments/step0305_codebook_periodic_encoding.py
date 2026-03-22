#!/usr/bin/env python3
"""
Step 305 -- Periodic encoding for ABSORB. The Tempest physics test.

Spec. Periodic encoding = physics of circular motion.
Same-class examples have IDENTICAL (sin, cos) → cosine = 1.0 → always absorb.
The substrate discovers the geometry; the encoding provides the physics.

Encoding: [sin(2πa/b), cos(2πa/b), one_hot(b, 20), one_hot(label, 20)]
Dim: 2 + 20 + 20 = 42

LOO: leave-one-out on training set (a,b in 1..20)
OOD: train on a in 1..20, test on a in 21..50 (periodic encoding naturally extends)

Kill: LOO < 50% or OOD < 50%.
Success: LOO >= 80% AND OOD >= 80%.
"""

import time
import numpy as np

TRAIN_MAX   = 20
MAX_CLASS   = TRAIN_MAX
ALPHA       = 0.1
SPAWN_DIST  = 0.3   # cosine distance threshold (1 - cosine_sim)

B_DIM       = TRAIN_MAX       # one_hot(b)
LABEL_DIM   = MAX_CLASS       # one_hot(label)
TOTAL_DIM   = 2 + B_DIM + LABEL_DIM   # sin, cos, b, label

FEAT_DIM    = 2 + B_DIM   # sin + cos + b (used for NN only in analysis)

STEP296_REF = 0.868
STEP300_REF = 0.952   # OOD reference (reflection spawn)


def encode_train(a, b, y):
    v = np.zeros(TOTAL_DIM, dtype=np.float32)
    v[0] = np.sin(2 * np.pi * a / b)
    v[1] = np.cos(2 * np.pi * a / b)
    v[2 + b - 1] = 1.0           # b one-hot (b in 1..20 → indices 2..21)
    v[2 + B_DIM + y] = 1.0       # label one-hot
    return v


def encode_infer(a, b):
    v = np.zeros(TOTAL_DIM, dtype=np.float32)
    v[0] = np.sin(2 * np.pi * a / b)
    v[1] = np.cos(2 * np.pi * a / b)
    v[2 + b - 1] = 1.0
    # label dims = 0
    return v


class Codebook:
    def __init__(self):
        self.V = []
        self.n_spawn = 0
        self.n_absorb = 0

    def copy_state(self):
        return [v.copy() for v in self.V]

    def restore_state(self, state):
        self.V = [v.copy() for v in state]

    @property
    def V_arr(self):
        if not self.V:
            return np.zeros((0, TOTAL_DIM), dtype=np.float32)
        return np.array(self.V, dtype=np.float32)


def absorb(cb, x, alpha=ALPHA, spawn_dist=SPAWN_DIST):
    if len(cb.V) == 0:
        cb.V.append(x.copy())
        cb.n_spawn += 1
        return 0, True

    V = cb.V_arr
    x_norm = x / (np.linalg.norm(x) + 1e-10)
    V_norms = np.linalg.norm(V, axis=1, keepdims=True)
    V_normed = V / (V_norms + 1e-10)
    cos_sims = V_normed @ x_norm
    star_idx = int(np.argmax(cos_sims))
    cos_dist = 1.0 - float(cos_sims[star_idx])

    if cos_dist > spawn_dist:
        cb.V.append(x.copy())
        cb.n_spawn += 1
        return len(cb.V) - 1, True
    else:
        cb.V[star_idx] = (1.0 - alpha) * cb.V[star_idx] + alpha * x
        cb.n_absorb += 1
        return star_idx, False


def predict(cb, x_inf):
    state = cb.copy_state()
    star_idx, _ = absorb(cb, x_inf)
    moved = cb.V[star_idx]
    label_dims = moved[2 + B_DIM:]
    pred = int(np.argmax(label_dims)) if np.max(np.abs(label_dims)) > 1e-10 else -1
    cb.restore_state(state)
    return pred


def build_codebook(train_data, spawn_dist=SPAWN_DIST):
    cb = Codebook()
    for x, y, a, b in train_data:
        absorb(cb, x, spawn_dist=spawn_dist)
    return cb


def make_train_data():
    data = []
    for a in range(1, TRAIN_MAX + 1):
        for b in range(1, TRAIN_MAX + 1):
            y = a % b
            data.append((encode_train(a, b, y), y, a, b))
    return data


def evaluate_phase2(cb, train_data):
    """Standard evaluation: absorb inference input, restore, read label."""
    correct = 0
    for x_train, true_y, a, b in train_data:
        x_inf = encode_infer(a, b)
        pred = predict(cb, x_inf)
        if pred == true_y:
            correct += 1
    return correct / len(train_data)


def evaluate_loo(spawn_dist=SPAWN_DIST):
    """
    LOO: for each (held_a, held_b), train on all others, test prediction.
    Periodic encoding: same-class examples (different a) are identical → LOO
    only fails if (held_a%held_b, held_b) class has NO other training examples.
    """
    correct = 0
    total = 0
    by_b = {}   # track per-b accuracy

    for held_a in range(1, TRAIN_MAX + 1):
        for held_b in range(1, TRAIN_MAX + 1):
            held_y = held_a % held_b

            cb = Codebook()
            for a in range(1, TRAIN_MAX + 1):
                for b in range(1, TRAIN_MAX + 1):
                    if a == held_a and b == held_b:
                        continue
                    y = a % b
                    absorb(cb, encode_train(a, b, y), spawn_dist=spawn_dist)

            x_inf = encode_infer(held_a, held_b)
            pred = predict(cb, x_inf)

            if held_b not in by_b:
                by_b[held_b] = [0, 0]
            by_b[held_b][1] += 1
            if pred == held_y:
                correct += 1
                by_b[held_b][0] += 1
            total += 1

    return correct / total, by_b


def evaluate_ood(cb_train, ood_a_range=(21, 51)):
    """
    OOD: test on a in [ood_a_range), b in 1..20.
    Periodic encoding: sin(2π*a/b) = sin(2π*(a+kb)/b) for any integer k.
    So OOD encoding = in-distribution encoding for same class.
    """
    correct = 0
    total = 0
    by_b = {}

    for a in range(ood_a_range[0], ood_a_range[1]):
        for b in range(1, TRAIN_MAX + 1):
            true_y = a % b
            x_inf = encode_infer(a, b)
            pred = predict(cb_train, x_inf)

            if b not in by_b:
                by_b[b] = [0, 0]
            by_b[b][1] += 1
            if pred == true_y:
                correct += 1
                by_b[b][0] += 1
            total += 1

    return correct / total, by_b


def main():
    t0 = time.time()
    np.random.seed(42)

    print("Step 305 -- Periodic Encoding: The Tempest Physics Test", flush=True)
    print(f"f(State, D) = absorb. Physics = circular motion.", flush=True)
    print(f"Encoding: [sin(2pi*a/b), cos(2pi*a/b), one_hot(b), one_hot(label)], {TOTAL_DIM}d", flush=True)
    print(f"Same-class examples: identical sin/cos -> cosine = 1.0 -> always absorb.", flush=True)
    print(f"References: phi LOO={STEP296_REF*100:.1f}%, reflection OOD={STEP300_REF*100:.1f}%\n", flush=True)

    train_data = make_train_data()

    # ─── Phase 1: Build codebook ───────────────────────────────────────────
    print("=== Phase 1: Codebook structure ===", flush=True)
    cb = build_codebook(train_data)
    print(f"  Vectors: {len(cb.V)}  Spawns: {cb.n_spawn}  Absorbs: {cb.n_absorb}", flush=True)

    V = cb.V_arr
    lp = V[:, 2 + B_DIM:]
    mx = np.max(lp, axis=1)
    print(f"  Label signal: mean max={np.mean(mx):.3f}, min={np.min(mx):.3f}", flush=True)

    # Expected: ~num distinct (b, class) pairs. For a,b in 1..20, a%b:
    # max distinct (b, class) = sum over b of (number of classes for that b)
    print(flush=True)

    # ─── Phase 2: In-distribution accuracy ────────────────────────────────
    print("=== Phase 2: In-distribution accuracy ===", flush=True)
    t2 = time.time()
    acc_indist = evaluate_phase2(cb, train_data)
    print(f"  Accuracy: {acc_indist*100:.1f}%  (reference: {STEP296_REF*100:.1f}%)", flush=True)
    print(f"  Time: {time.time()-t2:.1f}s", flush=True)
    print(flush=True)

    # ─── Spawn threshold sweep ─────────────────────────────────────────────
    print("=== Spawn threshold sweep ===", flush=True)
    print(f"{'thresh':>8} | {'CB':>5} | {'Acc':>7}", flush=True)
    print("-" * 28, flush=True)
    for thresh in [0.05, 0.1, 0.2, 0.3, 0.4, 0.5]:
        cb_t = build_codebook(train_data, spawn_dist=thresh)
        acc_t = evaluate_phase2(cb_t, train_data)
        print(f"  {thresh:>6.2f} | {len(cb_t.V):>5} | {acc_t*100:>6.1f}%", flush=True)
    print(flush=True)

    # ─── Phase 3: OOD (a in 21..50) ───────────────────────────────────────
    print("=== Phase 3: OOD — a in 21..50 ===", flush=True)
    t3 = time.time()
    acc_ood, by_b_ood = evaluate_ood(cb)
    print(f"  OOD accuracy: {acc_ood*100:.1f}%  (reference: {STEP300_REF*100:.1f}%)", flush=True)
    print(f"  Time: {time.time()-t3:.1f}s", flush=True)

    # Per-b OOD breakdown
    print(f"\n  Per-b OOD:")
    print(f"  {'b':>3} | {'correct':>7} | {'total':>5} | {'acc':>7}", flush=True)
    print(f"  " + "-" * 32, flush=True)
    for b in sorted(by_b_ood.keys()):
        c, tot = by_b_ood[b]
        print(f"  {b:>3} | {c:>7} | {tot:>5} | {c/tot*100:>6.1f}%", flush=True)
    print(flush=True)

    # ─── Phase 4: LOO (slower) ─────────────────────────────────────────────
    print("=== Phase 4: LOO (leave-one-out) ===", flush=True)
    print("  (Running 400 LOO folds...)", flush=True)
    t4 = time.time()
    acc_loo, by_b_loo = evaluate_loo()
    print(f"  LOO accuracy: {acc_loo*100:.1f}%  (reference: {STEP296_REF*100:.1f}%)", flush=True)
    print(f"  Time: {time.time()-t4:.1f}s", flush=True)

    # Per-b LOO breakdown (just show b=1..5 and b=15..20)
    print(f"\n  Per-b LOO (selected b):")
    for b in list(range(1, 6)) + list(range(15, 21)):
        c, tot = by_b_loo[b]
        print(f"  b={b:>2}: {c}/{tot} = {c/tot*100:.0f}%", flush=True)
    print(flush=True)

    # ─── Summary ──────────────────────────────────────────────────────────
    elapsed = time.time() - t0
    print("=" * 65, flush=True)
    print("STEP 305 SUMMARY — Periodic Physics Test", flush=True)
    print("=" * 65, flush=True)
    print(f"In-dist:  {acc_indist*100:.1f}%", flush=True)
    print(f"OOD:      {acc_ood*100:.1f}%  (Step 300 ref: {STEP300_REF*100:.1f}%)", flush=True)
    print(f"LOO:      {acc_loo*100:.1f}%  (Step 296 ref: {STEP296_REF*100:.1f}%)", flush=True)
    print(flush=True)

    all_pass = acc_loo >= 0.80 and acc_ood >= 0.80
    either_kill = acc_loo < 0.50 or acc_ood < 0.50

    print("TEMPEST PHYSICS TEST:", flush=True)
    if all_pass:
        print(f"  SUCCESS -- LOO >= 80% AND OOD >= 80%.", flush=True)
        print(f"  Periodic physics + absorb substrate = confirmed.", flush=True)
        print(f"  The encoding (physics) was the missing piece.", flush=True)
    elif either_kill:
        print(f"  KILLED -- LOO={acc_loo*100:.1f}% or OOD={acc_ood*100:.1f}% < 50%.", flush=True)
        print(f"  Periodic physics does not rescue absorption.", flush=True)
    else:
        print(f"  PARTIAL -- Above kill bar, below 80% success.", flush=True)
        print(f"  Mechanism works but not fully realized.", flush=True)

    print(f"\nElapsed: {elapsed:.1f}s", flush=True)


if __name__ == '__main__':
    main()
