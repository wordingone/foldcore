#!/usr/bin/env python3
"""
Step 303 -- The ABSORB operation. Atomic substrate.

Spec. One operation, no output, no labels. The codebook is just
vectors in R^d. The prediction IS the state change (where did the codebook move?).

absorb(codebook, x):
    v_star = nearest(codebook, x)
    blend = (1 - alpha) * v_star + alpha * x
    codebook.replace(v_star, blend)

Training: feed (features, label_onehot). Labels absorbed into codebook positions.
Inference: feed (features, zeros). Observe which vector moved. Prediction = argmax
           of label dims in the moved vector (built up by prior absorptions).

Phase 1: Build codebook from empty (spawn if cosine_dist > 0.3).
Phase 2: Classify — absorb, observe moved vector, restore. Accuracy vs Step 296.
Phase 3: Stream — 100 inference inputs, NO restore. Does codebook degrade?

Kill: Phase 2 < 50%, or Phase 3 degrades to chance.
Success: Phase 2 >= 80% AND Phase 3 stable.
"""

import time
import numpy as np

# ─── Config ────────────────────────────────────────────────────────────────────

TRAIN_MAX   = 20
MAX_CLASS   = TRAIN_MAX            # max a%b = 19
ALPHA       = 0.1
SPAWN_DIST  = 0.3                  # cosine distance threshold for spawning

FEAT_DIM    = 2 * TRAIN_MAX        # therm(a) ++ therm(b)
TOTAL_DIM   = FEAT_DIM + MAX_CLASS # features ++ label

STEP296_REF = 0.868

# ─── Encoding ─────────────────────────────────────────────────────────────────

def therm(val, max_val):
    v = np.zeros(max_val, dtype=np.float32)
    v[:val] = 1.0
    return v


def encode_train(a, b, y):
    v = np.zeros(TOTAL_DIM, dtype=np.float32)
    v[:TRAIN_MAX] = therm(a, TRAIN_MAX)
    v[TRAIN_MAX:FEAT_DIM] = therm(b, TRAIN_MAX)
    v[FEAT_DIM + y] = 1.0
    return v


def encode_infer(a, b):
    v = np.zeros(TOTAL_DIM, dtype=np.float32)
    v[:TRAIN_MAX] = therm(a, TRAIN_MAX)
    v[TRAIN_MAX:FEAT_DIM] = therm(b, TRAIN_MAX)
    # label dims = zero (inference)
    return v

# ─── Codebook state ───────────────────────────────────────────────────────────

class Codebook:
    def __init__(self):
        self.V = []          # list of float32 vectors
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

# ─── The ABSORB operation ─────────────────────────────────────────────────────

def absorb(cb, x, alpha=ALPHA, spawn_dist=SPAWN_DIST):
    """
    The single atomic operation. No output.
    Returns: (star_idx, spawned) for analysis only.
    """
    if len(cb.V) == 0:
        cb.V.append(x.copy())
        cb.n_spawn += 1
        return 0, True

    V = cb.V_arr
    # Cosine distance: 1 - cosine_similarity
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

# ─── Phase 1: Build codebook ──────────────────────────────────────────────────

def build_codebook(shuffle=False):
    cb = Codebook()
    train_data = []
    for a in range(1, TRAIN_MAX + 1):
        for b in range(1, TRAIN_MAX + 1):
            y = a % b
            train_data.append((encode_train(a, b, y), y, a, b))

    if shuffle:
        import random
        random.shuffle(train_data)

    for x, y, a, b in train_data:
        absorb(cb, x)

    return cb, train_data

# ─── Phase 2: Classify ────────────────────────────────────────────────────────

def predict(cb, x_inf):
    """
    Absorb inference input, observe which vector moved, read label dims.
    Does NOT permanently modify cb (restores state after).
    """
    state = cb.copy_state()
    star_idx, _ = absorb(cb, x_inf)

    # Prediction: argmax of label dims in the moved vector (post-blend)
    moved = cb.V[star_idx]
    label_dims = moved[FEAT_DIM:]
    pred = int(np.argmax(label_dims))

    cb.restore_state(state)
    return pred, star_idx


def evaluate_phase2(cb, train_data):
    correct = 0
    total = 0
    for x_train, true_y, a, b in train_data:
        x_inf = encode_infer(a, b)
        pred, _ = predict(cb, x_inf)
        if pred == true_y:
            correct += 1
        total += 1
    return correct / total


def evaluate_1nn(cb, train_data):
    """Baseline: NN classification using only feature dims."""
    V = cb.V_arr
    V_feat = V[:, :FEAT_DIM]
    correct = 0
    for x_train, true_y, a, b in train_data:
        x_inf = encode_infer(a, b)
        x_feat = x_inf[:FEAT_DIM]
        dists = np.sum((V_feat - x_feat)**2, axis=1)
        nn_idx = int(np.argmin(dists))
        # Predict from label dims of nearest codebook vector
        label_dims = V[nn_idx, FEAT_DIM:]
        pred = int(np.argmax(label_dims))
        if pred == true_y:
            correct += 1
    return correct / len(train_data)

# ─── Phase 3: Stream ─────────────────────────────────────────────────────────

def evaluate_phase3(cb, train_data, n_stream=100):
    """
    Feed n_stream random inference inputs WITHOUT restoring codebook.
    Track accuracy over time. Tests self-correction / degradation.
    """
    import random
    stream = random.choices(train_data, k=n_stream)

    accuracies = []
    window = 10
    correct_window = 0

    for i, (x_train, true_y, a, b) in enumerate(stream):
        x_inf = encode_infer(a, b)

        # Absorb WITHOUT restoring (permanent modification)
        star_idx, spawned = absorb(cb, x_inf)
        moved = cb.V[star_idx]
        label_dims = moved[FEAT_DIM:]
        pred = int(np.argmax(label_dims))

        correct = (pred == true_y)
        correct_window += int(correct)
        if (i + 1) % window == 0:
            acc = correct_window / window
            accuracies.append(acc)
            correct_window = 0

    return accuracies

# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    t0 = time.time()
    np.random.seed(42)

    print("Step 303 -- ABSORB: Atomic Substrate", flush=True)
    print(f"Operation: blend = (1-alpha)*v_star + alpha*x, alpha={ALPHA}", flush=True)
    print(f"Spawn threshold: cosine_dist > {SPAWN_DIST}", flush=True)
    print(f"Encoding: therm(a) ++ therm(b) ++ one_hot(label), {TOTAL_DIM}d", flush=True)
    print(f"References: Step 296 in-dist={STEP296_REF*100:.1f}%", flush=True)
    print(flush=True)

    # ─── Phase 1: Build codebook ───────────────────────────────────────────
    print("=== Phase 1: Build codebook from empty ===", flush=True)
    cb, train_data = build_codebook()
    print(f"  Training examples fed: {len(train_data)}", flush=True)
    print(f"  Codebook size: {len(cb.V)} vectors", flush=True)
    print(f"  Spawns: {cb.n_spawn}  Absorptions: {cb.n_absorb}", flush=True)

    # Analyze codebook: label dim structure
    V = cb.V_arr
    label_part = V[:, FEAT_DIM:]
    # Is there a dominant class per vector?
    max_label = np.max(label_part, axis=1)
    has_label = np.sum(max_label > 0.01)
    print(f"  Vectors with non-zero label dims: {has_label}/{len(cb.V)}", flush=True)
    print(f"  Mean max label dim: {np.mean(max_label):.3f}  Min: {np.min(max_label):.3f}",
          flush=True)
    print(flush=True)

    # ─── Phase 2: Classify ────────────────────────────────────────────────
    print("=== Phase 2: Classification accuracy ===", flush=True)
    t2 = time.time()
    acc_phase2 = evaluate_phase2(cb, train_data)
    acc_1nn = evaluate_1nn(cb, train_data)
    print(f"  ABSORB prediction: {acc_phase2*100:.1f}%", flush=True)
    print(f"  1-NN in codebook:  {acc_1nn*100:.1f}%", flush=True)
    print(f"  Step 296 reference: {STEP296_REF*100:.1f}%", flush=True)
    print(f"  Time: {time.time()-t2:.1f}s", flush=True)
    print(flush=True)

    # ─── Spawn threshold sweep ────────────────────────────────────────────
    print("=== Spawn threshold sweep (Phase 2) ===", flush=True)
    print(f"{'threshold':>10} | {'CB size':>8} | {'Spawns':>7} | {'Acc':>7}", flush=True)
    print("-" * 42, flush=True)
    for thresh in [0.1, 0.2, 0.3, 0.4, 0.5, 0.7]:
        cb_t = Codebook()
        for x, y, a, b in train_data:
            absorb(cb_t, x, spawn_dist=thresh)
        acc_t = evaluate_phase2(cb_t, train_data)
        print(f"  {thresh:>8.1f} | {len(cb_t.V):>8} | {cb_t.n_spawn:>7} | {acc_t*100:>6.1f}%",
              flush=True)
    print(flush=True)

    # ─── Phase 3: Stream stability ────────────────────────────────────────
    print("=== Phase 3: Stream stability (100 inference inputs, no restore) ===",
          flush=True)
    import copy
    cb3 = Codebook()
    cb3.V = [v.copy() for v in cb.V]
    accs3 = evaluate_phase3(cb3, train_data, n_stream=100)
    print(f"  Accuracy by 10-input windows:", flush=True)
    for i, a in enumerate(accs3):
        print(f"    Window {(i+1)*10:>3}: {a*100:.0f}%", flush=True)
    first10 = accs3[0] if accs3 else 0
    last10 = accs3[-1] if accs3 else 0
    degraded = last10 < first10 - 0.2
    print(flush=True)
    print(f"  First window: {first10*100:.0f}%  Last window: {last10*100:.0f}%", flush=True)
    print(f"  Trend: {'DEGRADED' if degraded else 'STABLE'} ({(last10-first10)*100:+.0f}pp)",
          flush=True)
    print(flush=True)

    # ─── Multiple passes (multi-epoch training) ──────────────────────────
    print("=== Multi-pass training (N epochs) ===", flush=True)
    print(f"{'N passes':>9} | {'CB size':>8} | {'Acc':>7}", flush=True)
    print("-" * 32, flush=True)
    cb_mp = Codebook()
    for n_pass in range(1, 6):
        for x, y, a, b in train_data:
            absorb(cb_mp, x)
        acc_mp = evaluate_phase2(cb_mp, train_data)
        print(f"  {n_pass:>7} | {len(cb_mp.V):>8} | {acc_mp*100:>6.1f}%", flush=True)
    print(flush=True)

    # ─── Summary ──────────────────────────────────────────────────────────
    elapsed = time.time() - t0
    print("=" * 65, flush=True)
    print("STEP 303 SUMMARY", flush=True)
    print("=" * 65, flush=True)
    print(f"Phase 2 accuracy: {acc_phase2*100:.1f}%  (reference: {STEP296_REF*100:.1f}%)",
          flush=True)
    print(f"Phase 3 trend:    {'STABLE' if not degraded else 'DEGRADED'}  "
          f"({first10*100:.0f}% -> {last10*100:.0f}%)", flush=True)
    print(flush=True)
    print("KILL CRITERION:", flush=True)
    if acc_phase2 < 0.50:
        print(f"  KILLED -- Phase 2 ({acc_phase2*100:.1f}%) < 50%", flush=True)
        print(f"  Absorption-based classification fails.", flush=True)
    elif degraded:
        print(f"  KILLED -- Phase 3 degrades to chance.", flush=True)
        print(f"  The stream is self-destructive.", flush=True)
    else:
        print(f"  PASSES -- Phase 2 ({acc_phase2*100:.1f}%) >= 50% AND Phase 3 stable.",
              flush=True)
        if acc_phase2 >= 0.80:
            print(f"  SUCCESS -- Phase 2 >= 80%.", flush=True)
            print(f"  The atomic ABSORB operation classifies non-Lipschitz functions.", flush=True)

    print(f"\nElapsed: {elapsed:.1f}s", flush=True)


if __name__ == '__main__':
    main()
