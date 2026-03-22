#!/usr/bin/env python3
"""
Step 304 -- Feature-only NN in ABSORB. One hypothesis, one test.

The Step 303 failure diagnosis:
  Training D = [features | label_onehot]
  Inference D = [features | zeros]
  Cosine NN over full 60d vector → inference input looks UNLIKE training
  vectors in label dims → NN finds wrong vector → prediction biased toward
  vectors with near-zero label signal (averaged-away absorptions).

Fix: cosine NN on feature dims only. Blend still on full vector.
This means the substrate decides "which vector to update" based on feature
similarity alone — the label dims accumulate independently of the NN.

Kill: Phase 2 < 50%. (Same as Step 303.)
Question: does separating NN-space from blend-space rescue ABSORB?
"""

import time
import numpy as np

TRAIN_MAX   = 20
MAX_CLASS   = TRAIN_MAX
ALPHA       = 0.1
SPAWN_DIST  = 0.3

FEAT_DIM    = 2 * TRAIN_MAX   # therm(a) ++ therm(b)
TOTAL_DIM   = FEAT_DIM + MAX_CLASS

STEP303_REF = 0.260
STEP296_REF = 0.868


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


def absorb_feature_nn(cb, x, alpha=ALPHA, spawn_dist=SPAWN_DIST):
    """
    ABSORB with feature-only NN.
    NN decision: cosine sim on feature dims only (first FEAT_DIM).
    Blend: full TOTAL_DIM vector.
    """
    if len(cb.V) == 0:
        cb.V.append(x.copy())
        cb.n_spawn += 1
        return 0, True

    V = cb.V_arr
    # NN on feature dims only
    x_feat = x[:FEAT_DIM]
    V_feat = V[:, :FEAT_DIM]

    x_norm = x_feat / (np.linalg.norm(x_feat) + 1e-10)
    V_norms = np.linalg.norm(V_feat, axis=1, keepdims=True)
    V_normed = V_feat / (V_norms + 1e-10)
    cos_sims = V_normed @ x_norm
    star_idx = int(np.argmax(cos_sims))
    cos_dist = 1.0 - float(cos_sims[star_idx])

    if cos_dist > spawn_dist:
        cb.V.append(x.copy())
        cb.n_spawn += 1
        return len(cb.V) - 1, True
    else:
        # Blend full vector
        cb.V[star_idx] = (1.0 - alpha) * cb.V[star_idx] + alpha * x
        cb.n_absorb += 1
        return star_idx, False


def absorb_full_nn(cb, x, alpha=ALPHA, spawn_dist=SPAWN_DIST):
    """Original ABSORB: NN on full vector (Step 303 baseline)."""
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


def build_codebook(absorb_fn, shuffle=False):
    cb = Codebook()
    train_data = []
    for a in range(1, TRAIN_MAX + 1):
        for b in range(1, TRAIN_MAX + 1):
            y = a % b
            train_data.append((encode_train(a, b, y), y, a, b))
    if shuffle:
        import random; random.shuffle(train_data)
    for x, y, a, b in train_data:
        absorb_fn(cb, x)
    return cb, train_data


def evaluate(cb, train_data, absorb_fn):
    """Phase 2: predict via absorb, observe changed vector, restore."""
    correct = 0
    for x_train, true_y, a, b in train_data:
        x_inf = encode_infer(a, b)
        state = cb.copy_state()
        star_idx, _ = absorb_fn(cb, x_inf)
        moved = cb.V[star_idx]
        label_dims = moved[FEAT_DIM:]
        pred = int(np.argmax(label_dims))
        cb.restore_state(state)
        if pred == true_y:
            correct += 1
    return correct / len(train_data)


def threshold_sweep(absorb_fn, train_data, label):
    print(f"\n{label}")
    print(f"{'thresh':>8} | {'CB':>5} | {'Acc':>7}")
    print("-" * 28)
    for thresh in [0.1, 0.2, 0.3, 0.4, 0.5, 0.7]:
        cb = Codebook()
        for x, y, a, b in train_data:
            absorb_fn(cb, x, spawn_dist=thresh)
        acc = evaluate(cb, train_data, lambda cb2, x2: absorb_fn(cb2, x2, spawn_dist=thresh))
        print(f"  {thresh:>6.1f} | {len(cb.V):>5} | {acc*100:>6.1f}%")


def main():
    t0 = time.time()
    np.random.seed(42)

    print("Step 304 -- Feature-only NN in ABSORB", flush=True)
    print(f"Hypothesis: cosine NN on feat dims only rescues label accumulation", flush=True)
    print(f"Encoding: therm(a,20) ++ therm(b,20) ++ one_hot(label,20), {TOTAL_DIM}d", flush=True)
    print(f"References: Step 303={STEP303_REF*100:.1f}%, Step 296={STEP296_REF*100:.1f}%\n", flush=True)

    # Build training data
    cb_feat, train_data = build_codebook(absorb_feature_nn)
    cb_full, _ = build_codebook(absorb_full_nn)

    print("=== Phase 1: Codebook structure ===", flush=True)
    print(f"  Feature-only NN: {len(cb_feat.V)} vectors, "
          f"{cb_feat.n_spawn} spawns, {cb_feat.n_absorb} absorbs", flush=True)

    V_feat_arr = cb_feat.V_arr
    lp = V_feat_arr[:, FEAT_DIM:]
    mx = np.max(lp, axis=1)
    print(f"  Label signal: mean max={np.mean(mx):.3f}, min={np.min(mx):.3f}\n", flush=True)

    print(f"  Full-vector NN: {len(cb_full.V)} vectors, "
          f"{cb_full.n_spawn} spawns, {cb_full.n_absorb} absorbs", flush=True)

    V_full_arr = cb_full.V_arr
    lp2 = V_full_arr[:, FEAT_DIM:]
    mx2 = np.max(lp2, axis=1)
    print(f"  Label signal: mean max={np.mean(mx2):.3f}, min={np.min(mx2):.3f}\n", flush=True)

    # Phase 2
    print("=== Phase 2: Classification accuracy ===", flush=True)
    acc_feat = evaluate(cb_feat, train_data, absorb_feature_nn)
    acc_full = evaluate(cb_full, train_data, absorb_full_nn)
    print(f"  Feature-only NN: {acc_feat*100:.1f}%", flush=True)
    print(f"  Full-vector NN:  {acc_full*100:.1f}%  (Step 303 default)", flush=True)
    print(f"  Step 296 ref:    {STEP296_REF*100:.1f}%\n", flush=True)

    # Threshold sweep — feature NN only
    threshold_sweep(absorb_feature_nn, train_data, "Threshold sweep (feature-only NN):")

    # Multi-pass
    print("\n=== Multi-pass training (feature-only NN) ===", flush=True)
    print(f"{'N':>4} | {'CB':>5} | {'Acc':>7}", flush=True)
    print("-" * 22, flush=True)
    cb_mp = Codebook()
    for n_pass in range(1, 6):
        for x, y, a, b in train_data:
            absorb_feature_nn(cb_mp, x)
        acc_mp = evaluate(cb_mp, train_data, absorb_feature_nn)
        print(f"  {n_pass:>2} | {len(cb_mp.V):>5} | {acc_mp*100:>6.1f}%", flush=True)

    elapsed = time.time() - t0
    print(f"\n{'='*60}", flush=True)
    print("STEP 304 SUMMARY", flush=True)
    print(f"{'='*60}", flush=True)
    print(f"Feature-only NN: {acc_feat*100:.1f}%  vs  Full-vector NN: {acc_full*100:.1f}%", flush=True)
    print(f"Delta: {(acc_feat-acc_full)*100:+.1f}pp", flush=True)

    if acc_feat >= 0.80:
        print(f"\nSUCCESS -- Feature-only NN >= 80%.", flush=True)
        print(f"Encoding that separates NN-space from blend-space rescues ABSORB.", flush=True)
    elif acc_feat >= 0.50:
        print(f"\nPASSES kill criterion (>= 50%). Not yet 80%.", flush=True)
    else:
        print(f"\nKILLED -- {acc_feat*100:.1f}% < 50%.", flush=True)
        print(f"Feature-only NN does not rescue label accumulation.", flush=True)

    print(f"\nElapsed: {elapsed:.1f}s", flush=True)


if __name__ == '__main__':
    main()
