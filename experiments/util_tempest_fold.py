#!/usr/bin/env python3
"""
Tempest Fold — State(t+1) = f(State(t), D)

f = absorb. The nearest vector blends with the input.
D = stream of inputs. Training: features + label. Inference: features + zeros.
The observer reads the geometry. The substrate doesn't predict.

No labels. No output port. No metadata. Just vectors and transitions.
The prediction is where the state changed, observed from outside.
"""

import numpy as np
from collections import defaultdict


# === The Substrate ===

class TempestFold:
    """State = codebook of vectors. f = absorb nearest toward input."""

    def __init__(self, dim, alpha=0.1, spawn_radius=0.3):
        self.dim = dim
        self.alpha = alpha
        self.spawn_radius = spawn_radius
        self.state = np.zeros((0, dim), dtype=np.float32)  # empty state

    def f(self, D):
        """
        State(t+1) = f(State(t), D)

        The only operation. Absorb D into state.
        Returns the INDEX of the changed vector (for the observer).
        The substrate doesn't know what the index means.
        """
        D = D.astype(np.float32)

        if len(self.state) == 0:
            # First input — becomes the first state vector
            self.state = D.reshape(1, -1).copy()
            return 0

        # Find nearest
        sims = self.state @ D
        norms_s = np.linalg.norm(self.state, axis=1, keepdims=False)
        norms_d = np.linalg.norm(D)
        if norms_d < 1e-15:
            return -1  # zero input, no absorption
        cos_sims = sims / (norms_s * norms_d + 1e-15)
        winner_idx = int(np.argmax(cos_sims))
        max_sim = cos_sims[winner_idx]

        if max_sim < self.spawn_radius:
            # Nothing close enough — D becomes new state vector
            self.state = np.vstack([self.state, D.reshape(1, -1)])
            return len(self.state) - 1

        # Absorb: blend winner with D
        self.state[winner_idx] = (1 - self.alpha) * self.state[winner_idx] + self.alpha * D
        return winner_idx

    @property
    def size(self):
        return len(self.state)


# === The Observer (outside the substrate) ===

class Observer:
    """
    Watches state transitions. Maps geometry to classes.
    The observer is US — external to the substrate.
    """

    def __init__(self, n_feature_dims, n_label_dims):
        self.n_feat = n_feature_dims
        self.n_label = n_label_dims
        # Track which regions absorbed which training inputs
        self.region_class_counts = defaultdict(lambda: defaultdict(int))

    def encode_training(self, a, b, label, max_a=20, n_classes=20):
        """Encode (a, b, label) as D vector: features + label_onehot."""
        feat = np.zeros(self.n_feat, dtype=np.float32)
        # One-hot encoding for a and b (orthogonal, no similarity between different values)
        feat[a - 1] = 1.0
        feat[max_a + b - 1] = 1.0

        lab = np.zeros(self.n_label, dtype=np.float32)
        lab[label] = 1.0

        return np.concatenate([feat, lab])

    def encode_inference(self, a, b, max_a=20):
        """Encode (a, b) without label — zeros in label dims."""
        feat = np.zeros(self.n_feat, dtype=np.float32)
        feat[a - 1] = 1.0
        feat[max_a + b - 1] = 1.0

        lab = np.zeros(self.n_label, dtype=np.float32)
        return np.concatenate([feat, lab])

    def read_prediction(self, substrate, changed_idx):
        """
        Observer reads the geometry at the changed location.
        The prediction is the argmax of the label dimensions
        of the changed vector.
        """
        if changed_idx < 0 or changed_idx >= substrate.size:
            return -1
        vec = substrate.state[changed_idx]
        label_dims = vec[self.n_feat:]
        if np.max(np.abs(label_dims)) < 1e-10:
            return -1  # no label signal
        return int(np.argmax(label_dims))

    def observe_training(self, substrate, changed_idx, true_label):
        """Observer notes which class was absorbed where."""
        self.region_class_counts[changed_idx][true_label] += 1


# === The Experiment ===

def run_tempest_fold():
    MAX_A = 20
    N_CLASSES = MAX_A  # max possible remainder
    N_FEAT = 2 * MAX_A  # thermometer(a) + thermometer(b)
    DIM = N_FEAT + N_CLASSES

    # Sweep alpha and spawn_radius
    configs = [
        (0.01, 0.3),
        (0.01, 0.5),
        (0.01, 0.7),
        (0.05, 0.3),
        (0.05, 0.5),
        (0.05, 0.7),
        (0.1, 0.5),
        (0.1, 0.7),
        (0.1, 0.8),
        (0.2, 0.7),
        (0.2, 0.8),
        (0.5, 0.8),
        (0.5, 0.9),
    ]

    print("TEMPEST FOLD — State(t+1) = f(State(t), D)")
    print(f"State: codebook in R^{DIM}")
    print(f"f: absorb (blend nearest with input)")
    print(f"D: (a,b) in 1..{MAX_A}, label = a%b")
    print(f"Observer: reads label dims of changed vector")
    print()

    # Build training stream
    training = []
    for a in range(1, MAX_A + 1):
        for b in range(1, MAX_A + 1):
            training.append((a, b, a % b))

    # Shuffle training order (multiple seeds)
    rng = np.random.RandomState(42)

    best_acc = 0
    best_cfg = None

    for alpha, spawn_r in configs:
        # --- Phase 1: Training (absorb with labels) ---
        substrate = TempestFold(DIM, alpha=alpha, spawn_radius=spawn_r)
        observer = Observer(N_FEAT, N_CLASSES)

        order = list(range(len(training)))
        rng.shuffle(order)

        for idx in order:
            a, b, y = training[idx]
            D = observer.encode_training(a, b, y, MAX_A, N_CLASSES)
            changed = substrate.f(D)
            observer.observe_training(substrate, changed, y)

        # --- Phase 2: Inference (absorb without labels, observe) ---
        # Save state for restoration
        saved_state = substrate.state.copy()

        correct = 0
        total = 0
        for a in range(1, MAX_A + 1):
            for b in range(1, MAX_A + 1):
                true_y = a % b
                D_inf = observer.encode_inference(a, b, MAX_A)

                # Save, absorb, observe, restore
                pre_state = substrate.state.copy()
                changed = substrate.f(D_inf)
                pred = observer.read_prediction(substrate, changed)
                substrate.state = pre_state  # restore (non-destructive observation)

                if pred == true_y:
                    correct += 1
                total += 1

        acc = correct / total
        if acc > best_acc:
            best_acc = acc
            best_cfg = (alpha, spawn_r)

        print(f"  alpha={alpha:.2f} spawn={spawn_r:.1f}: "
              f"CB={substrate.size:>4}, acc={acc*100:.1f}%")

    print()
    print(f"Best: alpha={best_cfg[0]}, spawn={best_cfg[1]}, acc={best_acc*100:.1f}%")
    print()

    # --- Phase 3: Stream mode (no restore) with best config ---
    alpha, spawn_r = best_cfg
    substrate = TempestFold(DIM, alpha=alpha, spawn_radius=spawn_r)
    observer = Observer(N_FEAT, N_CLASSES)

    # Train
    order = list(range(len(training)))
    rng = np.random.RandomState(42)
    rng.shuffle(order)
    for idx in order:
        a, b, y = training[idx]
        D = observer.encode_training(a, b, y, MAX_A, N_CLASSES)
        substrate.f(D)

    print(f"Phase 3: Stream mode (alpha={alpha}, spawn={spawn_r})")
    print(f"  Post-training CB: {substrate.size}")

    # Stream 5 passes of inference WITHOUT restoring state
    for pass_num in range(5):
        correct = 0
        total = 0
        rng2 = np.random.RandomState(pass_num)
        test_order = list(range(len(training)))
        rng2.shuffle(test_order)

        for idx in test_order:
            a, b, y = training[idx]
            D_inf = observer.encode_inference(a, b, MAX_A)
            changed = substrate.f(D_inf)
            pred = observer.read_prediction(substrate, changed)
            if pred == y:
                correct += 1
            total += 1

        acc = correct / total
        print(f"  Pass {pass_num}: CB={substrate.size}, acc={acc*100:.1f}%")

    # --- Phase 4: Multi-epoch training + inference ---
    print()
    print("Phase 4: Multi-epoch training (5 epochs) then inference")
    substrate = TempestFold(DIM, alpha=alpha, spawn_radius=spawn_r)
    observer = Observer(N_FEAT, N_CLASSES)

    for epoch in range(5):
        rng_e = np.random.RandomState(epoch * 7)
        order = list(range(len(training)))
        rng_e.shuffle(order)
        for idx in order:
            a, b, y = training[idx]
            D = observer.encode_training(a, b, y, MAX_A, N_CLASSES)
            substrate.f(D)

    # Inference (with restore)
    correct = 0
    total = 0
    for a in range(1, MAX_A + 1):
        for b in range(1, MAX_A + 1):
            true_y = a % b
            D_inf = observer.encode_inference(a, b, MAX_A)
            pre = substrate.state.copy()
            changed = substrate.f(D_inf)
            pred = observer.read_prediction(substrate, changed)
            substrate.state = pre
            if pred == true_y:
                correct += 1
            total += 1

    print(f"  5-epoch CB: {substrate.size}, acc={correct/total*100:.1f}%")

    # --- Phase 5: OOD test ---
    # One-hot can't represent a>20 directly. OOD test skipped for one-hot.
    # Instead: test LOO (leave-one-out) to verify the substrate isn't just memorizing.
    print()
    print("Phase 5: LOO test (remove each training example, test prediction)")
    substrate_loo = TempestFold(DIM, alpha=alpha, spawn_radius=spawn_r)
    observer_loo = Observer(N_FEAT, N_CLASSES)

    correct = 0
    total = 0
    for held_a in range(1, MAX_A + 1):
        for held_b in range(1, MAX_A + 1):
            sub = TempestFold(DIM, alpha=alpha, spawn_radius=spawn_r)
            obs = Observer(N_FEAT, N_CLASSES)
            # Train on everything EXCEPT (held_a, held_b)
            for a in range(1, MAX_A + 1):
                for b in range(1, MAX_A + 1):
                    if a == held_a and b == held_b:
                        continue
                    D = obs.encode_training(a, b, a % b, MAX_A, N_CLASSES)
                    sub.f(D)
            # Test on held-out
            true_y = held_a % held_b
            D_inf = obs.encode_inference(held_a, held_b, MAX_A)
            pre = sub.state.copy()
            changed = sub.f(D_inf)
            pred = obs.read_prediction(sub, changed)
            sub.state = pre
            if pred == true_y:
                correct += 1
            total += 1
            if total % 100 == 0:
                print(f"    {total}/400: {correct/total*100:.1f}%", flush=True)

    print(f"  LOO acc: {correct/total*100:.1f}%")
    print()
    print("Done.")


if __name__ == '__main__':
    run_tempest_fold()
