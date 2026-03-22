"""
Step 190: End-to-end atomic substrate — Fibonacci mod 10.

Discovers features via LOO-scored random cosines on raw 2D integer input,
then iterates the 1-step predictor for multi-step computation.
Achieves 100% at 50 steps with 5 discovered features.

Usage:
    python experiments/run_step190_fibonacci_e2e.py
"""

import torch
import torch.nn.functional as F

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def loo_accuracy(V, labels, n_classes, k=5):
    """Leave-one-out accuracy — the self-evaluation signal."""
    V_n = F.normalize(V, dim=1)
    sims = V_n @ V_n.T
    sims.fill_diagonal_(-1e9)
    scores = torch.zeros(V.shape[0], n_classes, device=device)
    for c in range(n_classes):
        m = labels == c
        cs = sims[:, m]
        if cs.shape[1] == 0:
            continue
        scores[:, c] = cs.topk(min(k, cs.shape[1]), dim=1).values.sum(dim=1)
    return (scores.argmax(1) == labels).float().mean().item()


def discover_features(states, next_states, mod_n, d=2, max_rounds=30,
                      n_candidates=150, threshold=0.0003, k=5):
    """Discover random cosine features that improve LOO accuracy."""
    V = states.clone()
    discovered = []

    for step in range(max_rounds):
        current_loo = (
            loo_accuracy(F.normalize(V, dim=1), next_states[:, 0].long(), mod_n, k) +
            loo_accuracy(F.normalize(V, dim=1), next_states[:, 1].long(), mod_n, k)
        ) / 2

        best_w = None
        best_b = None
        best_loo = current_loo

        for _ in range(n_candidates):
            w = torch.randn(d, device=device)
            b = torch.rand(1, device=device) * mod_n
            feat = torch.cos(states @ w + b).unsqueeze(1)
            aug = F.normalize(torch.cat([V, feat], dim=1), dim=1)
            l = (
                loo_accuracy(aug, next_states[:, 0].long(), mod_n, k) +
                loo_accuracy(aug, next_states[:, 1].long(), mod_n, k)
            ) / 2
            if l > best_loo + threshold:
                best_loo = l
                best_w = w.clone()
                best_b = b.clone()

        if best_w is None:
            break
        discovered.append((best_w, best_b))
        V = torch.cat([V, torch.cos(states @ best_w + best_b).unsqueeze(1)], dim=1)

    return discovered


def augment(raw, discovered, d=2):
    """Apply discovered features to raw input."""
    aug = raw.clone() if raw.dim() == 2 else raw.unsqueeze(0)
    for w, b in discovered:
        aug = torch.cat([aug, torch.cos(aug[:, :d] @ w + b).unsqueeze(1)], dim=1)
    return F.normalize(aug, dim=1)


def predict_step(state_raw, states_tr, next_states_tr, discovered, mod_n, d=2, k=5):
    """Predict next state via classification-based k-NN on augmented features."""
    X_tr_aug = augment(states_tr, discovered, d)
    state_aug = augment(state_raw.unsqueeze(0), discovered, d)
    sims = state_aug @ X_tr_aug.T

    pred = torch.zeros(2, device=device)
    for dim in range(2):
        targets = next_states_tr[:, dim].long()
        scores = torch.zeros(mod_n, device=device)
        for c in range(mod_n):
            m = targets == c
            cs = sims[0, m]
            if cs.shape[0] == 0:
                continue
            scores[c] = cs.topk(min(k, cs.shape[0])).values.sum()
        pred[dim] = scores.argmax().float()
    return pred


def main():
    mod_n = 10
    d = 2
    n_train = 800
    k = 5

    # Generate training data: (a, b) -> (b, (a+b) % mod_n)
    states = torch.randint(0, mod_n, (n_train, 2), device=device).float()
    next_states = torch.stack([states[:, 1], (states[:, 0] + states[:, 1]) % mod_n], dim=1)

    print(f"Fibonacci mod {mod_n}: discovering features...")
    discovered = discover_features(states, next_states, mod_n, d)
    print(f"Discovered {len(discovered)} features (d={d} -> d={d + len(discovered)})")

    # Test iterated prediction
    n_test = 100
    print(f"\nIterated k-NN with discovered features:")
    for n_steps in [1, 5, 10, 20, 50]:
        correct = 0
        for _ in range(n_test):
            a, b = torch.randint(0, mod_n, (2,)).tolist()
            true_a, true_b = a, b
            for _ in range(n_steps):
                true_a, true_b = true_b, (true_a + true_b) % mod_n

            current = torch.tensor([float(a), float(b)], device=device)
            for _ in range(n_steps):
                current = predict_step(current, states, next_states, discovered, mod_n, d, k)

            if int(current[0].item()) == true_a and int(current[1].item()) == true_b:
                correct += 1

        print(f"  {n_steps:2d} steps: {correct}% correct")


if __name__ == '__main__':
    main()
