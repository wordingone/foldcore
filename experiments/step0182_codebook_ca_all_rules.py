"""
Step 182: Iterated k-NN across all 10 elementary CA rules.

9/10 rules perfectly iterated to 10 steps. Only Rule 225 fails.
Includes Rule 110 (Turing-complete).

Usage:
    python experiments/run_step182_ca_all_rules.py
"""

import torch
import torch.nn.functional as F

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def test_iterated_ca(rule_num, width=15, n_train=1000, n_test=100, n_steps=10, k=5):
    rule = {((i >> 2) & 1, (i >> 1) & 1, i & 1): (rule_num >> i) & 1 for i in range(8)}

    def evolve1(row):
        new = torch.zeros_like(row)
        for i in range(1, len(row) - 1):
            nb = (row[i - 1].item(), row[i].item(), row[i + 1].item())
            new[i] = rule[nb]
        return new

    # Train 1-step predictor
    X_tr = torch.randint(0, 2, (n_train, width), device=device).float()
    y_tr = torch.stack([evolve1(X_tr[i]) for i in range(n_train)]).to(device)

    def predict_step(row):
        pred = torch.zeros(width, device=device)
        for cell in range(1, width - 1):
            feat_tr = X_tr[:, cell - 1:cell + 2]
            feat_te = row[cell - 1:cell + 2].unsqueeze(0)
            sims = F.normalize(feat_te, dim=1) @ F.normalize(feat_tr, dim=1).T
            scores = torch.zeros(1, 2, device=device)
            for c in range(2):
                m = y_tr[:, cell] == c
                cs = sims[:, m]
                if cs.shape[1] == 0:
                    continue
                scores[:, c] = cs.topk(min(k, cs.shape[1]), dim=1).values.sum(dim=1)
            pred[cell] = scores.argmax(1).float()
        return pred

    # Test iterated prediction
    X_te = torch.randint(0, 2, (n_test, width), device=device).float()
    true_final = X_te.clone()
    for _ in range(n_steps):
        true_final = torch.stack([evolve1(true_final[i]) for i in range(n_test)])

    correct = 0
    for i in range(n_test):
        current = X_te[i].clone()
        for _ in range(n_steps):
            current = predict_step(current)
        correct += (current[1:-1] == true_final[i, 1:-1].to(device)).sum().item()

    return correct / (n_test * (width - 2)) * 100


def main():
    print(f"Iterated k-NN across elementary CA rules (10 steps, width=15):")
    print(f"{'Rule':>5s} | {'Accuracy':>8s}")
    print(f"------|----------")

    perfect = 0
    total = 0
    for rule_num in [30, 54, 60, 90, 110, 150, 182, 210, 225, 250]:
        acc = test_iterated_ca(rule_num)
        total += 1
        if acc == 100:
            perfect += 1
        print(f"{rule_num:5d} | {acc:7.1f}%")

    print(f"\nPerfect: {perfect}/{total}")


if __name__ == '__main__':
    main()
