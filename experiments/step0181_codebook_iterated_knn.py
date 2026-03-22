"""
Step 181: Iterated k-NN closes the computation gap.

Direct N-step prediction degrades (70% at 10 steps).
Iterated 1-step k-NN maintains 100% by composing the rule through self-application.

Usage:
    python experiments/run_step181_iterated_knn.py
"""

import torch
import torch.nn.functional as F

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def make_ca_rule(rule_num):
    return {((i >> 2) & 1, (i >> 1) & 1, i & 1): (rule_num >> i) & 1 for i in range(8)}


def evolve_row(row, rule):
    new = torch.zeros_like(row)
    for i in range(1, len(row) - 1):
        nb = (row[i - 1].item(), row[i].item(), row[i + 1].item())
        new[i] = rule[nb]
    return new


def train_1step(rule, width=15, n_train=1000):
    """Generate 1-step CA training data."""
    X, y = [], []
    rows = [torch.randint(0, 2, (width,), device=device).float() for _ in range(n_train // (width - 2) + 1)]
    for row in rows:
        next_row = evolve_row(row, rule)
        for i in range(1, width - 1):
            X.append(row[i - 1:i + 2].clone())
            y.append(next_row[i].item())
    X = torch.stack(X[:n_train])
    y = torch.tensor(y[:n_train], device=device, dtype=torch.long)
    return X, y


def predict_one_step(row, X_train, y_train, width, k=5):
    """Predict next CA row using local 3-cell k-NN."""
    pred = torch.zeros(width, device=device)
    for cell in range(1, width - 1):
        feat_te = row[cell - 1:cell + 2].unsqueeze(0)
        sims = F.normalize(feat_te, dim=1) @ F.normalize(X_train, dim=1).T
        scores = torch.zeros(1, 2, device=device)
        for c in range(2):
            m = y_train == c
            cs = sims[:, m]
            if cs.shape[1] == 0:
                continue
            scores[:, c] = cs.topk(min(k, cs.shape[1]), dim=1).values.sum(dim=1)
        pred[cell] = scores.argmax(1).float()
    return pred


def test_direct(rule, X_train_direct, y_train_direct, X_te, y_te, width, n_steps, k=5):
    """Direct N-step prediction (predict N steps ahead in one shot)."""
    correct = total = 0
    for i in range(X_te.shape[0]):
        for cell in range(1, width - 1):
            left = max(0, cell - n_steps)
            right = min(width, cell + n_steps + 1)
            feat_tr = X_train_direct[:, left:right]
            feat_te = X_te[i, left:right].unsqueeze(0)
            sims = F.normalize(feat_te, dim=1) @ F.normalize(feat_tr, dim=1).T
            scores = torch.zeros(1, 2, device=device)
            for c in range(2):
                m = y_train_direct[:, cell] == c
                cs = sims[:, m]
                if cs.shape[1] == 0:
                    continue
                scores[:, c] = cs.topk(min(k, cs.shape[1]), dim=1).values.sum(dim=1)
            if scores.argmax(1) == y_te[i, cell]:
                correct += 1
            total += 1
    return correct / total * 100


def main():
    width = 15
    n_train = 1000
    n_test = 100
    k = 5
    rule_num = 110

    rule = make_ca_rule(rule_num)

    # Train 1-step predictor
    X_train, y_train = train_1step(rule, width, n_train)

    print(f"Iterated vs Direct k-NN on Rule {rule_num} (width={width}):")
    print(f"{'Steps':>5s} | {'Direct':>7s} | {'Iterated':>8s} | Delta")
    print(f"------|---------|----------|------")

    for n_steps in [1, 2, 3, 5, 10]:
        # Generate test data
        X_te = torch.randint(0, 2, (n_test, width), device=device).float()
        true_final = X_te.clone()
        for _ in range(n_steps):
            true_final = torch.stack([evolve_row(true_final[i], rule) for i in range(n_test)])

        # Direct: train on N-step data
        X_tr_direct = torch.randint(0, 2, (n_train, width), device=device).float()
        y_tr_direct = X_tr_direct.clone()
        for _ in range(n_steps):
            y_tr_direct = torch.stack([evolve_row(y_tr_direct[i], rule) for i in range(n_train)])
        acc_direct = test_direct(rule, X_tr_direct, y_tr_direct, X_te, true_final, width, n_steps, k)

        # Iterated: apply 1-step predictor N times
        correct_iter = 0
        for i in range(n_test):
            current = X_te[i].clone()
            for _ in range(n_steps):
                current = predict_one_step(current, X_train, y_train, width, k)
            correct_iter += (current[1:-1] == true_final[i, 1:-1].to(device)).sum().item()
        acc_iter = correct_iter / (n_test * (width - 2)) * 100

        print(f"{n_steps:5d} | {acc_direct:6.1f}% | {acc_iter:7.1f}% | {acc_iter - acc_direct:+.1f}pp")


if __name__ == '__main__':
    main()
