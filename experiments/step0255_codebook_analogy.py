"""
Step 255: Analogy reasoning — A:B :: C:?

The substrate discovers the offset relationship between A and B,
and transfers it to predict D from C. 40% → 83% (+43pp).

Usage:
    python experiments/run_step255_analogy.py
"""

import torch
import torch.nn.functional as F

device = 'cuda' if torch.cuda.is_available() else 'cpu'
vocab = 15
templates = {
    'cos': lambda x, w, b: torch.cos(x @ w + b),
    'abs': lambda x, w, b: torch.abs(x @ w + b),
}


def discover_and_test(X_tr, y_tr, X_te, y_te, n_cls, max_layers=5, n_cand=100):
    def loo(V, labels):
        V_n = F.normalize(V, dim=1); sims = V_n @ V_n.T; sims.fill_diagonal_(-1e9)
        scores = torch.zeros(V.shape[0], n_cls, device=device)
        for c in range(n_cls):
            m = labels == c; cs = sims[:, m]
            if cs.shape[1] == 0: continue
            scores[:, c] = cs.topk(min(5, cs.shape[1]), dim=1).values.sum(dim=1)
        return (scores.argmax(1) == labels).float().mean().item()

    def knn(V, labels, te, yte):
        sims = F.normalize(te, dim=1) @ F.normalize(V, dim=1).T
        scores = torch.zeros(te.shape[0], n_cls, device=device)
        for c in range(n_cls):
            m = labels == c; cs = sims[:, m]
            if cs.shape[1] == 0: continue
            scores[:, c] = cs.topk(min(5, cs.shape[1]), dim=1).values.sum(dim=1)
        return (scores.argmax(1) == yte).float().mean().item() * 100

    base = knn(X_tr, y_tr, X_te, y_te)
    V = X_tr.clone(); layers = []
    for _ in range(max_layers):
        cd = V.shape[1]; bl = loo(V, y_tr); best = None
        for tn, tf in templates.items():
            for _ in range(n_cand // len(templates)):
                w = torch.randn(cd, device=device) / (cd ** 0.5)
                b = torch.rand(1, device=device) * n_cls
                try:
                    feat = tf(V, w, b).unsqueeze(1)
                    aug = F.normalize(torch.cat([V, feat], 1), dim=1)
                    l = loo(aug, y_tr)
                    if l > bl + 0.002: bl = l; best = (tn, w.clone(), b.clone())
                except: pass
        if best is None: break
        tn, w, b = best; layers.append((tn, w, b))
        V = torch.cat([V, templates[tn](V, w, b).unsqueeze(1)], 1)

    Vte = X_te.clone(); Vtr = X_tr.clone()
    for tn, w, b in layers:
        Vtr = torch.cat([Vtr, templates[tn](Vtr, w, b).unsqueeze(1)], 1)
        Vte = torch.cat([Vte, templates[tn](Vte, w, b).unsqueeze(1)], 1)
    sub = knn(F.normalize(Vtr, dim=1), y_tr, F.normalize(Vte, dim=1), y_te)
    return base, sub, len(layers)


def main():
    n_train = 2000; d = 3

    X = torch.zeros(n_train, d, device=device)
    y = torch.zeros(n_train, device=device, dtype=torch.long)
    for i in range(n_train):
        offset = torch.randint(-5, 6, (1,)).item()
        a = torch.randint(0, vocab, (1,)).item()
        c = torch.randint(0, vocab, (1,)).item()
        b = max(0, min(vocab - 1, a + offset))
        d_val = max(0, min(vocab - 1, c + offset))
        X[i] = torch.tensor([a, b, c], device=device, dtype=torch.float)
        y[i] = d_val

    Xte = []; yte = []
    for offset in range(-5, 6):
        for a in range(vocab):
            for c in range(vocab):
                b = max(0, min(vocab - 1, a + offset))
                d_val = max(0, min(vocab - 1, c + offset))
                Xte.append([float(a), float(b), float(c)])
                yte.append(d_val)
    idx = torch.randperm(len(Xte))[:500]
    Xte = torch.tensor(Xte, device=device)[idx]
    yte = torch.tensor(yte, device=device, dtype=torch.long)[idx]

    base, sub, n_layers = discover_and_test(X, y, Xte, yte, vocab)
    print(f"Analogy A:B :: C:?")
    print(f"  Base k-NN: {base:.1f}%")
    print(f"  Substrate: {sub:.1f}% ({n_layers} layers, +{sub - base:.1f}pp)")


if __name__ == '__main__':
    main()
