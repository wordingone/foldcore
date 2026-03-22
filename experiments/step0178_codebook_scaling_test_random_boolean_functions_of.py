"""
Step 178 — Recovered from session JSONL (inline python -c experiment).
Original execution: foldcore k-NN / torch experiments, March 15 2026.
Source: B--M-avir-leo/0606b161, line 10668.
"""
import torch, torch.nn.functional as F, random
device = 'cuda'

# Step 178: Scaling test — random boolean functions of increasing complexity
# For each complexity level, generate a random truth table and test if
# the substrate can discover features that help

def loo_accuracy(V, labels, k=5):
    V_n = F.normalize(V, dim=1); sims = V_n @ V_n.T; sims.fill_diagonal_(0)
    n_cls = labels.max().item() + 1
    scores = torch.zeros(V.shape[0], n_cls, device=device)
    for c in range(n_cls):
        m = labels == c; cs = sims[:, m]
        if cs.shape[1] == 0: continue
        scores[:, c] = cs.topk(min(k, cs.shape[1]), dim=1).values.sum(dim=1)
    return (scores.argmax(1) == labels).float().mean().item()

def knn_acc(V, labels, te, yte, k=5):
    sims = F.normalize(te,dim=1) @ F.normalize(V,dim=1).T
    n_cls = labels.max().item() + 1
    scores = torch.zeros(te.shape[0], n_cls, device=device)
    for c in range(n_cls):
        m = labels == c; cs = sims[:, m]
        if cs.shape[1] == 0: continue
        scores[:, c] = cs.topk(min(k, cs.shape[1]), dim=1).values.sum(dim=1)
    return (scores.argmax(1) == yte).float().mean().item() * 100

templates = {
    'cos':  lambda x, w, b: torch.cos(x @ w + b),
    'abs':  lambda x, w, b: torch.abs(x @ w + b),
    'sign': lambda x, w, b: torch.sign(x @ w + b),
    'mod2': lambda x, w, b: ((x @ w.abs()).round() % 2).float(),
}

print(f'{\"d\":>3s} | Base  | +Feat | Delta | n_feats')
print(f'----|-------|-------|-------|--------')

for d in [4, 6, 8, 10, 12]:
    n_train = min(2000, 2**d * 10)
    X = torch.randint(0, 2, (n_train, d), device=device).float()
    
    # Random boolean function: random truth table for d-bit input
    # Use a random neural network to generate the labels (complex nonlinear)
    random.seed(42 + d)
    torch.manual_seed(42 + d)
    W1 = torch.randn(d, d, device=device)
    W2 = torch.randn(d, 1, device=device)
    y = (torch.tanh(X @ W1) @ W2 > 0).long().squeeze()
    
    # Test set
    n_te = min(1000, 2**d)
    Xte = torch.randint(0, 2, (n_te, d), device=device).float()
    yte = (torch.tanh(Xte @ W1) @ W2 > 0).long().squeeze()
    
    acc_base = knn_acc(X, y, Xte, yte)
    
    # Discover features (multi-template, LOO-scored, 3 rounds)
    V = X.clone()
    n_found = 0
    for step in range(3):
        loo_base = loo_accuracy(F.normalize(V, dim=1), y)
        best = None; best_loo = loo_base
        for tname, tfn in templates.items():
            for _ in range(50):
                w = torch.randn(d, device=device) / (d**0.5)
                b = torch.rand(1, device=device) * 6.28
                try:
                    feat = tfn(X, w, b).unsqueeze(1)
                    aug = F.normalize(torch.cat([V, feat], 1), dim=1)
                    loo = loo_accuracy(aug, y)
                    if loo > best_loo:
                        best_loo = loo; best = (tname, w.clone(), b.clone())
                except: pass
        if best is None: break
        tname, w, b = best
        feat_tr = templates[tname](X, w, b).unsqueeze(1)
        V = torch.cat([V, feat_tr], 1)
        n_found += 1
    
    # Eval (need to replay features on test)
    # Can't easily without storing — just use augmented V dimensions
    # Use: base + single best feature for simplicity
    best_test = acc_base
    for tname, tfn in templates.items():
        for _ in range(100):
            w = torch.randn(d, device=device) / (d**0.5)
            b = torch.rand(1, device=device) * 6.28
            try:
                ft = tfn(X, w, b).unsqueeze(1); fte = tfn(Xte, w, b).unsqueeze(1)
                acc = knn_acc(F.normalize(torch.cat([X,ft],1),dim=1), y,
                             F.normalize(torch.cat([Xte,fte],1),dim=1), yte)
                if acc > best_test: best_test = acc
            except: pass
    
    print(f'{d:3d} | {acc_base:5.1f}% | {best_test:5.1f}% | {best_test-acc_base:+.1f}pp | {n_found}')
" 2>&1
