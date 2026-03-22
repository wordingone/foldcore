"""
Step 173 — Recovered from session JSONL (inline python -c experiment).
Original execution: foldcore k-NN / torch experiments, March 15 2026.
Source: B--M-avir-leo/0606b161, line 10587.
"""
import torch, torch.nn.functional as F
device = 'cuda'

# Step 173: Stage 6 — adaptive metric selection via LOO
d = 8; n_train = 1000; k = 5
X = torch.randint(0, 2, (n_train, d), device=device).float()
y = (X.sum(1) % 2).long()
Xte = torch.zeros(256, d, device=device)
for i in range(256):
    for b in range(d): Xte[i,b]=(i>>b)&1
yte = (Xte.sum(1) % 2).long()

def loo_with_metric(V, labels, metric, k=5):
    if metric == 'cosine':
        sims = F.normalize(V,dim=1) @ F.normalize(V,dim=1).T
    elif metric == 'l2':
        sims = -torch.cdist(V, V)
    elif metric == 'l1':
        sims = -torch.cdist(V, V, p=1)
    sims.fill_diagonal_(-1e9)
    n_cls = labels.max().item() + 1
    scores = torch.zeros(V.shape[0], n_cls, device=device)
    for c in range(n_cls):
        m = labels == c; cs = sims[:, m]
        if cs.shape[1] == 0: continue
        scores[:, c] = cs.topk(min(k, cs.shape[1]), dim=1).values.sum(dim=1)
    return (scores.argmax(1) == labels).float().mean().item()

def knn_acc(V, labels, te, y_te, metric, k=5):
    if metric == 'cosine':
        sims = F.normalize(te,dim=1) @ F.normalize(V,dim=1).T
    elif metric == 'l2':
        sims = -torch.cdist(te, V)
    elif metric == 'l1':
        sims = -torch.cdist(te, V, p=1)
    n_cls = labels.max().item() + 1
    scores = torch.zeros(te.shape[0], n_cls, device=device)
    for c in range(n_cls):
        m = labels == c; cs = sims[:, m]
        if cs.shape[1] == 0: continue
        scores[:, c] = cs.topk(min(k, cs.shape[1]), dim=1).values.sum(dim=1)
    return (scores.argmax(1) == y_te).float().mean().item() * 100

# LOO-guided metric selection
print('Stage 6: Adaptive metric selection on parity')
best_metric = None; best_loo = 0
for metric in ['cosine', 'l2', 'l1']:
    loo = loo_with_metric(X, y, metric)
    acc = knn_acc(X, y, Xte, yte, metric)
    selected = ''
    if loo > best_loo:
        best_loo = loo; best_metric = metric
        selected = ' <-- SELECTED'
    print(f'  {metric:8s}: LOO={loo:.4f} test={acc:.1f}%{selected}')

print(f'\\nAdaptive selected: {best_metric}')
print(f'Test acc with selected metric: {knn_acc(X, y, Xte, yte, best_metric):.1f}%')
print(f'Test acc with cosine (default): {knn_acc(X, y, Xte, yte, \"cosine\"):.1f}%')
print(f'Delta: {knn_acc(X, y, Xte, yte, best_metric) - knn_acc(X, y, Xte, yte, \"cosine\"):+.1f}pp')
" 2>&1
