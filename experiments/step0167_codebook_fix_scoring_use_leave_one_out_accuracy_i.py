"""
Step 167 — Recovered from session JSONL (inline python -c experiment).
Original execution: foldcore k-NN / torch experiments, March 15 2026.
Source: B--M-avir-leo/0606b161, line 10523.
"""
import torch, torch.nn.functional as F
device = 'cuda'

# Step 167: Fix scoring — use LEAVE-ONE-OUT accuracy instead of self-eval
# LOO: for each training sample, predict using ALL OTHER samples
# This gives a realistic accuracy estimate without overfitting

d = 8
X = torch.randint(0, 2, (1000, d), device=device).float()
y = (X.sum(1) % 2).long()
X_te = torch.zeros(256, d, device=device)
for i in range(256):
    for b in range(d):
        X_te[i, b] = (i >> b) & 1
y_te = (X_te.sum(1) % 2).long()

def loo_accuracy(V, labels, k=5):
    '''Leave-one-out accuracy: exclude self from neighbors.'''
    V_n = F.normalize(V, dim=1)
    sims = V_n @ V_n.T
    # Zero out self-similarity
    sims.fill_diagonal_(0)
    n_cls = labels.max().item() + 1
    scores = torch.zeros(V.shape[0], n_cls, device=device)
    for c in range(n_cls):
        m = labels == c
        cs = sims[:, m]
        if cs.shape[1] == 0: continue
        scores[:, c] = cs.topk(min(k, cs.shape[1]), dim=1).values.sum(dim=1)
    preds = scores.argmax(1)
    return (preds == labels).float().mean().item()

def knn_acc(V, labels, te, y_te, k=5):
    sims = F.normalize(te,dim=1) @ F.normalize(V,dim=1).T
    n_cls = labels.max().item() + 1
    scores = torch.zeros(te.shape[0], n_cls, device=device)
    for c in range(n_cls):
        m = labels == c; cs = sims[:, m]
        if cs.shape[1] == 0: continue
        scores[:, c] = cs.topk(min(k, cs.shape[1]), dim=1).values.sum(dim=1)
    return (scores.argmax(1) == y_te).float().mean().item() * 100

# LOO on raw features
V = X.clone()
loo_base = loo_accuracy(F.normalize(V, dim=1), y)
acc_base = knn_acc(V, y, X_te, y_te)
print(f'Base: LOO={loo_base:.4f} test={acc_base:.1f}%')

# Discover with LOO scoring
best_w = None; best_b = None; best_loo = loo_base
for _ in range(200):
    w = torch.randn(d, device=device) / (d**0.5)
    b = torch.rand(1, device=device) * 6.28
    feat = torch.cos(X @ w + b).unsqueeze(1)
    aug = F.normalize(torch.cat([V, feat], 1), dim=1)
    loo = loo_accuracy(aug, y)
    if loo > best_loo:
        best_loo = loo; best_w = w.clone(); best_b = b.clone()

if best_w is not None:
    feat_tr = torch.cos(X @ best_w + best_b).unsqueeze(1)
    feat_te = torch.cos(X_te @ best_w + best_b).unsqueeze(1)
    V_aug = F.normalize(torch.cat([V, feat_tr], 1), dim=1)
    te_aug = F.normalize(torch.cat([X_te, feat_te], 1), dim=1)
    acc = knn_acc(V_aug, y, te_aug, y_te)
    print(f'After discovery: LOO={best_loo:.4f} test={acc:.1f}% (+{acc-acc_base:.1f}pp)')
else:
    print('No improvement found')
" 2>&1
