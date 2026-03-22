"""
Step 171 — Recovered from session JSONL (inline python -c experiment).
Original execution: foldcore k-NN / torch experiments, March 15 2026.
Source: B--M-avir-leo/0606b161, line 10571.
"""
import torch, torch.nn.functional as F
device = 'cuda'

# Step 171: Stage 5 — Adaptive topology via feature-gated neighborhoods
# Instead of flat k-NN over ALL vectors, use discovered features to
# partition the codebook into neighborhoods. Query only the relevant neighborhood.

d = 20; n_train = 2000; n_test = 500; k = 5

# XOR task
X_tr = torch.randint(0, 2, (n_train, d), device=device).float()
y_tr = (X_tr[:, 0].long() ^ X_tr[:, 1].long())
X_te = torch.randn(n_test, d, device=device).clamp(0,1).round()
y_te = (X_te[:, 0].long() ^ X_te[:, 1].long())

def knn_acc(V, labels, te, y_te, k=5):
    sims = F.normalize(te,dim=1) @ F.normalize(V,dim=1).T
    scores = torch.zeros(te.shape[0], 2, device=device)
    for c in range(2):
        m = labels == c; cs = sims[:, m]
        if cs.shape[1] == 0: continue
        scores[:, c] = cs.topk(min(k, cs.shape[1]), dim=1).values.sum(dim=1)
    return (scores.argmax(1) == y_te).float().mean().item() * 100

# Base k-NN (flat topology)
acc_flat = knn_acc(X_tr, y_tr, X_te, y_te)

# Adaptive topology: partition codebook by a discovered feature
# Use the best random cosine feature to SPLIT the codebook into 2 halves
# Then classify within each half separately

# Find best splitting feature
best_w = None; best_b = None; best_acc = acc_flat
for _ in range(200):
    w = torch.randn(d, device=device) / (d**0.5)
    b = torch.rand(1, device=device) * 6.28
    split_val = torch.cos(X_tr @ w + b)
    
    # Split codebook at median
    median = split_val.median()
    group_tr = (split_val > median)
    
    # For each test sample, determine group and classify within it
    split_te = torch.cos(X_te @ w + b)
    group_te = (split_te > median)
    
    # Classify within group
    preds = torch.zeros(n_test, device=device, dtype=torch.long)
    for g in [True, False]:
        te_mask = group_te == g
        tr_mask = group_tr == g
        if te_mask.sum() == 0 or tr_mask.sum() < k: continue
        V_g = X_tr[tr_mask]; labels_g = y_tr[tr_mask]
        te_g = X_te[te_mask]; yte_g = y_te[te_mask]
        sims = F.normalize(te_g,dim=1) @ F.normalize(V_g,dim=1).T
        scores = torch.zeros(te_g.shape[0], 2, device=device)
        for c in range(2):
            m = labels_g == c; cs = sims[:, m]
            if cs.shape[1] == 0: continue
            scores[:, c] = cs.topk(min(k, cs.shape[1]), dim=1).values.sum(dim=1)
        preds[te_mask] = scores.argmax(1)
    
    acc = (preds == y_te).float().mean().item() * 100
    if acc > best_acc:
        best_acc = acc; best_w = w.clone(); best_b = b.clone()

print(f'XOR d={d}:')
print(f'  Flat k-NN (all vectors): {acc_flat:.1f}%')
print(f'  Adaptive topology (split): {best_acc:.1f}% (delta={best_acc-acc_flat:+.1f}pp)')
print(f'  Feature-augmented k-NN:  ', end='')

# For comparison: just adding the feature (Step 163 approach)
best_feat_acc = acc_flat
for _ in range(200):
    w = torch.randn(d, device=device) / (d**0.5)
    b = torch.rand(1, device=device) * 6.28
    ft = torch.cos(X_tr @ w + b).unsqueeze(1)
    fte = torch.cos(X_te @ w + b).unsqueeze(1)
    acc = knn_acc(F.normalize(torch.cat([X_tr,ft],1),dim=1), y_tr,
                 F.normalize(torch.cat([X_te,fte],1),dim=1), y_te)
    if acc > best_feat_acc: best_feat_acc = acc
print(f'{best_feat_acc:.1f}%')

print(f'')
print(f'Topology adaptation vs feature augmentation: {best_acc-best_feat_acc:+.1f}pp')
" 2>&1
