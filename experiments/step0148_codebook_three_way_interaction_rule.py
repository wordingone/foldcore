"""
Step 148 — Recovered from session JSONL (inline python -c experiment).
Original execution: foldcore k-NN / torch experiments, March 15 2026.
Source: B--M-avir-leo/0606b161, line 10250.
"""
import torch, torch.nn.functional as F
device = 'cuda'

# Step 148: Three-way interaction rule
# Class = (x[0] AND x[1] AND x[2]) vs everything else
# Requires a 3-way feature: x[0]*x[1]*x[2]

d = 10; n_train = 2000
X_tr = torch.randint(0, 2, (n_train, d), device=device).float()
y_tr = (X_tr[:, 0] * X_tr[:, 1] * X_tr[:, 2]).long()  # 3-way AND

X_te = torch.zeros(1024, d, device=device)
for i in range(1024):
    for b in range(d):
        X_te[i, b] = (i >> b) & 1
y_te = (X_te[:, 0] * X_te[:, 1] * X_te[:, 2]).long()

def knn_acc(V, labels, test_x, test_y, k=5):
    V_n = F.normalize(V, dim=1); te = F.normalize(test_x, dim=1)
    sims = te @ V_n.T
    n_cls = labels.max().item() + 1
    scores = torch.zeros(test_x.shape[0], n_cls, device=device)
    for c in range(n_cls):
        m = labels == c; cs = sims[:, m]
        if cs.shape[1] == 0: continue
        scores[:, c] = cs.topk(min(k, cs.shape[1]), dim=1).values.sum(dim=1)
    return (scores.argmax(1) == test_y).float().mean().item() * 100

def knn_margin_score(V, labels, k=5):
    sims = V @ V.T
    n_cls = labels.max().item() + 1
    scores = torch.zeros(V.shape[0], n_cls, device=device)
    for c in range(n_cls):
        m = labels == c; cs = sims[:, m]
        if cs.shape[1] == 0: continue
        scores[:, c] = cs.topk(min(k, cs.shape[1]), dim=1).values.sum(dim=1)
    margin = scores.sort(1, descending=True).values[:, 0] - scores.sort(1, descending=True).values[:, 1]
    return margin.mean().item()

acc_base = knn_acc(X_tr, y_tr, X_te, y_te)
print(f'3-way AND task (d={d}): base k-NN = {acc_base:.1f}%')
print(f'Class dist: {[(y_tr==c).sum().item() for c in range(2)]}')

# Step 1: pairwise products (can't fully capture 3-way)
raw = X_tr.clone()
V = X_tr.clone()
margin_base = knn_margin_score(F.normalize(V, dim=1), y_tr)

# Find best pairwise product
best_pair = None; best_m = margin_base
for i in range(d):
    for j in range(i+1, d):
        feat = (raw[:, i] * raw[:, j]).unsqueeze(1)
        aug = F.normalize(torch.cat([V, feat], 1), dim=1)
        m = knn_margin_score(aug, y_tr)
        if m > best_m:
            best_m = m; best_pair = (i, j)

if best_pair:
    feat_tr = (raw[:, best_pair[0]] * raw[:, best_pair[1]]).unsqueeze(1)
    feat_te = (X_te[:, best_pair[0]] * X_te[:, best_pair[1]]).unsqueeze(1)
    V = torch.cat([V, feat_tr], 1)
    te1 = torch.cat([X_te, feat_te], 1)
    acc1 = knn_acc(F.normalize(V, dim=1), y_tr, F.normalize(te1, dim=1), y_te)
    print(f'Step 1: pair {best_pair} -> {acc1:.1f}%')
    
    # Step 2: product of discovered feature with remaining dims
    # Now V has [raw, pair_product]. Try multiplying pair_product with each remaining dim
    aug_dim = V.shape[1]
    best_triple = None; best_m2 = knn_margin_score(F.normalize(V, dim=1), y_tr)
    
    for i in range(d):
        # Triple product: pair_product * x[i] = x[best_pair[0]] * x[best_pair[1]] * x[i]
        triple_feat = (raw[:, best_pair[0]] * raw[:, best_pair[1]] * raw[:, i]).unsqueeze(1)
        aug2 = F.normalize(torch.cat([V, triple_feat], 1), dim=1)
        m2 = knn_margin_score(aug2, y_tr)
        if m2 > best_m2:
            best_m2 = m2; best_triple = i
    
    if best_triple is not None:
        triple_tr = (raw[:, best_pair[0]] * raw[:, best_pair[1]] * raw[:, best_triple]).unsqueeze(1)
        triple_te = (X_te[:, best_pair[0]] * X_te[:, best_pair[1]] * X_te[:, best_triple]).unsqueeze(1)
        V2 = torch.cat([V, triple_tr], 1)
        te2 = torch.cat([te1, triple_te], 1)
        acc2 = knn_acc(F.normalize(V2, dim=1), y_tr, F.normalize(te2, dim=1), y_te)
        target = best_pair[0]==0 and best_pair[1]==1 and best_triple==2
        print(f'Step 2: triple ({best_pair[0]},{best_pair[1]},{best_triple}) -> {acc2:.1f}% {\"*** EXACT RULE ***\" if target else \"\"}')
    else:
        print('Step 2: no triple improvement')
else:
    print('No pairwise improvement found')

# Oracle: exact 3-way feature
feat_oracle_tr = (raw[:, 0] * raw[:, 1] * raw[:, 2]).unsqueeze(1)
feat_oracle_te = (X_te[:, 0] * X_te[:, 1] * X_te[:, 2]).unsqueeze(1)
acc_oracle = knn_acc(F.normalize(torch.cat([X_tr, feat_oracle_tr], 1), dim=1), y_tr,
                     F.normalize(torch.cat([X_te, feat_oracle_te], 1), dim=1), y_te)
print(f'Oracle (exact 3-way): {acc_oracle:.1f}%')
" 2>&1
