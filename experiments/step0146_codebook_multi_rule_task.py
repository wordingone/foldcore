"""
Step 146 — Recovered from session JSONL (inline python -c experiment).
Original execution: foldcore k-NN / torch experiments, March 15 2026.
Source: B--M-avir-leo/0606b161, line 10219.
"""
import torch, torch.nn.functional as F
device = 'cuda'

# Step 146: Multi-rule task
# 4 classes, each defined by a different rule on 10 binary features:
# Class 0: x[0] XOR x[1] = 1
# Class 1: x[2] AND x[3] = 1  
# Class 2: x[4] OR x[5] = 0 (NOR)
# Class 3: everything else

d = 10
n_train = 2000

X_tr = torch.randint(0, 2, (n_train, d), device=device).float()

# Assign labels by rules (priority order)
y_tr = torch.full((n_train,), 3, device=device, dtype=torch.long)  # default class 3
mask_xor = ((X_tr[:, 0].long() ^ X_tr[:, 1].long()) == 1)
mask_and = ((X_tr[:, 2].long() & X_tr[:, 3].long()) == 1)
mask_nor = ((X_tr[:, 4].long() | X_tr[:, 5].long()) == 0)

# Assign in priority order (non-overlapping for clarity)
for i in range(n_train):
    if mask_xor[i] and not mask_and[i] and not mask_nor[i]:
        y_tr[i] = 0
    elif mask_and[i] and not mask_xor[i] and not mask_nor[i]:
        y_tr[i] = 1
    elif mask_nor[i] and not mask_xor[i] and not mask_and[i]:
        y_tr[i] = 2

# Generate test
n_test = 1024
X_te = torch.zeros(n_test, d, device=device)
for i in range(n_test):
    for b in range(d):
        X_te[i, b] = (i >> b) & 1 if b < 10 else 0
y_te = torch.full((n_test,), 3, device=device, dtype=torch.long)
for i in range(n_test):
    xor_i = int(X_te[i, 0]) ^ int(X_te[i, 1])
    and_i = int(X_te[i, 2]) & int(X_te[i, 3])
    nor_i = 1 - (int(X_te[i, 4]) | int(X_te[i, 5]))
    if xor_i and not and_i and not nor_i: y_te[i] = 0
    elif and_i and not xor_i and not nor_i: y_te[i] = 1
    elif nor_i and not xor_i and not and_i: y_te[i] = 2

print(f'Class distribution train: {[(y_tr==c).sum().item() for c in range(4)]}')
print(f'Class distribution test:  {[(y_te==c).sum().item() for c in range(4)]}')

def knn_with_margin(V, labels, queries, query_labels, k=5):
    V_n = F.normalize(V, dim=1); te = F.normalize(queries, dim=1)
    sims = te @ V_n.T
    n_cls = labels.max().item() + 1
    scores = torch.zeros(queries.shape[0], n_cls, device=device)
    for c in range(n_cls):
        m = labels == c; cs = sims[:, m]
        if cs.shape[1] == 0: continue
        scores[:, c] = cs.topk(min(k, cs.shape[1]), dim=1).values.sum(dim=1)
    preds = scores.argmax(1)
    margin = scores.sort(1, descending=True).values[:, 0] - scores.sort(1, descending=True).values[:, 1]
    acc = (preds == query_labels).float().mean().item() * 100
    return acc, margin.mean().item()

acc_base, margin_base = knn_with_margin(X_tr, y_tr, X_te, y_te)
print(f'\\nBase k-NN: {acc_base:.1f}%')

# Greedy margin-guided discovery
V = X_tr.clone(); raw = X_tr.clone(); raw_te = X_te.clone()
test_current = X_te.clone()
discovered = []

for step in range(5):
    best_pair = None; best_margin = margin_base
    
    # Try all pairs (d=10, only 45 pairs)
    for i in range(d):
        for j in range(i+1, d):
            if (i,j) in discovered: continue
            feat_tr = (raw[:, i] * raw[:, j]).unsqueeze(1)
            aug = F.normalize(torch.cat([V, feat_tr], 1), dim=1)
            _, m = knn_with_margin(aug, y_tr, aug, y_tr)
            if m > best_margin:
                best_margin = m
                best_pair = (i, j)
    
    if best_pair is None: break
    discovered.append(best_pair)
    feat_tr = (raw[:, best_pair[0]] * raw[:, best_pair[1]]).unsqueeze(1)
    feat_te = (raw_te[:, best_pair[0]] * raw_te[:, best_pair[1]]).unsqueeze(1)
    V = torch.cat([V, feat_tr], 1)
    test_current = torch.cat([test_current, feat_te], 1)
    
    acc, _ = knn_with_margin(F.normalize(V, dim=1), y_tr, F.normalize(test_current, dim=1), y_te)
    
    # Check which rule this pair relates to
    rule = 'unknown'
    if best_pair in [(0,1)]: rule = 'XOR(0,1)'
    elif best_pair in [(2,3)]: rule = 'AND(2,3)'
    elif best_pair in [(4,5)]: rule = 'NOR(4,5) via prod'
    print(f'Step {step+1}: pair {best_pair} ({rule}) test_acc={acc:.1f}%')

print(f'\\nFinal: {acc:.1f}% (was {acc_base:.1f}%, +{acc-acc_base:.1f}pp)')
print(f'Discovered: {discovered}')
" 2>&1
