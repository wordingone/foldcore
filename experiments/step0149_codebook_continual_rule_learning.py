"""
Step 149 — Recovered from session JSONL (inline python -c experiment).
Original execution: foldcore k-NN / torch experiments, March 15 2026.
Source: B--M-avir-leo/0606b161, line 10256.
"""
import torch, torch.nn.functional as F
device = 'cuda'

# Step 149: Continual rule learning
# Task 0: class = x[0] XOR x[1]
# Task 1: class = x[2] AND x[3]
# Task 2: class = x[4] + x[5] > 0 (OR)
# Each task arrives sequentially. The substrate must discover each rule
# without forgetting previous discoveries.

d = 10

def make_task(rule_fn, n=1000):
    X = torch.randint(0, 2, (n, d), device=device).float()
    y = rule_fn(X).long()
    return X, y

rules = [
    ('XOR(0,1)', lambda X: (X[:, 0].long() ^ X[:, 1].long())),
    ('AND(2,3)', lambda X: (X[:, 2] * X[:, 3])),
    ('OR(4,5)',  lambda X: ((X[:, 4] + X[:, 5]) > 0).float()),
]

# Test set covers all 1024 binary patterns
X_te = torch.zeros(1024, d, device=device)
for i in range(1024):
    for b in range(d):
        X_te[i, b] = (i >> b) & 1

def knn_acc_on_task(V, labels, test_x, rule_fn, k=5):
    y_te = rule_fn(test_x).long()
    V_n = F.normalize(V, dim=1); te = F.normalize(test_x, dim=1)
    sims = te @ V_n.T
    n_cls = labels.max().item() + 1
    scores = torch.zeros(test_x.shape[0], n_cls, device=device)
    for c in range(n_cls):
        m = labels == c; cs = sims[:, m]
        if cs.shape[1] == 0: continue
        scores[:, c] = cs.topk(min(k, cs.shape[1]), dim=1).values.sum(dim=1)
    return (scores.argmax(1) == y_te).float().mean().item() * 100

def knn_margin_score(V, labels, k=5):
    V_n = F.normalize(V, dim=1)
    sims = V_n @ V_n.T
    n_cls = labels.max().item() + 1
    scores = torch.zeros(V.shape[0], n_cls, device=device)
    for c in range(n_cls):
        m = labels == c; cs = sims[:, m]
        if cs.shape[1] == 0: continue
        scores[:, c] = cs.topk(min(k, cs.shape[1]), dim=1).values.sum(dim=1)
    margin = scores.sort(1, descending=True).values[:, 0] - scores.sort(1, descending=True).values[:, 1]
    return margin.mean().item()

print('=== Step 149: Continual Rule Learning ===')

V = torch.empty(0, d, device=device)
labels = torch.empty(0, dtype=torch.long, device=device)
all_raw = torch.empty(0, d, device=device)
discovered_feats = []

for t, (rule_name, rule_fn) in enumerate(rules):
    # Get new task data
    X_t, y_t = make_task(rule_fn, n=500)
    
    # Map labels to task-specific range (task 0: 0,1; task 1: 2,3; task 2: 4,5)
    y_t_mapped = y_t + t * 2
    
    # Append to codebook  
    V = torch.cat([V, X_t])
    labels = torch.cat([labels, y_t_mapped])
    all_raw = torch.cat([all_raw, X_t])
    
    # Eval on ALL tasks seen so far (before discovery)
    accs_before = []
    for t2 in range(t+1):
        rn, rf = rules[t2]
        y_te_t2 = rf(X_te).long() + t2 * 2
        acc = knn_acc_on_task(V, labels, X_te, lambda x, rf=rf, t2=t2: rf(x) + t2*2)
        accs_before.append(acc)
    
    # Discover features for current task via margin
    margin_base = knn_margin_score(F.normalize(V, dim=1), labels)
    best_pair = None; best_m = margin_base
    for i in range(d):
        for j in range(i+1, d):
            if (i,j) in discovered_feats: continue
            feat = (all_raw[:, i] * all_raw[:, j]).unsqueeze(1)
            aug = F.normalize(torch.cat([V, feat], 1), dim=1)
            m = knn_margin_score(aug, labels)
            if m > best_m:
                best_m = m; best_pair = (i, j)
    
    if best_pair:
        discovered_feats.append(best_pair)
        feat = (all_raw[:, best_pair[0]] * all_raw[:, best_pair[1]]).unsqueeze(1)
        V = torch.cat([V, feat], 1)
    
    # Eval after discovery
    # Need to augment test set with all discovered features
    te_aug = X_te.clone()
    for (pi, pj) in discovered_feats:
        te_aug = torch.cat([te_aug, (X_te[:, pi] * X_te[:, pj]).unsqueeze(1)], 1)
    
    accs_after = []
    for t2 in range(t+1):
        rn, rf = rules[t2]
        y_te_t2 = (rf(X_te) + t2 * 2).long()
        V_n = F.normalize(V, dim=1); te_n = F.normalize(te_aug, dim=1)
        sims = te_n @ V_n.T
        n_cls = labels.max().item() + 1
        scores = torch.zeros(1024, n_cls, device=device)
        for c in range(n_cls):
            m2 = labels == c; cs = sims[:, m2]
            if cs.shape[1] == 0: continue
            scores[:, c] = cs.topk(min(5, cs.shape[1]), dim=1).values.sum(dim=1)
        preds = scores.argmax(1)
        acc = (preds == y_te_t2).float().mean().item() * 100
        accs_after.append(acc)
    
    feat_str = f'discovered {best_pair}' if best_pair else 'no feature'
    print(f'Task {t} ({rule_name}): {feat_str}')
    for t2 in range(t+1):
        print(f'  {rules[t2][0]}: before={accs_before[t2]:.1f}% after={accs_after[t2]:.1f}%')
print(f'\\nDiscovered features: {discovered_feats}')
" 2>&1
