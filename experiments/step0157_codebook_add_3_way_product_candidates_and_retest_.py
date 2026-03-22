"""
Step 157 — Recovered from session JSONL (inline python -c experiment).
Original execution: foldcore k-NN / torch experiments, March 15 2026.
Source: B--M-avir-leo/0606b161, line 10397.
"""
import sys; sys.path.insert(0, 'substrates/topk-fold')
from self_improving_substrate import SelfImprovingSubstrate
import torch, torch.nn.functional as F

# Step 157: Add 3-way product candidates and retest failed rules
d = 3

# Monkey-patch to add 3-way products
original_discover = SelfImprovingSubstrate._discover_feature

def enhanced_discover(self):
    V_current = self._augment(self.raw)
    margin_base = self._margin_score(V_current, self.labels)
    best_pair = None; best_margin = margin_base

    # Original candidates
    candidates = []
    for i in range(self.d):
        for j in range(i+1, self.d):
            if ('prod', i, j) not in self.features:
                candidates.append(('prod', i, j))
    for name in ['sum', 'cos_sum_pi', 'sum_mod2']:
        if (name,) not in self.features:
            candidates.append((name,))
    
    # NEW: 3-way products
    for i in range(self.d):
        for j in range(i+1, self.d):
            for k in range(j+1, self.d):
                if ('triple', i, j, k) not in self.features:
                    candidates.append(('triple', i, j, k))
    
    # NEW: threshold features
    for thresh in [1, 2]:
        name = f'sum_gt{thresh}'
        if (name,) not in self.features:
            candidates.append((name,))

    for cand in candidates:
        feat = enhanced_compute(self, self.raw, cand)
        if feat is None: continue
        aug = F.normalize(torch.cat([V_current, feat], dim=1), dim=1)
        m = self._margin_score(aug, self.labels)
        if m > best_margin:
            best_margin = m; best_pair = cand
    return best_pair, best_margin - margin_base

original_compute = SelfImprovingSubstrate._compute_feature

def enhanced_compute(self, X, spec):
    result = original_compute(X, spec)
    if result is not None: return result
    if spec[0] == 'triple':
        return (X[:, spec[1]] * X[:, spec[2]] * X[:, spec[3]]).unsqueeze(1)
    if spec[0] == 'sum_gt1':
        return (X.sum(1) > 1).float().unsqueeze(1)
    if spec[0] == 'sum_gt2':
        return (X.sum(1) > 2).float().unsqueeze(1)
    return None

SelfImprovingSubstrate._discover_feature = enhanced_discover
SelfImprovingSubstrate._compute_feature = enhanced_compute

# Test on the 4 failed rules
failed_rules = [90, 210, 225, 250]
for rule_num in failed_rules:
    rule_table = {}
    for i in range(8):
        rule_table[((i>>2)&1, (i>>1)&1, i&1)] = (rule_num >> i) & 1
    
    width = 30; row = torch.zeros(width, dtype=torch.int); row[width//2] = 1
    X_train, y_train = [], []
    for step in range(100):
        new_row = torch.zeros(width, dtype=torch.int)
        for i in range(1, width-1):
            nb = (row[i-1].item(), row[i].item(), row[i+1].item())
            new_row[i] = rule_table[nb]
            X_train.append([float(row[i-1]), float(row[i]), float(row[i+1])])
            y_train.append(new_row[i].item())
        row = new_row
    
    X_train = torch.tensor(X_train, dtype=torch.float)
    y_train = torch.tensor(y_train, dtype=torch.long)
    
    X_te = torch.tensor([[i>>2&1, i>>1&1, i&1] for i in range(8)], dtype=torch.float)
    y_te = torch.tensor([rule_table[tuple(X_te[j].int().tolist())] for j in range(8)], dtype=torch.long)
    
    if y_te.sum() == 0 or y_te.sum() == 8: continue
    
    sub = SelfImprovingSubstrate(d=d, max_features=5)
    sub.train(X_train, y_train)
    acc = (sub.predict(X_te).cpu() == y_te).float().mean().item() * 100
    print(f'Rule {rule_num}: {acc:.1f}% feats={sub.features}')
" 2>&1
