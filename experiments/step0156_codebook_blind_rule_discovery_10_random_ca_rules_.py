"""
Step 156 — Recovered from session JSONL (inline python -c experiment).
Original execution: foldcore k-NN / torch experiments, March 15 2026.
Source: B--M-avir-leo/0606b161, line 10377.
"""
import sys; sys.path.insert(0, 'substrates/topk-fold')
from self_improving_substrate import SelfImprovingSubstrate
import torch

# Step 156: Blind rule discovery — 10 random CA rules, substrate doesn't know which
# For each rule, train on CA evolution data, test on all 8 neighborhoods

d = 3  # 3-cell neighborhood

results = []
for rule_num in [30, 54, 60, 90, 110, 150, 182, 210, 225, 250]:
    # Decode rule number to lookup table
    rule_table = {}
    for i in range(8):
        neighborhood = ((i>>2)&1, (i>>1)&1, i&1)
        rule_table[neighborhood] = (rule_num >> i) & 1
    
    # Generate CA evolution
    width = 30
    row = torch.zeros(width, dtype=torch.int)
    row[width//2] = 1
    
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
    
    # Test: all 8 neighborhoods
    X_te = torch.tensor([[i>>2&1, i>>1&1, i&1] for i in range(8)], dtype=torch.float)
    y_te = torch.tensor([rule_table[tuple(X_te[j].int().tolist())] for j in range(8)], dtype=torch.long)
    
    # Skip trivial rules (all 0 or all 1)
    if y_te.sum() == 0 or y_te.sum() == 8:
        continue
    
    sub = SelfImprovingSubstrate(d=d, max_features=3)
    sub.train(X_train, y_train)
    preds = sub.predict(X_te)
    acc = (preds.cpu() == y_te).float().mean().item() * 100
    
    results.append((rule_num, acc, sub.features, y_train.float().mean().item()))
    
print(f'Rule  | Acc   | Class bal | Discovered features')
print(f'------|-------|----------|--------------------')
for rule, acc, feats, bal in results:
    print(f'{rule:5d} | {acc:5.1f}% | {bal:.2f}     | {feats}')

perfect = sum(1 for _, acc, _, _ in results if acc == 100)
print(f'\\nPerfect: {perfect}/{len(results)} rules')
" 2>&1
