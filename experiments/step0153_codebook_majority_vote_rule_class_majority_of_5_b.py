"""
Step 153 — Recovered from session JSONL (inline python -c experiment).
Original execution: foldcore k-NN / torch experiments, March 15 2026.
Source: B--M-avir-leo/0606b161, line 10346.
"""
import sys; sys.path.insert(0, 'substrates/topk-fold')
from self_improving_substrate import SelfImprovingSubstrate
import torch

# Step 153: Majority vote rule — class = majority of 5 bits
# Requires understanding that 3+ out of 5 bits determine the class
# Not capturable by any single pairwise product
d = 10
n_train = 2000

X = torch.randint(0, 2, (n_train, d)).float()
# Class = majority(x[0:5]) — 1 if sum >= 3, else 0
y = (X[:, :5].sum(1) >= 3).long()

X_te = torch.zeros(1024, d)
for i in range(1024):
    for b in range(d):
        X_te[i, b] = (i >> b) & 1
y_te = (X_te[:, :5].sum(1) >= 3).long()

print(f'Majority vote on 5 bits (d={d}):')
print(f'Class dist train: {[(y==c).sum().item() for c in range(2)]}')

# Base
sub_base = SelfImprovingSubstrate(d=d, max_features=0)
sub_base.train(X, y)
acc_base = (sub_base.predict(X_te).cpu() == y_te).float().mean().item() * 100

# Self-improving
sub = SelfImprovingSubstrate(d=d, max_features=5)
sub.train(X, y)
acc = (sub.predict(X_te).cpu() == y_te).float().mean().item() * 100

print(f'Base k-NN:      {acc_base:.1f}%')
print(f'Self-improving: {acc:.1f}% (+{acc-acc_base:.1f}pp)')
print(f'Discovered: {sub.features}')

# Also test: threshold function (x[0]+x[1]+x[2] > 1)
X2 = torch.randint(0, 2, (n_train, d)).float()
y2 = (X2[:, :3].sum(1) > 1).long()
X_te2 = X_te.clone()
y_te2 = (X_te2[:, :3].sum(1) > 1).long()

sub2 = SelfImprovingSubstrate(d=d, max_features=5)
sub2.train(X2, y2)
acc2 = (sub2.predict(X_te2).cpu() == y_te2).float().mean().item() * 100

sub2_base = SelfImprovingSubstrate(d=d, max_features=0)
sub2_base.train(X2, y2)
acc2_base = (sub2_base.predict(X_te2).cpu() == y_te2).float().mean().item() * 100

print(f'')
print(f'Threshold (sum(0:3) > 1):')
print(f'Base k-NN:      {acc2_base:.1f}%')
print(f'Self-improving: {acc2:.1f}% (+{acc2-acc2_base:.1f}pp)')
print(f'Discovered: {sub2.features}')
" 2>&1
