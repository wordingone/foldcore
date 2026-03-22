"""
Step 155 — Recovered from session JSONL (inline python -c experiment).
Original execution: foldcore k-NN / torch experiments, March 15 2026.
Source: B--M-avir-leo/0606b161, line 10355.
"""
import sys; sys.path.insert(0, 'substrates/topk-fold')
from self_improving_substrate import SelfImprovingSubstrate
import torch

# Step 155: Elementary Cellular Automaton Rule Discovery
# Given a 3-cell neighborhood, predict the next state of the center cell
# Rule 110 (Turing-complete): look up table for 8 possible neighborhoods

d = 3  # 3-cell neighborhood (left, center, right)

# Rule 110 lookup table
rule_110 = {
    (1,1,1): 0, (1,1,0): 1, (1,0,1): 1, (1,0,0): 0,
    (0,1,1): 1, (0,1,0): 1, (0,0,1): 1, (0,0,0): 0,
}

# Generate training data from Rule 110
n_train = 500
X = torch.randint(0, 2, (n_train, d)).float()
y = torch.tensor([rule_110[tuple(X[i].int().tolist())] for i in range(n_train)], dtype=torch.long)

# All 8 possible inputs (exhaustive test)
X_te = torch.tensor([[i>>2 & 1, i>>1 & 1, i & 1] for i in range(8)], dtype=torch.float)
y_te = torch.tensor([rule_110[tuple(X_te[i].int().tolist())] for i in range(8)], dtype=torch.long)

print(f'Rule 110 truth table:')
for i in range(8):
    print(f'  {X_te[i].int().tolist()} -> {y_te[i].item()}')

# Base k-NN (d=3, should be trivial with enough data)
sub_base = SelfImprovingSubstrate(d=d, max_features=0)
sub_base.train(X, y)
acc_base = (sub_base.predict(X_te).cpu() == y_te).float().mean().item() * 100

# Self-improving
sub = SelfImprovingSubstrate(d=d, max_features=3)
sub.train(X, y)
acc = (sub.predict(X_te).cpu() == y_te).float().mean().item() * 100

print(f'')
print(f'Base k-NN:      {acc_base:.1f}%')
print(f'Self-improving: {acc:.1f}%')
print(f'Discovered: {sub.features}')

# Now test: can it discover the CA rule from a LONGER sequence?
# Given a full row of cells, predict the next row
# This requires the substrate to work on local neighborhoods

width = 20
n_steps = 50

# Generate CA evolution
row = torch.zeros(width, dtype=torch.int)
row[width//2] = 1  # single seed

rows = [row.clone()]
for _ in range(n_steps):
    new_row = torch.zeros(width, dtype=torch.int)
    for i in range(1, width-1):
        neighborhood = (row[i-1].item(), row[i].item(), row[i+1].item())
        new_row[i] = rule_110[neighborhood]
    row = new_row
    rows.append(row.clone())

# Training data: (neighborhood) -> next_state from actual CA evolution
X_ca = []
y_ca = []
for t in range(len(rows)-1):
    for i in range(1, width-1):
        X_ca.append([rows[t][i-1].float(), rows[t][i].float(), rows[t][i+1].float()])
        y_ca.append(rows[t+1][i].item())

X_ca = torch.tensor(X_ca, dtype=torch.float)
y_ca = torch.tensor(y_ca, dtype=torch.long)
print(f'')
print(f'CA evolution data: {X_ca.shape[0]} samples from {n_steps} steps x {width-2} cells')
print(f'Class dist: {[(y_ca==c).sum().item() for c in range(2)]}')

sub_ca = SelfImprovingSubstrate(d=3, max_features=3)
sub_ca.train(X_ca, y_ca)
acc_ca = (sub_ca.predict(X_te).cpu() == y_te).float().mean().item() * 100
print(f'CA-trained accuracy on all 8 neighborhoods: {acc_ca:.1f}%')
print(f'Discovered: {sub_ca.features}')
" 2>&1
