"""
Step 179 — Recovered from session JSONL (inline python -c experiment).
Original execution: foldcore k-NN / torch experiments, March 15 2026.
Source: B--M-avir-leo/0606b161, line 10689.
"""
import torch, torch.nn.functional as F
device = 'cuda'

# Step 179: Can the substrate predict CA evolution at the ROW level?
# Input: full row of cells. Output: next row.
# This is a SEQUENCE of local rules applied simultaneously.

width = 15  # small enough for exhaustive testing
rule_110 = {((i>>2)&1,(i>>1)&1,i&1): (110>>i)&1 for i in range(8)}

def evolve_row(row):
    new = torch.zeros_like(row)
    for i in range(1, len(row)-1):
        nb = (row[i-1].item(), row[i].item(), row[i+1].item())
        new[i] = rule_110[nb]
    return new

# Generate training data: (current_row -> next_row)
n_train = 500
X_tr = torch.randint(0, 2, (n_train, width), device=device).float()
y_tr = torch.stack([evolve_row(X_tr[i]) for i in range(n_train)]).to(device)

# Test: different random rows
n_test = 200
X_te = torch.randint(0, 2, (n_test, width), device=device).float()
y_te = torch.stack([evolve_row(X_te[i]) for i in range(n_test)]).to(device)

# Predict each cell independently using k-NN with local neighborhood
# For cell i, use (x[i-1], x[i], x[i+1]) as features
correct_total = 0
total_cells = 0

for cell in range(1, width-1):
    # Features: 3-cell neighborhood
    feat_tr = X_tr[:, cell-1:cell+2]  # (n_train, 3)
    label_tr = y_tr[:, cell]  # (n_train,)
    feat_te = X_te[:, cell-1:cell+2]
    label_te = y_te[:, cell]
    
    # k-NN
    sims = F.normalize(feat_te,dim=1) @ F.normalize(feat_tr,dim=1).T
    scores = torch.zeros(n_test, 2, device=device)
    for c in range(2):
        m = label_tr == c; cs = sims[:, m]
        if cs.shape[1] == 0: continue
        scores[:, c] = cs.topk(min(5, cs.shape[1]), dim=1).values.sum(dim=1)
    preds = scores.argmax(1)
    correct = (preds == label_te).sum().item()
    correct_total += correct
    total_cells += n_test

acc_local = correct_total / total_cells * 100

# Now try: use FULL ROW as features for each cell prediction
# This captures global context that local neighborhoods miss
correct_global = 0
for cell in range(1, width-1):
    label_tr = y_tr[:, cell]
    label_te = y_te[:, cell]
    
    sims = F.normalize(X_te,dim=1) @ F.normalize(X_tr,dim=1).T
    scores = torch.zeros(n_test, 2, device=device)
    for c in range(2):
        m = label_tr == c; cs = sims[:, m]
        if cs.shape[1] == 0: continue
        scores[:, c] = cs.topk(min(5, cs.shape[1]), dim=1).values.sum(dim=1)
    preds = scores.argmax(1)
    correct_global += (preds == label_te).sum().item()

acc_global = correct_global / total_cells * 100

# Per-row accuracy (all cells correct)
preds_row = torch.zeros(n_test, width, device=device, dtype=torch.long)
for cell in range(1, width-1):
    feat_tr = X_tr[:, cell-1:cell+2]
    label_tr = y_tr[:, cell]
    feat_te = X_te[:, cell-1:cell+2]
    sims = F.normalize(feat_te,dim=1) @ F.normalize(feat_tr,dim=1).T
    scores = torch.zeros(n_test, 2, device=device)
    for c in range(2):
        m = label_tr == c; cs = sims[:, m]
        if cs.shape[1] == 0: continue
        scores[:, c] = cs.topk(min(5, cs.shape[1]), dim=1).values.sum(dim=1)
    preds_row[:, cell] = scores.argmax(1)

row_correct = sum(1 for i in range(n_test) if (preds_row[i,1:-1] == y_te[i,1:-1]).all())
acc_row = row_correct / n_test * 100

print(f'CA Row Prediction (Rule 110, width={width}):')
print(f'  Per-cell (local 3-cell):  {acc_local:.1f}%')
print(f'  Per-cell (global row):    {acc_global:.1f}%')
print(f'  Full-row correct:         {acc_row:.1f}%')
print(f'  Random baseline (cell):   50.0%')
print(f'  Random baseline (row):    {100*(0.5**(width-2)):.4f}%')
" 2>&1
