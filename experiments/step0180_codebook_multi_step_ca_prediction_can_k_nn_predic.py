"""
Step 180 — Recovered from session JSONL (inline python -c experiment).
Original execution: foldcore k-NN / torch experiments, March 15 2026.
Source: B--M-avir-leo/0606b161, line 10695.
"""
import torch, torch.nn.functional as F
device = 'cuda'

# Step 180: Multi-step CA prediction — can k-NN predict N steps ahead?
width = 15
rule_110 = {((i>>2)&1,(i>>1)&1,i&1): (110>>i)&1 for i in range(8)}

def evolve(row, steps):
    for _ in range(steps):
        new = torch.zeros_like(row)
        for i in range(1, len(row)-1):
            nb = (row[i-1].item(), row[i].item(), row[i+1].item())
            new[i] = rule_110[nb]
        row = new
    return row

n_train = 1000

print(f'Multi-step CA prediction (Rule 110, width={width}):')
print(f'{\"Steps\":>5s} | Per-cell | Full-row')
print(f'------|---------|--------')

for n_steps in [1, 2, 3, 5, 10]:
    X_tr = torch.randint(0, 2, (n_train, width), device=device).float()
    y_tr = torch.stack([evolve(X_tr[i], n_steps) for i in range(n_train)]).to(device)
    
    n_test = 200
    X_te = torch.randint(0, 2, (n_test, width), device=device).float()
    y_te = torch.stack([evolve(X_te[i], n_steps) for i in range(n_test)]).to(device)
    
    # Local k-NN per cell (neighborhood size adapts to steps)
    # For n_steps ahead, need neighborhood of 2*n_steps+1 cells
    hood = min(2*n_steps+1, width)
    
    correct = total = 0
    preds_row = torch.zeros(n_test, width, device=device, dtype=torch.long)
    
    for cell in range(1, width-1):
        # Expanded neighborhood
        left = max(0, cell - n_steps)
        right = min(width, cell + n_steps + 1)
        feat_tr = X_tr[:, left:right]
        feat_te = X_te[:, left:right]
        label_tr = y_tr[:, cell]
        label_te = y_te[:, cell]
        
        sims = F.normalize(feat_te,dim=1) @ F.normalize(feat_tr,dim=1).T
        scores = torch.zeros(n_test, 2, device=device)
        for c in range(2):
            m = label_tr == c; cs = sims[:, m]
            if cs.shape[1] == 0: continue
            scores[:, c] = cs.topk(min(5, cs.shape[1]), dim=1).values.sum(dim=1)
        preds = scores.argmax(1)
        preds_row[:, cell] = preds
        correct += (preds == label_te).sum().item()
        total += n_test
    
    acc_cell = correct / total * 100
    row_ok = sum(1 for i in range(n_test) if (preds_row[i,1:-1] == y_te[i,1:-1]).all())
    acc_row = row_ok / n_test * 100
    
    print(f'{n_steps:5d} | {acc_cell:6.1f}% | {acc_row:5.1f}%')
" 2>&1
