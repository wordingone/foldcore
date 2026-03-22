"""
Step 195 — Recovered from session JSONL (inline python -c experiment).
Original execution: foldcore k-NN / torch experiments, March 15 2026.
Source: B--M-avir-leo/0606b161, line 10994.
"""
import torch, torch.nn.functional as F
device = 'cuda'

# Step 195: Can LOO determine iteration count?
# For a CA prediction task, the 'right' number of iterations matches
# the number of steps being predicted. Can the system figure this out?

# Setup: Rule 110, width=15, train on 1-step data
width = 15; n_train = 1000; k = 5
rule = {((i>>2)&1,(i>>1)&1,i&1): (110>>i)&1 for i in range(8)}

def evolve1(row):
    new = torch.zeros_like(row)
    for i in range(1, len(row)-1):
        nb = (row[i-1].item(), row[i].item(), row[i+1].item())
        new[i] = rule[nb]
    return new

X_tr = torch.randint(0, 2, (n_train, width), device=device).float()
y_tr = torch.stack([evolve1(X_tr[i]) for i in range(n_train)]).to(device)

def predict_step(row):
    pred = torch.zeros(width, device=device)
    for cell in range(1, width-1):
        feat_tr = X_tr[:, cell-1:cell+2]
        feat_te = row[cell-1:cell+2].unsqueeze(0)
        sims = F.normalize(feat_te,dim=1) @ F.normalize(feat_tr,dim=1).T
        scores = torch.zeros(1, 2, device=device)
        for c in range(2):
            m = y_tr[:, cell] == c; cs = sims[:, m]
            if cs.shape[1] == 0: continue
            scores[:, c] = cs.topk(min(k, cs.shape[1]), dim=1).values.sum(dim=1)
        pred[cell] = scores.argmax(1).float()
    return pred

# Test: given a (start_row, target_row) pair, can the system determine
# HOW MANY iterations are needed to get from start to target?

# Generate test pairs with known step counts
print('Step 195: Can the system determine iteration count?')
print(f'{\"True steps\":>10s} | Predicted | Correct')
print(f'{\"----------\":>10s}-|-----------|--------')

for true_steps in [1, 3, 5, 8]:
    correct = 0; n_test = 20
    for _ in range(n_test):
        start = torch.randint(0, 2, (width,), device=device).float()
        target = start.clone()
        for _ in range(true_steps):
            target = evolve1(target)
        
        # Try different iteration counts, pick the one that matches target best
        best_n = 0; best_dist = float('inf')
        for n in range(1, 12):
            current = start.clone()
            for _ in range(n):
                current = predict_step(current)
            dist = (current[1:-1] - target[1:-1].to(device)).abs().sum().item()
            if dist < best_dist:
                best_dist = dist; best_n = n
            if dist == 0: break  # perfect match
        
        if best_n == true_steps: correct += 1
    
    print(f'{true_steps:10d} | varies    | {correct}/{n_test} ({correct/n_test*100:.0f}%)')
" 2>&1
