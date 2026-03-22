"""
Step 160 — Recovered from session JSONL (inline python -c experiment).
Original execution: foldcore k-NN / torch experiments, March 15 2026.
Source: B--M-avir-leo/0606b161, line 10423.
"""
import torch, torch.nn.functional as F
device = 'cuda'

d = 3
rule_90 = {((i>>2)&1,(i>>1)&1,i&1): (90>>i)&1 for i in range(8)}

width=30; row=torch.zeros(width,dtype=torch.int); row[width//2]=1
X_tr, y_tr = [], []
for _ in range(100):
    new_row = torch.zeros(width,dtype=torch.int)
    for i in range(1,width-1):
        nb=(row[i-1].item(),row[i].item(),row[i+1].item())
        new_row[i]=rule_90[nb]
        X_tr.append([float(row[i-1]),float(row[i]),float(row[i+1])])
        y_tr.append(new_row[i].item())
    row=new_row
X_tr=torch.tensor(X_tr,dtype=torch.float,device=device)
y_tr=torch.tensor(y_tr,dtype=torch.long,device=device)
X_te = torch.tensor([[i>>2&1, i>>1&1, i&1] for i in range(8)], dtype=torch.float, device=device)
y_te = torch.tensor([rule_90[tuple(X_te[j].int().tolist())] for j in range(8)], dtype=torch.long, device=device)

def knn_margin(V, labels, k=5):
    V_n = F.normalize(V, dim=1); sims = V_n @ V_n.T
    scores = torch.zeros(V.shape[0], labels.max().item()+1, device=device)
    for c in range(labels.max().item()+1):
        m = labels == c; cs = sims[:, m]
        if cs.shape[1] == 0: continue
        scores[:, c] = cs.topk(min(k, cs.shape[1]), dim=1).values.sum(dim=1)
    return (scores.sort(1,descending=True).values[:,0] - scores.sort(1,descending=True).values[:,1]).mean().item()

def knn_acc(V, labels, te, y_te, k=5):
    sims = F.normalize(te,dim=1) @ F.normalize(V,dim=1).T
    scores = torch.zeros(te.shape[0], labels.max().item()+1, device=device)
    for c in range(labels.max().item()+1):
        m = labels == c; cs = sims[:, m]
        if cs.shape[1] == 0: continue
        scores[:, c] = cs.topk(min(k, cs.shape[1]), dim=1).values.sum(dim=1)
    return (scores.argmax(1) == y_te).float().mean().item() * 100

# Step 160: GREEDY PARTIAL-SUM CONSTRUCTION
# Start empty. Each step: add the dimension whose inclusion most improves margin.
print('Step 160: Greedy partial-sum construction for Rule 90')

V = X_tr.clone()
selected_dims = []

for step in range(d):
    best_dim = None; best_m = -999
    for dim in range(d):
        if dim in selected_dims: continue
        trial_dims = selected_dims + [dim]
        # Partial sum of selected dims, then cos(sum*pi) as the feature
        partial_sum = X_tr[:, trial_dims].sum(1)
        feat = torch.cos(partial_sum * 3.14159).unsqueeze(1)
        aug = F.normalize(torch.cat([V, feat], 1), dim=1)
        m = knn_margin(aug, y_tr)
        if m > best_m:
            best_m = m; best_dim = dim
    
    selected_dims.append(best_dim)
    partial_sum = X_tr[:, selected_dims].sum(1)
    feat_tr = torch.cos(partial_sum * 3.14159).unsqueeze(1)
    feat_te = torch.cos(X_te[:, selected_dims].sum(1) * 3.14159).unsqueeze(1)
    aug_tr = F.normalize(torch.cat([V, feat_tr], 1), dim=1)
    aug_te = F.normalize(torch.cat([X_te, feat_te], 1), dim=1)
    acc = knn_acc(aug_tr, y_tr, aug_te, y_te)
    print(f'  Step {step+1}: add dim {best_dim}, dims={selected_dims}, acc={acc:.1f}%')
    if acc >= 100: break

correct = set(selected_dims) == {0, 2}
print(f'\\nDiscovered dims: {selected_dims}')
print(f'Correct (should be [0,2]): {correct}')
" 2>&1
