"""
Step 161 — Recovered from session JSONL (inline python -c experiment).
Original execution: foldcore k-NN / torch experiments, March 15 2026.
Source: B--M-avir-leo/0606b161, line 10442.
"""
import torch, torch.nn.functional as F
device = 'cuda'

# Step 161: Random nonlinear feature generation + margin selection
# Instead of fixed templates, generate RANDOM features and let margin pick

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

V = X_tr.clone()
m_base = knn_margin(F.normalize(V,dim=1), y_tr)
acc_base = knn_acc(F.normalize(V,dim=1), y_tr, F.normalize(X_te,dim=1), y_te)
print(f'Rule 90: base acc={acc_base:.1f}%')

# Generate 500 random nonlinear features: cos(w @ x + b) with random w, b
n_candidates = 500
best_w = None; best_b = None; best_m = m_base

for _ in range(n_candidates):
    w = torch.randn(d, device=device)
    b = torch.rand(1, device=device) * 6.28  # random phase
    feat = torch.cos(X_tr @ w + b).unsqueeze(1)
    aug = F.normalize(torch.cat([V, feat], 1), dim=1)
    m = knn_margin(aug, y_tr)
    if m > best_m:
        best_m = m; best_w = w.clone(); best_b = b.clone()

if best_w is not None:
    feat_tr = torch.cos(X_tr @ best_w + best_b).unsqueeze(1)
    feat_te = torch.cos(X_te @ best_w + best_b).unsqueeze(1)
    aug_tr = F.normalize(torch.cat([V, feat_tr], 1), dim=1)
    aug_te = F.normalize(torch.cat([X_te, feat_te], 1), dim=1)
    acc = knn_acc(aug_tr, y_tr, aug_te, y_te)
    print(f'Best random feature: acc={acc:.1f}% margin_delta=+{best_m-m_base:.4f}')
    print(f'  w = {best_w.cpu().tolist()}')
    print(f'  b = {best_b.item():.4f}')
    
    # Interpret: which dimensions does w weight?
    w_abs = best_w.abs()
    w_normalized = w_abs / w_abs.sum()
    print(f'  w importance: {[f\"d{i}:{w_normalized[i]:.3f}\" for i in range(d)]}')
    print(f'  Dominant dims: {[i for i in range(d) if w_normalized[i] > 0.3]}')
else:
    print('No improving feature found')
" 2>&1
