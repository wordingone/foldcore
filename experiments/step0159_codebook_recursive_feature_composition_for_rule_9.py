"""
Step 159 — Recovered from session JSONL (inline python -c experiment).
Original execution: foldcore k-NN / torch experiments, March 15 2026.
Source: B--M-avir-leo/0606b161, line 10420.
"""
import torch, torch.nn.functional as F
device = 'cuda'

# Step 159: Recursive feature composition for Rule 90 (XOR of left+right)
# Rule 90: next = left XOR right. Failed at 75% with fixed features.
# The XOR function requires: knowing that (left + right) % 2 is the answer.
# But on d=3 (left, center, right), the 'sum' feature sums ALL 3 bits.
# We need sum of SPECIFIC bits: left + right (excluding center).

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

# The key insight: Rule 90 = XOR(left, right) = (left + right) % 2
# This is parity of a SUBSET of features (dims 0 and 2, excluding dim 1).
# To discover this, the system needs PARTIAL SUMS — not just full sum.

# Recursive composition candidates:
# Layer 0: raw features (x0, x1, x2)
# Layer 1: pairwise sums (x0+x2, x0+x1, x1+x2) and their parities
# Layer 2: parity of layer 1 features

V = X_tr.clone()
m_base = knn_margin(F.normalize(V,dim=1), y_tr)

# Partial sum candidates
candidates = {
    'x0+x2': lambda x: (x[:,0]+x[:,2]).unsqueeze(1),
    'x0+x1': lambda x: (x[:,0]+x[:,1]).unsqueeze(1),
    'x1+x2': lambda x: (x[:,1]+x[:,2]).unsqueeze(1),
    '(x0+x2)%2': lambda x: ((x[:,0]+x[:,2])%2).unsqueeze(1),
    '(x0+x1)%2': lambda x: ((x[:,0]+x[:,1])%2).unsqueeze(1),
    '(x1+x2)%2': lambda x: ((x[:,1]+x[:,2])%2).unsqueeze(1),
    'cos((x0+x2)*pi)': lambda x: torch.cos((x[:,0]+x[:,2])*3.14159).unsqueeze(1),
}

print('Rule 90 (XOR of left+right):')
print(f'Base k-NN: {knn_acc(F.normalize(V,dim=1), y_tr, F.normalize(X_te,dim=1), y_te):.1f}%')

for name, fn in candidates.items():
    feat_tr = fn(X_tr); feat_te = fn(X_te)
    aug_tr = F.normalize(torch.cat([V, feat_tr], 1), dim=1)
    aug_te = F.normalize(torch.cat([X_te, feat_te], 1), dim=1)
    m = knn_margin(aug_tr, y_tr)
    acc = knn_acc(aug_tr, y_tr, aug_te, y_te)
    print(f'  +{name:20s}: margin_delta={m-m_base:+.4f} acc={acc:.1f}%')
" 2>&1
