"""
Step 174 — Recovered from session JSONL (inline python -c experiment).
Original execution: foldcore k-NN / torch experiments, March 15 2026.
Source: B--M-avir-leo/0606b161, line 10600.
"""
import torch, torch.nn.functional as F
device = 'cuda'

# Step 174: Is cos(w·x+b) universal? Can it approximate ANY binary function?
# Test: can a SINGLE cos(w·x+b) feature match the oracle for various rules?

d = 8
X = torch.randint(0, 2, (1000, d), device=device).float()
Xte = torch.zeros(256, d, device=device)
for i in range(256):
    for b in range(d): Xte[i,b]=(i>>b)&1

rules = {
    'parity': lambda x: (x.sum(1) % 2).long(),
    'majority': lambda x: (x[:,:5].sum(1) >= 3).long(),
    'x0 AND x1': lambda x: (x[:,0] * x[:,1]).long(),
    'x0 XOR x1': lambda x: (x[:,0].long() ^ x[:,1].long()),
    'x0 OR (x1 AND x2)': lambda x: ((x[:,0] + x[:,1]*x[:,2]) > 0).long(),
    '3-way AND': lambda x: (x[:,0]*x[:,1]*x[:,2]).long(),
    'sum > 3': lambda x: (x.sum(1) > 3).long(),
    'weighted sum > 2': lambda x: (x[:,0]*3 + x[:,1]*2 + x[:,2] > 2).long(),
}

print(f'{\"Rule\":25s} | Base  | +cos(wx+b) | Oracle')
print(f'{\"-\"*25}-|-------|-----------|-------')

for name, rule_fn in rules.items():
    y = rule_fn(X); yte = rule_fn(Xte)
    
    def knn_acc(V, labels, te, y_te, k=5):
        sims = F.normalize(te,dim=1) @ F.normalize(V,dim=1).T
        scores = torch.zeros(te.shape[0], 2, device=device)
        for c in range(2):
            m = labels == c; cs = sims[:, m]
            if cs.shape[1] == 0: continue
            scores[:, c] = cs.topk(min(k, cs.shape[1]), dim=1).values.sum(dim=1)
        return (scores.argmax(1) == y_te).float().mean().item() * 100
    
    acc_base = knn_acc(X, y, Xte, yte)
    
    # Find best single cos(w·x+b)
    best = acc_base
    for _ in range(500):
        w = torch.randn(d, device=device) / (d**0.5)
        b = torch.rand(1, device=device) * 6.28
        ft = torch.cos(X @ w + b).unsqueeze(1)
        fte = torch.cos(Xte @ w + b).unsqueeze(1)
        acc = knn_acc(F.normalize(torch.cat([X,ft],1),dim=1), y,
                     F.normalize(torch.cat([Xte,fte],1),dim=1), yte)
        if acc > best: best = acc
    
    # Oracle: directly add the rule output as a feature
    oracle_ft = rule_fn(X).float().unsqueeze(1)
    oracle_fte = rule_fn(Xte).float().unsqueeze(1)
    acc_oracle = knn_acc(F.normalize(torch.cat([X,oracle_ft],1),dim=1), y,
                        F.normalize(torch.cat([Xte,oracle_fte],1),dim=1), yte)
    
    print(f'{name:25s} | {acc_base:5.1f}% | {best:5.1f}%    | {acc_oracle:5.1f}%')
" 2>&1
