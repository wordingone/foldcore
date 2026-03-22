"""
Step 217 — Recovered from session JSONL (inline python -c experiment).
Original execution: foldcore k-NN / torch experiments, March 15 2026.
Source: B--M-avir-leo/0606b161, line 11207.
"""
import torch, torch.nn.functional as F
device = 'cuda'

# Step 217: Logical tasks — implication, biconditional, NAND, NOR
# These are ALL binary functions on 2 inputs — the substrate should handle them

d = 10; n_train = 1000; vocab = 2
templates = {'cos': lambda x,w,b: torch.cos(x@w+b), 'abs': lambda x,w,b: torch.abs(x@w+b), 'mod2': lambda x,w,b: ((x@w.abs()).round()%2).float()}

def full_test(X, y, Xte, yte, n_cls=2):
    def loo(V, labels):
        V_n = F.normalize(V,dim=1); sims = V_n @ V_n.T; sims.fill_diagonal_(-1e9)
        scores = torch.zeros(V.shape[0], n_cls, device=device)
        for c in range(n_cls):
            m = labels == c; cs = sims[:, m]
            if cs.shape[1] == 0: continue
            scores[:, c] = cs.topk(min(5, cs.shape[1]), dim=1).values.sum(dim=1)
        return (scores.argmax(1) == labels).float().mean().item()
    def knn(V, labels, te, yte):
        sims = F.normalize(te,dim=1) @ F.normalize(V,dim=1).T
        scores = torch.zeros(te.shape[0], n_cls, device=device)
        for c in range(n_cls):
            m = labels == c; cs = sims[:, m]
            if cs.shape[1] == 0: continue
            scores[:, c] = cs.topk(min(5, cs.shape[1]), dim=1).values.sum(dim=1)
        return (scores.argmax(1) == yte).float().mean().item() * 100
    
    base = knn(X, y, Xte, yte)
    V = X.clone(); n = 0
    for _ in range(5):
        cd = V.shape[1]; bl = loo(V, y); best = None
        for tn, tf in templates.items():
            for _ in range(50):
                w = torch.randn(cd, device=device)/(cd**0.5); b = torch.rand(1,device=device)*6.28
                try:
                    feat = tf(V, w, b).unsqueeze(1); aug = F.normalize(torch.cat([V,feat],1),dim=1)
                    l = loo(aug, y)
                    if l > bl+0.005: bl=l; best=(tn,w.clone(),b.clone())
                except: pass
        if best is None: break
        tn,w,b = best; n += 1
        V = torch.cat([V, templates[tn](V,w,b).unsqueeze(1)], 1)
    
    V_te = Xte.clone(); V_tr = X.clone()
    for tn,w,b in [(best[0],best[1],best[2])] if best else []:  # simplified
        pass
    sub = knn(F.normalize(V,dim=1), y, F.normalize(Xte,dim=1), yte)
    return base, sub, n

X = torch.randint(0, 2, (n_train, d), device=device).float()
Xte = torch.zeros(1024, d, device=device)
for i in range(1024):
    for b in range(d): Xte[i,b]=(i>>b)&1

logic_ops = {
    'AND': lambda x: (x[:,0] * x[:,1]).long(),
    'OR':  lambda x: ((x[:,0] + x[:,1]) > 0).long(),
    'XOR': lambda x: (x[:,0].long() ^ x[:,1].long()),
    'NAND': lambda x: (1 - x[:,0] * x[:,1]).long(),
    'NOR': lambda x: (1 - ((x[:,0] + x[:,1]) > 0).float()).long(),
    'IMPL': lambda x: (1 - x[:,0] * (1-x[:,1])).long(),  # a -> b = NOT(a AND NOT b)
    'BICOND': lambda x: (1 - (x[:,0].long() ^ x[:,1].long())).long(),  # XNOR
    'Parity3': lambda x: (x[:,:3].sum(1) % 2).long(),
    'Majority3': lambda x: (x[:,:3].sum(1) >= 2).long(),
}

print(f'{\"Logic\":12s} | Base  | Sub   | Delta')
print(f'{\"-\"*12}-|-------|-------|------')
for name, fn in logic_ops.items():
    y = fn(X); yte = fn(Xte)
    b, s, n = full_test(X, y, Xte, yte)
    print(f'{name:12s} | {b:5.1f}% | {s:5.1f}% | {s-b:+.1f}pp')
" 2>&1
