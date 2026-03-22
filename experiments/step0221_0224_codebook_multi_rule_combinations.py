"""
Steps 221-224 — Recovered from CC session 0606b161 (inline Bash execution).
foldcore k-NN / torch experiments, March 15 2026.
Source: line 11253.
"""
import torch, torch.nn.functional as F
device = 'cuda'

# Steps 221-224: Full arithmetic suite from raw integers
templates = {'cos': lambda x,w,b: torch.cos(x@w+b), 'abs': lambda x,w,b: torch.abs(x@w+b), 'mod2': lambda x,w,b: ((x@w.abs()).round()%2).float()}

def substrate_raw(X_tr, y_tr, X_te, y_te, n_cls, max_layers=10, n_cand=100):
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
    
    base = knn(X_tr, y_tr, X_te, y_te)
    V = X_tr.clone(); layers = []
    for _ in range(max_layers):
        cd = V.shape[1]; bl = loo(V, y_tr); best = None
        for tn, tf in templates.items():
            for _ in range(n_cand//len(templates)):
                w = torch.randn(cd, device=device)/(cd**0.5); b = torch.rand(1,device=device)*n_cls
                try:
                    feat = tf(V, w, b).unsqueeze(1); aug = F.normalize(torch.cat([V,feat],1),dim=1)
                    l = loo(aug, y_tr)
                    if l > bl+0.001: bl=l; best=(tn,w.clone(),b.clone())
                except: pass
        if best is None: break
        tn,w,b = best; layers.append((tn,w,b))
        V = torch.cat([V, templates[tn](V,w,b).unsqueeze(1)], 1)
    
    V_te = X_te.clone(); V_tr = X_tr.clone()
    for tn,w,b in layers:
        V_tr = torch.cat([V_tr, templates[tn](V_tr,w,b).unsqueeze(1)], 1)
        V_te = torch.cat([V_te, templates[tn](V_te,w,b).unsqueeze(1)], 1)
    sub = knn(F.normalize(V_tr,dim=1), y_tr, F.normalize(V_te,dim=1), y_te)
    return base, sub, len(layers)

n = 10
print(f'{\"Task\":25s} | Base  | Sub   | Delta | #L')
print(f'{\"-\"*25}-|-------|-------|-------|---')

# 221: Multiplication (raw)
X = torch.randint(0, n, (800, 2), device=device).float()
y = (X[:,0] * X[:,1]).long()
Xte = torch.zeros(100, 2, device=device); yte = torch.zeros(100, device=device, dtype=torch.long)
for i in range(100): Xte[i] = torch.tensor([i//10, i%10], dtype=torch.float, device=device); yte[i] = (i//10)*(i%10)
b,s,l = substrate_raw(X, y, Xte, yte, n*n)
print(f'{\"Multiplication\":25s} | {b:5.1f}% | {s:5.1f}% | {s-b:+.1f}pp | {l}')

# 222: Subtraction (raw, result mod 10)
y = ((X[:,0] - X[:,1]) % n).long()
yte = ((Xte[:,0] - Xte[:,1]) % n).long()
b,s,l = substrate_raw(X, y, Xte, yte, n)
print(f'{\"Subtraction mod 10\":25s} | {b:5.1f}% | {s:5.1f}% | {s-b:+.1f}pp | {l}')

# 223: Max(a, b)
y = torch.max(X[:,0], X[:,1]).long()
yte = torch.max(Xte[:,0], Xte[:,1]).long()
b,s,l = substrate_raw(X, y, Xte, yte, n)
print(f'{\"Max(a,b)\":25s} | {b:5.1f}% | {s:5.1f}% | {s-b:+.1f}pp | {l}')

# 224: Absolute difference |a - b|
y = (X[:,0] - X[:,1]).abs().long()
yte = (Xte[:,0] - Xte[:,1]).abs().long()
b,s,l = substrate_raw(X, y, Xte, yte, n)
print(f'{\"Abs diff |a-b|\":25s} | {b:5.1f}% | {s:5.1f}% | {s-b:+.1f}pp | {l}')
" 2>&1
