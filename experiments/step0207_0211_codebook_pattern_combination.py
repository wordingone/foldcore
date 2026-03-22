"""
Steps 207-211 — Recovered from CC session 0606b161 (inline Bash execution).
foldcore k-NN / torch experiments, March 15 2026.
Source: line 11146.
"""
import torch, torch.nn.functional as F
device = 'cuda'

# Steps 207-209: Arithmetic tasks — addition, multiplication, modular arithmetic
# These are STRUCTURED discrete tasks where the substrate should shine

templates = {'cos': lambda x,w,b: torch.cos(x@w+b), 'abs': lambda x,w,b: torch.abs(x@w+b), 'mod2': lambda x,w,b: ((x@w.abs()).round()%2).float()}

def substrate_test(X_tr, y_tr, X_te, y_te, n_cls, name, max_layers=10, n_cand=100):
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
        cd = V.shape[1]; best_loo = loo(V, y_tr); best = None
        for tn, tf in templates.items():
            for _ in range(n_cand//len(templates)):
                w = torch.randn(cd, device=device)/(cd**0.5); b = torch.rand(1,device=device)*6.28
                try:
                    feat = tf(V, w, b).unsqueeze(1)
                    aug = F.normalize(torch.cat([V,feat],1),dim=1)
                    l = loo(aug, y_tr)
                    if l > best_loo+0.001: best_loo=l; best=(tn,w.clone(),b.clone())
                except: pass
        if best is None: break
        tn,w,b = best; layers.append((tn,w,b))
        V = torch.cat([V, templates[tn](V,w,b).unsqueeze(1)], 1)
    
    V_te = X_te.clone(); V_tr2 = X_tr.clone()
    for tn,w,b in layers:
        V_tr2 = torch.cat([V_tr2, templates[tn](V_tr2,w,b).unsqueeze(1)], 1)
        V_te = torch.cat([V_te, templates[tn](V_te,w,b).unsqueeze(1)], 1)
    sub = knn(F.normalize(V_tr2,dim=1), y_tr, F.normalize(V_te,dim=1), y_te)
    print(f'{name:30s} | {base:5.1f}% | {sub:5.1f}% | {sub-base:+.1f}pp | {len(layers)}')
    return base, sub

print(f'{\"Task\":30s} | Base  | Sub   | Delta | #L')
print(f'{\"-\"*30}-|-------|-------|-------|---')

# 207: Addition mod 5 — (a + b) mod 5
n = 500
X = torch.randint(0, 5, (n, 2), device=device).float()
y = ((X[:,0] + X[:,1]) % 5).long()
Xte = torch.randint(0, 5, (100, 2), device=device).float()
yte = ((Xte[:,0] + Xte[:,1]) % 5).long()
substrate_test(X, y, Xte, yte, 5, 'Addition mod 5 (d=2)')

# 208: Multiplication mod 7
X = torch.randint(0, 7, (700, 2), device=device).float()
y = ((X[:,0] * X[:,1]) % 7).long()
Xte = torch.randint(0, 7, (100, 2), device=device).float()
yte = ((Xte[:,0] * Xte[:,1]) % 7).long()
substrate_test(X, y, Xte, yte, 7, 'Multiplication mod 7 (d=2)')

# 209: Binary addition carry — predict carry bit of 4-bit addition
d = 8  # two 4-bit numbers
X = torch.randint(0, 2, (2000, d), device=device).float()
a = X[:, :4] @ torch.tensor([8,4,2,1], device=device, dtype=torch.float)
b = X[:, 4:] @ torch.tensor([8,4,2,1], device=device, dtype=torch.float)
y = ((a + b) >= 16).long()  # carry bit
Xte = torch.randint(0, 2, (500, d), device=device).float()
a_te = Xte[:, :4] @ torch.tensor([8,4,2,1], device=device, dtype=torch.float)
b_te = Xte[:, 4:] @ torch.tensor([8,4,2,1], device=device, dtype=torch.float)
yte = ((a_te + b_te) >= 16).long()
substrate_test(X, y, Xte, yte, 2, 'Binary addition carry (d=8)')

# 210: Comparison — a > b?
X = torch.randint(0, 10, (1000, 2), device=device).float()
y = (X[:,0] > X[:,1]).long()
Xte = torch.randint(0, 10, (200, 2), device=device).float()
yte = (Xte[:,0] > Xte[:,1]).long()
substrate_test(X, y, Xte, yte, 2, 'Comparison a>b (d=2)')

# 211: Max of 3
X = torch.randint(0, 10, (1000, 3), device=device).float()
y = X.argmax(1)
Xte = torch.randint(0, 10, (200, 3), device=device).float()
yte = Xte.argmax(1)
substrate_test(X, y, Xte, yte, 3, 'Argmax of 3 values (d=3)')
" 2>&1
