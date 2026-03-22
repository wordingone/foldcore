"""
Step 197 — Recovered from session JSONL (inline python -c experiment).
Original execution: foldcore k-NN / torch experiments, March 15 2026.
Source: B--M-avir-leo/0606b161, line 11020.
"""
import torch, torch.nn.functional as F
device = 'cuda'

# Step 197: Greedy composition — layer features on features
# Instead of random composition (Step 177), do LAYERED discovery:
# Layer 1: best f(w@x+b) feature
# Layer 2: best g(w@[x, f1]+b) feature — takes augmented input
# Each layer expands the input space for the next

d = 8
X = torch.randint(0, 2, (1000, d), device=device).float()
y = (X.sum(1) % 2).long()
Xte = torch.zeros(256, d, device=device)
for i in range(256):
    for b in range(d): Xte[i,b]=(i>>b)&1
yte = (Xte.sum(1) % 2).long()

templates = {
    'cos': lambda x,w,b: torch.cos(x@w+b),
    'abs': lambda x,w,b: torch.abs(x@w+b),
    'mod2': lambda x,w,b: ((x@w.abs()).round()%2).float(),
}

def loo(V, labels, k=5):
    V_n = F.normalize(V,dim=1); sims = V_n @ V_n.T; sims.fill_diagonal_(-1e9)
    scores = torch.zeros(V.shape[0], 2, device=device)
    for c in range(2):
        m = labels == c; cs = sims[:, m]
        if cs.shape[1] == 0: continue
        scores[:, c] = cs.topk(min(k, cs.shape[1]), dim=1).values.sum(dim=1)
    return (scores.argmax(1) == labels).float().mean().item()

def knn_acc(V, labels, te, yte, k=5):
    sims = F.normalize(te,dim=1) @ F.normalize(V,dim=1).T
    scores = torch.zeros(te.shape[0], 2, device=device)
    for c in range(2):
        m = labels == c; cs = sims[:, m]
        if cs.shape[1] == 0: continue
        scores[:, c] = cs.topk(min(k, cs.shape[1]), dim=1).values.sum(dim=1)
    return (scores.argmax(1) == yte).float().mean().item() * 100

# Layered discovery: each layer operates on the AUGMENTED input
V_tr = X.clone(); V_te = Xte.clone()
print('Step 197: Layered feature composition')
print(f'Base: test={knn_acc(V_tr, y, V_te, yte):.1f}%')

for layer in range(8):
    current_d = V_tr.shape[1]
    best_loo = loo(F.normalize(V_tr,dim=1), y)
    best = None
    
    for tname, tfn in templates.items():
        for _ in range(100):
            w = torch.randn(current_d, device=device) / (current_d**0.5)
            b = torch.rand(1, device=device) * 6.28
            try:
                feat = tfn(V_tr, w, b).unsqueeze(1)
                aug = F.normalize(torch.cat([V_tr, feat], 1), dim=1)
                l = loo(aug, y)
                if l > best_loo + 0.001:
                    best_loo = l; best = (tname, w.clone(), b.clone())
            except: pass
    
    if best is None: break
    tname, w, b = best
    V_tr = torch.cat([V_tr, templates[tname](V_tr, w, b).unsqueeze(1)], 1)
    V_te = torch.cat([V_te, templates[tname](V_te, w, b).unsqueeze(1)], 1)
    acc = knn_acc(F.normalize(V_tr,dim=1), y, F.normalize(V_te,dim=1), yte)
    print(f'Layer {layer+1}: +{tname}(d={current_d}) test={acc:.1f}% d={V_tr.shape[1]}')
" 2>&1
