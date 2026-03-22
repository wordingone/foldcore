"""
Step 193 — Recovered from session JSONL (inline python -c experiment).
Original execution: foldcore k-NN / torch experiments, March 15 2026.
Source: B--M-avir-leo/0606b161, line 10954.
"""
import torch, torch.nn.functional as F
device = 'cuda'

# Step 193: Template menu ablation
# Test each template ALONE and the full menu on parity + XOR + Fibonacci
# Quantify: LOO selection contribution vs template menu contribution

d_parity = 8
X_p = torch.randint(0, 2, (1000, d_parity), device=device).float()
y_p = (X_p.sum(1) % 2).long()
Xte_p = torch.zeros(256, d_parity, device=device)
for i in range(256):
    for b in range(d_parity): Xte_p[i,b]=(i>>b)&1
yte_p = (Xte_p.sum(1) % 2).long()

d_xor = 20
X_x = torch.randint(0, 2, (2000, d_xor), device=device).float()
y_x = (X_x[:,0].long() ^ X_x[:,1].long())
Xte_x = torch.randn(500, d_xor, device=device).clamp(0,1).round()
yte_x = (Xte_x[:,0].long() ^ Xte_x[:,1].long())

def test_templates(X, y, Xte, yte, d, templates, n_rounds=5, n_cand=200):
    def loo(V, labels):
        V_n = F.normalize(V,dim=1); sims = V_n @ V_n.T; sims.fill_diagonal_(-1e9)
        n_cls = labels.max().item()+1
        scores = torch.zeros(V.shape[0], n_cls, device=device)
        for c in range(n_cls):
            m = labels == c; cs = sims[:, m]
            if cs.shape[1] == 0: continue
            scores[:, c] = cs.topk(min(5, cs.shape[1]), dim=1).values.sum(dim=1)
        return (scores.argmax(1) == labels).float().mean().item()
    
    def knn_acc(V, labels, te, yte):
        sims = F.normalize(te,dim=1) @ F.normalize(V,dim=1).T
        n_cls = labels.max().item()+1
        scores = torch.zeros(te.shape[0], n_cls, device=device)
        for c in range(n_cls):
            m = labels == c; cs = sims[:, m]
            if cs.shape[1] == 0: continue
            scores[:, c] = cs.topk(min(5, cs.shape[1]), dim=1).values.sum(dim=1)
        return (scores.argmax(1) == yte).float().mean().item() * 100
    
    V = X.clone(); discovered = []
    for step in range(n_rounds):
        best_loo = loo(F.normalize(V,dim=1), y)
        best = None
        for tname, tfn in templates.items():
            for _ in range(n_cand // len(templates)):
                w = torch.randn(d, device=device) / (d**0.5)
                b = torch.rand(1, device=device) * 6.28
                try:
                    feat = tfn(X, w, b).unsqueeze(1)
                    if feat.isnan().any(): continue
                    aug = F.normalize(torch.cat([V, feat], 1), dim=1)
                    l = loo(aug, y)
                    if l > best_loo + 0.001:
                        best_loo = l; best = (tname, w.clone(), b.clone())
                except: pass
        if best is None: break
        tname, w, b = best
        discovered.append((tname, w, b))
        V = torch.cat([V, templates[tname](X, w, b).unsqueeze(1)], 1)
    
    # Eval
    te_aug = Xte.clone()
    for tname, w, b in discovered:
        te_aug = torch.cat([te_aug, templates[tname](Xte, w, b).unsqueeze(1)], 1)
    acc = knn_acc(F.normalize(V,dim=1), y, F.normalize(te_aug,dim=1), yte)
    base = knn_acc(X, y, Xte, yte)
    return base, acc, len(discovered), [d[0] for d in discovered]

# Template sets to test
template_sets = {
    'cos only':  {'cos': lambda x,w,b: torch.cos(x@w+b)},
    'abs only':  {'abs': lambda x,w,b: torch.abs(x@w+b)},
    'mod2 only': {'mod2': lambda x,w,b: ((x@w.abs()).round()%2).float()},
    'sign only': {'sign': lambda x,w,b: torch.sign(x@w+b)},
    'cos+abs':   {'cos': lambda x,w,b: torch.cos(x@w+b), 'abs': lambda x,w,b: torch.abs(x@w+b)},
    'FULL menu': {'cos': lambda x,w,b: torch.cos(x@w+b), 'abs': lambda x,w,b: torch.abs(x@w+b),
                  'sign': lambda x,w,b: torch.sign(x@w+b), 'mod2': lambda x,w,b: ((x@w.abs()).round()%2).float(),
                  'tanh': lambda x,w,b: torch.tanh(x@w+b)},
    'NO templates': {},
}

print(f'PARITY (d={d_parity}):')
print(f'{\"Templates\":20s} | Base  | After | Delta | #F | Types')
print(f'{\"-\"*20}-|-------|-------|-------|----|---------')
for name, tset in template_sets.items():
    if not tset:
        base = test_templates(X_p, y_p, Xte_p, yte_p, d_parity, {'cos': lambda x,w,b: torch.cos(x@w+b)}, n_rounds=0, n_cand=0)
        print(f'{name:20s} | {base[0]:5.1f}% | {base[0]:5.1f}% | {0:+.1f}pp | 0  | -')
        continue
    b, a, n, types = test_templates(X_p, y_p, Xte_p, yte_p, d_parity, tset)
    print(f'{name:20s} | {b:5.1f}% | {a:5.1f}% | {a-b:+.1f}pp | {n}  | {types}')

print(f'\\nXOR (d={d_xor}):')
print(f'{\"Templates\":20s} | Base  | After | Delta | #F')
print(f'{\"-\"*20}-|-------|-------|-------|---')
for name, tset in template_sets.items():
    if not tset:
        base = test_templates(X_x, y_x, Xte_x, yte_x, d_xor, {'cos': lambda x,w,b: torch.cos(x@w+b)}, n_rounds=0, n_cand=0)
        print(f'{name:20s} | {base[0]:5.1f}% | {base[0]:5.1f}% | {0:+.1f}pp | 0')
        continue
    b, a, n, _ = test_templates(X_x, y_x, Xte_x, yte_x, d_xor, tset)
    print(f'{name:20s} | {b:5.1f}% | {a:5.1f}% | {a-b:+.1f}pp | {n}')
" 2>&1
