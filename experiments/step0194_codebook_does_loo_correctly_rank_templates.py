"""
Step 194 — Recovered from session JSONL (inline python -c experiment).
Original execution: foldcore k-NN / torch experiments, March 15 2026.
Source: B--M-avir-leo/0606b161, line 10969.
"""
import torch, torch.nn.functional as F
device = 'cuda'

# Step 194: Does LOO correctly rank templates?
# For each task, compute LOO score with each template's BEST feature.
# Compare LOO ranking to test-set ranking.

templates = {
    'cos':  lambda x,w,b: torch.cos(x@w+b),
    'abs':  lambda x,w,b: torch.abs(x@w+b),
    'mod2': lambda x,w,b: ((x@w.abs()).round()%2).float(),
    'sign': lambda x,w,b: torch.sign(x@w+b),
    'tanh': lambda x,w,b: torch.tanh(x@w+b),
}

tasks = {
    'parity_d8': (
        torch.randint(0,2,(1000,8),device=device).float(),
        None,  # computed below
        torch.zeros(256,8,device=device),
        None,
        8
    ),
    'xor_d20': (
        torch.randint(0,2,(2000,20),device=device).float(),
        None, None, None, 20
    ),
}

# Setup tasks
X_p, _, Xte_p, _, d_p = tasks['parity_d8']
y_p = (X_p.sum(1)%2).long()
for i in range(256):
    for b in range(8): Xte_p[i,b]=(i>>b)&1
yte_p = (Xte_p.sum(1)%2).long()
tasks['parity_d8'] = (X_p, y_p, Xte_p, yte_p, d_p)

X_x, _, _, _, d_x = tasks['xor_d20']
y_x = (X_x[:,0].long()^X_x[:,1].long())
Xte_x = torch.randn(500,d_x,device=device).clamp(0,1).round()
yte_x = (Xte_x[:,0].long()^Xte_x[:,1].long())
tasks['xor_d20'] = (X_x, y_x, Xte_x, yte_x, d_x)

def best_feature_score(X, y, Xte, yte, d, tfn, n_cand=200):
    def loo(V, labels):
        V_n = F.normalize(V,dim=1); sims = V_n @ V_n.T; sims.fill_diagonal_(-1e9)
        n_cls = labels.max().item()+1
        scores = torch.zeros(V.shape[0], n_cls, device=device)
        for c in range(n_cls):
            m = labels == c; cs = sims[:, m]
            if cs.shape[1] == 0: continue
            scores[:, c] = cs.topk(min(5, cs.shape[1]), dim=1).values.sum(dim=1)
        return (scores.argmax(1) == labels).float().mean().item()
    
    def knn(V, labels, te, yte):
        sims = F.normalize(te,dim=1) @ F.normalize(V,dim=1).T
        n_cls = labels.max().item()+1
        scores = torch.zeros(te.shape[0], n_cls, device=device)
        for c in range(n_cls):
            m = labels == c; cs = sims[:, m]
            if cs.shape[1] == 0: continue
            scores[:, c] = cs.topk(min(5, cs.shape[1]), dim=1).values.sum(dim=1)
        return (scores.argmax(1) == yte).float().mean().item() * 100
    
    best_loo = 0; best_test = knn(X, y, Xte, yte)
    best_w = None; best_b = None
    for _ in range(n_cand):
        w = torch.randn(d, device=device)/(d**0.5); b = torch.rand(1,device=device)*6.28
        try:
            feat_tr = tfn(X, w, b).unsqueeze(1); feat_te = tfn(Xte, w, b).unsqueeze(1)
            aug_tr = F.normalize(torch.cat([X, feat_tr],1),dim=1)
            l = loo(aug_tr, y)
            if l > best_loo:
                best_loo = l; best_w = w; best_b = b
                best_test = knn(aug_tr, y, F.normalize(torch.cat([Xte,feat_te],1),dim=1), yte)
        except: pass
    return best_loo, best_test

print('Step 194: LOO ranking vs test ranking across templates')
for task_name, (X, y, Xte, yte, d) in tasks.items():
    print(f'\\n{task_name}:')
    results = []
    for tname, tfn in templates.items():
        loo_score, test_score = best_feature_score(X, y, Xte, yte, d, tfn)
        results.append((tname, loo_score, test_score))
    
    # Rank by LOO and by test
    by_loo = sorted(results, key=lambda x: -x[1])
    by_test = sorted(results, key=lambda x: -x[2])
    
    print(f'  {\"Template\":8s} | LOO    | Test   | LOO rank | Test rank')
    print(f'  {\"--------\":8s}-|--------|--------|----------|--------')
    for tname, loo_s, test_s in results:
        loo_rank = [r[0] for r in by_loo].index(tname) + 1
        test_rank = [r[0] for r in by_test].index(tname) + 1
        match = 'OK' if loo_rank == test_rank else f'MISMATCH'
        print(f'  {tname:8s} | {loo_s:.4f} | {test_s:5.1f}% | {loo_rank}        | {test_rank}       {match}')
    
    # Spearman correlation
    loo_ranks = [([r[0] for r in by_loo].index(r[0])+1) for r in results]
    test_ranks = [([r[0] for r in by_test].index(r[0])+1) for r in results]
    n = len(results)
    d_sq = sum((l-t)**2 for l,t in zip(loo_ranks, test_ranks))
    rho = 1 - 6*d_sq/(n*(n**2-1))
    print(f'  Spearman rho: {rho:.3f} ({\"STRONG\" if rho > 0.7 else \"WEAK\" if rho > 0.3 else \"NONE\"})')
" 2>&1
