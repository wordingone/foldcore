"""
Steps 213-215 — Recovered from CC session 0606b161 (inline Bash execution).
foldcore k-NN / torch experiments, March 15 2026.
Source: line 11175.
"""
import torch, torch.nn.functional as F
device = 'cuda'

# Step 213: Sort a sequence of 4 numbers (vocab=10)
# Step 214: Copy task — identity mapping
# Step 215: Shift-by-1 — each output = input shifted right

seq_len = 4; vocab = 10; n_train = 3000; n_test = 200
templates = {'cos': lambda x,w,b: torch.cos(x@w+b), 'abs': lambda x,w,b: torch.abs(x@w+b)}

def test_seq_task(X_tr, y_tr, X_te, y_te, name, vocab_out):
    def knn(V, labels, te, yte, n_cls, k=5):
        sims = F.normalize(te,dim=1) @ F.normalize(V,dim=1).T
        scores = torch.zeros(te.shape[0], n_cls, device=device)
        for c in range(n_cls):
            m = labels == c; cs = sims[:, m]
            if cs.shape[1] == 0: continue
            scores[:, c] = cs.topk(min(k, cs.shape[1]), dim=1).values.sum(dim=1)
        return (scores.argmax(1) == yte).float().mean().item() * 100
    
    def loo(V, labels, n_cls, k=5):
        V_n = F.normalize(V,dim=1); sims = V_n @ V_n.T; sims.fill_diagonal_(-1e9)
        scores = torch.zeros(V.shape[0], n_cls, device=device)
        for c in range(n_cls):
            m = labels == c; cs = sims[:, m]
            if cs.shape[1] == 0: continue
            scores[:, c] = cs.topk(min(k, cs.shape[1]), dim=1).values.sum(dim=1)
        return (scores.argmax(1) == labels).float().mean().item()
    
    total_base = total_sub = 0
    for pos in range(seq_len):
        labels_tr = y_tr[:, pos].long(); labels_te = y_te[:, pos].long()
        acc_base = knn(X_tr, labels_tr, X_te, labels_te, vocab_out)
        
        V = X_tr.clone()
        for _ in range(3):
            cd = V.shape[1]; best_loo = loo(V, labels_tr, vocab_out); best = None
            for tn, tf in templates.items():
                for _ in range(30):
                    w = torch.randn(cd, device=device)/(cd**0.5); b = torch.rand(1,device=device)*vocab_out
                    try:
                        feat = tf(V, w, b).unsqueeze(1)
                        aug = F.normalize(torch.cat([V,feat],1),dim=1)
                        l = loo(aug, labels_tr, vocab_out)
                        if l > best_loo+0.005: best_loo=l; best=(tn,w.clone(),b.clone())
                    except: pass
            if best is None: break
            tn,w,b = best
            V = torch.cat([V, templates[tn](V,w,b).unsqueeze(1)], 1)
            V_te2 = X_te.clone()
            V_tr2 = X_tr.clone()
        
        total_base += acc_base; total_sub += knn(F.normalize(V,dim=1), labels_tr, F.normalize(X_te,dim=1), labels_te, vocab_out) if V.shape[1]==seq_len else acc_base
    
    avg_b = total_base/seq_len; avg_s = total_sub/seq_len
    print(f'{name:25s} | {avg_b:5.1f}% | {avg_s:5.1f}% | {avg_s-avg_b:+.1f}pp')

print(f'{\"Task\":25s} | Base  | Sub   | Delta')
print(f'{\"-\"*25}-|-------|-------|------')

# Sort
X = torch.randint(0, vocab, (n_train, seq_len), device=device).float()
y_sort = torch.sort(X.long(), dim=1).values
Xte = torch.randint(0, vocab, (n_test, seq_len), device=device).float()
yte_sort = torch.sort(Xte.long(), dim=1).values
test_seq_task(X, y_sort.float(), Xte, yte_sort.float(), 'Sort (len=4, v=10)', vocab)

# Copy (identity)
test_seq_task(X, X, Xte, Xte, 'Copy/Identity', vocab)

# Shift right by 1 (circular)
y_shift = torch.cat([X[:, -1:], X[:, :-1]], dim=1)
yte_shift = torch.cat([Xte[:, -1:], Xte[:, :-1]], dim=1)
test_seq_task(X, y_shift, Xte, yte_shift, 'Circular shift right', vocab)
" 2>&1
