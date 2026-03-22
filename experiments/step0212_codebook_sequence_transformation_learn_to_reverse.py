"""
Step 212 — Recovered from session JSONL (inline python -c experiment).
Original execution: foldcore k-NN / torch experiments, March 15 2026.
Source: B--M-avir-leo/0606b161, line 11163.
"""
import torch, torch.nn.functional as F
device = 'cuda'

# Step 212: Sequence transformation — learn to REVERSE a sequence
# Input: [a, b, c, d] -> Output: [d, c, b, a]
# This requires the substrate to learn a MAPPING, not just a classification

seq_len = 4; vocab = 5; n_train = 2000; n_test = 200

# Generate data: random sequences and their reverses
X_tr = torch.randint(0, vocab, (n_train, seq_len), device=device).float()
y_tr = X_tr.flip(1)  # reverse

X_te = torch.randint(0, vocab, (n_test, seq_len), device=device).float()
y_te = X_te.flip(1)

# Predict each output position independently
# For position i in output, predict the value using the full input sequence as features

templates = {'cos': lambda x,w,b: torch.cos(x@w+b), 'abs': lambda x,w,b: torch.abs(x@w+b)}

def loo(V, labels, n_cls, k=5):
    V_n = F.normalize(V,dim=1); sims = V_n @ V_n.T; sims.fill_diagonal_(-1e9)
    scores = torch.zeros(V.shape[0], n_cls, device=device)
    for c in range(n_cls):
        m = labels == c; cs = sims[:, m]
        if cs.shape[1] == 0: continue
        scores[:, c] = cs.topk(min(k, cs.shape[1]), dim=1).values.sum(dim=1)
    return (scores.argmax(1) == labels).float().mean().item()

def knn(V, labels, te, yte, n_cls, k=5):
    sims = F.normalize(te,dim=1) @ F.normalize(V,dim=1).T
    scores = torch.zeros(te.shape[0], n_cls, device=device)
    for c in range(n_cls):
        m = labels == c; cs = sims[:, m]
        if cs.shape[1] == 0: continue
        scores[:, c] = cs.topk(min(k, cs.shape[1]), dim=1).values.sum(dim=1)
    return (scores.argmax(1) == yte).float().mean().item() * 100

print(f'Sequence reversal (len={seq_len}, vocab={vocab}):')

total_base = total_sub = 0
for pos in range(seq_len):
    labels_tr = y_tr[:, pos].long()
    labels_te = y_te[:, pos].long()
    
    acc_base = knn(X_tr, labels_tr, X_te, labels_te, vocab)
    
    # Feature discovery for this position
    V = X_tr.clone(); layers = []
    for _ in range(5):
        cd = V.shape[1]; best_loo = loo(V, labels_tr, vocab); best = None
        for tn, tf in templates.items():
            for _ in range(50):
                w = torch.randn(cd, device=device)/(cd**0.5); b = torch.rand(1,device=device)*vocab
                try:
                    feat = tf(V, w, b).unsqueeze(1)
                    aug = F.normalize(torch.cat([V,feat],1),dim=1)
                    l = loo(aug, labels_tr, vocab)
                    if l > best_loo+0.005: best_loo=l; best=(tn,w.clone(),b.clone())
                except: pass
        if best is None: break
        tn,w,b = best; layers.append((tn,w,b))
        V = torch.cat([V, templates[tn](V,w,b).unsqueeze(1)], 1)
    
    V_te2 = X_te.clone(); V_tr2 = X_tr.clone()
    for tn,w,b in layers:
        V_tr2 = torch.cat([V_tr2, templates[tn](V_tr2,w,b).unsqueeze(1)], 1)
        V_te2 = torch.cat([V_te2, templates[tn](V_te2,w,b).unsqueeze(1)], 1)
    acc_sub = knn(F.normalize(V_tr2,dim=1), labels_tr, F.normalize(V_te2,dim=1), labels_te, vocab)
    
    total_base += acc_base; total_sub += acc_sub
    print(f'  pos {pos}: base={acc_base:.1f}% sub={acc_sub:.1f}% delta={acc_sub-acc_base:+.1f}pp layers={len(layers)}')

print(f'  AVG:  base={total_base/seq_len:.1f}% sub={total_sub/seq_len:.1f}% delta={total_sub/seq_len-total_base/seq_len:+.1f}pp')

# Full sequence accuracy
correct_base = correct_sub = 0
for i in range(n_test):
    pred_base = []; pred_sub = []
    for pos in range(seq_len):
        # Simplified: just use base k-NN per position
        sims = F.normalize(X_te[i:i+1],dim=1) @ F.normalize(X_tr,dim=1).T
        scores = torch.zeros(1, vocab, device=device)
        for c in range(vocab):
            m = y_tr[:, pos].long() == c; cs = sims[:, m]
            if cs.shape[1] == 0: continue
            scores[:, c] = cs.topk(min(5, cs.shape[1]), dim=1).values.sum(dim=1)
        pred_base.append(scores.argmax(1).item())
    if pred_base == y_te[i].long().tolist(): correct_base += 1

print(f'  Full-seq correct (base): {correct_base}/{n_test} ({correct_base/n_test*100:.1f}%)')
" 2>&1
