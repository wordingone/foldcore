"""
Step 227 — Recovered from session JSONL (inline python -c experiment).
Original execution: foldcore k-NN / torch experiments, March 15 2026.
Source: B--M-avir-leo/0606b161, line 11305.
"""
import torch, torch.nn.functional as F
device = 'cuda'

# Step 227: Simple language-like task — n-gram prediction
# Given context of 3 characters, predict the next character
# Train on a simple grammar: 'aab' always followed by 'c', 'bba' by 'a', etc.

vocab = 4  # a=0, b=1, c=2, d=3
context_len = 3
n_train = 2000

# Generate sequences from a simple Markov chain
# Transition: depends on last 3 chars
torch.manual_seed(42)
transition = torch.randint(0, vocab, (vocab**context_len,), device=device)

def ctx_to_idx(ctx):
    idx = 0
    for c in ctx: idx = idx * vocab + c
    return idx

# Generate training data
seqs = []
for _ in range(100):
    seq = torch.randint(0, vocab, (20,), device=device)
    # Override with deterministic transitions after first 3
    for t in range(context_len, len(seq)):
        ctx = seq[t-context_len:t].tolist()
        seq[t] = transition[ctx_to_idx(ctx)]
    seqs.append(seq)

# Extract (context, next_char) pairs
X_tr = []; y_tr = []
for seq in seqs[:80]:
    for t in range(context_len, len(seq)):
        ctx = seq[t-context_len:t].float()
        X_tr.append(ctx); y_tr.append(seq[t].item())
X_tr = torch.stack(X_tr); y_tr = torch.tensor(y_tr, device=device, dtype=torch.long)

X_te = []; y_te = []
for seq in seqs[80:]:
    for t in range(context_len, len(seq)):
        ctx = seq[t-context_len:t].float()
        X_te.append(ctx); y_te.append(seq[t].item())
X_te = torch.stack(X_te); y_te = torch.tensor(y_te, device=device, dtype=torch.long)

templates = {'cos': lambda x,w,b: torch.cos(x@w+b), 'abs': lambda x,w,b: torch.abs(x@w+b), 'mod2': lambda x,w,b: ((x@w.abs()).round()%2).float()}

def loo(V, labels, n_cls=vocab):
    V_n=F.normalize(V,dim=1); sims=V_n@V_n.T; sims.fill_diagonal_(-1e9)
    scores=torch.zeros(V.shape[0],n_cls,device=device)
    for c in range(n_cls):
        m=labels==c; cs=sims[:,m]
        if cs.shape[1]==0: continue
        scores[:,c]=cs.topk(min(5,cs.shape[1]),dim=1).values.sum(dim=1)
    return (scores.argmax(1)==labels).float().mean().item()

def knn(V,labels,te,yte,n_cls=vocab):
    sims=F.normalize(te,dim=1)@F.normalize(V,dim=1).T
    scores=torch.zeros(te.shape[0],n_cls,device=device)
    for c in range(n_cls):
        m=labels==c; cs=sims[:,m]
        if cs.shape[1]==0: continue
        scores[:,c]=cs.topk(min(5,cs.shape[1]),dim=1).values.sum(dim=1)
    return (scores.argmax(1)==yte).float().mean().item()*100

base = knn(X_tr, y_tr, X_te, y_te)

V = X_tr.clone(); layers = []
for _ in range(5):
    cd=V.shape[1]; bl=loo(V,y_tr); best=None
    for tn,tf in templates.items():
        for _ in range(50):
            w=torch.randn(cd,device=device)/(cd**0.5); b=torch.rand(1,device=device)*vocab
            try:
                feat=tf(V,w,b).unsqueeze(1); aug=F.normalize(torch.cat([V,feat],1),dim=1)
                l=loo(aug,y_tr)
                if l>bl+0.005: bl=l; best=(tn,w.clone(),b.clone())
            except: pass
    if best is None: break
    tn,w,b=best; layers.append((tn,w,b))
    V=torch.cat([V,templates[tn](V,w,b).unsqueeze(1)],1)

Vte=X_te.clone(); Vtr=X_tr.clone()
for tn,w,b in layers:
    Vtr=torch.cat([Vtr,templates[tn](Vtr,w,b).unsqueeze(1)],1)
    Vte=torch.cat([Vte,templates[tn](Vte,w,b).unsqueeze(1)],1)
sub = knn(F.normalize(Vtr,dim=1),y_tr,F.normalize(Vte,dim=1),y_te)

print(f'N-gram prediction (context={context_len}, vocab={vocab}):')
print(f'  Possible contexts: {vocab**context_len}')
print(f'  Train: {X_tr.shape[0]}, Test: {X_te.shape[0]}')
print(f'  Base k-NN: {base:.1f}%')
print(f'  Substrate: {sub:.1f}% ({len(layers)} layers, delta={sub-base:+.1f}pp)')
print(f'  Random: {100/vocab:.1f}%')
" 2>&1
