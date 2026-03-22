"""
Step 254 — Recovered from session JSONL (inline python -c experiment).
Original execution: foldcore k-NN / torch experiments, March 15 2026.
Source: B--M-avir-leo/0606b161, line 11882.
"""
import torch, torch.nn.functional as F
device = 'cuda'

# Step 254: Variable-length input via PADDING + LENGTH ENCODING
# Task: sum a variable-length list of digits
# Input: [len, d0, d1, ..., dn, pad, pad, ...] fixed-width with length prefix

max_len = 6; vocab = 5; d = max_len + 1  # length + padded digits
n_train = 2000

X = torch.zeros(n_train, d, device=device)
y = torch.zeros(n_train, device=device, dtype=torch.long)
for i in range(n_train):
    length = torch.randint(1, max_len+1, (1,)).item()
    digits = torch.randint(0, vocab, (length,))
    X[i, 0] = length
    X[i, 1:1+length] = digits.float()
    y[i] = digits.sum().item()

# Test
n_test = 500
Xte = torch.zeros(n_test, d, device=device)
yte = torch.zeros(n_test, device=device, dtype=torch.long)
for i in range(n_test):
    length = torch.randint(1, max_len+1, (1,)).item()
    digits = torch.randint(0, vocab, (length,))
    Xte[i, 0] = length
    Xte[i, 1:1+length] = digits.float()
    yte[i] = digits.sum().item()

n_cls = yte.max().item() + 1

templates = {'cos': lambda x,w,b: torch.cos(x@w+b), 'abs': lambda x,w,b: torch.abs(x@w+b)}

def loo(V,labels,n_cls):
    V_n=F.normalize(V,dim=1);sims=V_n@V_n.T;sims.fill_diagonal_(-1e9)
    scores=torch.zeros(V.shape[0],n_cls,device=device)
    for c in range(n_cls):
        m=labels==c;cs=sims[:,m]
        if cs.shape[1]==0:continue
        scores[:,c]=cs.topk(min(5,cs.shape[1]),dim=1).values.sum(dim=1)
    return(scores.argmax(1)==labels).float().mean().item()

def knn(V,labels,te,yte,n_cls):
    sims=F.normalize(te,dim=1)@F.normalize(V,dim=1).T
    scores=torch.zeros(te.shape[0],n_cls,device=device)
    for c in range(n_cls):
        m=labels==c;cs=sims[:,m]
        if cs.shape[1]==0:continue
        scores[:,c]=cs.topk(min(5,cs.shape[1]),dim=1).values.sum(dim=1)
    return(scores.argmax(1)==yte).float().mean().item()*100

base = knn(X,y,Xte,yte,n_cls)

V=X.clone();layers=[]
for _ in range(5):
    cd=V.shape[1];bl=loo(V,y,n_cls);best=None
    for tn,tf in templates.items():
        for _ in range(50):
            w=torch.randn(cd,device=device)/(cd**0.5);b=torch.rand(1,device=device)*n_cls
            try:
                feat=tf(V,w,b).unsqueeze(1);aug=F.normalize(torch.cat([V,feat],1),dim=1)
                l=loo(aug,y,n_cls)
                if l>bl+0.003:bl=l;best=(tn,w.clone(),b.clone())
            except:pass
    if best is None:break
    tn,w,b=best;layers.append((tn,w,b))
    V=torch.cat([V,templates[tn](V,w,b).unsqueeze(1)],1)

Vte=Xte.clone();Vtr=X.clone()
for tn,w,b in layers:
    Vtr=torch.cat([Vtr,templates[tn](Vtr,w,b).unsqueeze(1)],1)
    Vte=torch.cat([Vte,templates[tn](Vte,w,b).unsqueeze(1)],1)
sub=knn(F.normalize(Vtr,dim=1),y,F.normalize(Vte,dim=1),yte,n_cls)

print(f'Step 254: Variable-length list summing (padded input)')
print(f'  Max length={max_len}, vocab={vocab}, possible sums: 0-{vocab*max_len}')
print(f'  Base k-NN: {base:.1f}%')
print(f'  Substrate: {sub:.1f}% ({len(layers)} layers, delta={sub-base:+.1f}pp)')
" 2>&1
