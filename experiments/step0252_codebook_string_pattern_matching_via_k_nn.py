"""
Step 252 — Recovered from session JSONL (inline python -c experiment).
Original execution: foldcore k-NN / torch experiments, March 15 2026.
Source: B--M-avir-leo/0606b161, line 11840.
"""
import torch, torch.nn.functional as F
device = 'cuda'

# Step 252: String pattern matching via k-NN
# Task: given a binary string, does it contain the pattern '110'?
# Encode: string as raw bits

d = 8; n_train = 1000

X = torch.randint(0, 2, (n_train, d), device=device).float()
# Label: does the string contain '110'?
def has_pattern(x):
    bits = [int(x[i].item()) for i in range(d)]
    for i in range(d-2):
        if bits[i]==1 and bits[i+1]==1 and bits[i+2]==0:
            return 1
    return 0
y = torch.tensor([has_pattern(X[i]) for i in range(n_train)], device=device, dtype=torch.long)

Xte = torch.zeros(256, d, device=device)
for i in range(256):
    for b in range(d): Xte[i,b]=(i>>b)&1
yte = torch.tensor([has_pattern(Xte[i]) for i in range(256)], device=device, dtype=torch.long)

templates = {'cos': lambda x,w,b: torch.cos(x@w+b), 'abs': lambda x,w,b: torch.abs(x@w+b), 'mod2': lambda x,w,b: ((x@w.abs()).round()%2).float()}

def loo(V,labels):
    V_n=F.normalize(V,dim=1);sims=V_n@V_n.T;sims.fill_diagonal_(-1e9)
    scores=torch.zeros(V.shape[0],2,device=device)
    for c in range(2):
        m=labels==c;cs=sims[:,m]
        if cs.shape[1]==0:continue
        scores[:,c]=cs.topk(min(5,cs.shape[1]),dim=1).values.sum(dim=1)
    return(scores.argmax(1)==labels).float().mean().item()

def knn(V,labels,te,yte):
    sims=F.normalize(te,dim=1)@F.normalize(V,dim=1).T
    scores=torch.zeros(te.shape[0],2,device=device)
    for c in range(2):
        m=labels==c;cs=sims[:,m]
        if cs.shape[1]==0:continue
        scores[:,c]=cs.topk(min(5,cs.shape[1]),dim=1).values.sum(dim=1)
    return(scores.argmax(1)==yte).float().mean().item()*100

base = knn(X, y, Xte, yte)

V=X.clone();layers=[]
for _ in range(5):
    cd=V.shape[1];bl=loo(V,y);best=None
    for tn,tf in templates.items():
        for _ in range(50):
            w=torch.randn(cd,device=device)/(cd**0.5);b=torch.rand(1,device=device)*2
            try:
                feat=tf(V,w,b).unsqueeze(1);aug=F.normalize(torch.cat([V,feat],1),dim=1)
                l=loo(aug,y)
                if l>bl+0.005:bl=l;best=(tn,w.clone(),b.clone())
            except:pass
    if best is None:break
    tn,w,b=best;layers.append((tn,w,b))
    V=torch.cat([V,templates[tn](V,w,b).unsqueeze(1)],1)

Vte=Xte.clone();Vtr=X.clone()
for tn,w,b in layers:
    Vtr=torch.cat([Vtr,templates[tn](Vtr,w,b).unsqueeze(1)],1)
    Vte=torch.cat([Vte,templates[tn](Vte,w,b).unsqueeze(1)],1)
sub=knn(F.normalize(Vtr,dim=1),y,F.normalize(Vte,dim=1),yte)

print(f'Step 252: String pattern matching (contains \"110\")')
print(f'  Base k-NN: {base:.1f}%')
print(f'  Substrate: {sub:.1f}% ({len(layers)} layers, delta={sub-base:+.1f}pp)')
print(f'  Class dist: pos={y.sum().item()}, neg={len(y)-y.sum().item()}')
" 2>&1
