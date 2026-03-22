"""
Step 253 — Recovered from session JSONL (inline python -c experiment).
Original execution: foldcore k-NN / torch experiments, March 15 2026.
Source: B--M-avir-leo/0606b161, line 11861.
"""
import torch, torch.nn.functional as F
device = 'cuda'

# Step 253: Context-dependent classification
# Task: classify digit (0-9) but the RULE changes based on context
# Context 0: output = digit mod 3
# Context 1: output = digit mod 2 (parity)
# Context 2: output = digit > 5 ? 1 : 0
# The system must use CONTEXT to determine which rule to apply

d = 2  # (digit, context)
n_train = 1000; n_contexts = 3

X = torch.zeros(n_train, d, device=device)
X[:, 0] = torch.randint(0, 10, (n_train,), device=device).float()  # digit
X[:, 1] = torch.randint(0, n_contexts, (n_train,), device=device).float()  # context

def apply_rule(digit, context):
    if context == 0: return int(digit) % 3
    elif context == 1: return int(digit) % 2
    else: return 1 if digit > 5 else 0

y = torch.tensor([apply_rule(X[i,0].item(), X[i,1].item()) for i in range(n_train)], device=device, dtype=torch.long)

# Test: all (digit, context) combinations
Xte = []; yte = []
for digit in range(10):
    for ctx in range(n_contexts):
        Xte.append([float(digit), float(ctx)])
        yte.append(apply_rule(digit, ctx))
Xte = torch.tensor(Xte, device=device); yte = torch.tensor(yte, device=device, dtype=torch.long)

n_cls = max(y.max().item(), max(yte)) + 1

templates = {'cos': lambda x,w,b: torch.cos(x@w+b), 'abs': lambda x,w,b: torch.abs(x@w+b), 'mod2': lambda x,w,b: ((x@w.abs()).round()%2).float()}

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

base = knn(X, y, Xte, yte, n_cls)

V=X.clone();layers=[]
for _ in range(5):
    cd=V.shape[1];bl=loo(V,y,n_cls);best=None
    for tn,tf in templates.items():
        for _ in range(100):
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

print(f'Step 253: Context-dependent classification')
print(f'  3 rules: digit%3, digit%2, digit>5')
print(f'  Base k-NN: {base:.1f}%')
print(f'  Substrate: {sub:.1f}% ({len(layers)} layers, delta={sub-base:+.1f}pp)')
print(f'  30 total states (10 digits x 3 contexts)')
" 2>&1
