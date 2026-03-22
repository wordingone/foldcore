"""
Step 231 — Recovered from session JSONL (inline python -c experiment).
Original execution: foldcore k-NN / torch experiments, March 15 2026.
Source: B--M-avir-leo/0606b161, line 11349.
"""
import torch, torch.nn.functional as F
device = 'cuda'

# Step 231: Sparse training — only 16 of 32 states seen
d = 5
def rule(x):
    a,b,c,dd,e = int(x[0]),int(x[1]),int(x[2]),int(x[3]),int(x[4])
    return int((a and b) or (c ^ dd) or ((not e) and a))

# ALL 32 states
all_states = torch.zeros(32, d, device=device)
all_labels = torch.zeros(32, device=device, dtype=torch.long)
for i in range(32):
    for b in range(d): all_states[i,b] = (i>>b)&1
    all_labels[i] = rule(all_states[i])

templates = {'cos': lambda x,w,b: torch.cos(x@w+b), 'abs': lambda x,w,b: torch.abs(x@w+b), 'mod2': lambda x,w,b: ((x@w.abs()).round()%2).float()}

def knn(V,labels,te,yte):
    sims=F.normalize(te,dim=1)@F.normalize(V,dim=1).T
    scores=torch.zeros(te.shape[0],2,device=device)
    for c in range(2):
        m=labels==c;cs=sims[:,m]
        if cs.shape[1]==0:continue
        scores[:,c]=cs.topk(min(5,cs.shape[1]),dim=1).values.sum(dim=1)
    return(scores.argmax(1)==yte).float().mean().item()*100

def loo(V,labels):
    V_n=F.normalize(V,dim=1);sims=V_n@V_n.T;sims.fill_diagonal_(-1e9)
    scores=torch.zeros(V.shape[0],2,device=device)
    for c in range(2):
        m=labels==c;cs=sims[:,m]
        if cs.shape[1]==0:continue
        scores[:,c]=cs.topk(min(5,cs.shape[1]),dim=1).values.sum(dim=1)
    return(scores.argmax(1)==labels).float().mean().item()

print(f'Sparse training: varying fraction of 32 states seen')
print(f'{\"Train\":>5s} | Base  | Sub   | Delta')
print(f'------|-------|-------|------')

for n_seen in [10, 16, 20, 24]:
    # Random subset of states for training
    torch.manual_seed(42)
    perm = torch.randperm(32)
    tr_idx = perm[:n_seen]; te_idx = perm[n_seen:]  # test on UNSEEN states
    
    X_tr = all_states[tr_idx].repeat(10, 1)  # repeat for k-NN coverage
    y_tr = all_labels[tr_idx].repeat(10)
    X_te = all_states[te_idx]
    y_te = all_labels[te_idx]
    
    base = knn(X_tr, y_tr, X_te, y_te)
    
    V=X_tr.clone();layers=[]
    for _ in range(5):
        cd=V.shape[1];bl=loo(V,y_tr);best=None
        for tn,tf in templates.items():
            for _ in range(100):
                w=torch.randn(cd,device=device)/(cd**0.5);b=torch.rand(1,device=device)*2
                try:
                    feat=tf(V,w,b).unsqueeze(1);aug=F.normalize(torch.cat([V,feat],1),dim=1)
                    l=loo(aug,y_tr)
                    if l>bl+0.005:bl=l;best=(tn,w.clone(),b.clone())
                except:pass
        if best is None:break
        tn,w,b=best;layers.append((tn,w,b))
        V=torch.cat([V,templates[tn](V,w,b).unsqueeze(1)],1)
    
    Vte=X_te.clone();Vtr=X_tr.clone()
    for tn,w,b in layers:
        Vtr=torch.cat([Vtr,templates[tn](Vtr,w,b).unsqueeze(1)],1)
        Vte=torch.cat([Vte,templates[tn](Vte,w,b).unsqueeze(1)],1)
    sub=knn(F.normalize(Vtr,dim=1),y_tr,F.normalize(Vte,dim=1),y_te)
    
    print(f'{n_seen:5d} | {base:5.1f}% | {sub:5.1f}% | {sub-base:+.1f}pp')
" 2>&1
