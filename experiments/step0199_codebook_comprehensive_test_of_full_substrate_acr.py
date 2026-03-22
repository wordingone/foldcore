"""
Step 199 — Recovered from session JSONL (inline python -c experiment).
Original execution: foldcore k-NN / torch experiments, March 15 2026.
Source: B--M-avir-leo/0606b161, line 11046.
"""
import torch, torch.nn.functional as F
device = 'cuda'

# Step 199: Comprehensive test of full substrate across ALL domains
# One function, all tasks

templates = {
    'cos': lambda x,w,b: torch.cos(x@w+b),
    'abs': lambda x,w,b: torch.abs(x@w+b),
    'mod2': lambda x,w,b: ((x@w.abs()).round()%2).float(),
}

def full_substrate(X_tr, y_tr, X_te, y_te, n_cls, max_layers=10, n_cand=100, k=5):
    '''The complete self-improving substrate.'''
    def loo(V, labels):
        V_n = F.normalize(V,dim=1); sims = V_n @ V_n.T; sims.fill_diagonal_(-1e9)
        scores = torch.zeros(V.shape[0], n_cls, device=device)
        for c in range(n_cls):
            m = labels == c; cs = sims[:, m]
            if cs.shape[1] == 0: continue
            scores[:, c] = cs.topk(min(k, cs.shape[1]), dim=1).values.sum(dim=1)
        return (scores.argmax(1) == labels).float().mean().item()
    
    # Discover layered features
    V = X_tr.clone(); layers = []
    for _ in range(max_layers):
        cd = V.shape[1]
        best_loo = loo(F.normalize(V,dim=1), y_tr); best = None
        for tn, tf in templates.items():
            for _ in range(n_cand//len(templates)):
                w = torch.randn(cd, device=device)/(cd**0.5)
                b = torch.rand(1, device=device)*6.28
                try:
                    feat = tf(V, w, b).unsqueeze(1)
                    if feat.isnan().any(): continue
                    aug = F.normalize(torch.cat([V,feat],1),dim=1)
                    l = loo(aug, y_tr)
                    if l > best_loo+0.001: best_loo=l; best=(tn,w.clone(),b.clone())
                except: pass
        if best is None: break
        tn,w,b = best; layers.append((tn,w,b))
        V = torch.cat([V, templates[tn](V,w,b).unsqueeze(1)], 1)
    
    # Augment test
    V_te = X_te.clone()
    V_tr_aug = X_tr.clone()
    for tn,w,b in layers:
        V_tr_aug = torch.cat([V_tr_aug, templates[tn](V_tr_aug,w,b).unsqueeze(1)], 1)
        V_te = torch.cat([V_te, templates[tn](V_te,w,b).unsqueeze(1)], 1)
    
    # Classify
    sims = F.normalize(V_te,dim=1) @ F.normalize(V_tr_aug,dim=1).T
    scores = torch.zeros(X_te.shape[0], n_cls, device=device)
    for c in range(n_cls):
        m = y_tr == c; cs = sims[:, m]
        if cs.shape[1] == 0: continue
        scores[:, c] = cs.topk(min(k, cs.shape[1]), dim=1).values.sum(dim=1)
    acc = (scores.argmax(1) == y_te).float().mean().item() * 100
    
    # Base (no features)
    sims_b = F.normalize(X_te,dim=1) @ F.normalize(X_tr,dim=1).T
    scores_b = torch.zeros(X_te.shape[0], n_cls, device=device)
    for c in range(n_cls):
        m = y_tr == c; cs = sims_b[:, m]
        if cs.shape[1] == 0: continue
        scores_b[:, c] = cs.topk(min(k, cs.shape[1]), dim=1).values.sum(dim=1)
    base = (scores_b.argmax(1) == y_te).float().mean().item() * 100
    
    return base, acc, len(layers)

print('=== STEP 199: COMPREHENSIVE SUBSTRATE VALIDATION ===')
print(f'{\"Task\":30s} | Base  | Sub   | Delta | #L')
print(f'{\"-\"*30}-|-------|-------|-------|---')

# 1. Parity d=8
X=torch.randint(0,2,(1000,8),device=device).float(); y=(X.sum(1)%2).long()
Xte=torch.zeros(256,8,device=device)
for i in range(256):
    for b in range(8): Xte[i,b]=(i>>b)&1
yte=(Xte.sum(1)%2).long()
b,a,n=full_substrate(X,y,Xte,yte,2)
print(f'{\"Parity d=8\":30s} | {b:5.1f}% | {a:5.1f}% | {a-b:+.1f}pp | {n}')

# 2. XOR d=20
X=torch.randint(0,2,(2000,20),device=device).float(); y=(X[:,0].long()^X[:,1].long())
Xte=torch.randn(500,20,device=device).clamp(0,1).round(); yte=(Xte[:,0].long()^Xte[:,1].long())
b,a,n=full_substrate(X,y,Xte,yte,2)
print(f'{\"XOR d=20\":30s} | {b:5.1f}% | {a:5.1f}% | {a-b:+.1f}pp | {n}')

# 3. Multi-rule
X=torch.randint(0,2,(2000,10),device=device).float()
y=torch.full((2000,),3,device=device,dtype=torch.long)
for i in range(2000):
    xor=int(X[i,0])^int(X[i,1]); and_=int(X[i,2])&int(X[i,3])
    nor=1-(int(X[i,4])|int(X[i,5]))
    if xor and not and_ and not nor: y[i]=0
    elif and_ and not xor and not nor: y[i]=1
    elif nor and not xor and not and_: y[i]=2
Xte=torch.zeros(1024,10,device=device)
for i in range(1024):
    for bb in range(10): Xte[i,bb]=(i>>bb)&1
yte=torch.full((1024,),3,device=device,dtype=torch.long)
for i in range(1024):
    xor=int(Xte[i,0])^int(Xte[i,1]); and_=int(Xte[i,2])&int(Xte[i,3])
    nor=1-(int(Xte[i,4])|int(Xte[i,5]))
    if xor and not and_ and not nor: yte[i]=0
    elif and_ and not xor and not nor: yte[i]=1
    elif nor and not xor and not and_: yte[i]=2
b,a,n=full_substrate(X,y,Xte,yte,4)
print(f'{\"Multi-rule (XOR+AND+NOR)\":30s} | {b:5.1f}% | {a:5.1f}% | {a-b:+.1f}pp | {n}')

# 4. Majority vote
X=torch.randint(0,2,(2000,10),device=device).float(); y=(X[:,:5].sum(1)>=3).long()
Xte2=torch.zeros(1024,10,device=device)
for i in range(1024):
    for bb in range(10): Xte2[i,bb]=(i>>bb)&1
yte2=(Xte2[:,:5].sum(1)>=3).long()
b,a,n=full_substrate(X,y,Xte2,yte2,2)
print(f'{\"Majority vote (5 bits)\":30s} | {b:5.1f}% | {a:5.1f}% | {a-b:+.1f}pp | {n}')

# 5. MNIST (single task)
try:
    from torchvision import datasets
    mnist=datasets.MNIST('C:/tmp/mnist',train=True,download=False)
    mnist_t=datasets.MNIST('C:/tmp/mnist',train=False)
    Xm=F.normalize(mnist.data[:6000].float().view(-1,784).to(device),dim=1)
    ym=mnist.targets[:6000].to(device)
    Xmt=F.normalize(mnist_t.data[:1000].float().view(-1,784).to(device),dim=1)
    ymt=mnist_t.targets[:1000].to(device)
    b,a,n=full_substrate(Xm,ym,Xmt,ymt,10,max_layers=3,n_cand=50)
    print(f'{\"MNIST raw pixels\":30s} | {b:5.1f}% | {a:5.1f}% | {a-b:+.1f}pp | {n}')
except: print(f'{\"MNIST\":30s} | skip')
" 2>&1
