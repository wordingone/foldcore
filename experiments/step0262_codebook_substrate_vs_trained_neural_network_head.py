"""
Step 262 — Recovered from session JSONL (inline python -c experiment).
Original execution: foldcore k-NN / torch experiments, March 15 2026.
Source: B--M-avir-leo/0606b161, line 11981.
"""
import torch, torch.nn.functional as F
from torch import nn, optim
device = 'cuda'

# Step 262: Substrate vs trained neural network — head-to-head
# Same task, same data, same eval. Who wins?

# Task: addition mod 5 from raw integers (substrate's strong task)
d = 2; n_train = 500; vocab = 5; n_cls = 5

X_tr = torch.randint(0, vocab, (n_train, d), device=device).float()
y_tr = ((X_tr[:,0] + X_tr[:,1]) % n_cls).long()
X_te = torch.zeros(vocab*vocab, d, device=device)
y_te = torch.zeros(vocab*vocab, device=device, dtype=torch.long)
for i in range(vocab):
    for j in range(vocab):
        X_te[i*vocab+j] = torch.tensor([i,j], device=device, dtype=torch.float)
        y_te[i*vocab+j] = (i+j) % n_cls

# Method 1: Substrate (k-NN + feature discovery)
templates = {'cos': lambda x,w,b: torch.cos(x@w+b), 'abs': lambda x,w,b: torch.abs(x@w+b), 'mod2': lambda x,w,b: ((x@w.abs()).round()%2).float()}

def substrate_acc(X,y,Xte,yte,n_cls):
    def loo(V,labels):
        V_n=F.normalize(V,dim=1);sims=V_n@V_n.T;sims.fill_diagonal_(-1e9)
        scores=torch.zeros(V.shape[0],n_cls,device=device)
        for c in range(n_cls):
            m=labels==c;cs=sims[:,m]
            if cs.shape[1]==0:continue
            scores[:,c]=cs.topk(min(5,cs.shape[1]),dim=1).values.sum(dim=1)
        return(scores.argmax(1)==labels).float().mean().item()
    def knn(V,labels,te,yte):
        sims=F.normalize(te,dim=1)@F.normalize(V,dim=1).T
        scores=torch.zeros(te.shape[0],n_cls,device=device)
        for c in range(n_cls):
            m=labels==c;cs=sims[:,m]
            if cs.shape[1]==0:continue
            scores[:,c]=cs.topk(min(5,cs.shape[1]),dim=1).values.sum(dim=1)
        return(scores.argmax(1)==yte).float().mean().item()*100
    V=X.clone();layers=[]
    for _ in range(5):
        cd=V.shape[1];bl=loo(V,y);best=None
        for tn,tf in templates.items():
            for _ in range(50):
                w=torch.randn(cd,device=device)/(cd**0.5);b=torch.rand(1,device=device)*n_cls
                try:
                    feat=tf(V,w,b).unsqueeze(1);aug=F.normalize(torch.cat([V,feat],1),dim=1)
                    l=loo(aug,y)
                    if l>bl+0.003:bl=l;best=(tn,w.clone(),b.clone())
                except:pass
        if best is None:break
        tn,w,b=best;layers.append((tn,w,b))
        V=torch.cat([V,templates[tn](V,w,b).unsqueeze(1)],1)
    Vte=Xte.clone();Vtr=X.clone()
    for tn,w,b in layers:
        Vtr=torch.cat([Vtr,templates[tn](Vtr,w,b).unsqueeze(1)],1)
        Vte=torch.cat([Vte,templates[tn](Vte,w,b).unsqueeze(1)],1)
    return knn(F.normalize(Vtr,dim=1),y,F.normalize(Vte,dim=1),yte)

# Method 2: 2-layer MLP trained with backprop
def mlp_acc(X,y,Xte,yte,n_cls):
    model = nn.Sequential(nn.Linear(d,32),nn.ReLU(),nn.Linear(32,32),nn.ReLU(),nn.Linear(32,n_cls)).to(device)
    opt = optim.Adam(model.parameters(), lr=0.01)
    for epoch in range(500):
        logits = model(X)
        loss = F.cross_entropy(logits, y)
        opt.zero_grad(); loss.backward(); opt.step()
    return (model(Xte).argmax(1) == yte).float().mean().item() * 100

acc_sub = substrate_acc(X_tr, y_tr, X_te, y_te, n_cls)
acc_mlp = mlp_acc(X_tr, y_tr, X_te, y_te, n_cls)

print(f'Step 262: Substrate vs MLP on addition mod 5')
print(f'  Substrate: {acc_sub:.1f}%')
print(f'  MLP (2-layer, 500 epochs backprop): {acc_mlp:.1f}%')
print(f'  Winner: {\"SUBSTRATE\" if acc_sub > acc_mlp else \"MLP\" if acc_mlp > acc_sub else \"TIE\"}')

# Also test on parity
d2 = 8; X2 = torch.randint(0,2,(1000,d2),device=device).float()
y2 = (X2.sum(1)%2).long()
Xte2 = torch.zeros(256,d2,device=device)
for i in range(256):
    for b in range(d2): Xte2[i,b]=(i>>b)&1
yte2 = (Xte2.sum(1)%2).long()

acc_sub2 = substrate_acc(X2, y2, Xte2, yte2, 2)

model2 = nn.Sequential(nn.Linear(d2,32),nn.ReLU(),nn.Linear(32,32),nn.ReLU(),nn.Linear(32,2)).to(device)
opt2 = optim.Adam(model2.parameters(), lr=0.01)
for epoch in range(1000):
    logits = model2(X2); loss = F.cross_entropy(logits, y2)
    opt2.zero_grad(); loss.backward(); opt2.step()
acc_mlp2 = (model2(Xte2).argmax(1) == yte2).float().mean().item() * 100

print(f'\\nParity d=8:')
print(f'  Substrate: {acc_sub2:.1f}%')
print(f'  MLP (2-layer, 1000 epochs): {acc_mlp2:.1f}%')
print(f'  Winner: {\"SUBSTRATE\" if acc_sub2 > acc_mlp2 else \"MLP\" if acc_mlp2 > acc_sub2 else \"TIE\"}')
" 2>&1
