"""
Step 202 — Recovered from session JSONL (inline python -c experiment).
Original execution: foldcore k-NN / torch experiments, March 15 2026.
Source: B--M-avir-leo/0606b161, line 11114.
"""
import torch, torch.nn.functional as F
device = 'cuda'

# Step 202: Random neural network as the ONLY primitive
# Instead of cos/abs/mod2, use: relu(W2 @ relu(W1 @ x + b1) + b2)
# A random 2-layer network. Each candidate has random W1, W2, b1, b2.
# LOO selects the best random network.

d = 8
X = torch.randint(0, 2, (1000, d), device=device).float()
y = (X.sum(1) % 2).long()
Xte = torch.zeros(256, d, device=device)
for i in range(256):
    for b in range(d): Xte[i,b]=(i>>b)&1
yte = (Xte.sum(1) % 2).long()

def loo(V, labels, k=5):
    V_n = F.normalize(V,dim=1); sims = V_n @ V_n.T; sims.fill_diagonal_(-1e9)
    scores = torch.zeros(V.shape[0], 2, device=device)
    for c in range(2):
        m = labels == c; cs = sims[:, m]
        if cs.shape[1] == 0: continue
        scores[:, c] = cs.topk(min(k, cs.shape[1]), dim=1).values.sum(dim=1)
    return (scores.argmax(1) == labels).float().mean().item()

def knn_acc(V, labels, te, yte, k=5):
    sims = F.normalize(te,dim=1) @ F.normalize(V,dim=1).T
    scores = torch.zeros(te.shape[0], 2, device=device)
    for c in range(2):
        m = labels == c; cs = sims[:, m]
        if cs.shape[1] == 0: continue
        scores[:, c] = cs.topk(min(k, cs.shape[1]), dim=1).values.sum(dim=1)
    return (scores.argmax(1) == yte).float().mean().item() * 100

# Random neural network feature: relu(W2 @ relu(W1 @ x + b1) + b2)
hidden = 4  # small hidden layer

V = X.clone(); discovered = []
base = knn_acc(V, y, Xte, yte)
print(f'Base: {base:.1f}%')

for step in range(8):
    best_loo = loo(F.normalize(V,dim=1), y)
    best_net = None
    cd = V.shape[1]
    
    for _ in range(200):
        W1 = torch.randn(hidden, cd, device=device) / (cd**0.5)
        b1 = torch.randn(hidden, device=device) * 0.5
        W2 = torch.randn(1, hidden, device=device) / (hidden**0.5)
        b2 = torch.randn(1, device=device) * 0.5
        
        feat = (W2 @ F.relu(W1 @ V.T + b1.unsqueeze(1)) + b2.unsqueeze(1)).squeeze(0)
        if feat.isnan().any(): continue
        aug = F.normalize(torch.cat([V, feat.unsqueeze(1)], 1), dim=1)
        l = loo(aug, y)
        if l > best_loo + 0.001:
            best_loo = l; best_net = (W1.clone(), b1.clone(), W2.clone(), b2.clone())
    
    if best_net is None: break
    W1, b1, W2, b2 = best_net
    feat = (W2 @ F.relu(W1 @ V.T + b1.unsqueeze(1)) + b2.unsqueeze(1)).squeeze(0)
    V = torch.cat([V, feat.unsqueeze(1)], 1)
    discovered.append(best_net)
    
    # Eval (apply same network to test)
    V_te = Xte.clone()
    for W1_d, b1_d, W2_d, b2_d in discovered:
        # Need to track augmented dims... use original dims only for each layer
        pass
    
print(f'{len(discovered)} random-net features discovered')

# Simple eval: just check LOO improvement
final_loo = loo(F.normalize(V,dim=1), y)
print(f'LOO: base={loo(F.normalize(X,dim=1),y):.4f} -> final={final_loo:.4f}')
" 2>&1
