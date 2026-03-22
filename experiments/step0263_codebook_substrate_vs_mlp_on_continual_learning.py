"""
Step 263 — Recovered from session JSONL (inline python -c experiment).
Original execution: foldcore k-NN / torch experiments, March 15 2026.
Source: B--M-avir-leo/0606b161, line 12002.
"""
import torch, torch.nn.functional as F
from torch import nn, optim
device = 'cuda'

# Step 263: Substrate vs MLP on CONTINUAL LEARNING
# 5 sequential tasks, each with different rule. Eval on ALL previous tasks.

d = 8; n_per_task = 200; k = 5

rules = [
    lambda x: (x.sum(1) % 2).long(),                    # parity
    lambda x: (x[:,:4].sum(1) > 2).long(),               # majority first 4
    lambda x: (x[:,0].long() ^ x[:,1].long()),            # XOR(0,1)
    lambda x: (x[:,2] * x[:,3]).long(),                    # AND(2,3)
    lambda x: ((x.sum(1) > 4).long()),                    # threshold sum>4
]

# Substrate: append-all k-NN
V_sub = torch.empty(0, d, device=device)
y_sub = torch.empty(0, device=device, dtype=torch.long)
task_labels = torch.empty(0, device=device, dtype=torch.long)

# MLP: retrained on accumulated data
all_X = torch.empty(0, d, device=device)
all_y = torch.empty(0, device=device, dtype=torch.long)

print(f'Step 263: Substrate vs MLP on continual learning (5 tasks)')
print(f'{\"After task\":>10s} | {\"Substrate\":>9s} | {\"MLP\":>9s} | {\"MLP-fgt\":>7s}')
print(f'{\"----------\":>10s}-|-----------|-----------|-------')

for t in range(5):
    # New task data
    X_t = torch.randint(0, 2, (n_per_task, d), device=device).float()
    y_t = rules[t](X_t) + t * 2  # offset labels per task
    
    # Substrate: just append
    V_sub = torch.cat([V_sub, X_t])
    y_sub = torch.cat([y_sub, y_t])
    
    # MLP: retrain on all data
    all_X = torch.cat([all_X, X_t])
    all_y = torch.cat([all_y, y_t])
    
    n_cls = all_y.max().item() + 1
    model = nn.Sequential(nn.Linear(d,64),nn.ReLU(),nn.Linear(64,64),nn.ReLU(),nn.Linear(64,n_cls)).to(device)
    opt = optim.Adam(model.parameters(), lr=0.01)
    for epoch in range(300):
        logits = model(all_X); loss = F.cross_entropy(logits, all_y)
        opt.zero_grad(); loss.backward(); opt.step()
    
    # Eval on ALL tasks seen so far
    correct_sub = correct_mlp = total = 0
    for t2 in range(t+1):
        X_test = torch.randint(0, 2, (100, d), device=device).float()
        y_test = rules[t2](X_test) + t2 * 2
        
        # Substrate eval
        sims = F.normalize(X_test,dim=1) @ F.normalize(V_sub,dim=1).T
        scores = torch.zeros(100, n_cls, device=device)
        for c in range(n_cls):
            m = y_sub == c; cs = sims[:, m]
            if cs.shape[1] == 0: continue
            scores[:, c] = cs.topk(min(k, cs.shape[1]), dim=1).values.sum(dim=1)
        correct_sub += (scores.argmax(1) == y_test).sum().item()
        
        # MLP eval
        correct_mlp += (model(X_test).argmax(1) == y_test).sum().item()
        total += 100
    
    aa_sub = correct_sub / total * 100
    aa_mlp = correct_mlp / total * 100
    
    # MLP fine-tuned (only on latest task — measures forgetting)
    model_ft = nn.Sequential(nn.Linear(d,64),nn.ReLU(),nn.Linear(64,64),nn.ReLU(),nn.Linear(64,n_cls)).to(device)
    opt_ft = optim.Adam(model_ft.parameters(), lr=0.01)
    for epoch in range(300):
        logits = model_ft(X_t); loss = F.cross_entropy(logits, y_t)
        opt_ft.zero_grad(); loss.backward(); opt_ft.step()
    correct_ft = 0
    for t2 in range(t+1):
        X_test = torch.randint(0, 2, (100, d), device=device).float()
        y_test = rules[t2](X_test) + t2 * 2
        correct_ft += (model_ft(X_test).argmax(1) == y_test).sum().item()
    aa_ft = correct_ft / total * 100
    
    print(f'{t:10d} | {aa_sub:8.1f}% | {aa_mlp:8.1f}% | {aa_ft:6.1f}%')

print(f'\\nMLP-fgt = MLP fine-tuned on LATEST task only (catastrophic forgetting)')
" 2>&1
