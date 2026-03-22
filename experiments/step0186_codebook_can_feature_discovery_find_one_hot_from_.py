"""
Step 186 — Recovered from session JSONL (inline python -c experiment).
Original execution: foldcore k-NN / torch experiments, March 15 2026.
Source: B--M-avir-leo/0606b161, line 10797.
"""
import torch, torch.nn.functional as F
device = 'cuda'

# Step 186: Can feature discovery find one-hot from raw integers?
# State: (a, b) as raw floats. Next: (b, (a+b)%10)
# Feature discovery should find that DISCRETE identity matters, not magnitude

n_train = 500; k = 5; d = 2

states = torch.randint(0, 10, (n_train, 2), device=device).float()
next_a = states[:, 1]  # next a = current b
labels = next_a.long()  # predict next_a as classification over 10 classes

def loo_acc(V, labels, k=5):
    V_n = F.normalize(V, dim=1); sims = V_n @ V_n.T; sims.fill_diagonal_(-1e9)
    n_cls = labels.max().item() + 1
    scores = torch.zeros(V.shape[0], n_cls, device=device)
    for c in range(n_cls):
        m = labels == c; cs = sims[:, m]
        if cs.shape[1] == 0: continue
        scores[:, c] = cs.topk(min(k, cs.shape[1]), dim=1).values.sum(dim=1)
    return (scores.argmax(1) == labels).float().mean().item()

# Base: raw (a, b) features
loo_base = loo_acc(states, labels)
print(f'Task: predict next_a from (a, b)')
print(f'Base (raw floats): LOO={loo_base:.4f}')

# Feature discovery: try random nonlinear features
V = states.clone()
best_loo = loo_base
templates = {
    'cos': lambda x, w, b: torch.cos(x @ w + b),
    'mod': lambda x, w, b: ((x @ w.abs()).round() % 10).float(),
    'abs': lambda x, w, b: torch.abs(x @ w + b),
    'eq': lambda x, w, b: (x[:, int(w[0].abs().item()) % d] == w[1].round().item() % 10).float(),
}

for step in range(5):
    improved = False
    for tname, tfn in templates.items():
        for _ in range(100):
            w = torch.randn(d, device=device)
            b = torch.rand(1, device=device) * 10
            try:
                feat = tfn(states, w, b).unsqueeze(1)
                if feat.isnan().any(): continue
                aug = F.normalize(torch.cat([V, feat], 1), dim=1)
                loo = loo_acc(aug, labels)
                if loo > best_loo + 0.001:
                    best_loo = loo
                    V = torch.cat([V, feat], 1)
                    improved = True
                    print(f'  Step {step+1}: +{tname} LOO={loo:.4f} (d={V.shape[1]})')
                    break
            except: pass
        if improved: break
    if not improved: 
        print(f'  Step {step+1}: no improvement')
        break

print(f'\\nFinal LOO: {best_loo:.4f} (was {loo_base:.4f}, +{best_loo-loo_base:.4f})')
print(f'Features: d={V.shape[1]}')
" 2>&1
