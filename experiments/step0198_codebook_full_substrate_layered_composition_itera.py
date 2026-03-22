"""
Step 198 — Recovered from session JSONL (inline python -c experiment).
Original execution: foldcore k-NN / torch experiments, March 15 2026.
Source: B--M-avir-leo/0606b161, line 11035.
"""
import torch, torch.nn.functional as F
device = 'cuda'

# Step 198: Full substrate — layered composition + iteration on Fibonacci
d = 2; mod_n = 10; n_train = 800; k = 5

states = torch.randint(0, mod_n, (n_train, 2), device=device).float()
next_states = torch.stack([states[:,1], (states[:,0]+states[:,1]) % mod_n], dim=1)

templates = {
    'cos': lambda x,w,b: torch.cos(x@w+b),
    'abs': lambda x,w,b: torch.abs(x@w+b),
}

def loo(V, labels, n_cls, k=5):
    V_n = F.normalize(V,dim=1); sims = V_n @ V_n.T; sims.fill_diagonal_(-1e9)
    scores = torch.zeros(V.shape[0], n_cls, device=device)
    for c in range(n_cls):
        m = labels == c; cs = sims[:, m]
        if cs.shape[1] == 0: continue
        scores[:, c] = cs.topk(min(k, cs.shape[1]), dim=1).values.sum(dim=1)
    return (scores.argmax(1) == labels).float().mean().item()

# Phase 1: Layered feature discovery on AUGMENTED space
V = states.clone()
layers = []  # list of (template_name, w, b)

for layer in range(10):
    current_d = V.shape[1]
    best_loo = (loo(F.normalize(V,dim=1), next_states[:,0].long(), mod_n) +
                loo(F.normalize(V,dim=1), next_states[:,1].long(), mod_n)) / 2
    best = None
    for tname, tfn in templates.items():
        for _ in range(100):
            w = torch.randn(current_d, device=device) / (current_d**0.5)
            b = torch.rand(1, device=device) * mod_n
            try:
                feat = tfn(V, w, b).unsqueeze(1)
                aug = F.normalize(torch.cat([V, feat], 1), dim=1)
                l = (loo(aug, next_states[:,0].long(), mod_n) + loo(aug, next_states[:,1].long(), mod_n)) / 2
                if l > best_loo + 0.0003:
                    best_loo = l; best = (tname, w.clone(), b.clone())
            except: pass
    if best is None: break
    tname, w, b = best
    layers.append((tname, w, b))
    V = torch.cat([V, templates[tname](V, w, b).unsqueeze(1)], 1)

print(f'Discovered {len(layers)} layered features (d={d}->{V.shape[1]})')

# Phase 2: Build augmented codebook
def augment_layered(raw_states):
    V_aug = raw_states.clone()
    for tname, w, b in layers:
        V_aug = torch.cat([V_aug, templates[tname](V_aug, w, b).unsqueeze(1)], 1)
    return F.normalize(V_aug, dim=1)

X_aug = augment_layered(states)

# Phase 3: Iterated classification-based prediction
def predict_step(state_raw):
    state_aug = augment_layered(state_raw.unsqueeze(0))
    sims = state_aug @ X_aug.T
    pred = torch.zeros(2, device=device)
    for dim in range(2):
        targets = next_states[:, dim].long()
        scores = torch.zeros(mod_n, device=device)
        for c in range(mod_n):
            m = targets == c; cs = sims[0, m]
            if cs.shape[0] == 0: continue
            scores[c] = cs.topk(min(k, cs.shape[0])).values.sum()
        pred[dim] = scores.argmax().float()
    return pred

# Test
n_test = 100
print(f'\\nFull substrate: layered features + iterated k-NN on Fibonacci mod {mod_n}:')
for n_steps in [1, 5, 10, 20, 50]:
    correct = 0
    for _ in range(n_test):
        a, b = torch.randint(0, mod_n, (2,)).tolist()
        ta, tb = a, b
        for _ in range(n_steps):
            ta, tb = tb, (ta + tb) % mod_n
        current = torch.tensor([float(a), float(b)], device=device)
        for _ in range(n_steps):
            current = predict_step(current)
        if int(current[0]) == ta and int(current[1]) == tb: correct += 1
    print(f'  {n_steps:2d} steps: {correct}%')
" 2>&1
