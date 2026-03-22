"""
Step 189 — Recovered from session JSONL (inline python -c experiment).
Original execution: foldcore k-NN / torch experiments, March 15 2026.
Source: B--M-avir-leo/0606b161, line 10838.
"""
import torch, torch.nn.functional as F
device = 'cuda'

# Step 189: More features — 20 rounds of discovery, then iterated prediction
d = 2; n_train = 500; k = 5

states = torch.randint(0, 10, (n_train, 2), device=device).float()
next_states = torch.stack([states[:, 1], (states[:, 0] + states[:, 1]) % 10], dim=1)

# Also predict next_b to have a richer signal
labels_b = next_states[:, 1].long()

V = states.clone()
discovered = []
for step in range(20):
    def loo_acc(V_aug, labels):
        V_n = F.normalize(V_aug, dim=1); sims = V_n @ V_n.T; sims.fill_diagonal_(-1e9)
        scores = torch.zeros(V_aug.shape[0], 10, device=device)
        for c in range(10):
            m = labels == c; cs = sims[:, m]
            if cs.shape[1] == 0: continue
            scores[:, c] = cs.topk(min(k, cs.shape[1]), dim=1).values.sum(dim=1)
        return (scores.argmax(1) == labels).float().mean().item()
    
    best_w = None; best_b_val = None
    # Use BOTH labels to discover features useful for BOTH components
    best_loo = (loo_acc(F.normalize(V, dim=1), next_states[:,0].long()) + 
                loo_acc(F.normalize(V, dim=1), next_states[:,1].long())) / 2
    
    for _ in range(100):
        w = torch.randn(d, device=device); b = torch.rand(1, device=device) * 10
        feat = torch.cos(states @ w + b).unsqueeze(1)
        aug = F.normalize(torch.cat([V, feat], 1), dim=1)
        loo = (loo_acc(aug, next_states[:,0].long()) + loo_acc(aug, next_states[:,1].long())) / 2
        if loo > best_loo + 0.0005:
            best_loo = loo; best_w = w.clone(); best_b_val = b.clone()
    
    if best_w is None: break
    discovered.append((best_w, best_b_val))
    V = torch.cat([V, torch.cos(states @ best_w + best_b_val).unsqueeze(1)], 1)

print(f'Discovered {len(discovered)} features (d={V.shape[1]})')

# Augment + classify
def augment(raw):
    aug = raw.clone() if raw.dim() == 2 else raw.unsqueeze(0)
    for w, b in discovered:
        aug = torch.cat([aug, torch.cos(aug[:, :d] @ w + b).unsqueeze(1)], 1)
    return F.normalize(aug, dim=1)

X_tr_aug = augment(states)

def predict_classify(state_raw):
    state_aug = augment(state_raw.unsqueeze(0))
    sims = state_aug @ X_tr_aug.T
    pred = torch.zeros(2, device=device)
    for dim in range(2):
        targets = next_states[:, dim].long()
        scores = torch.zeros(10, device=device)
        for c in range(10):
            m = targets == c; cs = sims[0, m]
            if cs.shape[0] == 0: continue
            scores[c] = cs.topk(min(k, cs.shape[0])).values.sum()
        pred[dim] = scores.argmax().float()
    return pred

n_test = 100
for n_steps in [1, 5, 10, 20]:
    correct = 0
    for _ in range(n_test):
        a, b = torch.randint(0, 10, (2,)).tolist()
        true_a, true_b = a, b
        for _ in range(n_steps):
            true_a, true_b = true_b, (true_a + true_b) % 10
        current = torch.tensor([float(a), float(b)], device=device)
        for _ in range(n_steps):
            current = predict_classify(current)
        if int(current[0]) == true_a and int(current[1]) == true_b:
            correct += 1
    print(f'{n_steps:2d} steps: {correct}%')
" 2>&1
