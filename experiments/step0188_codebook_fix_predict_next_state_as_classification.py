"""
Step 188 — Recovered from session JSONL (inline python -c experiment).
Original execution: foldcore k-NN / torch experiments, March 15 2026.
Source: B--M-avir-leo/0606b161, line 10823.
"""
import torch, torch.nn.functional as F
device = 'cuda'

# Step 188: Fix — predict next state as CLASSIFICATION, not regression
# For each component (a, b) of the state, classify into {0..9}

d = 2; n_train = 500; k = 5

states = torch.randint(0, 10, (n_train, 2), device=device).float()
next_states = torch.stack([states[:, 1], (states[:, 0] + states[:, 1]) % 10], dim=1)

# Discover features (same as Step 187)
labels_a = next_states[:, 0].long()
V = states.clone()
discovered = []
for step in range(5):
    def loo_acc(V_aug, labels):
        V_n = F.normalize(V_aug, dim=1); sims = V_n @ V_n.T; sims.fill_diagonal_(-1e9)
        scores = torch.zeros(V_aug.shape[0], 10, device=device)
        for c in range(10):
            m = labels == c; cs = sims[:, m]
            if cs.shape[1] == 0: continue
            scores[:, c] = cs.topk(min(k, cs.shape[1]), dim=1).values.sum(dim=1)
        return (scores.argmax(1) == labels).float().mean().item()
    
    best_w = None; best_b = None; best_loo = loo_acc(F.normalize(V, dim=1), labels_a)
    for _ in range(200):
        w = torch.randn(d, device=device); b = torch.rand(1, device=device) * 10
        feat = torch.cos(states @ w + b).unsqueeze(1)
        aug = F.normalize(torch.cat([V, feat], 1), dim=1)
        loo = loo_acc(aug, labels_a)
        if loo > best_loo + 0.001:
            best_loo = loo; best_w = w.clone(); best_b = b.clone()
    if best_w is None: break
    discovered.append((best_w, best_b))
    V = torch.cat([V, torch.cos(states @ best_w + best_b).unsqueeze(1)], 1)

print(f'Discovered {len(discovered)} features')

# Augment function
def augment(raw):
    aug = raw.clone() if raw.dim() == 2 else raw.unsqueeze(0)
    for w, b in discovered:
        aug = torch.cat([aug, torch.cos(aug[:, :d] @ w + b).unsqueeze(1)], 1)
    return F.normalize(aug, dim=1)

X_tr_aug = augment(states)

# CLASSIFICATION-based iterated prediction
def predict_step_classify(state_raw):
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

# Test iterated classification
n_test = 100
print(f'Iterated CLASSIFICATION k-NN + discovered features (Fibonacci mod 10):')
for n_steps in [1, 5, 10, 20]:
    correct = 0
    for _ in range(n_test):
        a, b = torch.randint(0, 10, (2,)).tolist()
        true_a, true_b = a, b
        for _ in range(n_steps):
            true_a, true_b = true_b, (true_a + true_b) % 10
        
        current = torch.tensor([float(a), float(b)], device=device)
        for _ in range(n_steps):
            current = predict_step_classify(current)
        
        if int(current[0].item()) == true_a and int(current[1].item()) == true_b:
            correct += 1
    
    print(f'  {n_steps:2d} steps: {correct}% correct')
" 2>&1
