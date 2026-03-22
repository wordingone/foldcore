"""
Step 187 — Recovered from session JSONL (inline python -c experiment).
Original execution: foldcore k-NN / torch experiments, March 15 2026.
Source: B--M-avir-leo/0606b161, line 10820.
"""
import torch, torch.nn.functional as F
device = 'cuda'

# Step 187: END-TO-END — discover features, then iterate with discovered features
# Fibonacci mod 10 with RAW integer inputs + feature discovery + iteration

d = 2; n_train = 500; k = 5

# Training data: (a, b) -> (b, (a+b)%10)
states = torch.randint(0, 10, (n_train, 2), device=device).float()
next_states = torch.stack([states[:, 1], (states[:, 0] + states[:, 1]) % 10], dim=1)

# Phase 1: Discover features that help predict next state
# For simplicity, predict next_a (= current b) — the easier component
labels = next_states[:, 0].long()

V = states.clone()
discovered = []
for step in range(10):
    def loo_acc(V_aug, labels):
        V_n = F.normalize(V_aug, dim=1); sims = V_n @ V_n.T; sims.fill_diagonal_(-1e9)
        n_cls = labels.max().item() + 1
        scores = torch.zeros(V_aug.shape[0], n_cls, device=device)
        for c in range(n_cls):
            m = labels == c; cs = sims[:, m]
            if cs.shape[1] == 0: continue
            scores[:, c] = cs.topk(min(k, cs.shape[1]), dim=1).values.sum(dim=1)
        return (scores.argmax(1) == labels).float().mean().item()
    
    best_w = None; best_b = None; best_loo = loo_acc(F.normalize(V, dim=1), labels)
    for _ in range(200):
        w = torch.randn(d, device=device)
        b = torch.rand(1, device=device) * 10
        feat = torch.cos(states @ w + b).unsqueeze(1)
        aug = F.normalize(torch.cat([V, feat], 1), dim=1)
        loo = loo_acc(aug, labels)
        if loo > best_loo + 0.001:
            best_loo = loo; best_w = w.clone(); best_b = b.clone()
    
    if best_w is None: break
    discovered.append((best_w, best_b))
    V = torch.cat([V, torch.cos(states @ best_w + best_b).unsqueeze(1)], 1)

print(f'Discovered {len(discovered)} features. LOO: {best_loo:.4f}')

# Phase 2: Augment all states with discovered features
def augment(s):
    aug = s.clone() if s.dim() == 2 else s.unsqueeze(0)
    for w, b in discovered:
        aug = torch.cat([aug, torch.cos(aug[:, :d] @ w + b).unsqueeze(1)], 1)
    return aug

X_tr_aug = F.normalize(augment(states), dim=1)
y_tr_aug = F.normalize(augment(next_states), dim=1)

# Phase 3: Iterate — predict next state using augmented k-NN
def predict_step(state_raw):
    state_aug = F.normalize(augment(state_raw.unsqueeze(0)), dim=1)
    sims = state_aug @ X_tr_aug.T
    topk_idx = sims[0].topk(min(k, sims.shape[1])).indices
    # Average top-k next states (in raw space, not augmented)
    avg_next = next_states[topk_idx].mean(0)
    # Round to nearest integer
    return avg_next.round().clamp(0, 9)

# Test
n_test = 100
print(f'\\nIterated k-NN with discovered features (Fibonacci mod 10):')
for n_steps in [1, 5, 10, 20]:
    correct = 0
    for _ in range(n_test):
        a, b = torch.randint(0, 10, (2,)).tolist()
        true_a, true_b = a, b
        for _ in range(n_steps):
            true_a, true_b = true_b, (true_a + true_b) % 10
        
        current = torch.tensor([float(a), float(b)], device=device)
        for _ in range(n_steps):
            current = predict_step(current)
        
        if int(current[0].item()) == true_a and int(current[1].item()) == true_b:
            correct += 1
    
    print(f'  {n_steps:2d} steps: {correct}% correct')
" 2>&1
