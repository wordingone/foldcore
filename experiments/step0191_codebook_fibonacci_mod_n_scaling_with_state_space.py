"""
Step 191 — Recovered from session JSONL (inline python -c experiment).
Original execution: foldcore k-NN / torch experiments, March 15 2026.
Source: B--M-avir-leo/0606b161, line 10879.
"""
import torch, torch.nn.functional as F
device = 'cuda'

# Step 191: Fibonacci mod N — scaling with state space size
d = 2; k = 5

for mod_n in [5, 10, 20, 50]:
    n_states = mod_n * mod_n
    n_train = min(5000, n_states * 5)
    
    states = torch.randint(0, mod_n, (n_train, 2), device=device).float()
    next_states = torch.stack([states[:,1], (states[:,0]+states[:,1]) % mod_n], dim=1)
    
    V = states.clone(); discovered = []
    for step in range(30):
        def loo(V_aug, labels):
            V_n = F.normalize(V_aug,dim=1); sims = V_n @ V_n.T; sims.fill_diagonal_(-1e9)
            scores = torch.zeros(V_aug.shape[0], mod_n, device=device)
            for c in range(mod_n):
                m = labels == c; cs = sims[:, m]
                if cs.shape[1] == 0: continue
                scores[:, c] = cs.topk(min(k, cs.shape[1]), dim=1).values.sum(dim=1)
            return (scores.argmax(1) == labels).float().mean().item()
        
        best_loo = (loo(F.normalize(V,dim=1), next_states[:,0].long()) + loo(F.normalize(V,dim=1), next_states[:,1].long())) / 2
        best_w = None; best_b = None
        for _ in range(150):
            w = torch.randn(d, device=device); b = torch.rand(1, device=device) * mod_n
            feat = torch.cos(states @ w + b).unsqueeze(1)
            aug = F.normalize(torch.cat([V, feat], 1), dim=1)
            l = (loo(aug, next_states[:,0].long()) + loo(aug, next_states[:,1].long())) / 2
            if l > best_loo + 0.0003:
                best_loo = l; best_w = w.clone(); best_b = b.clone()
        if best_w is None: break
        discovered.append((best_w, best_b))
        V = torch.cat([V, torch.cos(states @ best_w + best_b).unsqueeze(1)], 1)
    
    def augment(raw):
        aug = raw.clone() if raw.dim() == 2 else raw.unsqueeze(0)
        for w, b in discovered:
            aug = torch.cat([aug, torch.cos(aug[:,:d] @ w + b).unsqueeze(1)], 1)
        return F.normalize(aug, dim=1)
    
    X_tr_aug = augment(states)
    
    def predict(state_raw):
        state_aug = augment(state_raw.unsqueeze(0))
        sims = state_aug @ X_tr_aug.T
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
    
    # Test 10-step iteration
    correct = 0; n_test = 50
    for _ in range(n_test):
        a, b = torch.randint(0, mod_n, (2,)).tolist()
        ta, tb = a, b
        for _ in range(10):
            ta, tb = tb, (ta + tb) % mod_n
        current = torch.tensor([float(a), float(b)], device=device)
        for _ in range(10):
            current = predict(current)
        if int(current[0]) == ta and int(current[1]) == tb: correct += 1
    
    print(f'mod={mod_n:3d} ({n_states:5d} states): {len(discovered)} feats, 10-step={correct}/{n_test} ({correct/n_test*100:.0f}%)')
" 2>&1
