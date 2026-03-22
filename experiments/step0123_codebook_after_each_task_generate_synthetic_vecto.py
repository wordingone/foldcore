"""
Step 123 — Recovered from session JSONL (inline python -c experiment).
Original execution: foldcore k-NN / torch experiments, March 15 2026.
Source: B--M-avir-leo/0606b161, line 9924.
"""
import torch, torch.nn.functional as F, numpy as np

device = 'cuda'
data = np.load('C:/Users/Admin/cifar100_resnet18_features.npz')
X_train = F.normalize(torch.tensor(data['X_train'], device=device, dtype=torch.float32), dim=1)
y_train = torch.tensor(data['y_train'], device=device, dtype=torch.long)
X_test = F.normalize(torch.tensor(data['X_test'], device=device, dtype=torch.float32), dim=1)
y_test = torch.tensor(data['y_test'], device=device, dtype=torch.long)

n_tasks, cpt, k = 10, 10, 5
task_train, task_test = [], []
for t in range(n_tasks):
    cs, ce = t*cpt, (t+1)*cpt
    task_train.append((X_train[(y_train>=cs)&(y_train<ce)], y_train[(y_train>=cs)&(y_train<ce)]))
    task_test.append((X_test[(y_test>=cs)&(y_test<ce)], y_test[(y_test>=cs)&(y_test<ce)]))

def topk_predict(V, labels, queries, k=5):
    sims = queries @ V.T
    n_cls = labels.max().item() + 1
    scores = torch.zeros(queries.shape[0], n_cls, device=device)
    for c in range(n_cls):
        m = labels == c
        if m.sum()==0: continue
        cs = sims[:, m]
        scores[:, c] = cs.topk(min(k, cs.shape[1]), dim=1).values.sum(dim=1)
    return scores.argmax(dim=1)

# Step 123: After each task, generate synthetic vectors by interpolating
# between same-class training vectors (mixup-style density injection)
for n_synth_per_class in [0, 50, 200, 500]:
    V = torch.empty(0, 512, device=device)
    labels = torch.empty(0, dtype=torch.long, device=device)
    
    for t in range(n_tasks):
        tx, ty = task_train[t]
        V = torch.cat([V, tx]); labels = torch.cat([labels, ty])
        
        if n_synth_per_class > 0:
            # Generate synthetic vectors for THIS task's classes
            for c in range(t*cpt, (t+1)*cpt):
                cmask = labels == c
                if cmask.sum() < 2: continue
                cvecs = V[cmask]
                # Random pairs, interpolate
                n = min(n_synth_per_class, cvecs.shape[0])
                idx1 = torch.randint(0, cvecs.shape[0], (n,), device=device)
                idx2 = torch.randint(0, cvecs.shape[0], (n,), device=device)
                alpha = torch.rand(n, 1, device=device)
                synth = F.normalize(alpha * cvecs[idx1] + (1-alpha) * cvecs[idx2], dim=1)
                V = torch.cat([V, synth])
                labels = torch.cat([labels, torch.full((n,), c, device=device, dtype=torch.long)])
    
    # Final eval
    c = tot = 0
    for t2 in range(n_tasks):
        tex, tey = task_test[t2]
        preds = topk_predict(V, labels, tex, k)
        c += (preds == tey).sum().item()
        tot += tey.shape[0]
    aa = c / tot * 100
    print(f'synth={n_synth_per_class:4d}/class: AA={aa:.1f}% CB={V.shape[0]}')
" 2>&1
