"""
Step 135 — Recovered from session JSONL (inline python -c experiment).
Original execution: foldcore k-NN / torch experiments, March 15 2026.
Source: B--M-avir-leo/0606b161, line 10060.
"""
import torch, torch.nn.functional as F, numpy as np

device = 'cuda'
data = np.load('C:/Users/Admin/cifar100_resnet18_features.npz')
X_train = F.normalize(torch.tensor(data['X_train'], device=device, dtype=torch.float32), dim=1)
y_train = torch.tensor(data['y_train'], device=device, dtype=torch.long)

# Step 135: Does the codebook discover STRUCTURE autonomously?
# Without labels, can cosine-based clustering find CIFAR-100 superclasses?

# Unsupervised: cluster by iterative prototype refinement
# Start with random prototypes, assign nearest, recompute centroids
n_clusters = 20  # CIFAR-100 has 20 superclasses of 5 classes each
n_samples = 10000

X = X_train[:n_samples]
Y = y_train[:n_samples]
superclass = Y // 5  # ground truth superclass

# K-means in cosine space
centroids = F.normalize(torch.randn(n_clusters, 512, device=device), dim=1)

for iteration in range(50):
    # Assign
    sims = X @ centroids.T
    assignments = sims.argmax(dim=1)
    
    # Recompute centroids
    new_centroids = torch.zeros_like(centroids)
    for c in range(n_clusters):
        m = assignments == c
        if m.sum() > 0:
            new_centroids[c] = F.normalize(X[m].mean(0), dim=0)
        else:
            new_centroids[c] = F.normalize(torch.randn(512, device=device), dim=0)
    centroids = new_centroids

# Measure: do discovered clusters align with superclasses?
# Compute normalized mutual information
from collections import Counter

def nmi(labels_a, labels_b):
    '''Simplified NMI computation'''
    n = len(labels_a)
    ca = Counter(labels_a.tolist())
    cb = Counter(labels_b.tolist())
    
    # Entropy
    def H(counter):
        total = sum(counter.values())
        return -sum(c/total * np.log(c/total + 1e-10) for c in counter.values())
    
    ha = H(ca)
    hb = H(cb)
    
    # Joint entropy
    joint = Counter(zip(labels_a.tolist(), labels_b.tolist()))
    hab = H(joint)
    
    # MI = H(A) + H(B) - H(A,B)
    mi = ha + hb - hab
    
    return 2 * mi / (ha + hb + 1e-10)

score = nmi(assignments.cpu(), superclass.cpu())
score_class = nmi(assignments.cpu(), Y.cpu())

# Also: how pure are the clusters?
purities = []
for c in range(n_clusters):
    m = assignments == c
    if m.sum() == 0: continue
    classes_in_cluster = Y[m]
    most_common = classes_in_cluster.mode().values.item()
    purity = (classes_in_cluster == most_common).float().mean().item()
    purities.append(purity)

print(f'K-means cosine (k={n_clusters}, 50 iterations):')
print(f'  NMI with superclasses (20): {score:.4f}')
print(f'  NMI with classes (100):     {score_class:.4f}')
print(f'  Mean cluster purity:        {np.mean(purities):.4f}')
print(f'  Cluster sizes: {[int((assignments==c).sum()) for c in range(n_clusters)]}')

# Compare: random assignment NMI
random_assign = torch.randint(0, 20, (n_samples,))
print(f'  Random assignment NMI:      {nmi(random_assign, superclass.cpu()):.4f}')
" 2>&1
