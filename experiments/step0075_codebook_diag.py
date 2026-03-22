#!/usr/bin/env python3
"""Diagnose delta distribution for EigenFold on P-MNIST."""
import sys, random, math
import numpy as np

sys.path.insert(0, 'B:/M/ArtificialArchitecture/the_singularity_search/src')
from rk import madd, msub, mscale, mmul, mtanh, frob, mclip

K=4; D_MID=384; D_MAT=16; ALPHA=1.2; BETA=0.8; INIT_STEPS=20; DT=0.03; MAX_NORM=3.0

def phi(M):
    return mtanh(madd(mscale(M, ALPHA), mscale(mmul(M,M), BETA/K)))

def cross_apply(Mi, Mj):
    return mtanh(madd(mscale(madd(Mi,Mj), ALPHA/2), mscale(mmul(Mi,Mj), BETA/K)))

def eigenform_steps(M, n=INIT_STEPS):
    for _ in range(n):
        M = madd(M, mscale(msub(phi(M), M), DT))
        M = mclip(M, MAX_NORM)
    return M

def to_matrix(vec, P2):
    flat = (P2 @ vec).tolist()
    return [flat[i*K:(i+1)*K] for i in range(K)]

def make_proj1(seed=12345):
    rng = np.random.RandomState(seed)
    return (rng.randn(D_MID, 784).astype(np.float32) / math.sqrt(784))

def make_proj2(seed=99999):
    rng = np.random.RandomState(seed)
    return (rng.randn(D_MAT, D_MID).astype(np.float32) / math.sqrt(D_MID))

def make_permutation(seed):
    perm = list(range(784))
    rng = random.Random(seed)
    rng.shuffle(perm)
    return np.array(perm, dtype=np.int64)

import torchvision
train_ds = torchvision.datasets.MNIST('C:/Users/Admin/mnist_data', train=True, download=True)
X_tr = train_ds.data.numpy().reshape(-1, 784).astype(np.float32) / 255.0
y_tr = train_ds.targets.numpy()

P1 = make_proj1(); P2 = make_proj2()
perm = make_permutation(1)

# Sample 50 points and run eigenform
N = 50
idx = np.random.RandomState(42).choice(len(X_tr), N)
mats = []
for i in idx:
    x = X_tr[i]
    xp = x[perm]
    mid = P1 @ xp
    mid /= (np.linalg.norm(mid) + 1e-15)
    R = to_matrix(mid.astype(np.float32), P2)
    M = eigenform_steps(R)
    mats.append((M, int(y_tr[i])))

print(f"Matrix Frobenius norms: ", end="")
norms = [frob(M) for M,_ in mats]
print(f"min={min(norms):.3f} max={max(norms):.3f} mean={sum(norms)/len(norms):.3f}")

# Cross-apply delta between random pairs (same class vs different class)
same_deltas = []
diff_deltas = []
for i in range(N):
    for j in range(i+1, N):
        Mi, li = mats[i]; Mj, lj = mats[j]
        d = frob(msub(cross_apply(Mi, Mj), Mi))
        if li == lj:
            same_deltas.append(d)
        else:
            diff_deltas.append(d)

if same_deltas:
    print(f"Same-class delta: min={min(same_deltas):.4f} max={max(same_deltas):.4f} mean={sum(same_deltas)/len(same_deltas):.4f}")
if diff_deltas:
    print(f"Diff-class delta: min={min(diff_deltas):.4f} max={max(diff_deltas):.4f} mean={sum(diff_deltas)/len(diff_deltas):.4f}")

# Also check: what's the delta for the FIRST spawned element vs subsequent same-class
print("\nFirst 10 deltas when codebook has 1 element (element 0):")
M0, l0 = mats[0]
for i in range(1, 11):
    Mi, li = mats[i]
    d = frob(msub(cross_apply(M0, Mi), M0))
    print(f"  [{l0}] vs [{li}]: delta={d:.4f}")
