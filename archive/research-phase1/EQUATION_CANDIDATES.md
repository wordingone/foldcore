# Candidate Equations for the Atomic Foundation

*[researcher], 2026-03-14. Raw exploration. Three categories: clean-slate, divergent, and evolutionary.*

---

## Category A: Clean Slate (Ignore Current FluxCore Entirely)

### A1. The Field Equation

Forget prototypes. The fundamental object is a continuous field F: R^d → R over the input space. The field value at any point represents "how much the system knows about this region."

```
Learning:     F(x) += delta(x - r)              # spike at input location
Retrieval:    label(r) = argmax_c integral F_c(x) * K(x, r) dx   # kernel-weighted vote per class
Generation:   r_{t+1} = argmax_x F(x) * K(x, r_t)               # follow the field gradient
Uncertainty:  U(r) = 1 / F(r)                   # low field = uncertain
```

The field grows by accumulating evidence. No prototypes, no vectors, no discrete entities. The codebook is the discretized approximation of this field (each vector is a delta function). But the FIELD is the fundamental object.

Discretized (practical): F is represented by weighted particles {(v_i, w_i)}. Each particle is a location v_i with weight w_i. This IS a codebook — but with weights. The equation becomes:

```
F(r) = sum_i w_i * K(v_i, r)     # kernel density estimate
w_i += K(v_i, r)                  # input reinforces nearby particles
spawn v_new if F(r) < threshold   # low-density region gets a new particle
```

K = exp(cos(v, r) / tau) is the kernel. This is kernel density estimation as a learning algorithm.

### A2. The Transformation Lattice

Each memory is a matrix M_i in R^(d x d) — a transformation, not a point. The system stores OPERATIONS, not exemplars.

```
Store:     M_new = r_out @ r_in^T / ||r_in||^2    # outer product: maps input to output
Retrieve:  output = (sum_i alpha_i * M_i) @ r      # weighted blend of transformations
           alpha_i = softmax(||M_i @ r|| / tau)     # weight by how well each transform fits
Learn:     M_i += lr * (target - M_i @ r) @ r^T    # Widrow-Hoff on each transformation
Spawn:     if max_i ||M_i @ r|| < threshold: store new M
```

The system learns input→output MAPPINGS, not input locations. Classification: M_i maps input to class logits. Generation: M_i maps a seed to the next state. Memory: M_i IS the learned function. Each transformation is simultaneously a weight, an architecture element, and a computation.

Cost: each M_i is d×d. At d=512, that's 262K parameters per memory. Expensive — but GPU handles it. With 1000 transformations: 262M parameters. Comparable to a small transformer.

### A3. The Resonance Equation

One equation that oscillates between two modes naturally:

```
z(t+1) = z(t) + dt * [alpha * phi(z) - beta * grad_E(z) + gamma * r(t)]
```

where:
- z is the system state (not input, not output — the system itself)
- phi(z) = autonomous dynamics (eigenform-like, drives toward attractor)
- grad_E(z) = energy gradient (drives toward stored patterns)
- r(t) = input injection (perception)
- alpha, beta, gamma gate the three forces

When input is present (gamma > 0): perception dominates, z moves toward input → LEARNING
When input is absent (gamma = 0): phi and E compete → GENERATION
When z is near a stored pattern: grad_E ≈ 0, phi dominates → STABLE MEMORY
When z is far from all patterns: grad_E is large → RETRIEVAL (convergence to nearest attractor)

The energy E is defined by the stored patterns:
```
E(z) = -log sum_i exp(cos(z, p_i) / tau)    # Hopfield energy over patterns p_i
```

Patterns p_i are themselves updated by the dynamics:
```
p_i += lr * softmax_i(cos(p_i, z) / tau) * (z - p_i)   # soft-competitive update
```

The resonance: z and {p_i} co-evolve. z settles toward patterns, patterns drift toward z. When z AND p converge simultaneously (z ≈ p_winner), the system is in resonance — stable state. When they diverge, the system is in search mode.

This is ART's resonance concept expressed as a continuous dynamical system with a Hopfield energy landscape.

### A4. The Compression Operator

The atomic operation is COMPRESSION itself. The system takes raw input and compresses it into its existing memory, or expands memory if compression fails.

```
Given input r:
1. Compress:  r_hat = Decode(Encode(r, memory))     # reconstruct r from memory
2. Error:     e = ||r - r_hat||                       # reconstruction error
3. If e < threshold:
     memory = Update(memory, r)                       # memory absorbs r (compression succeeds)
4. If e >= threshold:
     memory = Expand(memory, r)                       # memory grows (new pattern)
```

The system IS an autoencoder whose codebook is the latent space. Learning = improving compression. Forgetting = losing compression fidelity on old data. Generation = decoding from random latent points. Uncertainty = reconstruction error.

The "fold equation" would be the Update step: how does memory change when it absorbs an input? The Expand step: how does memory grow?

Concrete version (codebook as compressed memory):
```
Encode(r, V) = softmax(V @ r / tau)        # soft address in memory (attention)
Decode(a, V) = V^T @ a                      # reconstruct from soft address
Update: V += lr * (r - V^T @ softmax(V @ r / tau)) @ softmax(V @ r / tau)^T
```

This is dictionary learning / sparse coding with online updates. The update rule minimizes reconstruction error. Known — but the framing as "compression IS learning" connects to [director]'s thesis.

### A5. The Self-Referential Equation

The system applies itself to itself:

```
S(t+1) = F(S(t), S(t))
```

where S is the entire system state and F is a fixed operation. The system IS its own input. Each step, the system processes itself and produces the next version of itself.

For a codebook: S = {v_1, ..., v_n}. F computes pairwise interactions:

```
v_i(t+1) = normalize(v_i(t) + lr * sum_j w_ij * (v_j(t) - v_i(t)))
w_ij = softmax(cos(v_i, v_j) / tau)
```

Without input, this is a self-organizing system — prototypes rearrange based on their mutual relationships. With input:

```
v_i(t+1) = normalize(v_i(t) + lr * [sum_j w_ij * (v_j - v_i) + eta * w_ir * (r - v_i)])
w_ir = softmax(cos(v_i, r) / tau)
```

Input is treated as "another prototype" that participates in the same dynamics. The system doesn't distinguish between learning from data and reorganizing itself. Both are the same operation: pairwise interaction that minimizes an energy.

---

## Category B: Different Direction from FluxCore + RK

### B1. Prototype Matrices (State IS Transformation, Revisited)

Keep the codebook structure but make each entry a small matrix instead of a vector:

```
Codebook: {M_1, ..., M_n} where M_i in R^(k x k), living on the Stiefel manifold

Similarity:  s_i = ||M_i @ r_projected||     # how well M_i transforms the input
             (not cosine of vectors — magnitude of transformed output)

Update:      M_winner += lr * (r_out @ r_in^T)   # outer product update (Hebbian)
             M_winner = orthogonalize(M_winner)    # stay on Stiefel manifold

Classify:    output = M_winner @ r_projected       # APPLY the transformation
             label = argmax(output)                 # the output IS the classification

Spawn:       if max_i s_i < threshold: add new M = r_out @ r_in^T / ||...||
Generate:    r_{t+1} = M_composite @ r_t where M_composite = product of top-k matrices
```

This IS the RK thesis — state IS transformation — but applied to the codebook. Each memory is a learned function. Retrieval isn't "find the nearest point" but "find the transformation that best processes this input." The matrix product (composite) gives generation.

Key difference from current FluxCore: the matrix cells ARE the codebook entries. No separation. The "many-to-few" routing disappears because there's one set of entities.

Hyperparameters: k (matrix size), lr, spawn_thresh, merge_thresh. Four total.

### B2. Dual Codebook — Keys and Values

Separate what the system MATCHES on from what it STORES:

```
Keys:    K = {k_1, ..., k_n} in R^d     # what to match against (input geometry)
Values:  V = {v_1, ..., v_n} in R^d     # what to retrieve (output geometry)

Retrieve:  output = sum_i softmax(cos(k_i, r) / tau) * v_i   # attention: match on keys, read values

Learn keys:    k_i += lr * grad_k [cos(k_i, r)]   # keys learn to match relevant inputs
Learn values:  v_i += lr * (target - output)        # values learn to produce correct output

Spawn:  if max cos(k_i, r) < thresh: add k_new = r, v_new = target
```

This is literally the KV cache of a transformer — but adaptive. Keys and values co-evolve online. No frozen architecture. No backprop through layers. The system IS an attention head with growing memory.

The fold equation becomes TWO coupled updates: keys track input geometry, values track desired output. Current FluxCore collapses keys and values into one vector (the codebook entry is both what you match and what you retrieve). Separating them gives more expressive power at the cost of 2x memory.

### B3. Hierarchical Fold

Multiple codebook layers, each compressing the one below:

```
Layer 0 (raw):     V_0, spawn_thresh_0 = 0.95  (fine-grained, many vectors)
Layer 1 (abstract): V_1, spawn_thresh_1 = 0.7   (coarse, fewer vectors)
Layer 2 (concept):  V_2, spawn_thresh_2 = 0.3   (very coarse, few vectors)

Forward pass:
  winners_0 = softmax(V_0 @ r / tau)           # attend to raw prototypes
  r_1 = V_0^T @ winners_0                       # compress to layer-1 input
  winners_1 = softmax(V_1 @ r_1 / tau)         # attend at abstract level
  r_2 = V_1^T @ winners_1                       # compress to concept level
  output = V_2^T @ softmax(V_2 @ r_2 / tau)    # final output

Each layer spawns/merges/updates independently.
```

This is a deep fold — multiple layers of codebook attention. Each layer compresses the representation. The hierarchy emerges from spawn thresholds: fine-grained at bottom, coarse at top. No designed architecture — the hierarchy self-organizes based on data statistics.

### B4. The Oscillating Kernel

CPU and GPU execute different phases of one equation:

```
PERCEPTION (GPU, parallel, fast):
  sims = V @ R_batch^T                    # all similarities at once
  winners = sims.argmax(dim=0)             # batch winner selection
  V[winners] += lr * R_batch               # batch update
  V[winners] = normalize(V[winners])       # batch normalize
  spawns = (sims.max(dim=0) < thresh)      # batch spawn decisions

REFLECTION (CPU, sequential, deep):
  for i in range(n_codebook):
    neighbors = top_k_similar(V, V[i], k=5)
    if all labels in neighbors are same:
      V[i] = average(V[neighbors])          # consolidate (merge region)
    if error[i] > split_thresh:
      V_new = V[i] + noise; spawn(V_new)    # split bad prototype
    importance[i] = use_count[i] * mean_accuracy[i]
    lr[i] = base_lr / (1 + importance[i])   # stable prototypes resist drift

The system oscillates: perceive a batch (GPU), reflect on what was learned (CPU), perceive the next batch.
```

Perception is fast, wide, data-parallel. Reflection is slow, deep, structure-aware. The oscillation IS the computation — neither phase alone is sufficient. Perception without reflection drifts. Reflection without perception stagnates.

### B5. The Fold as Differential Equation

Express the codebook dynamics as a continuous ODE, not discrete updates:

```
dv_i/dt = -grad_v_i E(V, r)    for all i simultaneously

where E(V, r) = -log sum_j exp(cos(v_j, r) / tau)     # Hopfield energy
              + lambda * sum_{j != i} max(0, cos(v_i, v_j) - merge_thresh)^2   # repulsion between similar vectors
              + mu * sum_j ||v_j||^2                    # sphere constraint (Lagrange)

Spawn condition: E(V, r) > E_thresh   (energy too high = input not covered)
Merge condition: cos(v_i, v_j) > merge_thresh AND label_i == label_j
```

The entire system is gradient flow on one energy function. Every update provably decreases E. The energy landscape IS the system. Learning = reshaping the landscape. Retrieval = rolling downhill. Generation = following autonomous flow when r is absent. Uncertainty = height of E at current position.

This is the most mathematically principled formulation. It inherits convergence guarantees from gradient flow theory. The ODE can be solved with any integrator (Euler = current discrete step, RK4 = higher accuracy, adaptive = variable step size).

---

## Category C: Direct Evolution of Current FluxCore

### C1. Gradient-Aware Codebook (Current Direction)

The fold equation with label-conditional direction:

```
v_winner += lr * (prob_winner - target_winner) * perp(r, v_winner)
v_winner = normalize(v_winner)

where perp(r, v) = r - v * cos(r, v)     # perpendicular component (tangent space on S^{d-1})
```

Step 71 result: +1.2pp. Modest. The equation is correct but the bottleneck is elsewhere (readout).

### C2. Soft Readout (Step 72, In Progress)

Replace hard classify with attention:

```
classify(r) = argmax_c sum_{i: label_i = c} softmax(cos(v_i, r) / tau)
```

This is the minimum change that tests "fold IS attention." If tau matters: the codebook's energy landscape has structure that 1-NN ignores.

### C3. Full Evolution (Combine C1 + C2 + Error Accumulation)

```
# Learning step:
sims = V @ r                                           # (N,) cosine similarities
probs = softmax(sims / tau)                            # (N,) soft attention weights
error_signal = probs - one_hot(true_label, labels)     # (N,) per-prototype error
V -= lr * outer(error_signal, perp(r, V))              # gradient step on tangent bundle
V = row_normalize(V)                                   # stay on sphere

# Error tracking:
error_accum[argmax(sims)] += (labels[argmax(sims)] != true_label)

# Spawn (energy-gated):
E = -log(sum(exp(sims / tau)))
if E > spawn_thresh_energy: spawn(r, true_label)

# Merge (unchanged):
if max pairwise cos > merge_thresh and same label: fuse

# Classify:
per_class = scatter_add(softmax(V @ r / tau_infer), labels)
return argmax(per_class)
```

Hyperparameters: tau (learning temperature), tau_infer (inference temperature), lr, spawn energy threshold, merge_thresh. Five total vs current seventeen.

### C4. Codebook + Micro-Matrices

Keep the codebook but give each vector a small companion matrix:

```
Each entry: (v_i, A_i) where v_i in R^d (position), A_i in R^(k x k) (local dynamics)

Similarity:  s_i = cos(v_i, r) + lambda * cos(A_i, project(r))
Update v:    v_i += lr * r; normalize          # current fold rule
Update A:    A_i += dt * (phi(A_i) - A_i)      # eigenform drive (from RK)

Classify:    soft attention over v similarities, weighted by A magnitude
Generate:    product of top-k A matrices applied to seed
```

This preserves BOTH the codebook and the matrix dynamics but makes them per-prototype instead of shared. Each prototype has its own local dynamics. No separate routing — the matrix IS part of the prototype.

Cost: d + k^2 per prototype. At d=512, k=4: 528 params per entry. With 48K entries: 25M params. Feasible.

### C5. The Minimal Fold

Strip everything to the absolute minimum that could work:

```python
class AtomicFold:
    def __init__(self, d, tau=0.1):
        self.V = torch.empty(0, d, device='cuda')  # codebook on GPU
        self.labels = []
        self.tau = tau

    def step(self, r, label):
        if len(self.V) == 0 or self.energy(r) > threshold:
            self.V = torch.cat([self.V, F.normalize(r.unsqueeze(0))])
            self.labels.append(label)
            return

        # One equation: gradient of Hopfield energy w.r.t. codebook
        sims = self.V @ r
        weights = F.softmax(sims / self.tau, dim=0)
        target = torch.zeros_like(weights)
        for i, l in enumerate(self.labels):
            if l == label:
                target[i] = weights[i]  # redistribute weight to same-class
        target = target / (target.sum() + 1e-10)

        # Update: move all prototypes along energy gradient
        error = weights - target
        for i in range(len(self.V)):
            self.V[i] -= self.tau * error[i] * (r - self.V[i] * sims[i])
            self.V[i] = F.normalize(self.V[i], dim=0)

    def energy(self, r):
        if len(self.V) == 0: return float('inf')
        return -torch.logsumexp(self.V @ r / self.tau, dim=0).item()

    def classify(self, r):
        sims = self.V @ r
        weights = F.softmax(sims / self.tau, dim=0)
        per_class = {}
        for i, l in enumerate(self.labels):
            per_class[l] = per_class.get(l, 0) + weights[i].item()
        return max(per_class, key=per_class.get)
```

~30 lines. One class. One temperature parameter. Energy-gated spawning. Gradient-derived updates. Soft retrieval. No matrix, no projection, no coupling. The entire system.

This is what "atomic" might look like in code: small enough to read in one screen, complete enough to learn, classify, generate (via iterative self-retrieval), and know its own uncertainty (energy).

---

## Which Equations Excite Me

**B5 (Fold as Differential Equation)** — because it gives the codebook a PROVABLE energy landscape. Everything emerges from gradient flow on one function. Convergence is guaranteed. The energy IS the error signal. The gradient IS the update. Spawning is energy-gated. This is the most mathematically clean.

**A5 (Self-Referential)** — because the system processes itself. No external input needed for the dynamics to continue. The codebook reorganizes based on its own structure. This is closer to "the system IS its own substrate."

**B1 (Prototype Matrices)** — because it recovers "state IS transformation." Each memory DOES something, not just EXISTS somewhere. This connects back to the original RK thesis but in a unified way.

**C5 (Minimal Fold)** — because it's 30 lines and might actually work. The fastest path to empirical validation.

---

*These are candidates, not conclusions. The atomic equation will emerge from testing, not from theorizing. But the search space is now mapped.*
