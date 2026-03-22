#!/usr/bin/env python3
"""
Step 444b — Graph Substrate on P-MNIST (1 task, unsupervised).
Question: if different digits land on different nodes,
and node_idx is the prediction, does the graph discriminate?

Two modes:
A) Pure unsupervised: node_idx % 10 as prediction (no labels ever)
B) Labeled nodes: store majority label per node during training,
   predict via nearest node's stored label

5K training steps, frozen eval on test set. 5-min cap.
"""

import time, math, random
import numpy as np
import torch
import torch.nn.functional as F

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
N_CLASSES = 10


class GraphSubstrate:
    def __init__(self, d=256, n_actions=10, sim_thresh=0.99):
        self.nodes = []
        self.node_labels = []  # majority label per node
        self.node_label_counts = []  # list of {label: count} per node
        self.edges = {}
        self.n_actions = n_actions
        self.sim_thresh = sim_thresh
        self.prev_node = None
        self.prev_action = None

    def _find_nearest(self, x):
        if len(self.nodes) == 0:
            return None, -1.0
        V = torch.stack(self.nodes)
        sims = F.cosine_similarity(V, x.unsqueeze(0))
        best_idx = sims.argmax().item()
        return best_idx, sims[best_idx].item()

    def step(self, x, label=None):
        x = F.normalize(x.float().flatten(), dim=0)
        node_idx, sim = self._find_nearest(x)
        if node_idx is None or sim < self.sim_thresh:
            node_idx = len(self.nodes)
            self.nodes.append(x.clone())
            counts = {label: 1} if label is not None else {}
            self.node_label_counts.append(counts)
            self.node_labels.append(label if label is not None else 0)
        else:
            lr = max(0, 1 - sim)
            self.nodes[node_idx] = self.nodes[node_idx] + lr * (x - self.nodes[node_idx])
            self.nodes[node_idx] = F.normalize(self.nodes[node_idx], dim=0)
            if label is not None:
                counts = self.node_label_counts[node_idx]
                counts[label] = counts.get(label, 0) + 1
                self.node_labels[node_idx] = max(counts, key=counts.get)

        if self.prev_node is not None and self.prev_action is not None:
            key = (self.prev_node, self.prev_action)
            if key not in self.edges:
                self.edges[key] = {}
            self.edges[key][node_idx] = self.edges[key].get(node_idx, 0) + 1

        visit_counts = [sum(self.edges.get((node_idx, a), {}).values()) for a in range(self.n_actions)]
        min_count = min(visit_counts)
        candidates = [a for a, c in enumerate(visit_counts) if c == min_count]
        action = candidates[torch.randint(len(candidates), (1,)).item()]
        self.prev_node = node_idx
        self.prev_action = action
        return action

    def predict_unsupervised(self, x):
        """node_idx % n_classes — no labels used."""
        x = F.normalize(x.float().flatten(), dim=0)
        node_idx, _ = self._find_nearest(x)
        if node_idx is None: return 0
        return node_idx % N_CLASSES

    def predict_labeled(self, x):
        """nearest node's majority label."""
        x = F.normalize(x.float().flatten(), dim=0)
        node_idx, _ = self._find_nearest(x)
        if node_idx is None: return 0
        return self.node_labels[node_idx]


def load_mnist():
    import torchvision
    tr = torchvision.datasets.MNIST('./data/mnist', train=True, download=True)
    te = torchvision.datasets.MNIST('./data/mnist', train=False, download=True)
    X_tr = tr.data.numpy().reshape(-1, 784).astype(np.float32) / 255.0
    X_te = te.data.numpy().reshape(-1, 784).astype(np.float32) / 255.0
    return X_tr, tr.targets.numpy(), X_te, te.targets.numpy()


def make_projection(d_in=784, d_out=256, seed=12345):
    rng = np.random.RandomState(seed)
    P = rng.randn(d_out, d_in).astype(np.float32) / math.sqrt(d_in)
    return torch.from_numpy(P).to(DEVICE)


def make_permutation(seed):
    perm = list(range(784))
    random.Random(seed).shuffle(perm)
    return perm


def main():
    print(f"Step 444b: Graph Substrate on P-MNIST", flush=True)
    print(f"Device: {DEVICE}  sim_thresh=0.99  5K train steps", flush=True)
    print(flush=True)

    t0 = time.time()
    X_tr, y_tr, X_te, y_te = load_mnist()
    perm = make_permutation(seed=0)
    P = make_projection(784, 256, seed=12345)

    # One task (perm 0)
    X_tr_p = X_tr[:, perm]
    X_te_p = X_te[:, perm]

    sub = GraphSubstrate(d=256, n_actions=N_CLASSES, sim_thresh=0.99)

    # Training: 5K steps with labels (feed to substrate, store labels per node)
    n_train = min(5000, len(X_tr_p))
    for i in range(n_train):
        x = torch.from_numpy(X_tr_p[i]).to(DEVICE)
        x_proj = F.normalize(P @ x, dim=0).cpu()
        label = int(y_tr[i])
        sub.step(x_proj, label=label)

    print(f"Training done: nodes={len(sub.nodes)}  edges={len(sub.edges)}  {time.time()-t0:.1f}s", flush=True)

    # Evaluate on test set
    n_test = len(X_te_p)
    correct_unsup = correct_labeled = 0

    for i in range(n_test):
        x = torch.from_numpy(X_te_p[i]).to(DEVICE)
        x_proj = F.normalize(P @ x, dim=0).cpu()
        true_label = int(y_te[i])

        pred_unsup = sub.predict_unsupervised(x_proj)
        pred_labeled = sub.predict_labeled(x_proj)

        if pred_unsup == true_label: correct_unsup += 1
        if pred_labeled == true_label: correct_labeled += 1

    acc_unsup = correct_unsup / n_test * 100
    acc_labeled = correct_labeled / n_test * 100
    elapsed = time.time() - t0

    print(f"\n{'='*60}", flush=True)
    print("STEP 444b RESULTS (P-MNIST)", flush=True)
    print(f"{'='*60}", flush=True)
    print(f"nodes={len(sub.nodes)}  edges={len(sub.edges)}", flush=True)
    print(f"Mode A (node_idx % 10, no labels): {acc_unsup:.2f}%  (chance=10%)", flush=True)
    print(f"Mode B (majority label per node):   {acc_labeled:.2f}%  (codebook baseline ~94%)", flush=True)
    print(f"Elapsed: {elapsed:.1f}s", flush=True)

    if acc_unsup > 15:
        print(f"Mode A ABOVE CHANCE: digits cluster by cosine geometry!", flush=True)
    else:
        print(f"Mode A at chance: node indices don't align with labels.", flush=True)
    if acc_labeled > 25:
        print(f"Mode B PASS gate (>25%): labeled graph discriminates.", flush=True)


if __name__ == '__main__':
    main()
