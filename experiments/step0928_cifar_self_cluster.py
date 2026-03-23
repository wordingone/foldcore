"""
step0928_cifar_self_cluster.py -- CIFAR self-clustering via alpha-weighted L2 NN.

R3 hypothesis: Alpha-weighted encoding creates a metric space where similar
observations cluster. Consistent action assignment to clusters produces
above-chance classification WITHOUT labels.

Mechanism: L2 nearest neighbor in alpha-weighted encoding space.
  - Novel obs → new cluster, assign next action (round-robin mod n_actions)
  - Known obs (dist < THRESH) → return same action as nearest cluster
NOT codebook (no cosine, no repel/attract, no unit sphere). Pure L2 in alpha space.

Run on Split-CIFAR-100 (20 tasks, 500 img/task, 5 classes/task).
Metric: avg accuracy + backward transfer (BWT).
Compare against chance: 20% (random over 5 classes per task).

Also run on CIFAR-100 (100 classes) to see if clustering scales.
"""
import sys, time
sys.path.insert(0, 'B:/M/the-search')
sys.path.insert(0, 'B:/M/the-search/experiments')

import numpy as np
from collections import deque
from substrates.step0674 import _enc_frame

ENC_DIM = 256
ALPHA_LO = 0.10
ALPHA_HI = 5.00
ALPHA_UPDATE_DELAY = 50
ETA_W = 0.01
THRESH = 0.5         # L2 threshold for cluster membership

TEST_SEEDS = list(range(1, 6))


def one_hot(a, n):
    v = np.zeros(n, dtype=np.float32); v[a] = 1.0; return v


class SelfClusterSubstrate:
    """895h encoding + alpha-weighted L2 nearest-neighbor self-clustering."""

    def __init__(self, seed, n_actions):
        self._rng = np.random.RandomState(seed)
        self._n = n_actions
        self.W = np.zeros((ENC_DIM, ENC_DIM + n_actions), dtype=np.float32)
        self.alpha = np.ones(ENC_DIM, dtype=np.float32)
        self._running_mean = np.zeros(ENC_DIM, dtype=np.float32)
        self._n_obs = 0
        self._pred_errors = deque(maxlen=200)
        self._prev_enc = None; self._prev_action = None
        # Clusters: list of (alpha_weighted_enc, assigned_action)
        self._clusters = []

    def set_game(self, n_actions):
        self._n = n_actions
        self.W = np.zeros((ENC_DIM, ENC_DIM + n_actions), dtype=np.float32)
        self._clusters = []
        self._prev_enc = None; self._prev_action = None

    def _encode(self, obs):
        enc_raw = _enc_frame(np.asarray(obs, dtype=np.float32))
        self._n_obs += 1
        a = 1.0 / self._n_obs
        self._running_mean = (1 - a) * self._running_mean + a * enc_raw
        return enc_raw - self._running_mean

    def _update_alpha(self):
        if len(self._pred_errors) < ALPHA_UPDATE_DELAY: return
        me = np.mean(self._pred_errors, axis=0)
        if np.any(np.isnan(me)) or np.any(np.isinf(me)): return
        ra = np.sqrt(np.clip(me, 0, 1e6) + 1e-8); mr = np.mean(ra)
        if mr < 1e-8 or np.isnan(mr): return
        self.alpha = np.clip(ra / mr, ALPHA_LO, ALPHA_HI)

    def process(self, obs):
        enc = self._encode(obs)

        # Update W + alpha (signal learning — same as 895h)
        if self._prev_enc is not None and self._prev_action is not None:
            inp = np.concatenate([self._prev_enc * self.alpha,
                                   one_hot(self._prev_action, self._n)])
            pred = self.W @ inp
            error = (enc * self.alpha) - pred
            en = float(np.linalg.norm(error))
            if en > 10.0: error *= 10.0 / en
            if not np.any(np.isnan(error)):
                self.W -= ETA_W * np.outer(error, inp)
                self._pred_errors.append(np.abs(error))
                self._update_alpha()

        # Alpha-weighted nearest neighbor clustering
        weighted = enc * self.alpha

        if self._clusters:
            dists = np.array([float(np.linalg.norm(weighted - c[0]))
                               for c in self._clusters])
            nearest_idx = int(np.argmin(dists))
            if dists[nearest_idx] < THRESH:
                action = self._clusters[nearest_idx][1]
                self._prev_enc = enc.copy(); self._prev_action = action
                return action

        # New cluster
        action = len(self._clusters) % self._n
        self._clusters.append((weighted.copy(), action))
        self._prev_enc = enc.copy(); self._prev_action = action
        return action

    def on_level_transition(self):
        self._prev_enc = None; self._prev_action = None

    def alpha_conc(self):
        return float(np.max(self.alpha) / (np.min(self.alpha) + 1e-8))

    def n_clusters(self):
        return len(self._clusters)


def load_cifar():
    try:
        import torchvision, torchvision.transforms as T
        ds = torchvision.datasets.CIFAR100('B:/M/the-search/data', train=False,
                                            download=True, transform=T.ToTensor())
        imgs = np.array([np.array(ds[i][0]).transpose(1,2,0) for i in range(len(ds))],
                        dtype=np.float32)
        lbls = np.array([ds[i][1] for i in range(len(ds))], dtype=np.int32)
        return imgs, lbls
    except Exception as e:
        print(f"  CIFAR load failed: {e}"); return None, None


def run_split_cifar(sub, imgs, lbls, seed, n_images_per_task=500):
    """20 tasks, 5 classes each. Returns (task_accuracies, bwt)."""
    N_TASKS = 20; CLASSES_PER_TASK = 5
    rng = np.random.RandomState(seed)
    sub.set_game(CLASSES_PER_TASK)

    task0_acc = None
    task_accs = []
    for task_id in range(N_TASKS):
        class_start = task_id * CLASSES_PER_TASK
        class_end = class_start + CLASSES_PER_TASK
        mask = (lbls >= class_start) & (lbls < class_end)
        task_imgs = imgs[mask]; task_lbls = lbls[mask] - class_start
        idx = rng.choice(len(task_imgs), min(n_images_per_task, len(task_imgs)), replace=False)
        correct = 0
        for i in idx:
            action = sub.process(task_imgs[i]) % CLASSES_PER_TASK
            correct += int(action == task_lbls[i])
        acc = correct / len(idx)
        task_accs.append(acc)
        sub.on_level_transition()

    # BWT: re-eval task 0 after all tasks
    class_start = 0; class_end = CLASSES_PER_TASK
    mask0 = (lbls >= class_start) & (lbls < class_end)
    t0_imgs = imgs[mask0]; t0_lbls = lbls[mask0]
    idx0 = rng.choice(len(t0_imgs), min(n_images_per_task, len(t0_imgs)), replace=False)
    correct0 = sum(1 for i in idx0 if sub.process(t0_imgs[i]) % CLASSES_PER_TASK == t0_lbls[i])
    task0_acc_after = correct0 / len(idx0)
    bwt = task0_acc_after - task_accs[0]
    return task_accs, bwt


print("=" * 70)
print("STEP 928 — CIFAR SELF-CLUSTERING (alpha-weighted L2 nearest neighbor)")
print("=" * 70)
print(f"Split-CIFAR-100: 20 tasks, 5 classes/task, 500 imgs/task.")
print(f"Chance baseline: 20% (random over 5 classes).")
print(f"Threshold: L2 in alpha space < {THRESH}")
t0 = time.time()

imgs, lbls = load_cifar()
if imgs is None:
    print("CIFAR not available — cannot run")
    sys.exit(1)

all_accs = []; all_bwts = []; all_clusters = []

for seed in TEST_SEEDS:
    sub = SelfClusterSubstrate(seed=seed, n_actions=5)
    task_accs, bwt = run_split_cifar(sub, imgs, lbls, seed * 1000)
    avg_acc = np.mean(task_accs)
    all_accs.append(avg_acc)
    all_bwts.append(bwt)
    all_clusters.append(sub.n_clusters())
    print(f"  seed={seed}: avg_acc={avg_acc:.4f}  BWT={bwt:+.4f}  "
          f"n_clusters={sub.n_clusters()}  alpha_conc={sub.alpha_conc():.2f}")
    print(f"    task_accs (first 5): {[f'{x:.3f}' for x in task_accs[:5]]}")

print(f"\n{'='*70}")
print(f"STEP 928 RESULTS (Split-CIFAR-100 self-clustering):")
print(f"  avg_acc={np.mean(all_accs):.4f}  std={np.std(all_accs):.4f}")
print(f"  avg_BWT={np.mean(all_bwts):+.4f}")
print(f"  avg_clusters={np.mean(all_clusters):.0f}")
print(f"  Chance baseline: 0.2000 (20% random over 5 classes)")
print(f"  Above chance: {'YES' if np.mean(all_accs) > 0.22 else 'NO'} "
      f"(threshold: >22% = signal)")
print(f"Total elapsed: {time.time()-t0:.1f}s")
print("STEP 928 DONE")
