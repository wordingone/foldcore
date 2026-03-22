"""
Step 665 — Frame-diff at exit cell.

1 seed (seed 8: 420 exit-cell visits, gap 24198 steps from Step 652).
Log every frame at the exit cell during the run.
Cluster with k-means k=2,3,4. Which cluster contains the triggering frame?

If frames cluster into 2+ groups and triggering frame is in a distinct cluster:
  hidden state IS visible in the frame — LSH hash too coarse to distinguish.
If all frames are identical (L2 ~ 0):
  hidden state is truly invisible from observations.
If frames vary continuously (no clusters):
  hidden state affects frames but not in a clusterable way.
"""
import numpy as np
import time
import sys
from collections import defaultdict

K = 12
DIM = 256
N_A = 4
REFINE_EVERY = 5000
MIN_OBS = 8
H_SPLIT = 0.05
MAX_STEPS = 500_001
PER_SEED_TIME = 90  # single seed, need to reach step ~24235
SEED = 8


def enc_frame(frame):
    a = np.array(frame[0], dtype=np.float32) / 15.0
    x = a.reshape(16, 4, 16, 4).mean(axis=(1, 3)).flatten()
    return x - x.mean()


class Recode:
    def __init__(self, k=K, dim=DIM, seed=0):
        self.H = np.random.RandomState(seed).randn(k, dim).astype(np.float32)
        self.ref = {}
        self.G = {}
        self.C = {}
        self.live = set()
        self._pn = self._pa = self._px = None
        self.t = 0
        self.dim = dim
        self._cn = None
        # Frame logging: cell -> list of (step, enc_frame)
        self.cell_frames = defaultdict(list)
        self.MAX_FRAMES_PER_CELL = 600

    def _hash(self, x):
        return int(np.packbits(
            (self.H @ x > 0).astype(np.uint8), bitorder='big'
        ).tobytes().hex(), 16)

    def _node(self, x):
        n = self._hash(x)
        while n in self.ref:
            n = (n, int(self.ref[n] @ x > 0))
        return n

    def observe(self, frame):
        x = enc_frame(frame)
        n = self._node(x)
        self.live.add(n)
        self.t += 1
        if self._pn is not None:
            d = self.G.setdefault((self._pn, self._pa), {})
            d[n] = d.get(n, 0) + 1
            k = (self._pn, self._pa, n)
            s, c = self.C.get(k, (np.zeros(self.dim, np.float64), 0))
            self.C[k] = (s + self._px.astype(np.float64), c + 1)
        self._px = x
        self._cn = n
        # Log frame for this cell
        cf = self.cell_frames[n]
        if len(cf) < self.MAX_FRAMES_PER_CELL:
            cf.append((self.t, x.copy()))
        if self.t > 0 and self.t % REFINE_EVERY == 0:
            self._refine()
        return n

    def act(self):
        best_a, best_s = 0, float('inf')
        for a in range(N_A):
            s = sum(self.G.get((self._cn, a), {}).values())
            if s < best_s:
                best_s = s
                best_a = a
        self._pn = self._cn
        self._pa = best_a
        return best_a

    def on_reset(self):
        self._pn = None

    def _h(self, n, a):
        d = self.G.get((n, a))
        if not d or sum(d.values()) < 4:
            return 0.0
        v = np.array(list(d.values()), np.float64)
        p = v / v.sum()
        return float(-np.sum(p * np.log2(np.maximum(p, 1e-15))))

    def _refine(self):
        did = 0
        for (n, a), d in list(self.G.items()):
            if n not in self.live or n in self.ref:
                continue
            if len(d) < 2 or sum(d.values()) < MIN_OBS:
                continue
            if self._h(n, a) < H_SPLIT:
                continue
            top = sorted(d, key=d.get, reverse=True)[:2]
            r0 = self.C.get((n, a, top[0]))
            r1 = self.C.get((n, a, top[1]))
            if r0 is None or r1 is None or r0[1] < 3 or r1[1] < 3:
                continue
            diff = (r0[0] / r0[1]) - (r1[0] / r1[1])
            nm = np.linalg.norm(diff)
            if nm < 1e-8:
                continue
            self.ref[n] = (diff / nm).astype(np.float32)
            self.live.discard(n)
            did += 1
            if did >= 3:
                break


def kmeans_np(X, k, n_iter=30, seed=0):
    """Simple k-means in numpy. Returns (labels, centers)."""
    rng = np.random.RandomState(seed)
    idx = rng.choice(len(X), min(k, len(X)), replace=False)
    centers = X[idx].copy()
    labels = np.zeros(len(X), dtype=int)
    for _ in range(n_iter):
        # Assign
        dists = np.array([[np.linalg.norm(x - c) for c in centers] for x in X])
        new_labels = np.argmin(dists, axis=1)
        if np.all(new_labels == labels):
            break
        labels = new_labels
        # Update
        for j in range(k):
            mask = labels == j
            if mask.any():
                centers[j] = X[mask].mean(axis=0)
    return labels, centers


def analyze_frames(frames_with_steps, trigger_step):
    """Analyze frames at exit cell. frames_with_steps = list of (step, vec)."""
    if len(frames_with_steps) < 3:
        print(f"    Too few frames ({len(frames_with_steps)}) to cluster")
        return

    steps = [s for s, _ in frames_with_steps]
    X = np.array([v for _, v in frames_with_steps])
    n = len(X)

    # Basic stats
    dists = np.linalg.norm(X - X.mean(axis=0), axis=1)
    print(f"    n_frames={n}, L2_from_mean: avg={dists.mean():.4f} max={dists.max():.4f} min={dists.min():.4f}")

    # Find triggering frame (last frame logged at or before trigger_step)
    trig_idx = None
    for i, s in enumerate(steps):
        if s <= trigger_step:
            trig_idx = i
    if trig_idx is None:
        trig_idx = len(steps) - 1

    trig_dist_from_mean = float(np.linalg.norm(X[trig_idx] - X.mean(axis=0)))
    print(f"    Triggering frame index: {trig_idx}/{n-1}, dist_from_mean={trig_dist_from_mean:.4f}")

    # Pairwise distance of triggering frame vs others
    other_idx = [i for i in range(n) if i != trig_idx]
    trig_dists = np.linalg.norm(X[other_idx] - X[trig_idx], axis=1)
    other_dists = []
    for i in range(min(20, n)):
        if i != trig_idx:
            oth = [j for j in range(min(20, n)) if j != i and j != trig_idx]
            if oth:
                other_dists.append(np.linalg.norm(X[i] - X[oth], axis=1).mean())
    other_str = f"{np.mean(other_dists):.4f}" if other_dists else "N/A"
    print(f"    Trig avg_dist_to_others={trig_dists.mean():.4f}, "
          f"other avg_dist_to_others={other_str}")

    # K-means for k=2,3
    for k in [2, 3]:
        if n < k * 3:
            continue
        labels, centers = kmeans_np(X, k)
        trig_cluster = labels[trig_idx]
        cluster_sizes = [int((labels == j).sum()) for j in range(k)]
        # Intra-cluster variance
        intra = []
        for j in range(k):
            mask = labels == j
            if mask.sum() > 1:
                intra.append(float(np.linalg.norm(X[mask] - centers[j], axis=1).mean()))
        print(f"    k={k}: trig_cluster={trig_cluster} sizes={cluster_sizes} "
              f"intra_dist={np.mean(intra):.4f}")

        if k == 2:
            # Is the triggering frame alone in its cluster?
            trig_size = cluster_sizes[trig_cluster]
            other_cluster = 1 - trig_cluster
            other_size = cluster_sizes[other_cluster]
            if trig_size < n * 0.2:
                print(f"    -> FINDING: trig frame in MINORITY cluster ({trig_size}/{n})")
            else:
                print(f"    -> FINDING: trig frame in majority cluster ({trig_size}/{n})")

            # Distance between cluster centers
            center_dist = float(np.linalg.norm(centers[0] - centers[1]))
            print(f"    -> Center distance: {center_dist:.4f}")


def run(make):
    env = make()
    sub = Recode(seed=SEED * 1000)
    obs = env.reset(seed=SEED)
    level = 0
    l1 = None
    exit_cell = None
    t_start = time.time()

    for step in range(1, MAX_STEPS):
        if obs is None:
            obs = env.reset(seed=SEED)
            sub.on_reset()
            continue
        sub.observe(obs)
        action = sub.act()
        obs, reward, done, info = env.step(action)
        cl = info.get('level', 0) if isinstance(info, dict) else 0
        if cl > level:
            if cl == 1 and l1 is None:
                l1 = step
                exit_cell = sub._cn
            level = cl
            sub.on_reset()
        if done:
            obs = env.reset(seed=SEED)
            sub.on_reset()
        if time.time() - t_start > PER_SEED_TIME:
            break

    elapsed = time.time() - t_start
    print(f"  seed={SEED}: L1={l1} steps={sub.t} t={elapsed:.1f}s", flush=True)

    if l1 is None or exit_cell is None:
        print("  L1 not reached — cannot analyze exit cell")
        return

    frames = sub.cell_frames.get(exit_cell, [])
    print(f"  Exit cell frames logged: {len(frames)}")

    if not frames:
        print("  No frames logged at exit cell")
        return

    analyze_frames(frames, l1)


def main():
    try:
        sys.path.insert(0, '.')
        import arcagi3
        mk = lambda: arcagi3.make("LS20")
    except (ImportError, Exception) as e:
        print(f"arcagi3: {e}")
        return

    print(f"Frame-diff at exit cell: seed={SEED}, {PER_SEED_TIME}s cap")
    run(mk)


if __name__ == "__main__":
    main()
