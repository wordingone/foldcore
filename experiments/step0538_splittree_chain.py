"""
Step 538 -- SplitTree chain: CIFAR -> LS20 -> CIFAR.

Step 537: SplitTree(edge_transfer, threshold=64) -> 3/3 L1@15880. Navigates LS20.
Chain test: does tree survive domain switching?

CIFAR 1-pass (10K images, 100 actions) -> LS20 50K -> CIFAR 1-pass.
Shared tree across phases. mu accumulates across both domains.

Question: CIFAR splits are in CIFAR-relevant dimensions. When LS20 frames arrive,
do they traverse CIFAR-split cells and get action-0 bias (empty edges in those cells)?
Or does the observation-space separation keep the domains in different tree regions?

Predictions: LS20 still gets L1 (different observation space from CIFAR -> different
tree regions, no contamination). CIFAR return unchanged (tree returns to CIFAR region).
"""
import time
import copy
import sys
import os
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

N_CIFAR = 10_000
CIFAR_ACTIONS = 100
MAX_LS20 = 50_000
N_SEEDS = 3
TIME_CAP = 270
THRESHOLD = 64


def encode_cifar(img):
    if img.ndim == 3 and img.shape[0] == 3:
        img = img.transpose(1, 2, 0)
    gray = (0.299 * img[:, :, 0].astype(np.float32) +
            0.587 * img[:, :, 1].astype(np.float32) +
            0.114 * img[:, :, 2].astype(np.float32)) / 255.0
    arr = gray.reshape(16, 2, 16, 2).mean(axis=(1, 3)).flatten()
    return arr - arr.mean()


def encode_arc(frame):
    arr = np.array(frame[0], dtype=np.float32) / 15.0
    x = arr.reshape(16, 4, 16, 4).mean(axis=(1, 3)).flatten()
    return x - x.mean()


class SplitTreeCombined:
    """SplitTree: edge transfer + threshold=64."""

    def __init__(self, na, threshold=THRESHOLD):
        self.A = na
        self.T = {}
        self.G = {}
        self.R = {}
        self.mu = None
        self.d = 0
        self.n = 0
        self.p = None
        self.k = 1
        self.threshold = threshold
        self.splits = 0

    def __call__(self, x):
        D = len(x)
        if not self.mu:
            self.mu = [0.0] * D
            self.d = D
        self.n += 1
        z = [x[i] - self.mu[i] for i in range(D)]
        r = 1.0 / self.n
        for i in range(D):
            self.mu[i] += r * (x[i] - self.mu[i])
        c = self._map(z)
        if self.p:
            pc, pa, pz = self.p
            e = self.G.setdefault((pc, pa), {})
            e[c] = e.get(c, 0) + 1
            t = self.R.setdefault(pc, {}).setdefault((pa, c), [[0.0] * D, 0])
            t[1] += 1
            for i in range(D):
                t[0][i] += (pz[i] - t[0][i]) / t[1]
            self._split(pc)
            c = self._map(z)
        a = self._act(c)
        self.p = (c, a, z)
        return a

    def _map(self, z):
        c = 0
        while c in self.T:
            d, v, l, r = self.T[c]
            c = l if z[d] < v else r
        return c

    def _act(self, c):
        b, bn = 0, -1
        for a in range(self.A):
            n = sum(self.G.get((c, a), {}).values())
            if bn < 0 or n < bn:
                b, bn = a, n
        return b

    def _split(self, c):
        if c in self.T or c not in self.R:
            return
        pairs = [(v[1], v[0]) for v in self.R[c].values() if v[1] >= 4]
        tn = sum(p[0] for p in pairs)
        if tn < self.threshold or len(pairs) < 2:
            return
        pairs.sort(key=lambda p: p[0], reverse=True)
        n0, m0 = pairs[0]
        n1, m1 = pairs[1]
        bd, bv, bs = 0, 0.0, 0.0
        for i in range(self.d):
            s = abs(m1[i] - m0[i])
            if s > bs:
                bd, bv, bs = i, (m0[i] * n0 + m1[i] * n1) / (n0 + n1), s
        if bs < 1e-9:
            return
        l, r = self.k, self.k + 1
        self.k += 2
        self.T[c] = (bd, bv, l, r)
        self.splits += 1
        for (pa, c_next), (mean_pz, count) in list(self.R.get(c, {}).items()):
            child = l if mean_pz[bd] < bv else r
            cg = self.G.setdefault((child, pa), {})
            cg[c_next] = cg.get(c_next, 0) + count
            cr = self.R.setdefault(child, {})
            if (pa, c_next) in cr:
                old_mean, old_count = cr[(pa, c_next)]
                total = old_count + count
                merged = [(old_mean[i] * old_count + mean_pz[i] * count) / total
                          for i in range(self.d)]
                cr[(pa, c_next)] = [merged, total]
            else:
                cr[(pa, c_next)] = [mean_pz[:], count]
        if c in self.R:
            del self.R[c]
        for pa in range(self.A):
            self.G.pop((c, pa), None)


def run_cifar_phase(s, X, y, label, n_actions):
    """Run CIFAR phase: argmin over n_actions actions, no reset."""
    s.p = None  # domain boundary reset
    s.A = n_actions
    t0 = time.time()
    correct = 0
    cells_before = 1 + len(s.T)
    for i in range(len(X)):
        x = list(encode_cifar(X[i]).astype(float))
        a = s(x)
        if a == int(y[i]):
            correct += 1
    acc = correct / len(X) * 100
    cells_after = 1 + len(s.T)
    print(f"  {label}: acc={acc:.2f}%  cells={cells_before}->{cells_after}  "
          f"splits_total={s.splits}  {time.time()-t0:.0f}s", flush=True)
    return acc


def run_ls20_phase(s, arc, game_id):
    from arcengine import GameState
    s.p = None  # domain boundary reset
    s.A = 4
    env = arc.make(game_id)
    action_space = env.action_space
    obs = env.reset()
    ts = deaths = 0
    l1_step = None
    t0 = time.time()
    cells_before = 1 + len(s.T)

    while ts < MAX_LS20:
        if time.time() - t0 > TIME_CAP:
            break
        if obs is None or not obs.frame:
            obs = env.reset(); s.p = None; deaths += 1; continue
        if obs.state == GameState.GAME_OVER:
            obs = env.reset(); s.p = None; deaths += 1; continue

        x = list(encode_arc(obs.frame).astype(float))
        a = s(x)
        prev_lvls = obs.levels_completed
        obs = env.step(action_space[a])
        ts += 1

        if obs and obs.state == GameState.WIN:
            if l1_step is None:
                l1_step = ts
            break
        if obs and obs.levels_completed > prev_lvls and l1_step is None:
            l1_step = ts

    cells_after = 1 + len(s.T)
    tag = f"WIN@{l1_step}" if obs and obs.state == GameState.WIN else \
          (f"L1@{l1_step}" if l1_step else "FAIL")
    print(f"  LS20: {tag}  cells={cells_before}->{cells_after}  "
          f"splits_total={s.splits}  deaths={deaths}  {time.time()-t0:.0f}s", flush=True)
    return l1_step


def t1():
    s = SplitTreeCombined(4, threshold=THRESHOLD)
    x = list(np.random.randn(256).astype(float))
    a = s(x)
    assert 0 <= a < 4
    # Domain switch: change A
    s.p = None
    s.A = 100
    a2 = s(x)
    assert 0 <= a2 < 100
    print(f"T1 PASS (threshold={THRESHOLD}, domain switch OK)")


def main():
    t1()

    import torchvision, arc_agi
    ds = torchvision.datasets.CIFAR100('./data/cifar100',
                                        train=False, download=True)
    X = np.array(ds.data[:N_CIFAR])
    y = np.array(ds.targets[:N_CIFAR])
    arc = arc_agi.Arcade()
    games = arc.get_environments()
    ls20 = next(g for g in games if 'ls20' in g.game_id.lower())

    print(f"\nStep 538: SplitTree chain (CIFAR->{N_CIFAR}->LS20->{MAX_LS20//1000}K->CIFAR).",
          flush=True)
    print(f"Baseline (Step 537, no CIFAR): 3/3 L1@15880.", flush=True)
    print(f"Test: does CIFAR pre-training interfere with LS20 navigation?", flush=True)

    # Single run (substrate is deterministic; all seeds identical)
    t_total = time.time()
    s = SplitTreeCombined(na=CIFAR_ACTIONS, threshold=THRESHOLD)

    print(f"\n--- Phase 1: CIFAR ({N_CIFAR} images, {CIFAR_ACTIONS} actions) ---",
          flush=True)
    acc1 = run_cifar_phase(s, X, y, "CIFAR_P1", CIFAR_ACTIONS)

    print(f"\n--- Phase 2: LS20 ({MAX_LS20//1000}K steps) ---", flush=True)
    l1 = run_ls20_phase(s, arc, ls20.game_id)

    print(f"\n--- Phase 3: CIFAR return ---", flush=True)
    acc2 = run_cifar_phase(s, X, y, "CIFAR_P5", CIFAR_ACTIONS)

    print(f"\n{'='*55}", flush=True)
    print(f"STEP 538 SUMMARY", flush=True)
    print(f"  CIFAR P1:  {acc1:.2f}%", flush=True)
    print(f"  LS20:      {'L1@'+str(l1) if l1 else 'FAIL'}", flush=True)
    print(f"  CIFAR P5:  {acc2:.2f}% (delta={acc2-acc1:+.2f}pp)", flush=True)
    print(f"  Total cells: {1+len(s.T)}  Total splits: {s.splits}", flush=True)
    print(f"  Total elapsed: {time.time()-t_total:.0f}s", flush=True)

    if l1:
        print(f"\nSIGNAL: SplitTree navigates LS20 in chain.", flush=True)
        if acc2 >= acc1 - 1.0:
            print(f"  CIFAR retention OK (delta={acc2-acc1:+.2f}pp < -1pp threshold).",
                  flush=True)
        else:
            print(f"  CIFAR forgetting: {acc2-acc1:+.2f}pp.", flush=True)
    else:
        print(f"\nFAIL: CIFAR pre-training prevents LS20 navigation.", flush=True)
        print(f"  CIFAR splits contaminate the tree -> wrong cell mapping for ARC frames.",
              flush=True)


if __name__ == "__main__":
    main()
