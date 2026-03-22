"""
Step 543 -- Recode chain: CIFAR-100 (10K) -> LS20 (50K) -> CIFAR-100 (10K).

Step 542: Recode L1=5/5. Now test chain: does CIFAR refinement contaminate LS20?

SplitTree (Step 538) KILLED by CIFAR contamination. Recode hypothesis: CIFAR obs
hash to different base nodes than LS20 obs (different L2 scale), so CIFAR splits
should NOT affect LS20 nodes.

Encoding: avgpool16 + centered for both CIFAR and LS20. DIM=256 both.
CIFAR: 32x32x3 -> grayscale -> pool(2x2) -> 16x16 -> 256-dim, centered.
LS20:  64x64   ->           -> pool(4x4) -> 16x16 -> 256-dim, centered.

Predictions: LS20 3/5 L1 (baseline 5/5, CIFAR contamination slows by ~2x).
             CIFAR accuracy 1% (encoding-limited, random baseline 1/100).
             Zero forgetting (pass 2 acc >= pass 1 acc).
Kill: 0/3 L1 = CIFAR refinement breaks LS20 mapping (same as SplitTree).
"""
import numpy as np
import time
import sys
import os

N_A = 4
K = 16
DIM = 256
REFINE_EVERY = 5000
MIN_OBS = 8
H_SPLIT = 0.05
CIFAR_N = 10_000
LS20_STEPS = 50_000
TIME_CAP = 300  # 5 min per phase


def enc(frame):
    """Avgpool16 + centered. Works for LS20 frame list and CIFAR (H,W,C) uint8 array."""
    if isinstance(frame, np.ndarray) and frame.ndim == 3:
        # CIFAR: (32, 32, 3) uint8 -> grayscale -> pool 2x2 -> 16x16 = 256-dim
        gray = frame.mean(axis=2).astype(np.float32) / 255.0
        x = gray.reshape(16, 2, 16, 2).mean(axis=(1, 3)).flatten()
    else:
        # LS20: frame[0] is (64, 64), values 0-15 -> pool 4x4 -> 16x16 = 256-dim
        a = np.array(frame[0], dtype=np.float32) / 15.0
        x = a.reshape(16, 4, 16, 4).mean(axis=(1, 3)).flatten()
    return x - x.mean()


class Recode:

    def __init__(self, dim=DIM, k=K, seed=0):
        self.H = np.random.RandomState(seed).randn(k, dim).astype(np.float32)
        self.ref = {}
        self.G = {}
        self.C = {}
        self.live = set()
        self._pn = self._pa = self._px = None
        self.t = 0
        self.ns = 0

    def _base(self, x):
        return int(np.packbits(
            (self.H @ x > 0).astype(np.uint8), bitorder='big'
        ).tobytes().hex(), 16)

    def _node(self, x):
        n = self._base(x)
        while n in self.ref:
            n = (n, int(self.ref[n] @ x > 0))
        return n

    def observe(self, frame):
        x = enc(frame)
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
        self.dim = len(x)
        if self.t > 0 and self.t % REFINE_EVERY == 0:
            self._refine()
        return n

    def act(self):
        counts = [sum(self.G.get((self._cn, a), {}).values()) for a in range(N_A)]
        action = int(np.argmin(counts))
        self._pn = self._cn
        self._pa = action
        return action

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
            self.ns += 1
            did += 1
            if did >= 3:
                break

    def stats(self):
        return len(self.live), self.ns, len(self.G)


def t0():
    rng = np.random.RandomState(42)

    # Test enc() for both domains
    img = rng.randint(0, 256, (32, 32, 3)).astype(np.uint8)
    xc = enc(img)
    assert xc.shape == (256,), f"CIFAR enc shape {xc.shape}"
    assert abs(float(xc.mean())) < 1e-5, f"CIFAR enc not centered: {xc.mean()}"

    frame_ls20 = [rng.randint(0, 16, (64, 64))]
    xl = enc(frame_ls20)
    assert xl.shape == (256,), f"LS20 enc shape {xl.shape}"
    assert abs(float(xl.mean())) < 1e-5, f"LS20 enc not centered: {xl.mean()}"

    # CIFAR and LS20 should hash to DIFFERENT regions (different L2 scale)
    sub = Recode(seed=0)
    sub.dim = DIM
    nc = sub._base(xc)
    nl = sub._base(xl)
    # Not guaranteed different but we can verify both are valid ints
    assert isinstance(nc, int) and isinstance(nl, int), "base hash must be int"

    # Chain: CIFAR obs -> on_reset -> LS20 obs -> on_reset -> CIFAR obs (must not crash)
    sub2 = Recode(seed=1)
    img2 = rng.randint(0, 256, (32, 32, 3)).astype(np.uint8)
    sub2.observe(img2); sub2.act()       # CIFAR obs
    sub2.on_reset()
    sub2.observe(frame_ls20); sub2.act() # LS20 obs
    sub2.on_reset()
    sub2.observe(img2); sub2.act()       # CIFAR again
    nc, ns, ne = sub2.stats()
    assert nc >= 1, f"no live cells after chain: {nc}"

    print("T0 PASS")


def cifar_phase(sub, images, labels, desc):
    """Run 1-pass CIFAR: observe + act (action ignored). Returns node->label map."""
    rng = np.random.RandomState(42)
    idx = rng.choice(len(images), CIFAR_N, replace=False)
    node_label = {}  # node -> {class: count}
    t_start = time.time()
    for i in idx:
        node = sub.observe(images[i])
        sub.act()
        lbl = int(labels[i])
        if node not in node_label:
            node_label[node] = {}
        node_label[node][lbl] = node_label[node].get(lbl, 0) + 1
    nc, ns, ne = sub.stats()
    elapsed = time.time() - t_start
    print(f"  {desc}: c={nc} sp={ns} e={ne} nodes={len(node_label)} {elapsed:.1f}s",
          flush=True)
    return node_label


def cifar_accuracy(sub, images, labels, node_label_p1, desc):
    """Pass 2: predict class from node using pass-1 majority label."""
    rng = np.random.RandomState(99)
    idx = rng.choice(len(images), CIFAR_N, replace=False)
    correct = seen = known = 0
    for i in idx:
        node = sub.observe(images[i])
        sub.act()
        lbl = int(labels[i])
        seen += 1
        if node in node_label_p1:
            known += 1
            pred = max(node_label_p1[node], key=node_label_p1[node].get)
            if pred == lbl:
                correct += 1
    acc = correct / seen if seen > 0 else 0.0
    coverage = known / seen if seen > 0 else 0.0
    nc, ns, ne = sub.stats()
    print(f"  {desc}: acc={acc:.1%} cov={coverage:.1%} c={nc} sp={ns}", flush=True)
    return acc


def run_chain(seed, make_ls20, cifar_images, cifar_labels):
    sub = Recode(seed=seed * 1000)

    # --- Phase 1: CIFAR ---
    print(f"  s{seed} Phase 1 (CIFAR):", flush=True)
    node_label_p1 = cifar_phase(sub, cifar_images, cifar_labels,
                                 f"s{seed} CIFAR1")
    sub.on_reset()

    # --- Phase 2: LS20 ---
    print(f"  s{seed} Phase 2 (LS20):", flush=True)
    env = make_ls20()
    obs = env.reset(seed=seed)
    level = 0
    l1 = l2 = None
    go = 0
    t_ls = time.time()

    for step in range(1, LS20_STEPS + 1):
        if obs is None:
            obs = env.reset(seed=seed)
            sub.on_reset()
            continue

        sub.observe(obs)
        action = sub.act()
        obs, reward, done, info = env.step(action)

        if done:
            go += 1
            obs = env.reset(seed=seed)
            sub.on_reset()

        cl = info.get('level', 0) if isinstance(info, dict) else 0
        if cl > level:
            sub.on_reset()
            nc, ns, ne = sub.stats()
            if cl == 1 and l1 is None:
                l1 = step
                print(f"  s{seed} L1@{step} c={nc} sp={ns} e={ne} go={go}", flush=True)
            if cl == 2 and l2 is None:
                l2 = step
                print(f"  s{seed} L2@{step} c={nc} sp={ns} e={ne} go={go}", flush=True)
            level = cl

        if time.time() - t_ls > TIME_CAP:
            break

    nc_ls20, ns_ls20, ne_ls20 = sub.stats()
    elapsed = time.time() - t_ls
    print(f"  s{seed} LS20 done: @{step} c={nc_ls20} sp={ns_ls20} e={ne_ls20} "
          f"go={go} {elapsed:.0f}s", flush=True)
    sub.on_reset()

    # --- Phase 3: CIFAR accuracy ---
    print(f"  s{seed} Phase 3 (CIFAR acc):", flush=True)
    acc = cifar_accuracy(sub, cifar_images, cifar_labels, node_label_p1,
                         f"s{seed} CIFAR2")

    return dict(seed=seed, l1=l1, l2=l2, acc=acc,
                cells_ls20=nc_ls20, splits=ns_ls20)


def main():
    t0()

    # Load CIFAR-100
    try:
        import torchvision
        ds = torchvision.datasets.CIFAR100(
            './data/cifar100', train=True, download=True)
        cifar_images = np.array(ds.data)    # (50000, 32, 32, 3) uint8
        cifar_labels = np.array(ds.targets)
        print(f"CIFAR-100 loaded: {len(cifar_images)} images, "
              f"{len(set(cifar_labels.tolist()))} classes")
    except Exception as e:
        print(f"CIFAR-100 load failed: {e}")
        return

    # Load LS20
    try:
        sys.path.insert(0, '.')
        import arcagi3
        mk = lambda: arcagi3.make("LS20")
    except Exception as e:
        print(f"arcagi3: {e}")
        return

    R = []
    for seed in range(3):
        print(f"\nseed {seed}:", flush=True)
        R.append(run_chain(seed, mk, cifar_images, cifar_labels))

    print(f"\n{'='*60}")
    for r in R:
        tag = "L2" if r['l2'] else ("L1" if r['l1'] else "---")
        print(f"  s{r['seed']}: {tag:>3}  c={r['cells_ls20']:>5}  "
              f"sp={r['splits']:>3}  acc={r['acc']:.1%}")

    l1n = sum(1 for r in R if r['l1'])
    l2n = sum(1 for r in R if r['l2'])
    avg_acc = float(np.mean([r['acc'] for r in R]))
    mc = max(r['cells_ls20'] for r in R)

    print(f"\nL1={l1n}/3  L2={l2n}/3  max_cells={mc}  avg_acc={avg_acc:.1%}")

    if l1n == 0:
        print("KILL: 0/3 L1 — CIFAR refinement contaminated LS20 mapping.")
    else:
        status = "CLEAN" if l1n == 3 else "PARTIAL"
        print(f"{status}: {l1n}/3 L1. CIFAR contamination: "
              f"{'none' if l1n == 3 else 'partial'}.")
        if avg_acc < 0.05:
            print(f"CIFAR acc={avg_acc:.1%}: encoding-limited (expected ~1%).")


if __name__ == "__main__":
    main()
