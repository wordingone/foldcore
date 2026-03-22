"""
Step 700 — Chain 20-seed sweep.

Chain: CIFAR-100 P1 (1K obs) -> LS20 nav (120K steps) -> CIFAR-100 P3 (1K obs).
674 substrate with running-mean centering (per-domain centering, reset on episode end).
20 seeds (0-19).

Step 693 (5 seeds): L1=5/5, avg CIFAR acc=1.3%.
Question: does chain context help other seeds too? Compare L1: chain vs standalone.

Kill: L1 < 15/20 or contamination > 5pp.
"""
import numpy as np
import time
import sys

K_NAV = 12
K_FINE = 20
DIM = 256
N_A = 4
REFINE_EVERY = 5000
MIN_OBS = 8
H_SPLIT = 0.05
MIN_VISITS_ALIAS = 3
CIFAR_N = 1000
LS20_STEPS = 120_000


def enc_raw(frame):
    if isinstance(frame, np.ndarray) and frame.ndim == 3:
        gray = frame.mean(axis=2).astype(np.float32) / 255.0
        return gray.reshape(16, 2, 16, 2).mean(axis=(1, 3)).flatten()
    else:
        a = np.array(frame[0], dtype=np.float32) / 15.0
        return a.reshape(16, 4, 16, 4).mean(axis=(1, 3)).flatten()


class TransitionTriggeredChain:
    def __init__(self, seed=0):
        rng = np.random.RandomState(seed)
        self.H_nav = rng.randn(K_NAV, DIM).astype(np.float32)
        self.H_fine = rng.randn(K_FINE, DIM).astype(np.float32)
        self.ref = {}; self.G = {}; self.C = {}; self.live = set()
        self.G_fine = {}; self.aliased = set()
        self._pn = self._pa = self._px = None; self._pfn = None
        self.t = 0; self.dim = DIM; self._cn = None; self._fn = None
        self._mu = np.zeros(DIM, dtype=np.float32); self._mu_n = 0

    def _hash_nav(self, x):
        return int(np.packbits((self.H_nav @ x > 0).astype(np.uint8), bitorder='big').tobytes().hex(), 16)

    def _hash_fine(self, x):
        return int(np.packbits((self.H_fine @ x > 0).astype(np.uint8), bitorder='big').tobytes().hex(), 16)

    def _node(self, x):
        n = self._hash_nav(x)
        while n in self.ref: n = (n, int(self.ref[n] @ x > 0))
        return n

    def observe(self, frame):
        x_raw = enc_raw(frame)
        self._mu_n += 1
        self._mu = self._mu + (x_raw - self._mu) / self._mu_n
        x = x_raw - self._mu
        n = self._node(x); fn = self._hash_fine(x)
        self.live.add(n); self.t += 1
        if self._pn is not None:
            d = self.G.setdefault((self._pn, self._pa), {})
            d[n] = d.get(n, 0) + 1
            k = (self._pn, self._pa, n)
            s, c = self.C.get(k, (np.zeros(self.dim, np.float64), 0))
            self.C[k] = (s + self._px.astype(np.float64), c + 1)
            succ = self.G.get((self._pn, self._pa), {})
            if sum(succ.values()) >= MIN_VISITS_ALIAS and len(succ) >= 2:
                self.aliased.add(self._pn)
            if self._pn in self.aliased and self._pfn is not None:
                d2 = self.G_fine.setdefault((self._pfn, self._pa), {})
                d2[fn] = d2.get(fn, 0) + 1
        self._px = x; self._cn = n; self._fn = fn
        if self.t % REFINE_EVERY == 0: self._refine()
        return n

    def act(self):
        if self._cn in self.aliased and self._fn is not None:
            best_a, best_s = 0, float('inf')
            for a in range(N_A):
                s = sum(self.G_fine.get((self._fn, a), {}).values())
                if s < best_s: best_s = s; best_a = a
            self._pn = self._cn; self._pfn = self._fn; self._pa = best_a; return best_a
        best_a, best_s = 0, float('inf')
        for a in range(N_A):
            s = sum(self.G.get((self._cn, a), {}).values())
            if s < best_s: best_s = s; best_a = a
        self._pn = self._cn; self._pfn = self._fn; self._pa = best_a; return best_a

    def on_reset(self):
        self._pn = None; self._pfn = None
        self._mu = np.zeros(self.dim, dtype=np.float32); self._mu_n = 0

    def _h(self, n, a):
        d = self.G.get((n, a))
        if not d or sum(d.values()) < 4: return 0.0
        v = np.array(list(d.values()), np.float64); p = v/v.sum()
        return float(-np.sum(p * np.log2(np.maximum(p, 1e-15))))

    def _refine(self):
        did = 0
        for (n, a), d in list(self.G.items()):
            if n not in self.live or n in self.ref: continue
            if len(d) < 2 or sum(d.values()) < MIN_OBS: continue
            if self._h(n, a) < H_SPLIT: continue
            top = sorted(d, key=d.get, reverse=True)[:2]
            r0 = self.C.get((n, a, top[0])); r1 = self.C.get((n, a, top[1]))
            if r0 is None or r1 is None or r0[1] < 3 or r1[1] < 3: continue
            diff = (r0[0]/r0[1]) - (r1[0]/r1[1]); nm = np.linalg.norm(diff)
            if nm < 1e-8: continue
            self.ref[n] = (diff/nm).astype(np.float32); self.live.discard(n); did += 1
            if did >= 3: break


def cifar_phase(sub, images, labels, desc):
    rng = np.random.RandomState(42)
    idx = rng.choice(len(images), CIFAR_N, replace=False)
    node_label = {}
    aliased_before = len(sub.aliased)
    for i in idx:
        node = sub.observe(images[i])
        sub.act()
        lbl = int(labels[i])
        node_label.setdefault(node, {})[lbl] = node_label.get(node, {}).get(lbl, 0) + 1
    aliased_after = len(sub.aliased)
    print(f"  {desc}: nodes={len(node_label)} aliased={aliased_before}->{aliased_after}", flush=True)
    return node_label


def cifar_accuracy(sub, images, labels, node_label_p1, desc):
    rng = np.random.RandomState(99)
    idx = rng.choice(len(images), CIFAR_N, replace=False)
    correct = seen = known = 0
    aliased_before = len(sub.aliased)
    for i in idx:
        node = sub.observe(images[i])
        sub.act()
        lbl = int(labels[i])
        seen += 1
        if node in node_label_p1:
            known += 1
            pred = max(node_label_p1[node], key=node_label_p1[node].get)
            if pred == lbl: correct += 1
    acc = correct / seen if seen > 0 else 0.0
    cov = known / seen if seen > 0 else 0.0
    aliased_after = len(sub.aliased)
    print(f"  {desc}: acc={acc:.1%} cov={cov:.1%} aliased={aliased_before}->{aliased_after}", flush=True)
    return acc


def run_chain(seed, make_ls20, cifar_images, cifar_labels):
    sub = TransitionTriggeredChain(seed=seed * 1000)

    # Phase 1: CIFAR
    node_label_p1 = cifar_phase(sub, cifar_images, cifar_labels, f"s{seed} CIFAR-P1")
    cifar1_aliased = len(sub.aliased)
    sub.on_reset()

    # Phase 2: LS20
    env = make_ls20()
    obs = env.reset(seed=seed)
    level = 0; l1 = None

    for step in range(1, LS20_STEPS + 1):
        if obs is None:
            obs = env.reset(seed=seed); sub.on_reset(); continue
        sub.observe(obs); action = sub.act()
        obs, reward, done, info = env.step(action)
        cl = info.get('level', 0) if isinstance(info, dict) else 0
        if cl > level:
            if cl == 1 and l1 is None:
                l1 = step
                print(f"  s{seed} L1@{l1} aliased={len(sub.aliased)}", flush=True)
            level = cl; sub.on_reset()
        if done:
            obs = env.reset(seed=seed); sub.on_reset()

    ls20_aliased = len(sub.aliased)
    print(f"  s{seed} LS20 done: L1={l1} aliased={ls20_aliased}", flush=True)
    sub.on_reset()

    # Phase 3: CIFAR accuracy
    acc = cifar_accuracy(sub, cifar_images, cifar_labels, node_label_p1, f"s{seed} CIFAR-P3")

    return dict(seed=seed, l1=l1, acc=acc, cifar1_aliased=cifar1_aliased, ls20_aliased=ls20_aliased)


def main():
    try:
        import torchvision
        ds = torchvision.datasets.CIFAR100('./data/cifar100', train=True, download=True)
        cifar_images = np.array(ds.data)
        cifar_labels = np.array(ds.targets)
        print(f"CIFAR-100: {len(cifar_images)} images")
    except Exception as e:
        print(f"CIFAR-100 load failed: {e}"); return

    try:
        sys.path.insert(0, '.')
        import arcagi3
        mk = lambda: arcagi3.make("LS20")
    except Exception as e:
        print(f"arcagi3: {e}"); return

    print(f"Step 700: Chain 20-seed sweep. {CIFAR_N} CIFAR obs per phase, {LS20_STEPS} LS20 steps.")
    print(f"Game version: ls20/9607627b (verify in init log above)")
    R = []
    for seed in range(20):
        print(f"\n--- seed {seed} ---", flush=True)
        R.append(run_chain(seed, mk, cifar_images, cifar_labels))

    print(f"\n{'='*60}")
    l1n = sum(1 for r in R if r['l1'])
    avg_acc = float(np.mean([r['acc'] for r in R]))
    print(f"L1={l1n}/20  avg_CIFAR_acc={avg_acc:.1%}")
    print(f"Game version: ls20/9607627b")
    for r in R:
        print(f"  s{r['seed']:2d}: L1={r['l1']} cifar1_aliased={r['cifar1_aliased']} "
              f"ls20_aliased={r['ls20_aliased']} acc={r['acc']:.1%}")

    cifar1_contamination = [r['cifar1_aliased'] for r in R]
    max_cont = max(cifar1_contamination)
    print(f"CIFAR-P1 contamination: {cifar1_contamination} (max={max_cont})")

    # Compare to standalone Step 692/699
    # Step 692 standalone (17/20 at 25s, 20/20 at 120K)
    if l1n >= 15 and max_cont <= 5:
        print(f"FINDING: Chain 20-seed compatible. L1={l1n}/20, contamination OK")
    elif l1n < 15:
        print(f"KILL: Chain L1={l1n}/20 < 15/20 threshold")
    else:
        print(f"KILL: CIFAR contamination too high ({max_cont}pp > 5pp threshold)")


if __name__ == "__main__":
    main()
