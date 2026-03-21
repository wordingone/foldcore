"""
Step 596 -- Noise pre-training control.

Step 595 found: CIFAR pre-exposure accelerates LS20 (chain tax = -3592 steps, 2/5 faster).
This control distinguishes "warm start" from "domain transfer":
  - If random noise -> LS20 also accelerates: benefit is non-zero G counts, not structure
  - If only CIFAR -> LS20 accelerates: genuine cross-domain transfer

Three conditions, 5 seeds, 10K per phase:
  1. LS20 alone (baseline)
  2. Random noise (10K random 64x64 frames, values 0-15) -> LS20
  3. CIFAR (10K images) -> LS20

Shared G (same edge dict). Per-frame centering.
"""
import numpy as np
import time
import sys

K = 12
DIM = 256
N_A = 4
N_PRE = 10_000         # pre-training steps
MAX_LS20_STEPS = 10_000
TIME_CAP = 60
N_SEEDS = 5


def enc_ls20(frame):
    a = np.array(frame[0], dtype=np.float32) / 15.0
    x = a.reshape(16, 4, 16, 4).mean(axis=(1, 3)).flatten()
    return x - x.mean()


def enc_noise(arr):
    """Random 64x64 int array [0-15] -> 256D centered."""
    a = arr.astype(np.float32) / 15.0
    x = a.reshape(16, 4, 16, 4).mean(axis=(1, 3)).flatten()
    return x - x.mean()


def enc_cifar(img):
    """CIFAR (3,32,32) uint8 -> 256D centered."""
    if img.ndim == 3 and img.shape[0] == 3:
        img = img.transpose(1, 2, 0)
    gray = img.mean(axis=2).astype(np.float32) / 255.0
    x = gray.reshape(16, 2, 16, 2).mean(axis=(1, 3)).flatten()
    return x - x.mean()


def lsh_hash(x, H):
    bits = (H @ x > 0).astype(np.int64)
    return int(np.dot(bits, 1 << np.arange(len(bits))))


class LSH:
    def __init__(self, seed=0):
        self.H = np.random.RandomState(seed).randn(K, DIM).astype(np.float32)
        self.G = {}
        self._pn = self._pa = self._cn = None
        self.total_deaths = 0

    def observe(self, x):
        n = lsh_hash(x, self.H)
        if self._pn is not None:
            d = self.G.setdefault((self._pn, self._pa), {})
            d[n] = d.get(n, 0) + 1
        self._cn = n

    def act(self):
        counts = [sum(self.G.get((self._cn, a), {}).values()) for a in range(N_A)]
        action = int(np.argmin(counts))
        self._pn = self._cn; self._pa = action
        return action

    def on_death(self): self.total_deaths += 1
    def on_reset(self): self._pn = None


def pretrain_noise(sub, rng, n):
    """Pre-populate G with random noise frames."""
    sub.on_reset()
    for _ in range(n):
        arr = rng.randint(0, 16, (64, 64))
        x = enc_noise(arr)
        sub.observe(x)
        sub.act()


def pretrain_cifar(sub, cifar_imgs, n):
    """Pre-populate G with CIFAR images."""
    sub.on_reset()
    for i in range(min(n, len(cifar_imgs))):
        x = enc_cifar(cifar_imgs[i])
        sub.observe(x)
        sub.act()


def run_ls20(mk, seed, sub):
    env = mk()
    obs = env.reset(seed=seed)
    sub.on_reset()

    l1 = go = step = 0
    prev_cl = 0; fresh = True
    t0 = time.time()
    l1_step = None

    while step < MAX_LS20_STEPS and time.time() - t0 < TIME_CAP:
        if obs is None:
            obs = env.reset(seed=seed); sub.on_reset()
            prev_cl = 0; fresh = True; go += 1; continue
        x = enc_ls20(obs)
        sub.observe(x)
        action = sub.act()
        obs, _, done, info = env.step(action)
        step += 1
        if done:
            sub.on_death(); obs = env.reset(seed=seed); sub.on_reset()
            prev_cl = 0; fresh = True; go += 1; continue
        cl = info.get('level', 0) if isinstance(info, dict) else 0
        if fresh: prev_cl = cl; fresh = False
        elif cl >= 1 and prev_cl < 1:
            l1 += 1
            if l1_step is None: l1_step = step
        prev_cl = cl

    return l1, l1_step, go


def load_cifar100(n):
    try:
        import torchvision.datasets as dsets
        import torchvision.transforms as T
        import tempfile, os
        ds = dsets.CIFAR100(root=os.path.join(tempfile.gettempdir(), 'cifar100'),
                            train=True, download=True, transform=T.ToTensor())
        imgs = []
        for i in range(min(n, len(ds))):
            img, _ = ds[i]
            imgs.append((img.numpy() * 255).astype(np.uint8))
        return np.array(imgs)
    except Exception as e:
        print(f"  CIFAR unavailable ({e}). Using synthetic.", flush=True)
        rng = np.random.RandomState(42)
        return rng.randint(0, 256, (n, 3, 32, 32), dtype=np.uint8)


def main():
    try:
        sys.path.insert(0, '.')
        import arcagi3
        mk = lambda: arcagi3.make("LS20")
    except Exception as e:
        print(f"arcagi3: {e}"); return

    print(f"Step 596: Noise pre-training control", flush=True)
    print(f"  N_PRE={N_PRE} | {N_SEEDS} seeds | LS20={MAX_LS20_STEPS}", flush=True)

    cifar_imgs = load_cifar100(N_PRE)
    rng = np.random.RandomState(99)
    t_total = time.time()

    c1_l1, c1_steps, c2_l1, c2_steps, c3_l1, c3_steps = [], [], [], [], [], []

    for seed in range(N_SEEDS):
        print(f"\n--- seed {seed} ---", flush=True)

        # C1: LS20 alone
        sub1 = LSH(seed=seed * 100 + 7)
        l1, l1s, go = run_ls20(mk, seed, sub1)
        c1_l1.append(l1 > 0); c1_steps.append(l1s)
        print(f"  C1 alone:  L1={l1} @{l1s} go={go}", flush=True)

        # C2: noise -> LS20
        sub2 = LSH(seed=seed * 100 + 7)
        pretrain_noise(sub2, rng, N_PRE)
        l1, l1s, go = run_ls20(mk, seed, sub2)
        c2_l1.append(l1 > 0); c2_steps.append(l1s)
        print(f"  C2 noise:  L1={l1} @{l1s} go={go}", flush=True)

        # C3: CIFAR -> LS20
        sub3 = LSH(seed=seed * 100 + 7)
        pretrain_cifar(sub3, cifar_imgs, N_PRE)
        l1, l1s, go = run_ls20(mk, seed, sub3)
        c3_l1.append(l1 > 0); c3_steps.append(l1s)
        print(f"  C3 CIFAR:  L1={l1} @{l1s} go={go}", flush=True)

    def fmt(wins, steps):
        v = [s for s in steps if s is not None]
        s = f"avg@{np.mean(v):.0f}" if v else "never"
        return f"{sum(wins)}/{N_SEEDS} ({s})"

    print(f"\n{'='*60}", flush=True)
    print(f"Step 596: Noise pre-training control ({N_SEEDS} seeds)", flush=True)
    print(f"  C1 LS20 alone: {fmt(c1_l1, c1_steps)}", flush=True)
    print(f"  C2 Noise->LS20:{fmt(c2_l1, c2_steps)}", flush=True)
    print(f"  C3 CIFAR->LS20:{fmt(c3_l1, c3_steps)}", flush=True)

    c1v = [s for s in c1_steps if s is not None]
    c2v = [s for s in c2_steps if s is not None]
    c3v = [s for s in c3_steps if s is not None]

    if c2v and c1v and np.mean(c2v) < np.mean(c1v):
        print(f"\n  WARM START: Noise also accelerates ({np.mean(c2v):.0f} < {np.mean(c1v):.0f}).", flush=True)
        print(f"  Benefit is non-zero G counts, not CIFAR structure.", flush=True)
        if c3v and np.mean(c3v) < np.mean(c2v):
            print(f"  CIFAR additional benefit: {np.mean(c3v):.0f} < {np.mean(c2v):.0f}. Domain transfer on top.", flush=True)
    elif c3v and c1v and np.mean(c3v) < np.mean(c1v) and (not c2v or np.mean(c2v) >= np.mean(c1v)):
        print(f"\n  DOMAIN TRANSFER: Only CIFAR accelerates, not noise.", flush=True)
        print(f"  Cross-domain structure is the active ingredient.", flush=True)
    else:
        print(f"\n  NO CLEAR PATTERN. C1={np.mean(c1v) if c1v else 'N/A':.0f} "
              f"C2={np.mean(c2v) if c2v else 'N/A':.0f} "
              f"C3={np.mean(c3v) if c3v else 'N/A':.0f}", flush=True)

    print(f"\n  Total elapsed: {time.time()-t_total:.0f}s", flush=True)


if __name__ == "__main__":
    main()
