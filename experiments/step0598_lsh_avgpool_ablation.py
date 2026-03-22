"""
Step 598 -- Avgpool ablation: 16x16 vs 8x8 vs 4x4 resolution.

Current: 64x64 -> avgpool4 -> 16x16 = 256D (I1 constraint).
Step 470 showed raw 64x64 fails (1/5). Where exactly does resolution help?

Three encodings, LSH K=12, 5 seeds, 10K:
  E1: 16x16 = 256D (baseline)
  E2: 8x8   = 64D
  E3: 4x4   = 16D
"""
import numpy as np
import time
import sys

N_A = 4
MAX_STEPS = 10_000
TIME_CAP = 60
N_SEEDS = 5
K = 12


def enc_16x16(frame):
    """Current baseline: 256D"""
    a = np.array(frame[0], dtype=np.float32) / 15.0
    x = a.reshape(16, 4, 16, 4).mean(axis=(1, 3)).flatten()
    return x - x.mean()


def enc_8x8(frame):
    """8x8 = 64D"""
    a = np.array(frame[0], dtype=np.float32) / 15.0
    x = a.reshape(8, 8, 8, 8).mean(axis=(1, 3)).flatten()
    return x - x.mean()


def enc_4x4(frame):
    """4x4 = 16D"""
    a = np.array(frame[0], dtype=np.float32) / 15.0
    x = a.reshape(4, 16, 4, 16).mean(axis=(1, 3)).flatten()
    return x - x.mean()


def lsh_hash(x, H):
    bits = (H @ x > 0).astype(np.int64)
    return int(np.dot(bits, 1 << np.arange(len(bits))))


class Argmin:
    def __init__(self, seed=0, enc_fn=None, dim=256):
        self.enc_fn = enc_fn
        self.H = np.random.RandomState(seed).randn(K, dim).astype(np.float32)
        self.G = {}; self._pn = self._pa = self._cn = None
        self.cells = set(); self.total_deaths = 0

    def observe(self, frame):
        x = self.enc_fn(frame); n = lsh_hash(x, self.H)
        self.cells.add(n)
        if self._pn is not None:
            d = self.G.setdefault((self._pn, self._pa), {})
            d[n] = d.get(n, 0) + 1
        self._cn = n

    def act(self):
        counts = [sum(self.G.get((self._cn, a), {}).values()) for a in range(N_A)]
        action = int(np.argmin(counts))
        self._pn = self._cn; self._pa = action; return action

    def on_death(self): self.total_deaths += 1
    def on_reset(self): self._pn = None


def run_seed(mk, seed, enc_fn, dim):
    env = mk(); sub = Argmin(seed=seed * 100 + 7, enc_fn=enc_fn, dim=dim)
    obs = env.reset(seed=seed); sub.on_reset()
    l1 = go = step = 0; prev_cl = 0; fresh = True; t0 = time.time()

    while step < MAX_STEPS and time.time() - t0 < TIME_CAP:
        if obs is None:
            obs = env.reset(seed=seed); sub.on_reset()
            prev_cl = 0; fresh = True; go += 1; continue
        sub.observe(obs); action = sub.act()
        obs, _, done, info = env.step(action); step += 1
        if done:
            sub.on_death(); obs = env.reset(seed=seed); sub.on_reset()
            prev_cl = 0; fresh = True; go += 1; continue
        cl = info.get('level', 0) if isinstance(info, dict) else 0
        if fresh: prev_cl = cl; fresh = False
        elif cl >= 1 and prev_cl < 1: l1 += 1
        prev_cl = cl

    return dict(l1=l1, cells=len(sub.cells))


def run_enc(mk, label, enc_fn, dim):
    wins = 0; total_cells = []
    print(f"\n  {label} ({dim}D):", flush=True)
    for seed in range(N_SEEDS):
        r = run_seed(mk, seed, enc_fn, dim)
        wins += (r['l1'] > 0); total_cells.append(r['cells'])
        print(f"    s{seed}: L1={r['l1']} cells={r['cells']}", flush=True)
    avg_cells = np.mean(total_cells)
    print(f"  {label}: {wins}/{N_SEEDS}  avg_cells={avg_cells:.0f}", flush=True)
    return wins, avg_cells


def main():
    try:
        sys.path.insert(0, '.')
        import arcagi3
        mk = lambda: arcagi3.make("LS20")
    except Exception as e:
        print(f"arcagi3: {e}"); return

    print(f"Step 598: Avgpool ablation -- resolution effect", flush=True)
    print(f"  K={K} | {N_SEEDS} seeds x {MAX_STEPS} steps", flush=True)

    t0 = time.time()
    w1, c1 = run_enc(mk, "16x16", enc_16x16, 256)
    w2, c2 = run_enc(mk, "8x8",   enc_8x8,    64)
    w3, c3 = run_enc(mk, "4x4",   enc_4x4,    16)

    print(f"\n{'='*60}", flush=True)
    print(f"Step 598: Avgpool ablation", flush=True)
    print(f"  16x16 (256D): {w1}/{N_SEEDS}  avg_cells={c1:.0f}", flush=True)
    print(f"  8x8   (64D):  {w2}/{N_SEEDS}  avg_cells={c2:.0f}", flush=True)
    print(f"  4x4   (16D):  {w3}/{N_SEEDS}  avg_cells={c3:.0f}", flush=True)

    if w1 >= w2 >= w3:
        print(f"\n  MONOTONE: resolution degradation hurts navigation.", flush=True)
    elif w2 >= w1:
        print(f"\n  SURPRISE: 8x8 matches or beats 16x16. Over-specification possible.", flush=True)

    print(f"\n  Total elapsed: {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
