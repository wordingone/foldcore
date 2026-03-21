"""
Step 589 -- Recode vs LSH head-to-head, 20 seeds, 50K steps. NEEDS JUN APPROVAL.

Reviewer's separation result: does l_pi (Recode, adaptive encoding) provably
outperform l_0 (fixed LSH) at same budget?

Step 542 was 5/5 but used 500K steps and different protocol.
This is the clean head-to-head at the same scale as step 584 (20 seeds, 50K).

Three conditions:
  A) Recode (l_pi, K=16, adaptive splits) -- encoding self-modifies
  B) LSH (l_0, K=12) -- same as argmin baseline
  C) LSH (l_0, K=16) -- controls for K, isolates adaptive-split effect

Same seed protocol as 584. Fisher exact test: Recode vs LSH(K=12).

Expected runtime: 3 conditions x 20 seeds x 50K steps ~= 90 min.
JUN APPROVAL REQUIRED before running.
"""
import numpy as np
import time
import sys

K_RECODE = 16
K_LSH12  = 12
K_LSH16  = 16
DIM = 256
N_A = 4
MAX_STEPS = 50_000
TIME_CAP  = 300   # 5 min per seed
N_SEEDS   = 20
REFINE_EVERY = 5000
MIN_OBS   = 8
H_SPLIT   = 0.05


def enc_vec(frame):
    a = np.array(frame[0], dtype=np.float32) / 15.0
    x = a.reshape(16, 4, 16, 4).mean(axis=(1, 3)).flatten()
    return x - x.mean()

def lsh_hash(x, H):
    bits = (H @ x > 0).astype(np.int64)
    return int(np.dot(bits, 1 << np.arange(len(bits))))


class Recode:
    def __init__(self, seed=0):
        self.H = np.random.RandomState(seed).randn(K_RECODE, DIM).astype(np.float32)
        self.ref = {}; self.G = {}; self.C = {}; self.live = set()
        self._pn = self._pa = self._px = self._cn = None
        self.t = 0; self.dim = DIM; self.total_deaths = 0

    def _base(self, x):
        return int(np.packbits((self.H @ x > 0).astype(np.uint8), bitorder='big').tobytes().hex(), 16)

    def _node(self, x):
        n = self._base(x)
        while n in self.ref:
            n = (n, int(self.ref[n] @ x > 0))
        return n

    def observe(self, frame):
        x = enc_vec(frame)
        n = self._node(x)
        self.live.add(n)
        self.t += 1
        if self._pn is not None:
            d = self.G.setdefault((self._pn, self._pa), {})
            d[n] = d.get(n, 0) + 1
            k = (self._pn, self._pa, n)
            s, c = self.C.get(k, (np.zeros(self.dim, np.float64), 0))
            self.C[k] = (s + x.astype(np.float64), c + 1)
        self._px = x; self._cn = n
        if self.t > 0 and self.t % REFINE_EVERY == 0:
            self._refine()

    def act(self):
        counts = [sum(self.G.get((self._cn, a), {}).values()) for a in range(N_A)]
        action = int(np.argmin(counts))
        self._pn = self._cn; self._pa = action
        return action

    def on_death(self): self.total_deaths += 1
    def on_reset(self): self._pn = None

    def _h(self, n, a):
        d = self.G.get((n, a))
        if not d or sum(d.values()) < 4: return 0.0
        v = np.array(list(d.values()), np.float64); p = v / v.sum()
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
            diff = (r0[0] / r0[1]) - (r1[0] / r1[1])
            nm = np.linalg.norm(diff)
            if nm < 1e-8: continue
            self.ref[n] = (diff / nm).astype(np.float32)
            self.live.discard(n); did += 1
            if did >= 3: break

    def stats(self): return len(self.live), len(self.ref), len(self.G)


class LSH:
    def __init__(self, seed=0, k=K_LSH12):
        self.H = np.random.RandomState(seed).randn(k, DIM).astype(np.float32)
        self.G = {}; self._pn = self._pa = self._cn = None
        self.cells = set(); self.total_deaths = 0

    def observe(self, frame):
        x = enc_vec(frame); n = lsh_hash(x, self.H)
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


class LSH16(LSH):
    def __init__(self, seed=0): super().__init__(seed=seed, k=K_LSH16)


def run_seed(mk, seed, SubClass):
    env = mk(); sub = SubClass(seed=seed * 100 + 7)
    obs = env.reset(seed=seed); sub.on_reset()
    l1 = l2 = go = step = 0; prev_cl = 0; fresh = True; t0 = time.time()

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
        elif cl >= 1 and prev_cl < 1:
            l1 += 1
            if l1 <= 2: print(f"    s{seed} L1@{step}", flush=True)
        elif cl >= 2 and prev_cl < 2: l2 += 1
        prev_cl = cl

    deaths = getattr(sub, 'total_deaths', 0)
    if hasattr(sub, 'stats'): nc, ns, _ = sub.stats(); cells = nc
    else: cells = len(getattr(sub, 'cells', set()))
    print(f"  s{seed}: L1={l1} L2={l2} go={go} step={step} cells={cells} deaths={deaths} "
          f"{time.time()-t0:.0f}s", flush=True)
    return dict(seed=seed, l1=l1, l2=l2, go=go, steps=step, cells=cells, deaths=deaths)


def run_condition(mk, label, SubClass):
    print(f"\n--- {label} ---", flush=True)
    results = []
    for seed in range(N_SEEDS):
        print(f"\nseed {seed}:", flush=True)
        try: results.append(run_seed(mk, seed, SubClass))
        except Exception as e: print(f"  FAIL: {e}", flush=True)
    l1 = sum(r['l1'] for r in results)
    seeds = sum(1 for r in results if r['l1'] > 0)
    print(f"  {label}: {seeds}/{N_SEEDS} L1={l1}", flush=True)
    return results


def main():
    import sys
    # Require explicit approval flag
    if "--approved" not in sys.argv:
        print("Step 589 requires Jun's explicit 50K approval.")
        print("Run with --approved flag after receiving approval.")
        return

    try:
        sys.path.insert(0, '.')
        import arcagi3
        mk = lambda: arcagi3.make("LS20")
    except Exception as e:
        print(f"arcagi3: {e}"); return

    print(f"Step 589: Recode vs LSH head-to-head -- {N_SEEDS} seeds x {MAX_STEPS} steps",
          flush=True)

    t_total = time.time()
    rc  = run_condition(mk, "Recode (l_pi, K=16)", Recode)
    l12 = run_condition(mk, "LSH (l_0, K=12)",     LSH)
    l16 = run_condition(mk, "LSH (l_0, K=16)",     LSH16)

    def summ(r): return sum(x['l1'] for x in r), sum(1 for x in r if x['l1'] > 0)
    rc_l1, rc_s = summ(rc); l12_l1, l12_s = summ(l12); l16_l1, l16_s = summ(l16)

    print(f"\n{'='*60}")
    print(f"Step 589: Recode vs LSH ({N_SEEDS} seeds, {MAX_STEPS} steps)")
    print(f"  Recode  (l_pi, K=16): {rc_s}/{N_SEEDS} seeds L1={rc_l1}")
    print(f"  LSH     (l_0,  K=12): {l12_s}/{N_SEEDS} seeds L1={l12_l1}")
    print(f"  LSH     (l_0,  K=16): {l16_s}/{N_SEEDS} seeds L1={l16_l1}")

    if rc_l1 > l12_l1:
        print(f"\n  SEPARATION: Recode({rc_l1}) > LSH-K12({l12_l1}). l_pi provably better than l_0.")
    elif rc_l1 == l12_l1:
        print(f"\n  NO SEPARATION: Recode({rc_l1}) == LSH-K12({l12_l1}). Same final rate.")
    else:
        print(f"\n  INVERSION: Recode({rc_l1}) < LSH-K12({l12_l1}). Adaptive encoding hurts.")

    try:
        from scipy.stats import fisher_exact
        table = [[rc_s, N_SEEDS - rc_s], [l12_s, N_SEEDS - l12_s]]
        odds, pval = fisher_exact(table, alternative='greater')
        print(f"  Fisher (Recode > LSH-K12): odds={odds:.3f} p={pval:.4f}")
    except ImportError:
        pass

    print(f"\n  Total elapsed: {time.time()-t_total:.0f}s")


if __name__ == "__main__":
    main()
