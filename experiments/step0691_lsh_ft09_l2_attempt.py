"""
Step 691 — 674 on FT09, extended budget for L2.

674 substrate (transition-triggered dual-hash, k=12+k=20).
FT09 had only 1-4 aliased cells at L1 (Step 680).
Does L2 have more aliasing? Does mechanism adapt post-L1?

5 seeds, 300s cap.
Report every 30s: L1 step, L2 step, aliased_cells at each level.
"""
import numpy as np
import time
import sys

K_NAV = 12
K_FINE = 20
DIM = 256
REFINE_EVERY = 5000
MIN_OBS = 8
H_SPLIT = 0.05
MAX_STEPS = 2_000_001
PER_SEED_TIME = 300
N_SEEDS = 5
MIN_VISITS_ALIAS = 3
REPORT_INTERVAL = 30

# 8x8 click grid: pixel positions, action_int = x + y*64
GRID_ACTIONS = [(gx * 8 + 4) + (gy * 8 + 4) * 64
                for gy in range(8) for gx in range(8)]
N_A = len(GRID_ACTIONS)  # 64


def enc_frame(frame):
    a = np.array(frame[0], dtype=np.float32) / 15.0
    x = a.reshape(16, 4, 16, 4).mean(axis=(1, 3)).flatten()
    return x - x.mean()


class TransitionTriggeredFT09:
    def __init__(self, seed=0):
        rng = np.random.RandomState(seed)
        self.H_nav = rng.randn(K_NAV, DIM).astype(np.float32)
        self.H_fine = rng.randn(K_FINE, DIM).astype(np.float32)
        self.ref = {}; self.G = {}; self.C = {}; self.live = set()
        self.G_fine = {}; self.aliased = set()
        self._pn = self._pa = self._px = None; self._pfn = None
        self.t = 0; self.dim = DIM; self._cn = None; self._fn = None

    def _hash_nav(self, x):
        return int(np.packbits((self.H_nav @ x > 0).astype(np.uint8), bitorder='big').tobytes().hex(), 16)

    def _hash_fine(self, x):
        return int(np.packbits((self.H_fine @ x > 0).astype(np.uint8), bitorder='big').tobytes().hex(), 16)

    def _node(self, x):
        n = self._hash_nav(x)
        while n in self.ref: n = (n, int(self.ref[n] @ x > 0))
        return n

    def observe(self, frame):
        x = enc_frame(frame); n = self._node(x); fn = self._hash_fine(x)
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
        if self.t > 0 and self.t % REFINE_EVERY == 0: self._refine()
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

    def on_reset(self): self._pn = None; self._pfn = None

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


def run(seed, make):
    env = make()
    sub = TransitionTriggeredFT09(seed=seed * 1000)
    obs = env.reset(seed=seed)
    level = 0; l1 = l2 = None; l1_aliased = l2_aliased = None
    t_start = time.time(); next_report = REPORT_INTERVAL

    for step in range(1, MAX_STEPS):
        if obs is None:
            obs = env.reset(seed=seed); sub.on_reset(); continue
        sub.observe(obs)
        action_idx = sub.act()
        action_int = GRID_ACTIONS[action_idx]
        obs, reward, done, info = env.step(action_int)
        cl = info.get('level', 0) if isinstance(info, dict) else 0
        if cl > level:
            if cl == 1 and l1 is None:
                l1 = step; l1_aliased = len(sub.aliased)
                print(f"  s{seed} L1@{l1} aliased={l1_aliased} t={time.time()-t_start:.1f}s", flush=True)
            if cl == 2 and l2 is None:
                l2 = step; l2_aliased = len(sub.aliased)
                print(f"  s{seed} L2@{l2} aliased={l2_aliased} t={time.time()-t_start:.1f}s", flush=True)
            level = cl; sub.on_reset()
        if done:
            obs = env.reset(seed=seed); sub.on_reset()
        elapsed = time.time() - t_start
        if elapsed >= next_report:
            print(f"  s{seed} t={elapsed:.0f}s step={step} aliased={len(sub.aliased)} L1={l1} L2={l2}", flush=True)
            next_report += REPORT_INTERVAL
        if elapsed > PER_SEED_TIME:
            break

    print(f"  s{seed} DONE: L1={l1}(aliased={l1_aliased}) L2={l2}(aliased={l2_aliased}) final_aliased={len(sub.aliased)}", flush=True)
    return dict(seed=seed, l1=l1, l2=l2, l1_aliased=l1_aliased, l2_aliased=l2_aliased, final_aliased=len(sub.aliased))


def main():
    try:
        sys.path.insert(0, '.')
        import arcagi3
        mk = lambda: arcagi3.make("FT09")
    except Exception as e:
        print(f"arcagi3: {e}"); return

    print(f"Step 691: 674 on FT09, {N_SEEDS} seeds, {PER_SEED_TIME}s cap")

    results = []
    for seed in range(N_SEEDS):
        results.append(run(seed, mk))

    print(f"\n{'='*60}")
    l1_n = sum(1 for r in results if r['l1'])
    l2_n = sum(1 for r in results if r['l2'])
    print(f"L1={l1_n}/{N_SEEDS}  L2={l2_n}/{N_SEEDS}")
    for r in results:
        print(f"  s{r['seed']}: L1={r['l1']}(a={r['l1_aliased']}) L2={r['l2']}(a={r['l2_aliased']}) final_aliased={r['final_aliased']}")

    if l2_n >= 3: print("FINDING: L2 generalizes on FT09 — mechanism adapts post-L1")
    elif l2_n > 0: print("MARGINAL: Some L2 on FT09")
    elif l1_n >= 4: print("FINDING: L1 confirmed, L2 wall same as LS20")
    else: print(f"MARGINAL: L1={l1_n}/{N_SEEDS}")


if __name__ == "__main__":
    main()
