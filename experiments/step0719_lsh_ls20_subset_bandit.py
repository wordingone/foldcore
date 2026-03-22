"""
Step 719 — Action subset bandit on LS20.

R3 hypothesis: substrate discovers effective action space by running episodes
with random action subsets and tracking which subsets produce longer episodes.
Breaks argmin's equalization: different episodes use DIFFERENT action pools.

Mechanism:
- Base: 674+running-mean, raw 64x64, 68 universal actions
- Episode start: randomly select K=8 actions from 68 (episode's action pool)
- Argmin runs over only those K actions this episode
- Track: subset_scores[frozenset] = running avg episode length
- Per-action value = avg episode length of subsets containing that action
- Post MIN_EPISODES: bias subset selection toward high-value subsets
  (weighted sampling: P(subset) ∝ prod(action_value[a] for a in subset))

Kill: per-action std < 0.01 after MIN_EPISODES episodes → subsets don't discriminate.
Signal: dir_value > click_value → dirs correlate with longer episodes.

Runtime: 10K steps max (runtime-cap rule: LS20 10K max).
  ~6s/seed, ~30s total for 5 seeds. Signal by 10K or doesn't exist.
"""
import numpy as np
import time
import sys

K_NAV = 12
K_FINE = 20
DIM = 4096
REFINE_EVERY = 5000
MIN_OBS = 8
H_SPLIT = 0.05
MIN_VISITS_ALIAS = 3
WARMUP_STEPS = 500
MAX_STEPS = 10_001
N_SEEDS = 5
MIN_EPISODES = 20   # episodes before weighting kicks in
K_SUBSET = 8        # actions per episode subset
MAX_EP_STEPS = 500  # force episode end after N steps (prevents infinite click-only episodes)

DIR_ACTIONS = [0, 1, 2, 3]
GRID_ACTIONS = [(gx * 8 + 4) + (gy * 8 + 4) * 64
                for gy in range(8) for gx in range(8)]
UNIVERSAL_ACTIONS = DIR_ACTIONS + GRID_ACTIONS
N_UNIV = len(UNIVERSAL_ACTIONS)  # 68


def enc_raw(frame):
    a = np.array(frame[0], dtype=np.float32) / 15.0
    return a.flatten()


class SubsetBanditAD:
    def __init__(self, seed=0):
        rng = np.random.RandomState(seed)
        self.H_nav = rng.randn(K_NAV, DIM).astype(np.float32)
        self.H_fine = rng.randn(K_FINE, DIM).astype(np.float32)
        self.ref = {}; self.G = {}; self.C = {}; self.live_nodes = set()
        self.G_fine = {}; self.aliased = set()
        self._pn = self._pa = self._px = None; self._pfn = None
        self.t = 0; self.dim = DIM; self._cn = None; self._fn = None
        self._mu = np.zeros(DIM, dtype=np.float32); self._mu_n = 0
        # Subset bandit state
        self._rng = np.random.RandomState(seed + 999999)
        self.action_value = np.ones(N_UNIV, dtype=np.float64)  # per-action value
        self.action_ep_count = np.zeros(N_UNIV, dtype=np.int64)
        self.action_ep_total_len = np.zeros(N_UNIV, dtype=np.int64)
        self.subset_scores = {}   # frozenset -> (total_len, count)
        self.total_episodes = 0
        # Current episode state
        self._ep_step = 0
        self._ep_subset = list(range(N_UNIV))  # default: all actions
        self._probe_ptr = 0
        self.steps = 0

    def _new_subset(self):
        """Select K actions for next episode. Post-warmup: weighted by action_value."""
        if self.total_episodes < MIN_EPISODES:
            # Uniform random subset
            idx = sorted(self._rng.choice(N_UNIV, K_SUBSET, replace=False).tolist())
        else:
            # Weighted sampling: normalize action_value to probabilities
            weights = self.action_value / self.action_value.sum()
            idx = sorted(self._rng.choice(N_UNIV, K_SUBSET, replace=False,
                                          p=weights).tolist())
        self._ep_subset = idx

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
        self.live_nodes.add(n); self.t += 1
        self._ep_step += 1
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
        self.steps += 1
        active = self._ep_subset
        if self.steps <= WARMUP_STEPS:
            idx = active[self._probe_ptr % len(active)]
            self._probe_ptr += 1
        else:
            if self._cn in self.aliased and self._fn is not None:
                idx = min(active, key=lambda a: sum(self.G_fine.get((self._fn, a), {}).values()))
            else:
                idx = min(active, key=lambda a: sum(self.G.get((self._cn, a), {}).values()))
        self._pn = self._cn; self._pfn = self._fn; self._pa = idx
        return idx

    def on_episode_end(self):
        """Update subset scores and per-action values."""
        ep_len = self._ep_step
        if ep_len > 0 and self._ep_subset:
            self.total_episodes += 1
            sub_key = frozenset(self._ep_subset)
            total, count = self.subset_scores.get(sub_key, (0, 0))
            self.subset_scores[sub_key] = (total + ep_len, count + 1)
            for a in self._ep_subset:
                self.action_ep_count[a] += 1
                self.action_ep_total_len[a] += ep_len
            # Recompute per-action value = avg ep len for subsets containing a
            avg_overall = float(self.action_ep_total_len.sum()) / max(self.action_ep_count.sum(), 1)
            if avg_overall > 0:
                for a in range(N_UNIV):
                    if self.action_ep_count[a] > 0:
                        avg_when_in = self.action_ep_total_len[a] / self.action_ep_count[a]
                        self.action_value[a] = avg_when_in / avg_overall
                    # else: stays at 1.0 (neutral)

    def on_reset(self):
        self._pn = None; self._pfn = None
        self._mu = np.zeros(self.dim, dtype=np.float32); self._mu_n = 0
        self._ep_step = 0
        self._new_subset()  # new random subset for next episode

    def _h(self, n, a):
        d = self.G.get((n, a))
        if not d or sum(d.values()) < 4: return 0.0
        v = np.array(list(d.values()), np.float64); p = v/v.sum()
        return float(-np.sum(p * np.log2(np.maximum(p, 1e-15))))

    def _refine(self):
        did = 0
        for (n, a), d in list(self.G.items()):
            if n not in self.live_nodes or n in self.ref: continue
            if len(d) < 2 or sum(d.values()) < MIN_OBS: continue
            if self._h(n, a) < H_SPLIT: continue
            top = sorted(d, key=d.get, reverse=True)[:2]
            r0 = self.C.get((n, a, top[0])); r1 = self.C.get((n, a, top[1]))
            if r0 is None or r1 is None or r0[1] < 3 or r1[1] < 3: continue
            diff = (r0[0]/r0[1]) - (r1[0]/r1[1]); nm = np.linalg.norm(diff)
            if nm < 1e-8: continue
            self.ref[n] = (diff/nm).astype(np.float32); self.live_nodes.discard(n); did += 1
            if did >= 3: break


def run(seed, make):
    env = make()
    sub = SubsetBanditAD(seed=seed * 1000)
    obs = env.reset(seed=seed)
    level = 0; l1 = None
    t_start = time.time()
    sub._new_subset()  # initial subset

    for step in range(1, MAX_STEPS):
        if obs is None:
            obs = env.reset(seed=seed)
            sub.on_episode_end()
            sub.on_reset()
            continue
        sub.observe(obs)
        action_idx = sub.act()
        action_int = UNIVERSAL_ACTIONS[action_idx]
        try:
            obs_new, reward, done, info = env.step(action_int)
        except Exception:
            obs_new = obs; done = False; info = {}
        obs = obs_new
        cl = info.get('level', 0) if isinstance(info, dict) else 0
        if cl > level:
            if cl == 1 and l1 is None: l1 = step
            level = cl
            sub.on_episode_end()
            sub.on_reset()
        elif done:
            sub.on_episode_end()
            obs = env.reset(seed=seed)
            sub.on_reset()
        elif sub._ep_step >= MAX_EP_STEPS:
            # Timeout: episode ran too long (likely click-only subset, no movement/death)
            sub.on_episode_end()
            obs = env.reset(seed=seed)
            sub.on_reset()

    elapsed = time.time() - t_start

    dir_vals = [sub.action_value[a] for a in range(4)]
    click_vals = [sub.action_value[a] for a in range(4, N_UNIV)]
    avg_dir = float(np.mean(dir_vals))
    avg_click = float(np.mean(click_vals))
    val_std = float(np.std(sub.action_value))
    top5 = sorted(range(N_UNIV), key=lambda a: sub.action_value[a], reverse=True)[:5]
    bot5 = sorted(range(N_UNIV), key=lambda a: sub.action_value[a])[:5]
    bootloader = "PASS" if l1 else "FAIL"
    print(f"  s{seed:2d}: {bootloader} eps={sub.total_episodes} "
          f"avg_dir_val={avg_dir:.3f} avg_click_val={avg_click:.3f} "
          f"val_std={val_std:.4f} aliased={len(sub.aliased)} t={elapsed:.1f}s", flush=True)
    print(f"         top5={[(a, f'{sub.action_value[a]:.3f}') for a in top5]} "
          f"bot5={[(a, f'{sub.action_value[a]:.3f}') for a in bot5]}", flush=True)
    return dict(seed=seed, l1=l1, total_episodes=sub.total_episodes,
                avg_dir_val=avg_dir, avg_click_val=avg_click, val_std=val_std,
                top5=top5, bot5=bot5)


def main():
    try:
        sys.path.insert(0, '.')
        import arcagi3
        mk = lambda: arcagi3.make("LS20")
    except Exception as e:
        print(f"arcagi3: {e}"); return

    t_start = time.time()
    print(f"Step 719: Action subset bandit on LS20, {N_SEEDS} seeds, {MAX_STEPS-1} steps")
    print(f"K={K_SUBSET} actions/episode, MIN_EPISODES={MIN_EPISODES} before weighting")
    print(f"Kill: val_std < 0.01 across all seeds. Signal: dir_val > click_val.")
    results = []
    for seed in range(N_SEEDS):
        results.append(run(seed, mk))

    print(f"\n{'='*60}")
    boot_n = sum(1 for r in results if r['l1'])
    elapsed = time.time() - t_start
    print(f"Bootloader: {boot_n}/{N_SEEDS}  total_time={elapsed:.1f}s")
    for r in results:
        status = "PASS" if r['l1'] else "FAIL"
        print(f"  s{r['seed']:2d}: {status} eps={r['total_episodes']} "
              f"dir_val={r['avg_dir_val']:.3f} click_val={r['avg_click_val']:.3f} "
              f"std={r['val_std']:.4f}")

    any_signal = sum(1 for r in results if r['val_std'] >= 0.01)
    dir_beats_click = sum(1 for r in results if r['avg_dir_val'] > r['avg_click_val'])
    print(f"\nR3 result:")
    print(f"  Value discrimination (std>=0.01): {any_signal}/5")
    print(f"  Dir value > Click value: {dir_beats_click}/5")

    if any_signal >= 3 and dir_beats_click >= 3:
        print(f"SIGNAL: subset bandit discovers dirs are more valuable than clicks")
        print(f"Next: 719b on FT09/VC33 to verify cross-game")
    elif any_signal >= 3:
        print(f"PARTIAL: value variance present but dirs don't beat clicks")
    else:
        print(f"KILL: per-action values uniform — subsets don't discriminate")


if __name__ == "__main__":
    main()
