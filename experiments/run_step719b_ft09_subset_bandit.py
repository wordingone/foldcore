"""
Step 719b — Action subset bandit on FT09.

Same mechanism as 719 (LS20). Test if magic click positions get highest value.

R3 prediction: subsets containing magic clicks → level transitions → episodes
end via level-up (not death/timeout). Episode length = steps until level-up.
High or low value depends on whether level-up happens BEFORE or AFTER timeout.

Magic clicks: UNIV[35] (action 1852) and UNIV[43] (action 2364).
If magic-click subsets produce long episodes → magic clicks get high value → SIGNAL.
If magic-click subsets produce short episodes → magic clicks get low value → reversed.

NOTE: Survival metric may penalize productive actions (level-ups end episodes early).
This experiment tests what actually happens.

Game: ft09/0d8bbf25. 5 seeds, 10K steps.
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
MIN_EPISODES = 20
K_SUBSET = 8
MAX_EP_STEPS = 500

DIR_ACTIONS = [0, 1, 2, 3]
GRID_ACTIONS = [(gx * 8 + 4) + (gy * 8 + 4) * 64
                for gy in range(8) for gx in range(8)]
UNIVERSAL_ACTIONS = DIR_ACTIONS + GRID_ACTIONS
N_UNIV = len(UNIVERSAL_ACTIONS)

MAGIC_GRID_IDX_A = 3 * 8 + 7   # gy=3, gx=7 -> (60,28)
MAGIC_GRID_IDX_B = 4 * 8 + 7   # gy=4, gx=7 -> (60,36)
MAGIC_UNIV_IDX_A = 4 + MAGIC_GRID_IDX_A  # = 35
MAGIC_UNIV_IDX_B = 4 + MAGIC_GRID_IDX_B  # = 43


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
        self._rng = np.random.RandomState(seed + 999999)
        self.action_value = np.ones(N_UNIV, dtype=np.float64)
        self.action_ep_count = np.zeros(N_UNIV, dtype=np.int64)
        self.action_ep_total_len = np.zeros(N_UNIV, dtype=np.int64)
        self.total_episodes = 0
        self._ep_step = 0
        self._ep_subset = list(range(N_UNIV))
        self._probe_ptr = 0
        self.steps = 0

    def _new_subset(self):
        if self.total_episodes < MIN_EPISODES:
            idx = sorted(self._rng.choice(N_UNIV, K_SUBSET, replace=False).tolist())
        else:
            weights = self.action_value / self.action_value.sum()
            idx = sorted(self._rng.choice(N_UNIV, K_SUBSET, replace=False, p=weights).tolist())
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
        ep_len = self._ep_step
        if ep_len > 0 and self._ep_subset:
            self.total_episodes += 1
            for a in self._ep_subset:
                self.action_ep_count[a] += 1
                self.action_ep_total_len[a] += ep_len
            avg_overall = float(self.action_ep_total_len.sum()) / max(self.action_ep_count.sum(), 1)
            if avg_overall > 0:
                for a in range(N_UNIV):
                    if self.action_ep_count[a] > 0:
                        self.action_value[a] = (self.action_ep_total_len[a] /
                                                self.action_ep_count[a]) / avg_overall

    def on_reset(self):
        self._pn = None; self._pfn = None
        self._mu = np.zeros(self.dim, dtype=np.float32); self._mu_n = 0
        self._ep_step = 0
        self._new_subset()

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
    sub._new_subset()

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
            sub.on_episode_end()
            obs = env.reset(seed=seed)
            sub.on_reset()

    elapsed = time.time() - t_start

    magic_a_val = sub.action_value[MAGIC_UNIV_IDX_A]
    magic_b_val = sub.action_value[MAGIC_UNIV_IDX_B]
    dir_vals = [sub.action_value[a] for a in range(4)]
    click_vals = [sub.action_value[a] for a in range(4, N_UNIV)]
    avg_dir = float(np.mean(dir_vals))
    avg_click = float(np.mean(click_vals))
    val_std = float(np.std(sub.action_value))

    top5 = sorted(range(N_UNIV), key=lambda a: sub.action_value[a], reverse=True)[:5]
    magic_a_rank = sorted(range(N_UNIV), key=lambda a: sub.action_value[a], reverse=True).index(MAGIC_UNIV_IDX_A) + 1
    magic_b_rank = sorted(range(N_UNIV), key=lambda a: sub.action_value[a], reverse=True).index(MAGIC_UNIV_IDX_B) + 1

    bootloader = "PASS" if l1 else "FAIL"
    print(f"  s{seed:2d}: {bootloader} eps={sub.total_episodes} "
          f"magic_A=val{magic_a_val:.3f}(rank{magic_a_rank}) "
          f"magic_B=val{magic_b_val:.3f}(rank{magic_b_rank}) "
          f"std={val_std:.4f} t={elapsed:.1f}s", flush=True)
    print(f"         top5={[(a, f'{sub.action_value[a]:.3f}') for a in top5]} "
          f"avg_dir={avg_dir:.3f} avg_click={avg_click:.3f}", flush=True)
    return dict(seed=seed, l1=l1, total_episodes=sub.total_episodes,
                magic_a_val=magic_a_val, magic_b_val=magic_b_val,
                magic_a_rank=magic_a_rank, magic_b_rank=magic_b_rank,
                avg_dir=avg_dir, avg_click=avg_click, val_std=val_std, top5=top5)


def main():
    try:
        sys.path.insert(0, '.')
        import arcagi3
        mk = lambda: arcagi3.make("FT09")
    except Exception as e:
        print(f"arcagi3: {e}"); return

    t_start = time.time()
    print(f"Step 719b: Action subset bandit on FT09, {N_SEEDS} seeds, {MAX_STEPS-1} steps")
    print(f"K={K_SUBSET}, MAX_EP_STEPS={MAX_EP_STEPS}, MIN_EPISODES={MIN_EPISODES}")
    print(f"Magic: UNIV[{MAGIC_UNIV_IDX_A}] and UNIV[{MAGIC_UNIV_IDX_B}]")
    print(f"Signal: magic clicks in top 5 by value (high rank). Kill: magic in bottom.")
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
              f"magic_A=rank{r['magic_a_rank']}/68 magic_B=rank{r['magic_b_rank']}/68 "
              f"std={r['val_std']:.4f}")

    magic_top10_a = sum(1 for r in results if r['magic_a_rank'] <= 10)
    magic_top10_b = sum(1 for r in results if r['magic_b_rank'] <= 10)
    magic_bot10_a = sum(1 for r in results if r['magic_a_rank'] >= 59)
    magic_bot10_b = sum(1 for r in results if r['magic_b_rank'] >= 59)
    any_signal = sum(1 for r in results if r['val_std'] >= 0.01)

    print(f"\nR3 result:")
    print(f"  Value discrimination (std>=0.01): {any_signal}/5")
    print(f"  Magic A in top 10: {magic_top10_a}/5  |  bottom 10: {magic_bot10_a}/5")
    print(f"  Magic B in top 10: {magic_top10_b}/5  |  bottom 10: {magic_bot10_b}/5")

    if magic_top10_a >= 3 and magic_top10_b >= 3:
        print(f"SIGNAL: magic clicks rank high — subset bandit discovers productive actions")
    elif magic_bot10_a >= 3 or magic_bot10_b >= 3:
        print(f"REVERSED: magic clicks rank low — level-ups end episodes early (survival metric)")
        print(f"  Survival metric penalizes productive actions that terminate episodes via level-up")
    elif any_signal >= 3:
        print(f"PARTIAL: discrimination present but magic clicks not consistently ranked")
    else:
        print(f"KILL: no discrimination")


if __name__ == "__main__":
    main()
