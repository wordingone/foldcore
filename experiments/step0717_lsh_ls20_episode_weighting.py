"""
Step 717 — Episode-outcome action weighting (environmental R3) on LS20.

R3 hypothesis: the substrate discovers which actions are productive by correlating
action usage with episode outcomes. Actions appearing in longer episodes get higher
weight. R1-compliant — episode length is an environmental event.

Mechanism:
- Base: 674+running-mean, raw 64x64, 68 universal actions
- NO pruning. All 68 actions remain live.
- Per action a: track avg episode length in episodes where a was used
- action_value[a] = avg_length_when_used[a] / avg_length_overall
- Weighted argmin: min(actions, key=lambda a: edge_count(cn,a) / action_value[a])
  (= argmin of count * 1/value = prefers high-value actions at same visit count)
- Requires MIN_EPISODES before weighting kicks in

Key R3 property: action weights are SELF-DERIVED from episode outcomes.
Substrate correlates action usage with survival without being told which actions are good.

Kill: if action_value uniform across all types (std < 0.01) → outcomes don't correlate.

Test: LS20, 5 seeds, 120K steps.
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
MAX_STEPS = 120_001
N_SEEDS = 5
MIN_EPISODES_WEIGHT = 20  # don't apply weighting until this many episodes seen

DIR_ACTIONS = [0, 1, 2, 3]
GRID_ACTIONS = [(gx * 8 + 4) + (gy * 8 + 4) * 64
                for gy in range(8) for gx in range(8)]
UNIVERSAL_ACTIONS = DIR_ACTIONS + GRID_ACTIONS
N_UNIV = len(UNIVERSAL_ACTIONS)  # 68


def enc_raw(frame):
    a = np.array(frame[0], dtype=np.float32) / 15.0
    return a.flatten()


class EpisodeWeightAD:
    def __init__(self, seed=0):
        rng = np.random.RandomState(seed)
        self.H_nav = rng.randn(K_NAV, DIM).astype(np.float32)
        self.H_fine = rng.randn(K_FINE, DIM).astype(np.float32)
        self.ref = {}; self.G = {}; self.C = {}; self.live_nodes = set()
        self.G_fine = {}; self.aliased = set()
        self._pn = self._pa = self._px = None; self._pfn = None
        self.t = 0; self.dim = DIM; self._cn = None; self._fn = None
        self._mu = np.zeros(DIM, dtype=np.float32); self._mu_n = 0
        # Episode outcome tracking
        self.action_value = np.ones(N_UNIV, dtype=np.float64)  # start neutral
        self.action_ep_count = np.zeros(N_UNIV, dtype=np.int64)  # episodes where a was used
        self.action_ep_total_len = np.zeros(N_UNIV, dtype=np.int64)  # total length of those episodes
        self.total_episodes = 0
        self.total_ep_len = 0
        # Current episode tracking
        self._ep_step = 0
        self._ep_actions = set()  # actions used in current episode
        self._probe_ptr = 0
        self.steps = 0

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
        all_actions = list(range(N_UNIV))
        if self.steps <= WARMUP_STEPS:
            idx = all_actions[self._probe_ptr % N_UNIV]
            self._probe_ptr += 1
        else:
            if self._cn in self.aliased and self._fn is not None:
                if self.total_episodes >= MIN_EPISODES_WEIGHT:
                    idx = min(all_actions, key=lambda a:
                        sum(self.G_fine.get((self._fn, a), {}).values()) / self.action_value[a])
                else:
                    idx = min(all_actions, key=lambda a: sum(self.G_fine.get((self._fn, a), {}).values()))
            else:
                if self.total_episodes >= MIN_EPISODES_WEIGHT:
                    idx = min(all_actions, key=lambda a:
                        sum(self.G.get((self._cn, a), {}).values()) / self.action_value[a])
                else:
                    idx = min(all_actions, key=lambda a: sum(self.G.get((self._cn, a), {}).values()))
        self._ep_actions.add(idx)
        self._pn = self._cn; self._pfn = self._fn; self._pa = idx
        return idx

    def on_episode_end(self):
        """Call at end of each episode (done=True or level up). Updates action_value."""
        ep_len = self._ep_step
        if ep_len > 0:
            self.total_episodes += 1
            self.total_ep_len += ep_len
            for a in self._ep_actions:
                self.action_ep_count[a] += 1
                self.action_ep_total_len[a] += ep_len
            # Recompute action_value
            avg_overall = self.total_ep_len / self.total_episodes
            if avg_overall > 0:
                for a in range(N_UNIV):
                    if self.action_ep_count[a] > 0:
                        avg_when_used = self.action_ep_total_len[a] / self.action_ep_count[a]
                        self.action_value[a] = avg_when_used / avg_overall
                    # else: action_value[a] stays at 1.0 (neutral)

    def on_reset(self):
        self._pn = None; self._pfn = None
        self._mu = np.zeros(self.dim, dtype=np.float32); self._mu_n = 0
        self._ep_step = 0
        self._ep_actions = set()

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
    sub = EpisodeWeightAD(seed=seed * 1000)
    obs = env.reset(seed=seed)
    level = 0; l1 = None
    t_start = time.time()

    for step in range(1, MAX_STEPS):
        if obs is None:
            obs = env.reset(seed=seed); sub.on_reset(); continue
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
        if done:
            sub.on_episode_end()
            obs = env.reset(seed=seed)
            sub.on_reset()

    elapsed = time.time() - t_start

    # Report action_value distribution
    dir_vals = [sub.action_value[a] for a in range(4)]
    click_vals = [sub.action_value[a] for a in range(4, N_UNIV)]
    avg_dir = float(np.mean(dir_vals))
    avg_click = float(np.mean(click_vals))
    top5 = sorted(range(N_UNIV), key=lambda a: sub.action_value[a], reverse=True)[:5]
    val_std = float(np.std(sub.action_value))
    bootloader = "PASS" if l1 else "FAIL"
    print(f"  s{seed:2d}: {bootloader} eps={sub.total_episodes} avg_dir_val={avg_dir:.3f} "
          f"avg_click_val={avg_click:.3f} val_std={val_std:.4f} aliased={len(sub.aliased)} t={elapsed:.1f}s",
          flush=True)
    print(f"         top5_by_value={[(a, f'{sub.action_value[a]:.3f}') for a in top5]}", flush=True)
    return dict(seed=seed, l1=l1, total_episodes=sub.total_episodes,
                avg_dir_val=avg_dir, avg_click_val=avg_click, val_std=val_std,
                top5=top5)


def main():
    try:
        sys.path.insert(0, '.')
        import arcagi3
        mk = lambda: arcagi3.make("LS20")
    except Exception as e:
        print(f"arcagi3: {e}"); return

    t_start = time.time()
    print(f"Step 717: Episode-outcome action weighting on LS20, {N_SEEDS} seeds, {MAX_STEPS-1} steps")
    print(f"R3: action_value = avg_ep_len_when_used / avg_ep_len. Weighted argmin post {MIN_EPISODES_WEIGHT} eps.")
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
              f"dir_val={r['avg_dir_val']:.3f} click_val={r['avg_click_val']:.3f} std={r['val_std']:.4f}")

    # Kill criterion: action_value uniform (std < 0.01)
    any_signal = sum(1 for r in results if r['val_std'] >= 0.01)
    dir_beats_click = sum(1 for r in results if r['avg_dir_val'] > r['avg_click_val'])
    print(f"\nR3 result:")
    print(f"  Value discrimination (std>=0.01): {any_signal}/5 seeds")
    print(f"  Dir value > Click value: {dir_beats_click}/5 seeds")
    if any_signal >= 3 and dir_beats_click >= 3:
        print(f"SIGNAL: Episode outcomes correlate with action type — dirs valued over clicks")
        print(f"Next: 717b on FT09 and VC33 to verify cross-game")
    elif any_signal >= 3:
        print(f"PARTIAL: Value variance present but dirs don't beat clicks — correlation exists but wrong signal")
    else:
        print(f"KILL: action_value uniform (no episode-outcome correlation)")


if __name__ == "__main__":
    main()
