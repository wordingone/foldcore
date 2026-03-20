"""
Step 557 — Energy budget autopsy: is the 42-step budget the L2 blocker?

From ls20.py source analysis:
  - jvq.pca() decrements energy every step -> life lost when snw < 0
  - levels[i].get_data("vxy") = 42 = max energy steps per life
  - lbq = 3 lives -> ~129 total steps before done=True (full reset)
  - "iri" sprites refill energy via ggk.rzt() -> these are the "energy palettes"
  - Argmin policy: pure exploration, wastes all 129 steps without purposeful nav

Questions:
  1. How many steps until done=True from fresh start? (verify 129 budget)
  2. After L1, how many steps until done? (L2 budget)
  3. Does argmin ever reach an "iri" sprite (energy refill)?
  4. What does the frame look like at energy death (xhp event)?

Kill: if budget >= 1000 steps, energy isn't the issue.
Find: if budget ~= 129 steps, energy budget is the L2 blocker.
"""
import numpy as np
import time
import sys

N_A = 4
K = 16
DIM = 256
REFINE_EVERY = 2000
MIN_OBS = 4
H_SPLIT = 0.05


def enc(frame):
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
        self.dim = dim

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
            k_key = (self._pn, self._pa, n)
            s, c = self.C.get(k_key, (np.zeros(self.dim, np.float64), 0))
            self.C[k_key] = (s + self._px.astype(np.float64), c + 1)
        self._px = x
        self._cn = n
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
            if r0 is None or r1 is None or r0[1] < 2 or r1[1] < 2:
                continue
            diff = (r0[0] / r0[1]) - (r1[0] / r1[1])
            nm = np.linalg.norm(diff)
            if nm < 1e-8:
                continue
            self.ref[n] = (diff / nm).astype(np.float32)
            self.live.discard(n)


def t0():
    rng = np.random.RandomState(0)
    frame = [rng.randint(0, 16, (64, 64))]
    x = enc(frame)
    assert x.shape == (256,) and abs(float(x.mean())) < 1e-5
    sub = Recode(seed=0)
    sub.observe(frame)
    sub.act()
    assert sub._cn is not None
    print("T0 PASS")


def run_budget_test(env, sub, seed, max_steps, label=""):
    """Run until done=True, return steps taken."""
    obs = env.reset(seed=seed)
    sub.on_reset()
    episode_steps = []
    ep_len = 0
    level = 0
    l1_step = None
    total = 0
    done_count = 0

    for step in range(1, max_steps + 1):
        if obs is None:
            obs = env.reset(seed=seed)
            sub.on_reset()
            continue

        sub.observe(obs)
        action = sub.act()
        obs, reward, done, info = env.step(action)
        ep_len += 1
        total += 1

        cl = info.get('level', 0) if isinstance(info, dict) else 0
        if cl > level:
            level = cl
            sub.on_reset()
            if cl == 1 and l1_step is None:
                l1_step = step
                print(f"  {label} L1 at step={step} ep_len={ep_len}", flush=True)

        if done:
            done_count += 1
            episode_steps.append(ep_len)
            ep_len = 0
            obs = env.reset(seed=seed)
            sub.on_reset()
            if done_count <= 5 or done_count % 100 == 0:
                print(f"  {label} done #{done_count} at step={step} ep_len={episode_steps[-1]}", flush=True)
            if done_count >= 10:
                break

    return episode_steps, l1_step, level, done_count, total


def measure_random_budget(env, seed, max_steps=5000):
    """Purely random policy to get worst-case episode length distribution."""
    rng = np.random.RandomState(seed)
    obs = env.reset(seed=seed)
    episode_steps = []
    ep_len = 0

    for step in range(1, max_steps + 1):
        if obs is None:
            obs = env.reset(seed=seed)
            continue
        action = int(rng.randint(0, N_A))
        obs, reward, done, info = env.step(action)
        ep_len += 1

        if done:
            episode_steps.append(ep_len)
            obs = env.reset(seed=seed)
            ep_len = 0
            if len(episode_steps) >= 20:
                break

    return episode_steps


def frame_stats(obs):
    if obs is None:
        return "None"
    arr = np.array(obs[0], dtype=np.float32)
    return f"shape={arr.shape} min={arr.min():.0f} max={arr.max():.0f} mean={arr.mean():.2f} unique={len(np.unique(arr))}"


def main():
    t0()

    try:
        sys.path.insert(0, '.')
        import arcagi3
        env = arcagi3.make("LS20")
    except Exception as e:
        print(f"arcagi3: {e}")
        return

    t_start = time.time()

    # Phase 1: Random policy budget (no exploration bias)
    print("\nPhase 1: Random policy episode lengths (20 episodes)...", flush=True)
    env2 = arcagi3.make("LS20")
    rand_eps = measure_random_budget(env2, seed=0, max_steps=10000)
    if rand_eps:
        print(f"  Random: n={len(rand_eps)} min={min(rand_eps)} max={max(rand_eps)} "
              f"mean={np.mean(rand_eps):.1f} median={np.median(rand_eps):.1f}", flush=True)
    else:
        print("  Random: no episodes completed in 10K steps", flush=True)

    # Phase 2: Argmin budget (10 episodes)
    print("\nPhase 2: Argmin episode lengths (10 episodes)...", flush=True)
    sub = Recode(seed=0)
    ep_steps, l1_step, level_reached, done_count, total_steps = run_budget_test(
        env, sub, seed=0, max_steps=50000, label="argmin")
    if ep_steps:
        print(f"  Argmin: n={len(ep_steps)} min={min(ep_steps)} max={max(ep_steps)} "
              f"mean={np.mean(ep_steps):.1f}", flush=True)
    print(f"  L1 at step={l1_step} level_reached={level_reached} total_steps={total_steps}", flush=True)

    # Phase 3: Observe frame at energy death
    # Run env for exactly 45 steps from fresh reset (past 42-step budget)
    print("\nPhase 3: Frame at energy death (45 steps from reset)...", flush=True)
    env3 = arcagi3.make("LS20")
    obs3 = env3.reset(seed=0)
    frames = {}
    done_at = None

    for step in range(1, 200):
        if obs3 is None:
            break
        if step in (1, 10, 20, 30, 40, 42, 43, 44, 45, 50):
            frames[step] = obs3
        # Use action=0 repeatedly (pure right movement)
        obs3, reward, done3, info3 = env3.step(1)  # action 1 = move up
        if done3 and done_at is None:
            done_at = step
            print(f"  done=True at step={step} reward={reward} info={info3}", flush=True)
            if obs3 is not None:
                print(f"  frame at death: {frame_stats(obs3)}", flush=True)
            break

    print(f"  First done at step={done_at} (predicted ~43 from energy budget)", flush=True)

    # Phase 4: Count resets needed to reach L1
    print("\nPhase 4: How many resets does argmin need for L1?...", flush=True)
    env4 = arcagi3.make("LS20")
    sub4 = Recode(seed=0)
    obs4 = env4.reset(seed=0)
    level4 = 0
    resets = 0
    for step in range(1, 100001):
        if obs4 is None:
            obs4 = env4.reset(seed=0)
            sub4.on_reset()
            resets += 1
            continue
        sub4.observe(obs4)
        action = sub4.act()
        obs4, reward4, done4, info4 = env4.step(action)
        if done4:
            resets += 1
            obs4 = env4.reset(seed=0)
            sub4.on_reset()
        cl = info4.get('level', 0) if isinstance(info4, dict) else 0
        if cl > level4:
            level4 = cl
            sub4.on_reset()
            if cl == 1:
                print(f"  L1 at step={step} after {resets} resets (each ~129 steps budget)", flush=True)
                break
        if time.time() - t_start > 250:
            print(f"  Timeout at step={step} resets={resets} level={level4}", flush=True)
            break

    elapsed = time.time() - t_start
    print(f"\n{'='*60}")
    print(f"Elapsed: {elapsed:.0f}s")
    print(f"\nKEY FINDINGS:")
    if rand_eps:
        print(f"  Random episode length: mean={np.mean(rand_eps):.0f} steps "
              f"(budget={np.median(rand_eps):.0f} step estimate)")
    if done_at:
        print(f"  Confirmed energy death at step={done_at} (theory: ~43 for vxy=42)")
    if ep_steps:
        print(f"  Argmin episode length: mean={np.mean(ep_steps):.0f} steps")
    print(f"  L2 requires: navigate to energy palette within budget AND then find exit")
    print()
    if done_at and done_at < 150:
        print(f"CONFIRMED: Energy budget ~{done_at} steps is the L2 blocker.")
        print(f"Argmin uses all {done_at} steps randomly. No purposeful navigation possible.")
        print(f"L2 needs: object detection + nav to energy palette within budget.")
    elif done_at and done_at >= 1000:
        print(f"KILL: Budget={done_at} >> 42. Energy is NOT the L2 blocker. Other gap.")
    else:
        print(f"Budget={done_at}. Partial evidence.")


if __name__ == "__main__":
    main()
