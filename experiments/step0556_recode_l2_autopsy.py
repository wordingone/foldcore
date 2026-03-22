"""
Step 556 — L2 autopsy: what IS Level 2, and can any simple action reach it?

5-min cap. LS20 seed=0.

Phase 1: Reach L1 with aggressive Recode argmin (reliable 3/3 from Step 554).
Phase 2: Log L1 game state: info dict, frame stats.
Phase 3: From L1 state, try 4 single-action sequences x 1000 steps each.
Phase 4: Try 6 two-action alternating sequences x 500 steps each.
Phase 5: Frame comparison — are frames changing after L1?

Kill: All frames identical after L1 -> game frozen by design, not a substrate problem.
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
        self.ns = 0
        self.dim = dim
        self._last_visit = {}

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
        self._last_visit[n] = self.t
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
            self.ns += 1


def frame_stats(obs):
    if obs is None:
        return "None"
    arr = np.array(obs[0], dtype=np.float32)
    return (f"shape={arr.shape} min={arr.min():.0f} max={arr.max():.0f} "
            f"mean={arr.mean():.2f} std={arr.std():.2f} "
            f"unique={len(np.unique(arr))}")


def frames_equal(obs_a, obs_b):
    if obs_a is None or obs_b is None:
        return obs_a is obs_b
    return np.array_equal(np.array(obs_a[0]), np.array(obs_b[0]))


def frame_diff(obs_a, obs_b):
    """Mean absolute difference between frames."""
    if obs_a is None or obs_b is None:
        return float('nan')
    a = np.array(obs_a[0], dtype=np.float32)
    b = np.array(obs_b[0], dtype=np.float32)
    return float(np.abs(a - b).mean())


def t0():
    rng = np.random.RandomState(0)
    frame = [rng.randint(0, 16, (64, 64))]
    x = enc(frame)
    assert x.shape == (256,) and abs(float(x.mean())) < 1e-5
    s = frame_stats(frame)
    assert 'shape' in s
    f2 = [rng.randint(0, 16, (64, 64))]
    assert frames_equal(frame, frame)
    assert not frames_equal(frame, f2)
    d = frame_diff(frame, frame)
    assert d == 0.0
    d2 = frame_diff(frame, f2)
    assert d2 > 0.0
    print("T0 PASS")


def step_env(env, obs, action):
    """Step env, handle done via reset. Returns (obs, reward, done, info)."""
    if obs is None:
        obs = env.reset(seed=0)
        return obs, 0.0, False, {}
    obs, reward, done, info = env.step(action)
    if done:
        obs = env.reset(seed=0)
    return obs, reward, done, info


def main():
    t0()

    try:
        sys.path.insert(0, '.')
        import arcagi3
        env = arcagi3.make("LS20")
    except Exception as e:
        print(f"arcagi3: {e}")
        return

    # --- Phase 1: Reach L1 with argmin ---
    print("Phase 1: Reaching L1 with argmin (cap=200K steps)...", flush=True)
    sub = Recode(seed=0)
    obs = env.reset(seed=0)
    level = 0
    go = 0
    t_start = time.time()
    l1_step = None
    l1_obs = None
    l1_info = None

    for step in range(1, 200_001):
        if obs is None:
            obs = env.reset(seed=0)
            sub.on_reset()
            continue

        sub.observe(obs)
        action = sub.act()
        obs, reward, done, info = env.step(action)

        if done:
            go += 1
            obs = env.reset(seed=0)
            sub.on_reset()

        cl = info.get('level', 0) if isinstance(info, dict) else 0
        if cl > level:
            level = cl
            sub.on_reset()
            if cl == 1 and l1_step is None:
                l1_step = step
                l1_obs = obs
                l1_info = info
                print(f"L1 reached at step={step} go={go}", flush=True)
                print(f"  info keys: {list(info.keys()) if isinstance(info, dict) else type(info)}", flush=True)
                print(f"  info: {info}", flush=True)
                print(f"  frame: {frame_stats(obs)}", flush=True)
                break
            if cl >= 2:
                print(f"L2 reached during argmin at step={step}! info={info}", flush=True)
                return

        if time.time() - t_start > 120:
            print(f"Phase 1 timeout at {step} steps, level={level}", flush=True)
            break

    if l1_step is None:
        print(f"L1 not reached in 200K steps. Last frame: {frame_stats(obs)}", flush=True)
        print(f"Last info: {info}", flush=True)
        return

    # --- Phase 2: Frame comparison after L1 ---
    print(f"\nPhase 2: Frame comparison (continuing argmin for 5000 steps)...", flush=True)
    saved_frames = {0: l1_obs}  # relative step -> obs

    for rel in range(1, 5001):
        if obs is None:
            obs = env.reset(seed=0)
            sub.on_reset()
            continue

        sub.observe(obs)
        action = sub.act()
        obs, reward, done, info = env.step(action)

        if done:
            obs = env.reset(seed=0)
            sub.on_reset()

        cl = info.get('level', 0) if isinstance(info, dict) else 0
        if cl > level:
            level = cl
            if cl >= 2:
                print(f"L2 reached at L1+{rel}! info={info}", flush=True)
                break

        if rel in (100, 1000, 5000):
            saved_frames[rel] = obs
            eq = frames_equal(l1_obs, obs)
            diff = frame_diff(l1_obs, obs)
            print(f"  L1+{rel:>5}: {frame_stats(obs)} | same_as_L1={eq} diff={diff:.3f}", flush=True)

        if time.time() - t_start > 180:
            print(f"  Phase 2 timeout at L1+{rel}", flush=True)
            break

    all_same = all(frames_equal(l1_obs, v) for k, v in saved_frames.items() if k > 0)
    any_same = any(frames_equal(l1_obs, v) for k, v in saved_frames.items() if k > 0)
    print(f"  Frame summary: all_same={all_same} any_same={any_same}", flush=True)

    if level >= 2:
        return  # already found it

    # --- Phase 3: Single-action brute force ---
    print(f"\nPhase 3: Single-action brute force (1000 steps each)...", flush=True)
    for fixed_action in range(4):
        if time.time() - t_start > 230:
            print("  Phase 3 timeout")
            break
        l2_found = False
        for sub_step in range(1000):
            if obs is None:
                obs = env.reset(seed=0)
                continue
            obs, reward, done, info = env.step(fixed_action)
            if done:
                obs = env.reset(seed=0)
            cl = info.get('level', 0) if isinstance(info, dict) else 0
            if cl > level:
                level = cl
                if cl >= 2:
                    print(f"  action={fixed_action} -> L2 at sub-step={sub_step}! info={info}", flush=True)
                    l2_found = True
                    break
        if l2_found:
            break
        print(f"  action={fixed_action}: no L2 in 1000 steps. frame={frame_stats(obs)}", flush=True)

    if level >= 2:
        return

    # --- Phase 4: Two-action alternating ---
    print(f"\nPhase 4: Two-action alternating (500 steps each)...", flush=True)
    pairs = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
    for a0, a1 in pairs:
        if time.time() - t_start > 270:
            print("  Phase 4 timeout")
            break
        l2_found = False
        for sub_step in range(500):
            action = a0 if sub_step % 2 == 0 else a1
            if obs is None:
                obs = env.reset(seed=0)
                continue
            obs, reward, done, info = env.step(action)
            if done:
                obs = env.reset(seed=0)
            cl = info.get('level', 0) if isinstance(info, dict) else 0
            if cl > level:
                level = cl
                if cl >= 2:
                    print(f"  ({a0},{a1}) -> L2 at sub-step={sub_step}! info={info}", flush=True)
                    l2_found = True
                    break
        if l2_found:
            break
        print(f"  ({a0},{a1}): no L2 in 500 steps", flush=True)

    # --- Summary ---
    elapsed = time.time() - t_start
    print(f"\n{'='*60}")
    print(f"Elapsed: {elapsed:.0f}s")
    print(f"L1 reached: step={l1_step} go_before={go}")
    print(f"L2 reached: {'YES level=' + str(level) if level >= 2 else 'NO'}")
    print(f"L1 info: {l1_info}")

    if level < 2:
        if all_same and saved_frames.get(100) is not None:
            print("\nFROZEN: All frames identical after L1.")
            print("DIAGNOSIS: L2 unreachable by design — game frozen post-L1.")
            print("Not a substrate problem. Game version issue.")
        else:
            diffs = {k: frame_diff(l1_obs, v) for k, v in saved_frames.items() if k > 0}
            print(f"\nLIVE: Frames changing after L1. Diffs: {diffs}")
            print("STRUCTURAL GAP: Game is responsive but no action sequence reaches L2.")
            print("Simple policies (argmin, fixed, alternating) all fail.")
            print("L2 may require learned/complex navigation beyond simple action patterns.")
    else:
        print(f"\nL2 REACHABLE: Found via brute force.")


if __name__ == "__main__":
    main()
