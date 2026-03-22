"""
Step 577c -- Self-discovering substrate: online multi-buffer reinforcement.

Replaces evolutionary approach (577b) which failed: 0/1000 random programs
have the required threshold→label_cc→nearest→navigate pipeline sequence.
No fitness gradient → no evolution.

Architecture (Spec, 2026-03-20):
  - LSH k=12 + argmin (baseline, 80% of actions)
  - N=5 parallel buffers accumulating pixel statistics simultaneously:
      buf[0]: pixel_mode  -- argmax of per-pixel color histogram (mode map)
      buf[1]: pixel_mean  -- running average per pixel
      buf[2]: pixel_var   -- running variance per pixel (Welford's)
      buf[3]: diff_acc    -- cumulative |frame_t - frame_{t-1}|
      buf[4]: pixel_min   -- per-pixel minimum observed

  Each buffer produces targets via CC detection on its normalized 2D map.
  Action: 80% argmin, 20% navigate toward target from buffer[i].
  Buffer selection: softmax(buf_scores / BUF_TEMP).

Scoring (online, after each episode):
  Most-used buffer in the episode gets:
    delta = (ep_steps - avg_ep_len) / avg_ep_len  (normalized)
    buf_scores[active] += delta
  Longer episode → positive credit. Shorter → penalize.

Hypothesis: buf[0] (mode map) should accumulate the highest score.
Kill: buf_scores flat after 200K steps → no buffer provides useful signal.
Signal: any buf score clearly positive AND L1 reached.

5 seeds, TIME_CAP=280s each.
"""
import time
import numpy as np
import sys
from scipy.ndimage import label as ndlabel

K = 12
DIM = 256
N_A = 4
N_BUF = 5
BUF_NAMES = ['mode', 'mean', 'var', 'diff', 'min']
USE_BUF_PROB = 0.2     # fraction of steps using buffer nav (vs argmin)
BUF_TEMP = 1.0         # softmax temperature for buffer selection
MIN_CC = 2             # min CC size for targets
MAX_CC = 150           # max CC size for targets
WARMUP = 20            # frames before buffer-based navigation starts
MAX_STEPS = 200_000
TIME_CAP = 280


# ── target detection ──────────────────────────────────────────────────────────

def find_nearest_target(map2d, agent_yx):
    """
    Find nearest CC in a 2D map.
    Threshold: values >= mean + 0.5*std (highlights significant pixels).
    Returns (cy, cx) or None.
    """
    if map2d is None or agent_yx is None:
        return None
    std = float(map2d.std())
    if std < 1e-6:
        return None   # uniform map, no information
    thresh = float(map2d.mean()) + 0.5 * std
    binary = map2d >= thresh
    labeled, n = ndlabel(binary)
    if n == 0:
        return None
    ay, ax = agent_yx
    best_dist = 1e9
    bcy = bcx = None
    for cid in range(1, n + 1):
        region = labeled == cid
        sz = int(region.sum())
        if sz < MIN_CC or sz > MAX_CC:
            continue
        ys, xs = np.where(region)
        cy, cx = float(ys.mean()), float(xs.mean())
        d = (cy - ay)**2 + (cx - ax)**2
        if d < best_dist:
            best_dist = d; bcy = cy; bcx = cx
    if bcy is None:
        return None
    return (bcy, bcx)


# ── multi-buffer substrate ────────────────────────────────────────────────────

class MultiBufferSub:
    def __init__(self, lsh_seed=0):
        self.H = np.random.RandomState(lsh_seed).randn(K, DIM).astype(np.float32)
        self.G = {}
        self._pn = self._pa = self._cn = None
        self.cells = set()

        # 5 statistics buffers
        self.freq = np.zeros((64, 64, 16), dtype=np.int32)       # buf[0]: histogram
        self.mean_a = np.zeros((64, 64), dtype=np.float32)        # buf[1]: mean
        self.m2 = np.zeros((64, 64), dtype=np.float32)            # buf[2]: M2 (variance)
        self.diff_a = np.zeros((64, 64), dtype=np.float32)        # buf[3]: diff acc
        self.min_a = np.full((64, 64), 15, dtype=np.int32)        # buf[4]: min

        self.n_frames = 0
        self.prev_arr = None
        self.agent_yx = None

        # Buffer meta
        self.buf_scores = np.zeros(N_BUF, dtype=np.float32)
        self.buf_use_ep = np.zeros(N_BUF, dtype=np.int32)   # use count this episode

        # Episode tracking
        self.ep_steps = 0
        self.avg_ep_len = 200.0    # initial guess; updated via EMA
        self.total_episodes = 0
        self.l1_count = 0

        # For diagnostics
        self.nav_actions = 0    # actions from buffer navigation
        self.argmin_actions = 0

    def observe(self, frame):
        arr = np.array(frame[0], dtype=np.int32)

        # Agent position from pixel diff
        if self.prev_arr is not None:
            diff_px = np.abs(arr - self.prev_arr)
            nc = int((diff_px > 0).sum())
            if 1 <= nc < 200:
                ys, xs = np.where(diff_px > 0)
                self.agent_yx = (float(ys.mean()), float(xs.mean()))
            # buf[3]: cumulative diff
            self.diff_a += diff_px.astype(np.float32)

        self.prev_arr = arr.copy()
        self.n_frames += 1
        n = self.n_frames

        # buf[0]: pixel histogram
        r = np.arange(64)[:, None]; c = np.arange(64)[None, :]
        self.freq[r, c, arr] += 1

        # buf[1] + buf[2]: Welford's online mean and variance
        a = arr.astype(np.float32)
        delta = a - self.mean_a
        self.mean_a += delta / n
        delta2 = a - self.mean_a
        self.m2 += delta * delta2

        # buf[4]: per-pixel minimum
        self.min_a = np.minimum(self.min_a, arr)

        # LSH hash for cell counting and argmin
        x = arr.astype(np.float32).reshape(16, 4, 16, 4).mean(axis=(1, 3)).flatten() / 15.0
        x -= x.mean()
        cell = int(np.packbits((self.H @ x > 0).astype(np.uint8),
                               bitorder='big').tobytes().hex(), 16)
        self.cells.add(cell)
        if self._pn is not None:
            d = self.G.setdefault((self._pn, self._pa), {})
            d[cell] = d.get(cell, 0) + 1
        self._cn = cell

    def _get_map(self, buf_idx):
        if self.n_frames < WARMUP:
            return None
        if buf_idx == 0:
            return np.argmax(self.freq, axis=2).astype(np.float32)
        elif buf_idx == 1:
            return self.mean_a.copy()
        elif buf_idx == 2:
            if self.n_frames < 2: return None
            return (self.m2 / max(self.n_frames - 1, 1)).astype(np.float32)
        elif buf_idx == 3:
            return self.diff_a.copy()
        elif buf_idx == 4:
            return self.min_a.astype(np.float32)
        return None

    def act(self):
        # Decide: buffer navigation or argmin
        if (np.random.random() < USE_BUF_PROB
                and self.agent_yx is not None
                and self.n_frames >= WARMUP):
            # Select buffer by softmax over buf_scores
            s = self.buf_scores - self.buf_scores.max()
            probs = np.exp(s / BUF_TEMP); probs /= probs.sum()
            buf_idx = int(np.random.choice(N_BUF, p=probs))

            target = find_nearest_target(self._get_map(buf_idx), self.agent_yx)
            if target is not None:
                ty, tx = target; ay, ax = self.agent_yx
                dy, dx = ty - ay, tx - ax
                action = (0 if dy < 0 else 1) if abs(dy) >= abs(dx) else (2 if dx < 0 else 3)
                self.buf_use_ep[buf_idx] += 1
                self.nav_actions += 1
                self._pn = self._cn; self._pa = action
                return action

        # Argmin fallback
        counts = [sum(self.G.get((self._cn, a), {}).values()) for a in range(N_A)]
        action = int(np.argmin(counts))
        self.argmin_actions += 1
        self._pn = self._cn; self._pa = action
        return action

    def on_episode_end(self):
        """Update buffer scores based on episode length vs running average."""
        self.total_episodes += 1

        # Update EMA of episode length
        alpha = 0.05
        self.avg_ep_len = (1 - alpha) * self.avg_ep_len + alpha * self.ep_steps

        # Credit most-used buffer (if any buffer was used)
        if self.buf_use_ep.sum() > 0:
            active = int(np.argmax(self.buf_use_ep))
            delta = (self.ep_steps - self.avg_ep_len) / max(self.avg_ep_len, 1)
            self.buf_scores[active] += delta

        self.ep_steps = 0
        self.buf_use_ep[:] = 0

    def on_reset(self):
        self._pn = None
        self.prev_arr = None
        self.agent_yx = None


# ── tests ─────────────────────────────────────────────────────────────────────

def t0():
    sub = MultiBufferSub(lsh_seed=0)
    frame = [np.random.RandomState(1).randint(0, 16, (64, 64), dtype=np.uint8)]
    # Warm up buffers
    for _ in range(30):
        sub.observe(frame)
        sub.act()
    a = sub.act()
    assert 0 <= a < 4, f"Invalid action {a}"
    print(f"T0 PASS: action={a} nav_actions={sub.nav_actions} "
          f"argmin_actions={sub.argmin_actions}")

    # Verify target detection works
    m = np.zeros((64, 64), dtype=np.float32)
    m[20:24, 30:34] = 1.0   # small bright region
    t = find_nearest_target(m, (10.0, 10.0))
    assert t is not None, "Expected target"
    print(f"T0 PASS: find_nearest_target={t}")

    # Verify buf_scores update
    sub2 = MultiBufferSub(lsh_seed=99)
    for _ in range(30):
        sub2.observe(frame)
    sub2.agent_yx = (32.0, 32.0)
    sub2.buf_use_ep[0] = 5    # buf[0] used 5 times
    sub2.ep_steps = 400
    sub2.avg_ep_len = 200.0
    sub2.on_episode_end()
    assert sub2.buf_scores[0] > 0, f"Expected positive score, got {sub2.buf_scores}"
    print(f"T0 PASS: buf_scores after long episode={sub2.buf_scores.round(3)}")


# ── seed runner ────────────────────────────────────────────────────────────────

def run_seed(mk, seed, time_cap=TIME_CAP):
    env = mk()
    sub = MultiBufferSub(lsh_seed=seed * 100 + 7)
    obs = env.reset(seed=seed)
    sub.on_reset()
    prev_cl = 0; fresh = True
    l1 = l2 = go = step = 0
    t_start = time.time()

    while step < MAX_STEPS and time.time() - t_start < time_cap:
        if obs is None:
            sub.on_episode_end()
            obs = env.reset(seed=seed); sub.on_reset()
            prev_cl = 0; fresh = True; go += 1
            continue

        sub.observe(obs)
        action = sub.act()
        obs, _, done, info = env.step(action)
        step += 1; sub.ep_steps += 1

        if done:
            sub.on_episode_end()
            obs = env.reset(seed=seed); sub.on_reset()
            prev_cl = 0; fresh = True; go += 1
            continue

        cl = info.get('level', 0) if isinstance(info, dict) else 0
        if fresh:
            prev_cl = cl; fresh = False
        elif cl >= 1 and prev_cl < 1:
            l1 += 1
            if l1 <= 3:
                print(f"  s{seed} L1@{step} go={go} buf_scores={sub.buf_scores.round(2)}",
                      flush=True)
        elif cl >= 2 and prev_cl < 2:
            l2 += 1
        prev_cl = cl

    elapsed = time.time() - t_start
    nav_rate = sub.nav_actions / max(step, 1)
    print(f"  s{seed}: L1={l1} L2={l2} go={go} step={step} "
          f"cells={len(sub.cells)} nav_rate={nav_rate:.2f} {elapsed:.0f}s", flush=True)
    print(f"    buf_scores: {dict(zip(BUF_NAMES, sub.buf_scores.round(3)))}", flush=True)
    return dict(seed=seed, l1=l1, l2=l2, go=go, steps=step,
                cells=len(sub.cells), buf_scores=sub.buf_scores.copy(),
                nav_rate=nav_rate)


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    t0()

    try:
        sys.path.insert(0, '.')
        import arcagi3
        mk = lambda: arcagi3.make("LS20")
    except Exception as e:
        print(f"arcagi3: {e}"); return

    print(f"\nStep 577c: Multi-buffer substrate", flush=True)
    print(f"  N_BUF={N_BUF} USE_BUF_PROB={USE_BUF_PROB} BUF_TEMP={BUF_TEMP}", flush=True)

    results = []
    t_total = time.time()
    for seed in range(5):
        if time.time() - t_total > 1380:
            print("TOTAL TIME CAP HIT"); break
        print(f"\nseed {seed}:", flush=True)
        r = run_seed(mk, seed, time_cap=TIME_CAP)
        results.append(r)

    # Aggregate buf_scores across seeds
    all_scores = np.stack([r['buf_scores'] for r in results])
    avg_scores = all_scores.mean(axis=0)
    any_l1 = sum(1 for r in results if r['l1'] > 0)
    avg_cells = np.mean([r['cells'] for r in results])

    print(f"\n{'='*60}")
    print("STEP 577c: Online multi-buffer reinforcement")
    for r in results:
        print(f"  s{r['seed']}: L1={r['l1']} L2={r['l2']} "
              f"buf_scores={dict(zip(BUF_NAMES, r['buf_scores'].round(2)))}")
    print(f"\nAverage buf_scores: {dict(zip(BUF_NAMES, avg_scores.round(3)))}")
    print(f"L1 seeds: {any_l1}/{len(results)}")
    print(f"Avg cells: {avg_cells:.0f}")
    print(f"Baseline (argmin only): 6/10 at 50K (step 459)")

    winner = BUF_NAMES[int(np.argmax(avg_scores))]
    if avg_scores.max() > 0:
        print(f"\nDISCOVERED: buf[{winner}] most useful "
              f"(score={avg_scores.max():.3f})")
    else:
        print(f"\nFAIL: All buf_scores flat. No buffer provides useful navigation signal.")
        print(f"Possible causes: scoring mechanism not sensitive, "
              f"or all buffers equally useless/useful.")


if __name__ == "__main__":
    main()
