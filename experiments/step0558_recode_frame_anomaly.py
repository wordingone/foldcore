"""
Step 558 — Frame anomaly detection: is enc-space frame_diff bimodal?

Hypothesis: frame_diff = L2_norm(enc(t+1) - enc(t)) has two populations:
  - Low: normal movement (same region, small visual change)
  - High: significant events (deaths/resets, level transitions)

If bimodal: substrate can detect computation from noise via threshold.
If unimodal: no detectable signal — different encoding needed.

5-min cap. 200K steps. LS20 seed=0.
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
        return n, x  # return encoded x too

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


def detect_bimodal(diffs, n_bins=50):
    """
    Check for bimodality by finding a dip between two peaks.
    Returns (is_bimodal, threshold, low_count, high_count).
    """
    if len(diffs) < 100:
        return False, None, 0, 0

    arr = np.array(diffs)
    lo, hi = arr.min(), arr.max()
    bins = np.linspace(lo, hi, n_bins + 1)
    counts, edges = np.histogram(arr, bins=bins)
    centers = (edges[:-1] + edges[1:]) / 2

    # Find the two highest peaks
    peak1_idx = int(np.argmax(counts))
    # Zero out region around peak1 to find peak2
    masked = counts.copy()
    window = max(3, n_bins // 8)
    lo_mask = max(0, peak1_idx - window)
    hi_mask = min(n_bins, peak1_idx + window + 1)
    masked[lo_mask:hi_mask] = 0
    peak2_idx = int(np.argmax(masked))

    if masked[peak2_idx] == 0:
        return False, None, 0, 0

    # Two peaks found — find dip between them
    left_peak = min(peak1_idx, peak2_idx)
    right_peak = max(peak1_idx, peak2_idx)

    if right_peak <= left_peak + 1:
        return False, None, 0, 0

    valley_idx = left_peak + int(np.argmin(counts[left_peak:right_peak + 1]))
    valley_val = counts[valley_idx]
    threshold = centers[valley_idx]

    # Bimodal if valley is significantly lower than both peaks
    left_peak_val = counts[left_peak]
    right_peak_val = counts[right_peak]
    min_peak = min(left_peak_val, right_peak_val)
    bimodal = valley_val < 0.5 * min_peak and min_peak > 10

    low_count = int(np.sum(arr <= threshold))
    high_count = int(np.sum(arr > threshold))

    return bimodal, threshold, low_count, high_count


def percentiles(arr, ps):
    return {p: float(np.percentile(arr, p)) for p in ps}


def t0():
    rng = np.random.RandomState(0)
    f1 = [rng.randint(0, 16, (64, 64))]
    f2 = [rng.randint(0, 16, (64, 64))]
    x1 = enc(f1)
    x2 = enc(f2)
    assert x1.shape == (256,)
    diff = float(np.linalg.norm(x1 - x2))
    assert diff > 0.0

    # Same frame -> diff = 0
    diff_same = float(np.linalg.norm(x1 - x1))
    assert diff_same == 0.0

    # detect_bimodal: clearly bimodal (two separated clusters)
    lo = list(np.random.RandomState(1).normal(0.1, 0.01, 500))
    hi = list(np.random.RandomState(2).normal(1.0, 0.01, 100))
    bim, thresh, lc, hc = detect_bimodal(lo + hi)
    assert bim, f"Should detect bimodal clusters, thresh={thresh}"
    assert 0.1 < thresh < 0.95, f"Threshold should be between clusters: {thresh}"

    # Unimodal: should NOT detect bimodal
    uni = list(np.random.RandomState(3).normal(0.5, 0.1, 600))
    bim2, _, _, _ = detect_bimodal(uni)
    assert not bim2, "Unimodal should not be detected as bimodal"

    print("T0 PASS")


def main():
    t0()

    try:
        sys.path.insert(0, '.')
        import arcagi3
        env = arcagi3.make("LS20")
    except Exception as e:
        print(f"arcagi3: {e}")
        return

    sub = Recode(seed=0)
    obs = env.reset(seed=0)
    t_start = time.time()
    level = 0

    # Collect frame_diffs with event labels
    all_diffs = []          # all frame_diffs
    event_diffs = []        # (diff, event_type) for notable events
    prev_x = None
    go = 0
    l1_step = None

    print("Running 200K steps, tracking frame_diff...", flush=True)

    for step in range(1, 200_001):
        if obs is None:
            obs = env.reset(seed=0)
            sub.on_reset()
            prev_x = None
            continue

        node, x = sub.observe(obs)
        action = sub.act()
        obs, reward, done, info = env.step(action)

        # Compute frame_diff for THIS step (current obs -> next obs)
        if obs is not None and prev_x is not None:
            diff = float(np.linalg.norm(x - prev_x))
            all_diffs.append(diff)

            # Tag notable events
            cl = info.get('level', 0) if isinstance(info, dict) else 0
            if done:
                event_diffs.append((diff, 'death'))
            elif cl > level:
                event_diffs.append((diff, f'level_{cl}'))
            # else: regular movement (not tagged)

        prev_x = x

        cl = info.get('level', 0) if isinstance(info, dict) else 0
        if cl > level:
            level = cl
            sub.on_reset()
            if cl == 1 and l1_step is None:
                l1_step = step
                print(f"  L1 at step={step}", flush=True)

        if done:
            go += 1
            obs = env.reset(seed=0)
            sub.on_reset()
            prev_x = None

        if step % 50_000 == 0:
            elapsed = time.time() - t_start
            print(f"  @{step} go={go} diffs={len(all_diffs)} level={level} {elapsed:.0f}s", flush=True)

        if time.time() - t_start > 280:
            print(f"  Timeout at step={step}")
            break

    elapsed = time.time() - t_start
    arr = np.array(all_diffs, dtype=np.float64)
    print(f"\nCollected {len(arr)} frame_diffs in {elapsed:.0f}s", flush=True)

    if len(arr) < 100:
        print("Insufficient data.")
        return

    # Distribution analysis
    pcts = percentiles(arr, [1, 5, 10, 25, 50, 75, 90, 95, 99])
    print(f"\nframe_diff distribution:")
    print(f"  min={arr.min():.4f}  max={arr.max():.4f}  mean={arr.mean():.4f}  std={arr.std():.4f}")
    for p, v in pcts.items():
        print(f"  p{p:>2}={v:.4f}")

    # Histogram (ASCII)
    print(f"\nHistogram (50 bins, {len(arr)} samples):")
    hist_counts, hist_edges = np.histogram(arr, bins=30)
    hist_max = max(hist_counts)
    for i, (cnt, edge) in enumerate(zip(hist_counts, hist_edges[:-1])):
        bar = '#' * int(40 * cnt / hist_max)
        print(f"  {edge:.3f}-{hist_edges[i+1]:.3f}: {bar} ({cnt})")

    # Bimodality test
    bimodal, threshold, low_count, high_count = detect_bimodal(all_diffs)
    print(f"\nBimodality test:")
    print(f"  bimodal={bimodal}  threshold={threshold}  low={low_count}  high={high_count}")

    # Event analysis
    print(f"\nEvent correlations:")
    death_diffs = [d for d, t in event_diffs if t == 'death']
    level_diffs = [d for d, t in event_diffs if t.startswith('level_')]
    regular_sample = [d for d in all_diffs if d <= (threshold or arr.max())]

    if death_diffs:
        print(f"  Death transitions: n={len(death_diffs)} mean={np.mean(death_diffs):.4f} "
              f"min={np.min(death_diffs):.4f} max={np.max(death_diffs):.4f}")
    if level_diffs:
        print(f"  Level transitions: n={len(level_diffs)} mean={np.mean(level_diffs):.4f}")
    if threshold:
        death_above = sum(1 for d in death_diffs if d > threshold)
        level_above = sum(1 for d in level_diffs if d > threshold)
        print(f"  Deaths above threshold: {death_above}/{len(death_diffs)} "
              f"({death_above/len(death_diffs):.1%} if deaths)" if death_diffs else "")
        print(f"  Level transitions above threshold: {level_above}/{len(level_diffs)}" if level_diffs else "")

    # Summary
    print(f"\n{'='*60}")
    print(f"go={go} l1_step={l1_step} total_diffs={len(arr)}")
    if bimodal and threshold:
        print(f"\nBIMODAL: threshold={threshold:.4f}")
        print(f"  Low population ({low_count} samples): normal movement")
        print(f"  High population ({high_count} samples): significant events")
        frac_high = high_count / len(arr)
        print(f"  High fraction: {frac_high:.3%}")
        if death_diffs and sum(1 for d in death_diffs if d > threshold) > 0.5 * len(death_diffs):
            print("  FIND: Deaths correlate with high frame_diff. Threshold detects events!")
            print("  Substrate CAN detect significant transitions without external label.")
        else:
            print("  NOTE: High frame_diffs don't cleanly separate deaths from movement.")
    else:
        print(f"\nUNIMODAL: No clean bimodal separation detected.")
        print("  frame_diff does not separate events from movement in enc-space.")
        print("  KILL: Different encoding or signal needed for event detection.")


if __name__ == "__main__":
    main()
