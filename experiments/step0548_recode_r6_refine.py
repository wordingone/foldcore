"""
Step 548 — Recode R6 diagnostic: does self-refinement change policy post-saturation?

R6 requires meaningful self-modification. Post-500K, if refinements stop changing
argmin actions, Recode exits the feasible region — same saturation pattern as LSH.

For each split:
  - pre_action: argmin at the original node before split
  - post_child0/1_action: retroactive argmin at each child (redistribute G edges
    via the new hyperplane using stored mean observations)
  - changed: pre_action != post_a0 OR pre_action != post_a1

Metric: action_change_rate = splits_that_changed_argmin / total_splits
Compare pre-500K vs post-500K windows.

Predictions: pre-500K rate ~30-50%, post-500K rate <10%.
Kill: post-500K rate > 50% (refinement IS irredundant).

5-min cap. LS20. seed=0. Up to 1M steps.
"""
import numpy as np
import time
import sys

N_A = 4
K = 16
DIM = 256
REFINE_EVERY = 5000
MIN_OBS = 8
H_SPLIT = 0.05


def enc(frame):
    """Avgpool16 + centered."""
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
        self._refine_log = []  # list of dicts per split event

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
            self._refine(self.t)
        return n

    def act(self):
        counts = [sum(self.G.get((self._cn, a), {}).values()) for a in range(N_A)]
        action = int(np.argmin(counts))
        self._pn = self._cn
        self._pa = action
        return action

    def on_reset(self):
        self._pn = None

    def _act_at(self, n):
        """Argmin action at node n from current G."""
        counts = [sum(self.G.get((n, a), {}).values()) for a in range(N_A)]
        return int(np.argmin(counts))

    def _h(self, n, a):
        d = self.G.get((n, a))
        if not d or sum(d.values()) < 4:
            return 0.0
        v = np.array(list(d.values()), np.float64)
        p = v / v.sum()
        return float(-np.sum(p * np.log2(np.maximum(p, 1e-15))))

    def _refine(self, step):
        did = 0
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
            if r0 is None or r1 is None or r0[1] < 3 or r1[1] < 3:
                continue
            diff = (r0[0] / r0[1]) - (r1[0] / r1[1])
            nm = np.linalg.norm(diff)
            if nm < 1e-8:
                continue

            # Pre-split argmin at node n
            pre_action = self._act_at(n)

            # Commit split
            self.ref[n] = (diff / nm).astype(np.float32)
            ref_vec = self.ref[n]

            # Retroactively route G edges to children using stored obs means
            child_counts = [[0] * N_A, [0] * N_A]
            for (nn, aa), dd in self.G.items():
                if nn != n:
                    continue
                for succ, cnt in dd.items():
                    cm = self.C.get((n, aa, succ))
                    if cm is None or cm[1] < 1:
                        continue
                    obs_mean = (cm[0] / cm[1]).astype(np.float32)
                    side = int(ref_vec @ obs_mean > 0)
                    child_counts[side][aa] += cnt

            post_a0 = int(np.argmin(child_counts[0]))
            post_a1 = int(np.argmin(child_counts[1]))
            changed = (post_a0 != pre_action) or (post_a1 != pre_action)
            window = "post500K" if step > 500_000 else "pre500K"

            print(f"REFINE t={step} node={n} pre_action={pre_action} "
                  f"post_child0_action={post_a0} post_child1_action={post_a1} "
                  f"changed={changed} window={window}", flush=True)
            self._refine_log.append(dict(
                step=step, pre=pre_action,
                post0=post_a0, post1=post_a1,
                changed=changed, window=window,
            ))

            self.live.discard(n)
            self.ns += 1
            did += 1
            if did >= 3:
                break

    def stats(self):
        return len(self.live), self.ns, len(self.G)


def t0():
    rng = np.random.RandomState(42)
    sub = Recode(dim=4, k=2, seed=0)
    sub.H = rng.randn(2, 4).astype(np.float32)
    sub.dim = 4

    # Node n=0: bimodal on action=1, heavy traffic on actions 0,2,3
    n = 0
    sub.live.add(n)

    m_plus = rng.randn(4).astype(np.float64)
    m_plus /= np.linalg.norm(m_plus)
    m_minus = -m_plus

    sub.G[(n, 1)] = {10: 10, 11: 10}  # bimodal → triggers split
    sub.G[(n, 0)] = {5: 100}           # heavy on action 0
    sub.G[(n, 2)] = {7: 50}
    sub.G[(n, 3)] = {8: 60}
    # counts: a0=100, a1=20, a2=50, a3=60 → pre_action = argmin = 1

    sub.C[(n, 1, 10)] = (m_plus * 5, 5)
    sub.C[(n, 1, 11)] = (m_minus * 5, 5)
    sub.C[(n, 0, 5)] = (rng.randn(4).astype(np.float64) * 100, 100)
    sub.C[(n, 2, 7)] = (rng.randn(4).astype(np.float64) * 50, 50)
    sub.C[(n, 3, 8)] = (rng.randn(4).astype(np.float64) * 60, 60)

    # Trigger refine at step=1000 (pre-500K window)
    sub._refine(1000)

    assert len(sub._refine_log) >= 1, f"Expected refine event, got {sub._refine_log}"
    entry = sub._refine_log[0]
    assert entry['window'] == 'pre500K', f"Expected pre500K, got {entry['window']}"
    assert entry['pre'] == 1, f"Expected pre_action=1, got {entry['pre']}"
    assert entry['pre'] in range(N_A)
    assert entry['post0'] in range(N_A)
    assert entry['post1'] in range(N_A)
    assert isinstance(entry['changed'], bool)
    assert n in sub.ref, "Node 0 should be in ref after split"
    assert n not in sub.live, "Node 0 should be removed from live"

    # Test post-500K window label
    sub2 = Recode(dim=4, k=2, seed=0)
    sub2.H = sub.H.copy()
    sub2.dim = 4
    sub2.live.add(n)
    sub2.G = {(n, 1): {10: 10, 11: 10}, (n, 0): {5: 100},
              (n, 2): {7: 50}, (n, 3): {8: 60}}
    sub2.C = {
        (n, 1, 10): (m_plus * 5, 5),
        (n, 1, 11): (m_minus * 5, 5),
        (n, 0, 5): sub.C[(n, 0, 5)],
        (n, 2, 7): sub.C[(n, 2, 7)],
        (n, 3, 8): sub.C[(n, 3, 8)],
    }
    sub2._refine(600_000)
    assert len(sub2._refine_log) >= 1
    assert sub2._refine_log[0]['window'] == 'post500K', \
        f"Expected post500K, got {sub2._refine_log[0]['window']}"

    # Verify change detection: set up where changed=True is expected
    # child0 gets no a0 traffic but a1=10 → post_a0=1 (same); but if all a0→child1:
    # child1: {a0:100, a1:10, a2:50, a3:60} → post_a1=1; child0:{a0:0,a1:10,...} → post_a0=1
    # changed depends on geometry. Just verify it's a bool and consistent.
    changed_val = entry['changed']
    assert changed_val == ((entry['post0'] != entry['pre']) or
                           (entry['post1'] != entry['pre'])), "changed flag mismatch"

    print("T0 PASS")


def summarize(log):
    pre = [e for e in log if e['window'] == 'pre500K']
    post = [e for e in log if e['window'] == 'post500K']

    def rate(entries):
        if not entries:
            return 0.0, 0, 0
        changed = sum(1 for e in entries if e['changed'])
        return changed / len(entries), changed, len(entries)

    pre_rate, pre_c, pre_n = rate(pre)
    post_rate, post_c, post_n = rate(post)
    return pre_rate, pre_c, pre_n, post_rate, post_c, post_n


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
    level = 0
    l1 = l2 = None
    go = 0
    t_start = time.time()

    for step in range(1, 1_000_001):
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
            sub.on_reset()
            nc, ns, ne = sub.stats()
            if cl == 1 and l1 is None:
                l1 = step
                print(f"L1@{step} c={nc} sp={ns} e={ne} go={go}", flush=True)
            if cl == 2 and l2 is None:
                l2 = step
                print(f"L2@{step} c={nc} sp={ns} e={ne} go={go}", flush=True)
            level = cl

        if step % 100_000 == 0:
            nc, ns, ne = sub.stats()
            el = time.time() - t_start
            pre_r, pre_c, pre_n, post_r, post_c, post_n = summarize(sub._refine_log)
            print(f"@{step} c={nc} sp={ns} go={go} {el:.0f}s | "
                  f"pre={pre_c}/{pre_n}({pre_r:.0%}) post={post_c}/{post_n}({post_r:.0%})",
                  flush=True)

        if time.time() - t_start > 300:
            break

    nc, ns, ne = sub.stats()
    elapsed = time.time() - t_start
    pre_r, pre_c, pre_n, post_r, post_c, post_n = summarize(sub._refine_log)

    print(f"\n{'='*60}")
    tag = "L2" if l2 else ("L1" if l1 else "---")
    print(f"Result: {tag}  steps={step}  c={nc}  sp={ns}  go={go}  {elapsed:.0f}s")
    print(f"\nRefinement policy impact:")
    print(f"  pre-500K:  {pre_c}/{pre_n} changed  rate={pre_r:.1%}")
    print(f"  post-500K: {post_c}/{post_n} changed  rate={post_r:.1%}")
    print(f"  total:     {pre_c+post_c}/{pre_n+post_n} changed")

    if post_n == 0:
        print("\nNo post-500K refinements observed (cap hit before 500K).")
        print("R6 verdict: insufficient data for post-saturation window.")
    elif post_r > 0.50:
        print(f"\nKILL INVERTED: post-500K rate={post_r:.1%} > 50%. "
              "Refinement IS irredundant. R6 satisfied post-saturation.")
    elif post_r < 0.10:
        print(f"\nCONFIRMED: post-500K rate={post_r:.1%} < 10%. "
              "Refinements redundant. R6 not satisfied post-saturation.")
    else:
        print(f"\nAMBIGUOUS: post-500K rate={post_r:.1%}. Neither confirmed nor killed.")


if __name__ == "__main__":
    main()
