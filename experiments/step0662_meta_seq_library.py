"""
Step 662 — Sequence storage substrate.

No graph, no hash, no cells. Library of K action sequences.
Weighted random execution; mutate worst sequence every 10 episodes.
Fitness signal: level transitions from environment (R1-compliant).

If L1 reached: sequence representation more natural than topology for conjunction.
If fails: sequence search too slow (space too large).
"""
import numpy as np
import time
import sys

N_A = 4
MAX_STEPS = 500_001
PER_SEED_TIME = 25
N_SEEDS = 10
K_SEQS = 10
SEQ_LEN = 50
MUTATE_EVERY = 10


class SeqLib:
    def __init__(self, k=K_SEQS, seq_len=SEQ_LEN, seed=0):
        self.rng = np.random.RandomState(seed)
        self.seqs = [list(self.rng.randint(0, N_A, seq_len)) for _ in range(k)]
        self.failures = [0] * k
        self.success = [False] * k
        self.episode_count = 0
        self.step_in_ep = 0
        self.cur_idx = None
        self.unique_tried = set()
        self.k = k

    def _pick(self):
        weights = np.array([1.0 / (f + 1) for f in self.failures], dtype=np.float64)
        weights /= weights.sum()
        idx = int(self.rng.choice(self.k, p=weights))
        self.unique_tried.add(tuple(self.seqs[idx]))
        return idx

    def observe(self, frame):
        pass  # sequence substrate ignores observation

    def act(self):
        if self.cur_idx is None:
            self.cur_idx = self._pick()
        seq = self.seqs[self.cur_idx]
        if self.step_in_ep < len(seq):
            action = seq[self.step_in_ep]
        else:
            action = int(self.rng.randint(N_A))
        self.step_in_ep += 1
        return action

    def on_reset(self, success=False):
        if self.cur_idx is not None:
            if success:
                self.success[self.cur_idx] = True
            else:
                self.failures[self.cur_idx] += 1
        self.episode_count += 1
        self.step_in_ep = 0
        self.cur_idx = None
        if self.episode_count % MUTATE_EVERY == 0:
            self._mutate_worst()

    def _mutate_worst(self):
        # Find worst non-successful sequence
        worst = -1
        worst_f = -1
        for i in range(self.k):
            if not self.success[i] and self.failures[i] > worst_f:
                worst_f = self.failures[i]
                worst = i
        if worst < 0:
            return
        seq = list(self.seqs[worst])
        op = self.rng.randint(3)
        if op == 0 and len(seq) >= 2:
            # Swap 2 random positions
            i, j = self.rng.choice(len(seq), 2, replace=False)
            seq[i], seq[j] = seq[j], seq[i]
        elif op == 1:
            # Replace random position
            i = int(self.rng.randint(len(seq)))
            seq[i] = int(self.rng.randint(N_A))
        else:
            # Extend or shorten by 1
            if self.rng.rand() > 0.5 and len(seq) > 1:
                seq.pop(int(self.rng.randint(len(seq))))
            else:
                seq.append(int(self.rng.randint(N_A)))
        self.seqs[worst] = seq
        self.failures[worst] = 0


def run(seed, make):
    env = make()
    sub = SeqLib(seed=seed * 1000)
    obs = env.reset(seed=seed)
    level = 0
    l1 = None
    go = 0
    t_start = time.time()
    winning_seq = None

    for step in range(1, MAX_STEPS):
        if obs is None:
            obs = env.reset(seed=seed)
            sub.on_reset(success=False)
            continue

        sub.observe(obs)
        action = sub.act()
        obs, reward, done, info = env.step(action)

        cl = info.get('level', 0) if isinstance(info, dict) else 0
        if cl > level:
            if cl == 1 and l1 is None:
                l1 = step
                if sub.cur_idx is not None:
                    winning_seq = list(sub.seqs[sub.cur_idx])
            level = cl
            sub.on_reset(success=(cl == 1))

        if done:
            go += 1
            obs = env.reset(seed=seed)
            sub.on_reset(success=False)

        if time.time() - t_start > PER_SEED_TIME:
            break

    elapsed = time.time() - t_start
    print(f"  s{seed}: L1={l1} go={go} unique_seqs={len(sub.unique_tried)} "
          f"eps={sub.episode_count} t={elapsed:.1f}s", flush=True)
    if winning_seq:
        print(f"    winning_seq={winning_seq[:20]}...", flush=True)

    return dict(seed=seed, l1=l1, go=go,
                unique_tried=len(sub.unique_tried), eps=sub.episode_count)


def main():
    try:
        sys.path.insert(0, '.')
        import arcagi3
        mk = lambda: arcagi3.make("LS20")
    except (ImportError, Exception) as e:
        print(f"arcagi3: {e}")
        return

    print(f"Sequence library: K={K_SEQS} seqs len={SEQ_LEN}, {N_SEEDS} seeds, {PER_SEED_TIME}s cap")

    results = []
    for seed in range(N_SEEDS):
        results.append(run(seed, mk))

    print(f"\n{'='*60}")
    l1_n = sum(1 for r in results if r['l1'])
    avg_tried = np.mean([r['unique_tried'] for r in results])
    avg_eps = np.mean([r['eps'] for r in results])
    print(f"L1={l1_n}/{N_SEEDS}  avg_unique_seqs={avg_tried:.0f}  avg_eps={avg_eps:.0f}")

    if l1_n > 0:
        print("FINDING: Sequence representation reaches L1 — sequences beat topology for conjunction")
    else:
        print("FINDING: Sequence search fails — graph topology is necessary, or sequence space too large")


if __name__ == "__main__":
    main()
