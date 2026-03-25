"""
step0980_evolutionary_policy.py -- Evolutionary random policy search.

FAMILY: Evolutionary RL (completely different from 916-variants)
R3 HYPOTHESIS: Population of N=5 linear policies W_policy(n_actions, 256). Each runs
500 steps and scores by completions. Top-K=2 survive, N-K offspring mutate from them.
4 generations per game phase = 10K steps total. Self-modification: the POPULATION modifies
HOW the substrate selects actions (which policy wins) via selection pressure.

No prediction. No W_pred. No running_mean. No graph. Action = softmax(W_policy @ enc).

Ban-safe: W_policy is a parametric matrix. No per-(state,action) data. Selection by
episode completions (observable). Mutation = gaussian perturbation.
One config all games: N, K, sigma, steps_per_eval universal.

Kill: LS20 < 67.0 (chain baseline).
Success: FT09 > 0 OR VC33 > 0 (new family breaks FT09 wall).

Chain: CIFAR(1K) → LS20(10K) → FT09(10K) → VC33(10K) → CIFAR(1K). 10 seeds.
(CIFAR: 5 policies × 200 steps = 1K. ARC: 5 × 500 × 4 generations = 10K.)
"""
import sys, time
sys.path.insert(0, 'B:/M/the-search')
sys.path.insert(0, 'B:/M/the-search/experiments')

import numpy as np
from substrates.step0674 import _enc_frame

ENC_DIM       = 256   # _enc_frame output
N_POLICIES    = 5     # population size
K_SURVIVORS   = 2     # top-K survive each generation
SIGMA         = 0.1   # mutation std
STEPS_PER_EVAL = 500  # steps per policy evaluation (ARC)
N_GENERATIONS  = 4    # generations per ARC phase (5×500×4=10K)
CIFAR_STEPS_PER_EVAL = 200  # shorter for CIFAR (5×200=1K)
SOFTMAX_TEMP  = 1.0   # policy temperature (1.0 = explore while following direction)
TEST_SEEDS    = list(range(1, 11))
PHASE_STEPS   = 10_000
CIFAR_STEPS   = 1_000


def softmax_sel(logits, temp, rng):
    x = np.array(logits) / temp
    x -= np.max(x)
    e = np.exp(x)
    probs = e / (e.sum() + 1e-12)
    return int(rng.choice(len(logits), p=probs))


class EvolutionaryPolicy980:
    """N=5 linear policies, evolutionary selection by episode completions."""

    def __init__(self, seed):
        self._rng = np.random.RandomState(seed)
        self._init_rng = np.random.RandomState(seed + 50000)
        self._n_actions = 4
        self._steps_per_eval = STEPS_PER_EVAL

        # Will be initialized in set_game
        self.policies = None
        self._current_idx = 0
        self._step_count = 0
        self._scores = None

    def _init_policies(self, n_actions):
        """Initialize N random policies for given n_actions."""
        self.policies = [
            self._init_rng.randn(n_actions, ENC_DIM).astype(np.float32) * 0.01
            for _ in range(N_POLICIES)
        ]
        self._current_idx = 0
        self._step_count = 0
        self._scores = [0.0] * N_POLICIES

    def _evolve(self):
        """Select top-K survivors, generate N offspring by mutation."""
        sorted_idx = np.argsort(self._scores)[::-1]  # descending by score
        survivors = [self.policies[i].copy() for i in sorted_idx[:K_SURVIVORS]]

        new_policies = list(survivors)  # survivors carry over
        while len(new_policies) < N_POLICIES:
            parent = survivors[self._rng.randint(0, K_SURVIVORS)]
            child = parent + SIGMA * self._rng.randn(*parent.shape).astype(np.float32)
            new_policies.append(child)

        self.policies = new_policies
        self._scores = [0.0] * N_POLICIES

    def set_game(self, n_actions, steps_per_eval=None):
        self._n_actions = n_actions
        self._steps_per_eval = steps_per_eval or STEPS_PER_EVAL
        self._init_policies(n_actions)

    def process(self, obs):
        enc = _enc_frame(np.asarray(obs, dtype=np.float32))
        W = self.policies[self._current_idx]
        logits = W @ enc
        action = softmax_sel(logits, SOFTMAX_TEMP, self._rng)

        self._step_count += 1
        if self._step_count >= self._steps_per_eval:
            # Move to next policy in generation
            self._current_idx += 1
            self._step_count = 0
            if self._current_idx >= N_POLICIES:
                # End of generation: evolve
                self._evolve()
                self._current_idx = 0

        return action

    def on_completion(self, n=1):
        """Signal that the current policy achieved a completion."""
        if self._scores is not None:
            self._scores[self._current_idx] += n

    def on_level_transition(self):
        pass  # evolutionary substrate doesn't reset on level transition


def make_env(name):
    try:
        import arcagi3; return arcagi3.make(name)
    except Exception:
        import util_arcagi3; return util_arcagi3.make(name)


def load_cifar():
    try:
        import torchvision, torchvision.transforms as T
        ds = torchvision.datasets.CIFAR100('B:/M/the-search/data', train=False,
                                            download=True, transform=T.ToTensor())
        imgs = np.array([np.array(ds[i][0]).transpose(1, 2, 0) for i in range(len(ds))],
                        dtype=np.float32)
        lbls = np.array([ds[i][1] for i in range(len(ds))], dtype=np.int32)
        return imgs, lbls
    except Exception as e:
        print(f"  CIFAR load failed: {e}"); return None, None


def run_cifar(sub, imgs, lbls, seed, n_steps):
    if imgs is None:
        return 0.0
    sub.set_game(100, steps_per_eval=CIFAR_STEPS_PER_EVAL)
    rng = np.random.RandomState(seed)
    idx = rng.permutation(len(imgs))[:n_steps]
    correct = 0
    for i in idx:
        action = sub.process(imgs[i])
        if action % 100 == lbls[i]:
            correct += 1
            sub.on_completion(1)  # signal correct classification
    return correct / len(idx)


def run_arc(sub, game, n_actions, seed, n_steps):
    sub.set_game(n_actions, steps_per_eval=STEPS_PER_EVAL)
    env = make_env(game)
    obs = env.reset(seed=seed)
    step = 0; completions = 0; level = 0
    while step < n_steps:
        if obs is None:
            obs = env.reset(seed=seed); continue
        action = sub.process(np.asarray(obs, dtype=np.float32)) % n_actions
        obs, _, done, info = env.step(action); step += 1
        cl = info.get('level', 0) if isinstance(info, dict) else 0
        if cl > level:
            n_comp = cl - level
            completions += n_comp
            level = cl
            sub.on_completion(n_comp)  # signal completion to substrate
        if done:
            obs = env.reset(seed=seed); level = 0
    return completions


def run_chain(seeds, n_steps, cifar_steps, cifar_imgs, cifar_lbls):
    cifar1_list, ls20_list, ft09_list, vc33_list, cifar2_list = [], [], [], [], []
    for seed in seeds:
        sub = EvolutionaryPolicy980(seed=seed)
        c1 = run_cifar(sub, cifar_imgs, cifar_lbls, seed * 1000, cifar_steps)
        l = run_arc(sub, "LS20", 4, seed * 1000, n_steps)
        f = run_arc(sub, "FT09", 68, seed * 1000, n_steps)
        v = run_arc(sub, "VC33", 68, seed * 1000, n_steps)
        c2 = run_cifar(sub, cifar_imgs, cifar_lbls, seed * 1000 + 1, cifar_steps)
        cifar1_list.append(c1); ls20_list.append(l); ft09_list.append(f)
        vc33_list.append(v); cifar2_list.append(c2)
        best_score = max(sub._scores) if sub._scores else 0
        print(f"  seed={seed}: CIFAR1={c1:.3f} LS20={l:4d} FT09={f:4d} VC33={v:4d} CIFAR2={c2:.3f}"
              f"  best_score={best_score:.1f}")
    return cifar1_list, ls20_list, ft09_list, vc33_list, cifar2_list


if __name__ == "__main__":
    import os
    print("=" * 70)
    print("STEP 980 — EVOLUTIONARY POLICY SEARCH (N=5, K=2, sigma=0.1)")
    print("=" * 70)
    t0 = time.time()
    ls20_hash = next((d for d in os.listdir('B:/M/the-search/environment_files/ls20') if len(d) >= 8), '?')
    ft09_hash = next((d for d in os.listdir('B:/M/the-search/environment_files/ft09') if len(d) >= 8), '?')
    vc33_hash = next((d for d in os.listdir('B:/M/the-search/environment_files/vc33') if len(d) >= 8), '?')
    print(f"Game versions: LS20={ls20_hash}  FT09={ft09_hash}  VC33={vc33_hash}")
    print(f"Steps: ARC={PHASE_STEPS}  CIFAR={CIFAR_STEPS}  Seeds={TEST_SEEDS}")
    print(f"N={N_POLICIES}  K={K_SURVIVORS}  sigma={SIGMA}  steps/eval={STEPS_PER_EVAL}")
    print(f"Generations/game: {N_GENERATIONS} ({N_POLICIES}×{STEPS_PER_EVAL}×{N_GENERATIONS}=10K)")
    print(f"No W_pred. No running_mean. W_policy({'{n_actions}'},256). Fitness=completions.")
    print()

    cifar_imgs, cifar_lbls = load_cifar()

    c1, ls, ft, vc, c2 = run_chain(TEST_SEEDS, PHASE_STEPS, CIFAR_STEPS,
                                    cifar_imgs, cifar_lbls)

    print()
    print("=" * 70)
    print(f"STEP 980 RESULTS (965 chain: LS20=67.0, standalone 916@10K=74.7):")
    print(f"  CIFAR-1: {np.mean(c1):.3f} (chance=0.010)")
    print(f"  LS20:    {np.mean(ls):.1f}/seed  nonzero={sum(1 for x in ls if x > 0)}/10  {ls}")
    print(f"  FT09:    {np.mean(ft):.1f}/seed  nonzero={sum(1 for x in ft if x > 0)}/10  {ft}")
    print(f"  VC33:    {np.mean(vc):.1f}/seed  nonzero={sum(1 for x in vc if x > 0)}/10  {vc}")
    print(f"  CIFAR-2: {np.mean(c2):.3f}")
    ls20_baseline = 67.0
    chain_verdict = (f"LS20 PASS ({np.mean(ls):.1f} ≥ {ls20_baseline*0.9:.1f})"
                     if np.mean(ls) >= ls20_baseline * 0.9
                     else f"LS20 DEGRADED ({np.mean(ls):.1f} < {ls20_baseline*0.9:.1f})")
    ft09_verdict = (f"FT09 SIGNAL ({sum(1 for x in ft if x > 0)}/10 nonzero)"
                    if any(x > 0 for x in ft) else "FT09 ZERO (0/10)")
    vc33_verdict = (f"VC33 SIGNAL ({sum(1 for x in vc if x > 0)}/10 nonzero)"
                    if any(x > 0 for x in vc) else "VC33 ZERO (0/10)")
    print(f"  Chain verdict: {chain_verdict}")
    print(f"  FT09: {ft09_verdict}  |  VC33: {vc33_verdict}")
    print(f"Total elapsed: {time.time()-t0:.1f}s")
    print("STEP 980 DONE")
