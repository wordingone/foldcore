"""
step1002_osc_delta_obs.py -- Oscillatory substrate with obs-change modulator.

FAMILY: Oscillatory dynamics
R3 HYPOTHESIS: Phase-coherence gating provides better temporal credit than
running_mean when BOTH use the same base signal (observation change magnitude).
800b/916 uses running_mean[a] per action → Theorem 4: signal averages to 0 for
state-dependent actions. 1002 uses e_ij * delta_obs → credit goes to
phase-aligned oscillator pairs AT THE MOMENT of change. Phase alignment carries
temporal context that running_mean destroys.

1001 failure: compression-progress M (err_ema_slow - err_ema_fast) not
correlated with game-relevant events. Fix: replace M with delta_obs modulator
(above-average observation change), same signal as 916's delta_per_action but
credited via phase coherence rather than per-action mean.

Step 1002: LS20 only. Kill: LS20 < 20 at 10K. Success: any seed L1 > 0.
Chain: CIFAR(1K) → LS20(10K) → CIFAR(1K). 10 seeds.
"""
import sys, time
sys.path.insert(0, 'B:/M/the-search')
sys.path.insert(0, 'B:/M/the-search/experiments')

import numpy as np
from substrates.step0674 import _enc_frame

# Oscillator parameters
N_OSC = 32
MU = 1.0          # supercritical — stable limit cycle
DT = 0.1
N_SUBSTEPS = 5
TAU_F = 50        # eligibility trace time constant (game steps)
LR_W = 0.01       # coupling weight learning rate
LR_OUT = 0.01     # action readout learning rate
LR_PRED = 0.01    # W_pred learning rate
OMEGA_LO = 0.5
OMEGA_HI = 2.0

# Dimensions
ENC_DIM = 256
FEAT_DIM = N_OSC * 3  # [cos(phi), sin(phi), r] per oscillator = 96

# Observation-change modulator
DELTA_OBS_EMA = 0.99   # slow EMA of obs change magnitude

# Action selection
SOFTMAX_TEMP = 1.0

TEST_SEEDS = list(range(1, 11))
PHASE_STEPS = 10_000
CIFAR_STEPS = 1_000


def softmax_sample(logits, temp, rng):
    x = logits / temp
    x -= x.max()
    e = np.exp(x)
    probs = e / (e.sum() + 1e-12)
    return int(rng.choice(len(logits), p=probs))


class OscDeltaObs1002:
    """Stuart-Landau oscillators with phase-coherence credit assignment."""

    def __init__(self, seed):
        self._rng = np.random.RandomState(seed)
        rs = np.random.RandomState(seed + 10000)

        # Natural frequencies (diversity across oscillators)
        self._omega = rs.uniform(OMEGA_LO, OMEGA_HI, N_OSC).astype(np.float32)

        # Oscillator state: complex-valued, init on limit cycle with random phases
        phases = rs.uniform(0, 2 * np.pi, N_OSC).astype(np.float32)
        self._z = (np.cos(phases) + 1j * np.sin(phases)).astype(np.complex64)

        # Coupling matrix: zero initially, learned via three-factor plasticity
        self._W_ij = np.zeros((N_OSC, N_OSC), dtype=np.float32)

        # Input weights: each oscillator receives from enc (random init, small)
        self._W_in = rs.randn(N_OSC, ENC_DIM).astype(np.float32) * 0.1

        # Eligibility traces: EMA of phase coherence cos(phi_i - phi_j)
        self._e_ij = np.zeros((N_OSC, N_OSC), dtype=np.float32)

        # Prediction model: W_pred predicts enc (256-dim) from features (96-dim)
        # Same Hebbian update direction as 916. No alpha.
        self._W_pred = np.zeros((ENC_DIM, FEAT_DIM), dtype=np.float32)

        # Action readout
        self._n_actions = 4
        self._W_out = rs.randn(self._n_actions, FEAT_DIM).astype(np.float32) * 0.01

        # Observation-change modulator: M_t = max(0, delta_obs - delta_obs_ema)
        self._delta_obs_ema = 1.0

        # Previous enc for delta_obs computation
        self._prev_enc = None

        # Running mean for encoding (same as 916)
        self._running_mean = np.zeros(ENC_DIM, dtype=np.float32)
        self._n_obs = 0

        # Previous state for updates
        self._prev_features = None
        self._prev_action = None

    def set_game(self, n_actions):
        self._n_actions = n_actions
        rs = np.random.RandomState(int(self._rng.randint(1 << 31)))
        self._W_out = rs.randn(n_actions, FEAT_DIM).astype(np.float32) * 0.01
        phases = self._rng.uniform(0, 2 * np.pi, N_OSC).astype(np.float32)
        self._z = (np.cos(phases) + 1j * np.sin(phases)).astype(np.complex64)
        self._prev_features = None
        self._prev_action = None
        self._prev_enc = None

    def _encode(self, obs):
        enc_raw = _enc_frame(np.asarray(obs, dtype=np.float32))
        self._n_obs += 1
        a = 1.0 / self._n_obs
        self._running_mean = (1 - a) * self._running_mean + a * enc_raw
        return (enc_raw - self._running_mean).astype(np.float32)

    def _step_oscillators(self, enc):
        """5 Euler substeps of Stuart-Landau dynamics."""
        z = self._z.astype(np.complex128)
        omega = self._omega.astype(np.float64)
        W_ij = self._W_ij.astype(np.float64)
        input_drive = (self._W_in.astype(np.float64) @ enc.astype(np.float64)).astype(np.complex128)
        for _ in range(N_SUBSTEPS):
            r2 = np.real(z * np.conj(z))
            dz = (MU + 1j * omega) * z - r2 * z + W_ij @ z + input_drive
            z = z + DT * dz
            # Clip magnitude: prevent overflow from large input or coupling transients
            r = np.abs(z)
            too_large = r > 5.0
            if np.any(too_large):
                z[too_large] = z[too_large] / r[too_large] * 5.0
        # NaN guard: reset any diverged oscillators to unit circle
        bad = ~np.isfinite(np.abs(z))
        if np.any(bad):
            phases = self._rng.uniform(0, 2 * np.pi, np.sum(bad))
            z[bad] = np.cos(phases) + 1j * np.sin(phases)
        self._z = z.astype(np.complex64)

    def _update_eligibility(self):
        """EMA update of phase-coherence eligibility traces."""
        phi = np.angle(self._z).astype(np.float32)
        dphi = phi[:, None] - phi[None, :]  # (N_OSC, N_OSC)
        coherence = np.cos(dphi)
        alpha = 1.0 / TAU_F
        self._e_ij = (1 - alpha) * self._e_ij + alpha * coherence

    def _get_features(self):
        """[cos(phi_i), sin(phi_i), r_i] for each oscillator."""
        phi = np.angle(self._z).astype(np.float32)
        r = np.abs(self._z).astype(np.float32)
        return np.concatenate([np.cos(phi), np.sin(phi), r])

    def process(self, obs):
        enc = self._encode(obs)

        # Oscillator dynamics
        self._step_oscillators(enc)
        self._update_eligibility()
        features = self._get_features()

        # Observation-change modulator + learning updates
        M = 0.0
        if self._prev_features is not None:
            # W_pred: still learning (gradient descent), not used for M
            pred_enc = self._W_pred @ self._prev_features
            error = enc - pred_enc
            err_norm = float(np.linalg.norm(error))
            if err_norm > 10.0:
                error = error * (10.0 / err_norm)
            if not np.any(np.isnan(error)):
                self._W_pred += LR_PRED * np.outer(error, self._prev_features)

            # delta_obs modulator: above-average observation change triggers credit
            if self._prev_enc is not None:
                delta_obs = float(np.linalg.norm(enc - self._prev_enc))
                self._delta_obs_ema = DELTA_OBS_EMA * self._delta_obs_ema + (1 - DELTA_OBS_EMA) * delta_obs
                M = max(0.0, delta_obs - self._delta_obs_ema)

            # W_ij: three-factor plasticity gated by M
            if M > 1e-8:
                self._W_ij += LR_W * M * self._e_ij
                np.clip(self._W_ij, -5.0, 5.0, out=self._W_ij)

            # W_out: reinforce previous action when above-average change observed
            if self._prev_action is not None and M > 1e-8:
                self._W_out[self._prev_action] += LR_OUT * M * self._prev_features

        # Action selection
        logits = self._W_out @ features
        action = softmax_sample(logits, SOFTMAX_TEMP, self._rng)

        self._prev_features = features.copy()
        self._prev_enc = enc.copy()
        self._prev_action = action
        return action

    def on_level_transition(self):
        self._prev_features = None
        self._prev_action = None
        self._prev_enc = None

    def phase_diversity(self):
        """Std of oscillator phases."""
        phi = np.angle(self._z)
        return float(np.std(phi))

    def modulator_state(self):
        """Current delta_obs EMA and M — diagnostic."""
        return self._delta_obs_ema


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
        imgs = np.array([np.array(ds[i][0]).transpose(1, 2, 0) for i in range(len(ds))], dtype=np.float32)
        lbls = np.array([ds[i][1] for i in range(len(ds))], dtype=np.int32)
        return imgs, lbls
    except Exception as e:
        print(f"  CIFAR load failed: {e}"); return None, None


def run_cifar(sub, imgs, lbls, seed, n_steps):
    if imgs is None: return 0.0
    sub.set_game(100)
    rng = np.random.RandomState(seed)
    idx = rng.permutation(len(imgs))[:n_steps]
    correct = sum(1 for i in idx if sub.process(imgs[i]) % 100 == lbls[i])
    return correct / len(idx)


def run_arc(sub, game, n_actions, seed, n_steps):
    sub.set_game(n_actions)
    env = make_env(game); obs = env.reset(seed=seed)
    step = 0; completions = 0; level = 0
    while step < n_steps:
        if obs is None: obs = env.reset(seed=seed); sub.on_level_transition(); continue
        action = sub.process(np.asarray(obs, dtype=np.float32)) % n_actions
        obs, _, done, info = env.step(action); step += 1
        cl = info.get('level', 0) if isinstance(info, dict) else 0
        if cl > level:
            completions += cl - level; level = cl; sub.on_level_transition()
        if done: obs = env.reset(seed=seed); level = 0; sub.on_level_transition()
    return completions


def run_chain(seeds, n_steps, cifar_steps, cifar_imgs, cifar_lbls):
    c1l, lsl, c2l = [], [], []
    for seed in seeds:
        sub = OscDeltaObs1002(seed=seed)
        c1 = run_cifar(sub, cifar_imgs, cifar_lbls, seed * 1000, cifar_steps)
        l = run_arc(sub, "LS20", 4, seed * 1000, n_steps)
        c2 = run_cifar(sub, cifar_imgs, cifar_lbls, seed * 1000 + 1, cifar_steps)
        delta_ema = sub.modulator_state()
        div = sub.phase_diversity()
        c1l.append(c1); lsl.append(l); c2l.append(c2)
        print(f"  seed={seed}: C1={c1:.3f} LS20={l:4d} C2={c2:.3f}"
              f"  delta_ema={delta_ema:.3f} phase_div={div:.2f}")
    return c1l, lsl, c2l


if __name__ == "__main__":
    import os
    print("=" * 70)
    print("STEP 1002 — OSC + delta_obs modulator (obs-change replaces M)")
    print("=" * 70)
    ls20_hash = next((d for d in os.listdir('B:/M/the-search/environment_files/ls20') if len(d) >= 8), '?')
    print(f"Game version: LS20={ls20_hash}")
    print(f"N={N_OSC} oscillators, omega~U({OMEGA_LO},{OMEGA_HI}), mu={MU}, dt={DT}, substeps={N_SUBSTEPS}")
    print(f"Eligibility tau_f={TAU_F}, lr_w={LR_W}, lr_out={LR_OUT}, lr_pred={LR_PRED}")
    print(f"Modulator: delta_obs EMA={DELTA_OBS_EMA}, M_t=max(0, delta_obs - ema)")
    print(f"Kill: LS20<20. Success: any seed > 0.")
    print()

    t0 = time.time()
    cifar_imgs, cifar_lbls = load_cifar()
    c1, ls, c2 = run_chain(TEST_SEEDS, PHASE_STEPS, CIFAR_STEPS, cifar_imgs, cifar_lbls)

    print()
    print("=" * 70)
    print("STEP 1002 RESULTS:")
    print(f"  CIFAR-1: {np.mean(c1):.3f}")
    print(f"  LS20:    {np.mean(ls):.1f}/seed  nonzero={sum(1 for x in ls if x > 0)}/10  {ls}")
    print(f"  CIFAR-2: {np.mean(c2):.3f}")
    ls_v = "PASS (navigates)" if np.mean(ls) >= 20.0 else "KILL (no navigation)"
    ls_sig = f"SIGNAL ({sum(1 for x in ls if x > 0)}/10 seeds)" if any(x > 0 for x in ls) else "ZERO"
    print(f"  {ls_v}  |  {ls_sig}")
    print(f"Total elapsed: {time.time()-t0:.1f}s")
    print("STEP 1002 DONE")
