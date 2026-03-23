"""
step0903_elm_forward.py -- Extreme Learning Machine Forward Model.

R3 hypothesis: "Non-linear random features improve prediction accuracy beyond
linear W (19.9%) while maintaining fast online convergence (ELM = convex loss)."

Leo's diagnosis: MLP (step890) failed because online SGD on 98K-param non-convex
surface doesn't converge in 5K steps. ELM fixes this:
- W_hidden: FIXED random (never trained). Non-linear feature extraction.
- W_out: TRAINABLE linear readout. Convex → delta rule converges as fast as linear W.

Architecture:
- hidden=512. Input: (ENC_DIM + n_actions) = 260. W_hidden: (512, 260), ReLU.
- W_out: (256, 512). Delta rule, eta=0.01.
- Action: MSE-based novelty buffer (L2 min-dist to 500 stored obs). argmax novelty.
- 20% epsilon exploration.

Transfer: W_out only (W_hidden is fixed random — same across seeds with same init seed).

Key question: ELM pred_acc vs linear W (19.9% from step835)?
- ELM >> linear W → model family was bottleneck (non-linearity needed).
- ELM ≈ linear W → encoding is bottleneck (avgpool16 features saturated).

Protocol: LS20, R3_cf (cold vs warm W_out transfer), 10 seeds, 10K steps.
"""
import sys, time
sys.path.insert(0, 'B:/M/the-search')
sys.path.insert(0, 'B:/M/the-search/experiments')

import numpy as np
from substrates.base import BaseSubstrate
from substrates.step0674 import _enc_frame

PRETRAIN_SEEDS = list(range(1, 6))
TEST_SEEDS = list(range(6, 11))
PRETRAIN_STEPS = 5_000
TEST_STEPS = 10_000
N_ACTIONS = 4
ETA_OUT = 0.01
ENC_DIM = 256
HIDDEN = 512
NOVELTY_BUFFER_SIZE = 500
EPSILON = 0.20


def one_hot(a, n):
    v = np.zeros(n, dtype=np.float32); v[a] = 1.0
    return v


class ELMForward903(BaseSubstrate):
    """ELM: fixed random hidden layer + trainable linear readout."""

    def __init__(self, n_actions=N_ACTIONS, seed=0, epsilon=EPSILON):
        self._n_actions = n_actions
        self._seed = seed
        self._epsilon = epsilon
        rng = np.random.RandomState(seed)
        self._rng = np.random.RandomState(seed)
        inp_dim = ENC_DIM + n_actions
        # Fixed random hidden layer (He initialization)
        self.W_hidden = rng.randn(HIDDEN, inp_dim).astype(np.float32) * np.sqrt(2.0 / inp_dim)
        self.b_hidden = rng.randn(HIDDEN).astype(np.float32) * 0.1
        # Trainable output layer
        self.W_out = np.zeros((ENC_DIM, HIDDEN), dtype=np.float32)
        # Novelty buffer (MSE-based)
        self._buffer = np.zeros((NOVELTY_BUFFER_SIZE, ENC_DIM), dtype=np.float32)
        self._buf_count = 0
        # Running mean
        self._running_mean = np.zeros(ENC_DIM, np.float32)
        self._n_obs = 0
        self._prev_enc = None; self._prev_action = None; self._last_enc = None
        # Prediction error tracking
        self._pred_errors_mse = []  # for reporting

    def _encode(self, obs):
        enc_raw = _enc_frame(np.asarray(obs, dtype=np.float32))
        self._n_obs += 1
        a = 1.0 / self._n_obs
        self._running_mean = (1 - a) * self._running_mean + a * enc_raw
        return enc_raw - self._running_mean

    def _hidden(self, inp):
        """Fixed non-linear feature transformation."""
        return np.maximum(0.0, self.W_hidden @ inp + self.b_hidden)

    def _predict(self, enc, action):
        inp = np.concatenate([enc, one_hot(action, self._n_actions)])
        h = self._hidden(inp)
        return self.W_out @ h

    def _add_to_buffer(self, enc):
        self._buf_count += 1
        if self._buf_count <= NOVELTY_BUFFER_SIZE:
            self._buffer[self._buf_count - 1] = enc
        else:
            idx = self._rng.randint(0, self._buf_count)
            if idx < NOVELTY_BUFFER_SIZE:
                self._buffer[idx] = enc

    def _novelty(self, pred_enc):
        """Minimum L2 distance from pred_enc to buffer."""
        n_stored = min(self._buf_count, NOVELTY_BUFFER_SIZE)
        if n_stored == 0:
            return 1.0
        active = self._buffer[:n_stored]
        dists = np.linalg.norm(active - pred_enc, axis=1)
        return float(np.min(dists))

    def process(self, observation):
        enc = self._encode(observation)
        self._last_enc = enc
        self._add_to_buffer(enc)

        # Update W_out (delta rule, convex)
        if self._prev_enc is not None and self._prev_action is not None:
            inp = np.concatenate([self._prev_enc, one_hot(self._prev_action, self._n_actions)])
            h = self._hidden(inp)
            pred = self.W_out @ h
            error = enc - pred
            err_norm = float(np.linalg.norm(error))
            if err_norm > 10.0:
                error = error * (10.0 / err_norm)
            if not np.any(np.isnan(error)):
                self.W_out -= ETA_OUT * np.outer(error, h)
                self._pred_errors_mse.append(float(np.mean(error ** 2)))

        # Action: MSE-based novelty, argmax
        if self._rng.random() < self._epsilon:
            action = int(self._rng.randint(0, self._n_actions))
        else:
            best_a = 0; best_nov = -1.0
            for a in range(self._n_actions):
                pred_next = self._predict(enc, a)
                nov = self._novelty(pred_next)
                if nov > best_nov:
                    best_nov = nov; best_a = a
            action = best_a

        self._prev_enc = enc.copy()
        self._prev_action = action
        return action

    def pred_accuracy_pct(self):
        """Approximate pred accuracy from recent MSE errors."""
        if len(self._pred_errors_mse) < 10:
            return None
        recent = self._pred_errors_mse[-200:]
        mean_mse = np.mean(recent)
        # Compare to baseline: mean squared amplitude of centered enc
        # step835 reports as (1 - mean_mse / baseline_mse) * 100
        return float(-mean_mse)  # raw MSE for comparison

    @property
    def n_actions(self): return self._n_actions

    def reset(self, seed):
        rng = np.random.RandomState(seed)
        self._rng = np.random.RandomState(seed)
        inp_dim = ENC_DIM + self._n_actions
        self.W_hidden = rng.randn(HIDDEN, inp_dim).astype(np.float32) * np.sqrt(2.0 / inp_dim)
        self.b_hidden = rng.randn(HIDDEN).astype(np.float32) * 0.1
        self.W_out = np.zeros((ENC_DIM, HIDDEN), dtype=np.float32)
        self._buffer = np.zeros((NOVELTY_BUFFER_SIZE, ENC_DIM), dtype=np.float32)
        self._buf_count = 0
        self._running_mean = np.zeros(ENC_DIM, np.float32)
        self._n_obs = 0
        self._prev_enc = None; self._prev_action = None; self._last_enc = None
        self._pred_errors_mse = []

    def on_level_transition(self):
        self._prev_enc = None; self._prev_action = None

    def get_state(self):
        # W_hidden is fixed — only W_out transfers
        return {"W_out": self.W_out.copy(),
                "running_mean": self._running_mean.copy(), "n_obs": self._n_obs,
                "W_hidden": self.W_hidden.copy(), "b_hidden": self.b_hidden.copy()}

    def set_state(self, s):
        self.W_out = s["W_out"].copy()
        self._running_mean = s["running_mean"].copy(); self._n_obs = s["n_obs"]
        # W_hidden fixed — don't transfer (same init seed means same W_hidden anyway)

    def frozen_elements(self): return ["W_hidden"]


def make_game():
    try:
        import arcagi3; return arcagi3.make("LS20")
    except:
        import util_arcagi3; return util_arcagi3.make("LS20")


def run_phase(substrate, env_fn, n_actions, env_seed, n_steps):
    env = env_fn(); obs = env.reset(seed=env_seed)
    step = 0; completions = 0; current_level = 0
    while step < n_steps:
        if obs is None:
            obs = env.reset(seed=env_seed); substrate.on_level_transition(); continue
        action = substrate.process(np.asarray(obs, dtype=np.float32)) % n_actions
        obs, _, done, info = env.step(action); step += 1
        cl = info.get('level', 0) if isinstance(info, dict) else 0
        if cl > current_level:
            completions += (cl - current_level); current_level = cl
            substrate.on_level_transition()
        if done:
            obs = env.reset(seed=env_seed); current_level = 0
            substrate.on_level_transition()
    return completions


print("=" * 70)
print("STEP 903 — EXTREME LEARNING MACHINE (ELM) FORWARD MODEL")
print("=" * 70)
print(f"Fixed W_hidden (512,260) ReLU + trainable W_out (256,512). eta={ETA_OUT}.")
print(f"Action: MSE-based novelty (min L2 dist to 500-obs buffer). eps={EPSILON}.")
print(f"Transfer: W_out only. Key question: pred_acc vs linear W (19.9%)?")

t0 = time.time()

# Pretrain
sub_p = ELMForward903(n_actions=N_ACTIONS, seed=0)
sub_p.reset(0)
for ps in PRETRAIN_SEEDS:
    sub_p.on_level_transition()
    env = make_game(); obs = env.reset(seed=ps * 1000); s = 0
    while s < PRETRAIN_STEPS:
        if obs is None:
            obs = env.reset(seed=ps * 1000); sub_p.on_level_transition(); continue
        action = sub_p.process(np.asarray(obs, dtype=np.float32)) % N_ACTIONS
        obs, _, done, _ = env.step(action); s += 1
        if done:
            obs = env.reset(seed=ps * 1000); sub_p.on_level_transition()
saved = sub_p.get_state()
raw_mse = sub_p.pred_accuracy_pct()
print(f"  Pretrain done ({time.time()-t0:.1f}s). raw_MSE={raw_mse:.4f}" if raw_mse else
      f"  Pretrain done ({time.time()-t0:.1f}s). pred_acc=N/A")

cold_comps = []; warm_comps = []
cold_mse = []; warm_mse = []

for ts in TEST_SEEDS:
    sub_c = ELMForward903(n_actions=N_ACTIONS, seed=ts % 4)
    sub_c.reset(ts % 4)
    c_comp = run_phase(sub_c, make_game, N_ACTIONS, ts * 1000, TEST_STEPS)
    cold_comps.append(c_comp)
    cm = sub_c.pred_accuracy_pct()
    cold_mse.append(cm if cm else 0.0)

    sub_w = ELMForward903(n_actions=N_ACTIONS, seed=ts % 4)
    sub_w.reset(ts % 4)
    sub_w.W_out = saved["W_out"].copy()
    w_comp = run_phase(sub_w, make_game, N_ACTIONS, ts * 1000, TEST_STEPS)
    warm_comps.append(w_comp)
    wm = sub_w.pred_accuracy_pct()
    warm_mse.append(wm if wm else 0.0)
    print(f"  seed={ts}: cold L1={c_comp:4d} warm L1={w_comp:4d}")

mc = np.mean(cold_comps); mw = np.mean(warm_comps)
print(f"\ncold: L1={mc:.1f}/seed  raw_MSE_mean={np.mean(cold_mse):.4f}  {cold_comps}")
print(f"warm: L1={mw:.1f}/seed  raw_MSE_mean={np.mean(warm_mse):.4f}  {warm_comps}")
print(f"L1 delta (warm-cold): {mw-mc:+.1f}/seed")
print(f"\nNote: raw_MSE is negative pred_acc proxy. Linear W pred_acc = 19.9% (step835).")
print(f"If ELM raw_MSE << linear W MSE → ELM better. If similar → features are bottleneck.")
print(f"\nTotal elapsed: {time.time()-t0:.1f}s")
print("STEP 903 DONE")
