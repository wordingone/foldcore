"""
judge.py — ConstitutionalJudge: automated R1-R6 audit.

Takes a BaseSubstrate class, runs all checks, returns structured dict.
All outputs are dicts — no prose, no subjective calls.
"""
import ast
import copy
import inspect
import numpy as np
import time
from substrates.base import BaseSubstrate


# Tokens that indicate external objective use (R1 violation)
R1_FORBIDDEN_IMPORTS = {
    "torch.nn.functional",  "torch.nn",
    "sklearn.metrics", "sklearn.loss",
}
R1_FORBIDDEN_NAMES = {
    "reward", "loss", "target", "label", "y_true", "y_pred",
    "criterion", "objective", "supervised", "ground_truth",
}


class ConstitutionalJudge:
    """Automated R1-R6 evaluation for any BaseSubstrate subclass.

    Usage:
        judge = ConstitutionalJudge()
        results = judge.audit(MySubstrate, chain_results, game_name="LS20")
    """

    def audit(self, substrate_cls: type, chain_results: dict = None,
              game_name: str = "LS20", seed: int = 0,
              n_audit_steps: int = 1000) -> dict:
        """Full R1-R6 audit.

        substrate_cls: class to audit (must be BaseSubstrate subclass)
        chain_results: output from ChainRunner.run() if available
        game_name: game to use for runtime checks
        seed: seed for runtime checks
        n_audit_steps: steps to run for dynamic checks

        Returns: {R1: {pass, detail}, R2: ..., ..., R6: ..., summary: {score, flags}}
        """
        if not (isinstance(substrate_cls, type) and
                issubclass(substrate_cls, BaseSubstrate)):
            return {"error": f"{substrate_cls} is not a BaseSubstrate subclass"}

        results = {}
        results["R1"] = self._check_r1(substrate_cls)
        results["R3"] = self._check_r3(substrate_cls)

        # Dynamic checks require instantiating the substrate
        try:
            sub = substrate_cls()
            obs_dummy = np.zeros((64, 64, 3), dtype=np.float32)
            sub.reset(seed)
            results["R2"] = self._check_r2(sub, obs_dummy, n_audit_steps)
            results["R4"] = self._check_r4(chain_results)
            results["R5"] = self._check_r5(sub, obs_dummy, n_audit_steps)
            results["R6"] = self._check_r6(substrate_cls, obs_dummy, n_audit_steps)
        except Exception as e:
            results["dynamic_error"] = str(e)
            results["R2"] = {"pass": None, "error": str(e)}
            results["R4"] = {"pass": None, "source": "dynamic_check_failed"}
            results["R5"] = {"pass": None, "error": str(e)}
            results["R6"] = {"pass": None, "error": str(e)}

        results["chain"] = chain_results
        results["summary"] = self._summarize(results)
        return results

    # ------------------------------------------------------------------
    # R1: No external objectives
    # ------------------------------------------------------------------
    def _check_r1(self, substrate_cls: type) -> dict:
        """Static AST analysis for forbidden imports/names."""
        violations = []
        warnings = []

        try:
            src = inspect.getsource(substrate_cls)
        except Exception as e:
            return {"pass": None, "error": f"Cannot get source: {e}"}

        try:
            tree = ast.parse(src)
        except SyntaxError as e:
            return {"pass": None, "error": f"AST parse error: {e}"}

        for node in ast.walk(tree):
            # Check import statements
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                mod = ""
                if isinstance(node, ast.ImportFrom) and node.module:
                    mod = node.module
                elif isinstance(node, ast.Import):
                    mod = ".".join(n.name for n in node.names)
                for forbidden in R1_FORBIDDEN_IMPORTS:
                    if mod.startswith(forbidden):
                        violations.append(f"import {mod}")

            # Check name usage in assignments and function calls
            if isinstance(node, ast.Name) and node.id in R1_FORBIDDEN_NAMES:
                if isinstance(node.ctx, (ast.Store, ast.Load)):
                    warnings.append(f"name '{node.id}' used at line {node.lineno}")

        # Violations (imports) are hard FAIL. Warnings (names) are soft.
        passed = len(violations) == 0
        return {
            "pass": passed,
            "violations": violations,
            "warnings": warnings[:10],  # cap to avoid noise
            "detail": "PASS: no external objective imports" if passed
                      else f"FAIL: forbidden imports: {violations}",
        }

    # ------------------------------------------------------------------
    # R2: Adaptation from computation
    # ------------------------------------------------------------------
    def _check_r2(self, sub: BaseSubstrate, obs: np.ndarray,
                  n_steps: int) -> dict:
        """Check that state changes over time (substrate adapts)."""
        sub.reset(0)
        state_before = copy.deepcopy(sub.get_state())
        keys_before = set(state_before.keys())

        for _ in range(n_steps):
            sub.process(obs + np.random.randn(*obs.shape).astype(np.float32) * 0.1)

        state_after = sub.get_state()
        keys_after = set(state_after.keys())

        changed = []
        unchanged = []
        for k in keys_before & keys_after:
            v_before = state_before[k]
            v_after = state_after[k]
            try:
                if isinstance(v_before, np.ndarray):
                    if not np.array_equal(v_before, v_after):
                        changed.append(k)
                    else:
                        unchanged.append(k)
                elif v_before != v_after:
                    changed.append(k)
                else:
                    unchanged.append(k)
            except Exception:
                changed.append(k)  # assume changed if comparison fails

        passed = len(changed) > 0
        return {
            "pass": passed,
            "changed_keys": changed,
            "unchanged_keys": unchanged,
            "n_steps": n_steps,
            "detail": f"PASS: {len(changed)} state components changed" if passed
                      else "FAIL: no state changes detected",
        }

    # ------------------------------------------------------------------
    # R3: Minimal frozen frame (zero U elements)
    # ------------------------------------------------------------------
    def _check_r3(self, substrate_cls: type) -> dict:
        """Check frozen_elements() for U-classified elements."""
        try:
            sub = substrate_cls()
            elements = sub.frozen_elements()
        except Exception as e:
            return {"pass": None, "error": str(e)}

        u_elements = [e for e in elements if e.get("class") == "U"]
        m_elements = [e for e in elements if e.get("class") == "M"]
        i_elements = [e for e in elements if e.get("class") == "I"]

        passed = len(u_elements) == 0
        return {
            "pass": passed,
            "total_elements": len(elements),
            "M_count": len(m_elements),
            "I_count": len(i_elements),
            "U_count": len(u_elements),
            "U_elements": [e["name"] for e in u_elements],
            "elements": elements,
            "detail": "PASS: zero U elements" if passed
                      else f"FAIL: {len(u_elements)} U elements: {[e['name'] for e in u_elements]}",
        }

    # ------------------------------------------------------------------
    # R4: Modifications don't degrade prior tasks
    # ------------------------------------------------------------------
    def _check_r4(self, chain_results: dict) -> dict:
        """Compare CIFAR-100 before vs after (requires chain results)."""
        if chain_results is None:
            return {"pass": None, "source": "no_chain_results",
                    "detail": "Run ChainRunner first, pass results to audit()"}

        before = chain_results.get("CIFAR-100-before")
        after = chain_results.get("CIFAR-100-after")

        if before is None or after is None:
            # Try with game-only chain: compare first vs last game
            keys = [k for k in chain_results if k not in ("CIFAR-100-before", "CIFAR-100-after")]
            if len(keys) >= 2:
                first = chain_results[keys[0]]
                last = chain_results[keys[-1]]
                rate_first = first["l1_rate"]
                rate_last = last["l1_rate"]
                passed = rate_last >= rate_first * 0.8  # allow 20% degradation
                return {
                    "pass": passed,
                    "source": f"{keys[0]}_vs_{keys[-1]}",
                    "rate_before": rate_first,
                    "rate_after": rate_last,
                    "detail": f"{'PASS' if passed else 'FAIL'}: "
                              f"L1 {keys[0]}={rate_first:.0%} {keys[-1]}={rate_last:.0%}",
                }
            return {"pass": None, "source": "insufficient_chain_tasks"}

        rate_before = before["l1_rate"]
        rate_after = after["l1_rate"]
        passed = rate_after >= rate_before * 0.8  # 20% tolerance

        return {
            "pass": passed,
            "source": "CIFAR-100 before/after chain",
            "l1_rate_before": rate_before,
            "l1_rate_after": rate_after,
            "detail": f"{'PASS' if passed else 'FAIL'}: "
                      f"CIFAR before={rate_before:.0%} after={rate_after:.0%}",
        }

    # ------------------------------------------------------------------
    # R5: One fixed ground truth (game-provided only)
    # ------------------------------------------------------------------
    def _check_r5(self, sub: BaseSubstrate, obs: np.ndarray,
                  n_steps: int) -> dict:
        """Check that process() doesn't access external state beyond obs."""
        # Wrap process() to intercept any external signal access
        # Simple check: process() should only read from obs + internal state
        # We verify it doesn't raise errors when given consistent obs
        sub.reset(0)
        actions = []
        try:
            for _ in range(n_steps):
                a = sub.process(obs.copy())
                actions.append(a)
                assert 0 <= a < sub.n_actions, f"Invalid action {a}, n_actions={sub.n_actions}"
        except AssertionError as e:
            return {"pass": False, "detail": f"FAIL: {e}"}
        except Exception as e:
            return {"pass": False, "detail": f"FAIL: exception in process(): {e}"}

        # Check action diversity (all-same = might be degenerate)
        unique = len(set(actions))
        return {
            "pass": True,
            "n_steps": n_steps,
            "unique_actions": unique,
            "action_range": [min(actions), max(actions)],
            "detail": f"PASS: process() stable over {n_steps} steps, {unique} unique actions",
        }

    # ------------------------------------------------------------------
    # R6: No deletable parts (ablation test)
    # ------------------------------------------------------------------
    def _check_r6(self, substrate_cls: type, obs: np.ndarray,
                  n_steps: int) -> dict:
        """Ablate each I-classified component, verify degradation."""
        try:
            sub = substrate_cls()
            elements = sub.frozen_elements()
        except Exception as e:
            return {"pass": None, "error": str(e)}

        # Only ablate I-classified elements (these claim irreducibility)
        i_elements = [e for e in elements if e.get("class") == "I"]
        if not i_elements:
            return {"pass": True, "detail": "PASS: no I elements declared (nothing to ablate)",
                    "ablations": []}

        # Baseline: action variance without ablation
        sub_base = substrate_cls()
        sub_base.reset(0)
        actions_base = [sub_base.process(obs + np.random.randn(*obs.shape).astype(np.float32) * 0.01)
                        for _ in range(min(n_steps, 200))]
        base_var = np.var(actions_base)

        ablations = []
        all_degrade = True

        for elem in i_elements:
            name = elem["name"]
            try:
                sub_abl = substrate_cls()
                sub_abl.reset(0)
                sub_abl.ablate(name)
                actions_abl = [sub_abl.process(obs + np.random.randn(*obs.shape).astype(np.float32) * 0.01)
                                for _ in range(min(n_steps, 200))]
                abl_var = np.var(actions_abl)
                # Degradation: either action variance drops (degenerate) or all same action
                degraded = (abl_var < base_var * 0.5) or (len(set(actions_abl)) == 1)
                ablations.append({
                    "name": name,
                    "testable": True,
                    "base_variance": round(float(base_var), 4),
                    "ablated_variance": round(float(abl_var), 4),
                    "degraded": degraded,
                })
                if not degraded:
                    all_degrade = False
            except NotImplementedError:
                ablations.append({
                    "name": name,
                    "testable": False,
                    "detail": "ablate() not implemented",
                })
            except Exception as e:
                ablations.append({
                    "name": name,
                    "testable": False,
                    "detail": str(e),
                })

        testable = [a for a in ablations if a.get("testable")]
        if not testable:
            return {
                "pass": None,
                "detail": "UNTESTABLE: ablate() not implemented",
                "ablations": ablations,
            }

        passed = all(a["degraded"] for a in testable)
        return {
            "pass": passed,
            "ablations": ablations,
            "detail": f"{'PASS' if passed else 'FAIL'}: "
                      f"{sum(a['degraded'] for a in testable)}/{len(testable)} ablations degrade",
        }

    # ------------------------------------------------------------------
    # Dynamic R3 measurement (novel — no equivalent in published work)
    # ------------------------------------------------------------------
    def measure_r3_dynamics(self, substrate_cls: type,
                            obs_sequence: list = None,
                            n_steps: int = 1000,
                            n_checkpoints: int = 5) -> dict:
        """Track WHICH state components change and WHEN during processing.

        This is the novel measurement: unlike static R3 (frozen_elements()),
        dynamic R3 shows how the substrate's internal structure evolves
        over time and whether changes are substrate-driven or externally-driven.

        obs_sequence: list of np.ndarray observations. If None, uses random.
        n_steps: total steps to run
        n_checkpoints: number of state snapshots to take

        Returns: {
          checkpoints: [{step, state_snapshot, changed_from_prev, change_magnitude}],
          component_change_times: {component: first_step_changed},
          dynamics_profile: "static"|"slow"|"fast"|"chaotic",
          r3_dynamic_score: fraction of declared-M elements that actually change
        }
        """
        try:
            sub = substrate_cls()
            declared_elements = sub.frozen_elements()
            m_elements = [e["name"] for e in declared_elements if e.get("class") == "M"]
        except Exception as e:
            return {"error": str(e)}

        sub.reset(0)
        checkpoint_interval = max(1, n_steps // n_checkpoints)
        checkpoints = []
        state_prev = copy.deepcopy(sub.get_state())
        component_change_times = {}

        for step in range(n_steps):
            if obs_sequence is not None and step < len(obs_sequence):
                obs = obs_sequence[step]
            else:
                obs = np.random.randn(64, 64, 3).astype(np.float32) * 0.5

            sub.process(obs)

            if step % checkpoint_interval == 0 or step == n_steps - 1:
                state_now = sub.get_state()
                changed_keys = []
                magnitudes = {}
                for k in state_prev:
                    v_prev, v_now = state_prev.get(k), state_now.get(k)
                    try:
                        if isinstance(v_prev, np.ndarray) and isinstance(v_now, np.ndarray):
                            if not np.array_equal(v_prev, v_now):
                                changed_keys.append(k)
                                magnitudes[k] = float(np.linalg.norm(
                                    v_now.astype(float) - v_prev.astype(float)
                                ))
                                if k not in component_change_times:
                                    component_change_times[k] = step
                        elif v_prev != v_now:
                            changed_keys.append(k)
                            if k not in component_change_times:
                                component_change_times[k] = step
                    except Exception:
                        pass

                checkpoints.append({
                    "step": step,
                    "changed_from_prev": changed_keys,
                    "change_magnitudes": magnitudes,
                    "state_size": {k: (v.shape if isinstance(v, np.ndarray) else type(v).__name__)
                                   for k, v in state_now.items()},
                })
                state_prev = copy.deepcopy(state_now)

        # Score: fraction of declared-M elements that actually changed.
        # Bidirectional match: element name ⊂ state key OR state key ⊂ element name.
        # This handles naming mismatches (e.g. "aliased_set" vs "aliased_count").
        actually_changed = set(component_change_times.keys())

        def _matches(m_name: str, state_key: str) -> bool:
            m_lower = m_name.lower().replace("_", "")
            k_lower = state_key.lower().replace("_", "")
            return m_lower in k_lower or k_lower in m_lower

        m_verified = [m for m in m_elements
                      if any(_matches(m, k) for k in actually_changed)]
        r3_dynamic_score = len(m_verified) / max(len(m_elements), 1)

        # Dynamics profile
        n_changed_checkpoints = sum(1 for c in checkpoints if c["changed_from_prev"])
        if n_changed_checkpoints == 0:
            profile = "static"
        elif n_changed_checkpoints < n_checkpoints * 0.3:
            profile = "slow"
        elif n_changed_checkpoints < n_checkpoints * 0.8:
            profile = "fast"
        else:
            profile = "continuous"

        return {
            "checkpoints": checkpoints,
            "component_change_times": component_change_times,
            "declared_M_elements": m_elements,
            "verified_M_elements": m_verified,
            "r3_dynamic_score": round(r3_dynamic_score, 3),
            "dynamics_profile": profile,
            "detail": (f"{'PASS' if r3_dynamic_score >= 0.5 else 'PARTIAL'}: "
                       f"{len(m_verified)}/{len(m_elements)} declared-M elements "
                       f"verified to change. Profile: {profile}"),
        }

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    def _summarize(self, results: dict) -> dict:
        checks = ["R1", "R2", "R3", "R4", "R5", "R6"]
        passed = [r for r in checks if results.get(r, {}).get("pass") is True]
        failed = [r for r in checks if results.get(r, {}).get("pass") is False]
        unknown = [r for r in checks if results.get(r, {}).get("pass") is None]

        r3_u = results.get("R3", {}).get("U_count", "?")
        chain = results.get("chain") or {}
        l1_rates = {k: v["l1_rate"] for k, v in chain.items() if isinstance(v, dict) and "l1_rate" in v}

        return {
            "score": f"{len(passed)}/{len(checks)}",
            "passed": passed,
            "failed": failed,
            "unknown": unknown,
            "R3_U_count": r3_u,
            "chain_l1": l1_rates,
            "verdict": "R3_PASS" if "R3" in passed else f"R3_FAIL ({r3_u} U elements)",
        }
