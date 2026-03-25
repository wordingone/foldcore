"""
DEPRECATED — use templates/step_template.py instead.

Moved 2026-03-24 per Jun directive. The working template is at:
  B:/M/the-search/templates/step_template.py

Key fixes in the new template:
- CIFAR wrappers included (SplitCIFAR100Wrapper via make_prism())
- Substrate interface correct: set_game(n_actions), process(obs), on_level_transition()
- compute_chain_kill() call for automatic PASS/KILL/FAIL verdict
- save_results() enforced
"""
raise ImportError("Use templates/step_template.py — this file is deprecated.")
