"""
Step 462 — LLM navigation benchmark (meta-experiment, no Python substrate code).

NOT a Python substrate experiment. Three Claude models (haiku/sonnet/opus)
ran through ARC-AGI-3 API via separate CC sessions to test LLM navigation.

Results (git commit 82ae083, 2026-03-18): 0/3 models navigated LS20. 0 levels.
- Haiku: 97 steps, 100% ACTION1 dominance, mapped coordinates as letter grid
- Sonnet: 4 steps, spatial awareness but read frame data directly (cheated)
- Opus: 11 steps, spatial confusion, explicitly gave up ("I cannot solve this")

KEY FINDING: LLMs lack systematic exploration mechanism. LSH argmin covers
state space unintelligently but systematically. LLMs exploit by default.
Intelligence without systematic exploration = exploitation trap.

Infrastructure: llm-benchmark/parse_transcripts.py (commit 028dc28)
Sessions: B--M-the-search-llm-benchmark project directory.
Leo summary: "The mechanism that makes LSH work is its stupidity, not its
intelligence. LLMs are too smart to explore randomly."
"""
# LLM meta-experiment. See llm-benchmark/ for parse_transcripts.py.
# No Python substrate code for this step.
