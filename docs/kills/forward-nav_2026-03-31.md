# Kill: Forward Navigation (BFS to frontier)

**Date:** 2026-03-31
**Experiments:** 1 (API, 25 games, 5-min budget)
**Result:** K=2/25 (lp85, r11l)
**Kill criterion:** K ≤ 5 (doesn't beat state-argmin K=5/25 baseline)

## Mechanism

State-argmin + transition recording + BFS to nearest state with untried actions. Navigate forward through known transitions instead of wandering randomly.

## Why it failed

5-min API budget = ~5K steps at 60ms latency. At 5K steps, states have 7 KB actions each. Most states visited only 1-2 times — never exhaust their actions. BFS frontier navigation fired 0-163 times out of 5000 steps. The mechanism is irrelevant at this step count.

lp85 and r11l are a subset of the K=5/25 baseline set. No new games found. 3 games lost (ft09, vc33, ls20) due to step budget.

## What it taught

Navigation efficiency is not the bottleneck at 5K steps. Visual salience (which actions to try first) matters more than traversal efficiency (how to reach untested states). This motivated the full dolphin spec (CC segmentation + priority tiers).

## Status: KILLED (not ABANDONED)

1 experiment. Kill criterion met. Root cause understood (step budget makes navigation irrelevant). Superseded by full dolphin approach.
