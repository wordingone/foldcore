#!/usr/bin/env python3
"""
Parse LLM benchmark JSONL transcripts from ARC-AGI-3 game sessions.
Extracts: model, steps, levels, action distribution, reasoning snippets.
Streams files — never loads full base64 frames into memory.
"""
import json
import os
import re
from ast import literal_eval

PROJECTS_DIR = os.path.expanduser("~/.claude/projects/B--M-the-search-llm-benchmark")


def parse_transcript(path: str) -> dict:
    model = None
    reasoning_snippets = []
    gave_up = False
    final_diagnostics = None
    last_levels = 0

    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue

            t = obj.get("type")

            if t == "assistant":
                msg = obj.get("message", {})
                if model is None:
                    model = msg.get("model")
                for block in msg.get("content", []):
                    btype = block.get("type")
                    if btype in ("text", "thinking"):
                        text = block.get("text", "")
                        if text and len(text) > 20 and "base64" not in text[:200]:
                            snippet = text[:200].replace("\n", " ").strip()
                            if snippet and len(reasoning_snippets) < 6:
                                reasoning_snippets.append(snippet)
                        if any(kw in text.lower() for kw in [
                            "give up", "cannot solve", "stuck", "no progress",
                            "random", "don't know", "no clear", "unclear", "impossible"
                        ]):
                            gave_up = True

            elif t == "user":
                msg = obj.get("message", {})
                content = msg.get("content", [])
                if not isinstance(content, list):
                    continue
                for c in content:
                    if not isinstance(c, dict) or c.get("type") != "tool_result":
                        continue
                    inner = c.get("content", [])
                    if not isinstance(inner, list):
                        continue
                    for x in inner:
                        if not isinstance(x, dict) or x.get("type") != "text":
                            continue
                        text = x.get("text", "")
                        # Parse diagnostics from game_act results
                        if "'diagnostics'" in text or '"diagnostics"' in text:
                            # Parse whole result dict, then extract diagnostics
                            try:
                                result_dict = literal_eval(text)
                                diag = result_dict.get("diagnostics")
                                if diag:
                                    final_diagnostics = diag
                                    last_levels = diag.get("levels_completed", last_levels)
                            except Exception:
                                pass
                        # Also check for game_start result with levels
                        elif "'levels_completed'" in text:
                            m = re.search(r"'levels_completed':\s*(\d+)", text)
                            if m:
                                lvl = int(m.group(1))
                                if lvl > last_levels:
                                    last_levels = lvl

    # Build result
    result = {
        "file": os.path.basename(path),
        "model": model or "unknown",
        "reasoning_snippets": reasoning_snippets,
        "gave_up_signal": gave_up,
    }

    if final_diagnostics:
        result.update({
            "steps": final_diagnostics.get("steps", 0),
            "levels_completed": final_diagnostics.get("levels_completed", last_levels),
            "level_steps": final_diagnostics.get("level_steps", []),
            "action_counts": final_diagnostics.get("action_counts", {}),
            "dom_pct": final_diagnostics.get("dom_pct", 0.0),
            "game_overs": final_diagnostics.get("game_overs", 0),
        })
    else:
        result.update({
            "steps": 0, "levels_completed": last_levels, "level_steps": [],
            "action_counts": {}, "dom_pct": 0.0, "game_overs": 0,
        })

    return result


def format_result(r: dict) -> str:
    lines = [
        f"Model: {r['model']}",
        f"Steps: {r['steps']}  Levels: {r['levels_completed']}  Game-overs: {r['game_overs']}",
        f"Level steps: {r['level_steps']}",
        f"Actions: {r['action_counts']}  dom={r['dom_pct']}%",
        f"Gave-up signal: {r['gave_up_signal']}",
    ]
    if r['reasoning_snippets']:
        lines.append(f"Reasoning [{len(r['reasoning_snippets'])} snippets]:")
        for s in r['reasoning_snippets'][:4]:
            lines.append(f"  - {s[:180]}")
    return "\n".join(lines)


def main():
    files = sorted([
        os.path.join(PROJECTS_DIR, f)
        for f in os.listdir(PROJECTS_DIR)
        if f.endswith(".jsonl")
    ])

    if not files:
        print("No JSONL files found in", PROJECTS_DIR)
        return []

    results = []
    for path in files:
        size_mb = os.path.getsize(path) / 1e6
        print(f"Parsing {os.path.basename(path)} ({size_mb:.1f} MB)...", flush=True)
        r = parse_transcript(path)
        results.append(r)
        print(f"  {r['model']}  steps={r['steps']}  levels={r['levels_completed']}"
              f"  dom={r['dom_pct']}%  gave_up={r['gave_up_signal']}", flush=True)

    return results


if __name__ == "__main__":
    results = main()
    print("\n" + "="*70)
    print("LLM BENCHMARK SUMMARY")
    print("="*70)
    for r in results:
        print()
        print(format_result(r))
