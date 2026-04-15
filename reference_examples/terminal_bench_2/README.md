# Terminal-Bench 2

Paper reference experiment for Meta-Harness on Terminal-Bench 2.0. The outer loop evolves full agent scaffold files in `agents/`, then evaluates them with Harbor on the TB2 hard split.

## Quick Start

Install:

```bash
cd reference_examples/terminal_bench_2
uv sync
```

Read setup details first:

```bash
sed -n '1,200p' SETUP.md
```

Run one evolve iteration:

```bash
uv run python meta_harness.py --iterations 1 --trials 2
```

Run one candidate on the `extract-elf` smoke task:

```bash
uv run bash scripts/run_eval.sh agents.baseline_kira:AgentHarness full 1 1 -i extract-elf
```

## Key Files

- `.claude/skills/meta-harness-terminal-bench-2/SKILL.md`: proposer prior used by `meta_harness.py`.
- `agents/`: baseline agents and the write target for generated candidates.
- `prompt-templates/terminus-kira.txt`: prompt template used by `baseline_kira.py`.

## Runtime And Cost

The source notes budgeted the hard-split run at roughly `$70/iteration` and about `~4h/iteration`, dominated by Harbor/Runloop evaluation over 30 hard tasks with 2 trials each. The proposer cost is minor relative to the benchmark cost.
