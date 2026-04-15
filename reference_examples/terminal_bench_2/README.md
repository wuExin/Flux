# Terminal-Bench 2

Paper reference experiment for Meta-Harness on Terminal-Bench 2.0. The outer loop evolves full agent scaffold files in `agents/`, then evaluates them with Harbor on the full 89-task TB2 dataset used in the paper runs.

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

With Opus with a high-tier API key, each iteration takes about 4-6 hours to complete. The cost for a single iteration is approximately $500.
