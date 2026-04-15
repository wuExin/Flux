# Terminal-Bench 2 Setup

This directory is the paper reference scaffold for Terminal-Bench 2.0. The shipped scripts target Harbor with the `runloop` environment:

```bash
uv run harbor run ... -d "terminal-bench@2.0" -e runloop
```

## Requirements

- A working Anthropic API key.
- A Runloop/Daytona account with enough sandbox quota for the concurrency you plan to use.
- The `terminal-bench` package installed from this directory's `pyproject.toml`.
- Python 3.12.

Install:

```bash
cd reference_examples/terminal_bench_2
uv sync
```

## Environment Variables

The repo root ships `.env.example`, but the shell wrappers in `scripts/` source `.env` from this directory. In practice:

- Put the same `.env` file in this directory, or
- Export the needed variables in your shell before running the scripts.

As shipped, the code expects at least `ANTHROPIC_API_KEY`. Harbor/Runloop credentials may also need to be present in your environment depending on your setup.

## Task Set

- Dataset ID in the paper code: `terminal-bench@2.0`
- `meta_harness.py` uses the full TB2 dataset by default: 89 tasks, 2 trials each
- The paper submission metadata targeted `laude-institute/terminal-bench-2` commit `69671fbaac6d67a7ef0dfec016cc38a64ef7a77c`

## Smoke Check

```bash
uv run bash scripts/run_eval.sh agents.baseline_kira:AgentHarness full 1 1 -i extract-elf
```

The shell wrappers use `timeout` when available and fall back to `gtimeout` if
GNU coreutils is installed on macOS. If neither command is present, the Harbor
run still works but will not have an outer wall-clock timeout.

## Local Vs Remote Sandbox

As released, the scripts use `-e runloop`. If you want a different Harbor environment, change the environment flag in the shell wrappers or invoke `harbor run` directly with your local environment choice. The paper code does not ship a second local-sandbox path here.

## Version Note

The paper code did not ship a separate pinned `tb-cli` version. This release
keeps parity and installs the `terminal-bench` package declared in
`pyproject.toml`; `harbor` must be available from that install.
