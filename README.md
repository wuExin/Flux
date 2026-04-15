# Meta-Harness

Meta-Harness is a framework for automated search over task-specific model harnesses: the code around a fixed base model that decides what to store, retrieve, and show the model while it works. This repo is the public framework release with two reference experiments from the paper.
The paper is [Meta-Harness: End-to-End Optimization of Model Harnesses](https://arxiv.org/abs/2603.28052).

## What this repo is

- The reusable framework and onboarding flow for applying Meta-Harness to a new domain.
- Two paper reference experiments under `reference_examples/`:
  - [`reference_examples/text_classification/`](reference_examples/text_classification/README.md): memory-system search for text classification.
  - [`reference_examples/terminal_bench_2/`](reference_examples/terminal_bench_2/README.md): scaffold evolution for Terminal-Bench 2.0.

## Quick Start

Text classification:

```bash
cd reference_examples/text_classification
uv sync
uv run python meta_harness.py --iterations 1
```

Terminal-Bench 2 smoke task:

```bash
cd reference_examples/terminal_bench_2
uv sync
uv run bash scripts/run_eval.sh agents.baseline_kira:AgentHarness full 1 1 -i extract-elf
```

For full setup details, expected runtime, and additional one-candidate commands, use the subdir READMEs.

## Applying Meta-Harness To A New Domain

Start by pointing your coding assistant to [`ONBOARDING.md`](ONBOARDING.md) and having a conversation with it.
This should produce a `domain_spec.md` file with concrete details on how to proceed with implementing Meta-Harness for your domain.

## Citation

If this repository is useful for your research, please cite:

```bibtex
@misc{lee2026metaharnessendtoendoptimizationmodel,
      title={Meta-Harness: End-to-End Optimization of Model Harnesses},
      author={Yoonho Lee and Roshen Nair and Qizheng Zhang and Kangwook Lee and Omar Khattab and Chelsea Finn},
      year={2026},
      eprint={2603.28052},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2603.28052},
}
```
