# ASCC efficiency-v1 benchmark

This branch preserves the first semantics-preserving efficiency pass for the canonical GroupOpt/ASCC implementation.

## What changed

1. The all-source head proposal used only to score candidate tails is computed under `torch.no_grad()`.
2. After the tail is selected, only that tail's conditional head distribution is recomputed with autograd enabled.
3. Static AM head-query projections are cached across the construction rollout.
4. Entropy/statistics used only for logging are detached from autograd.

The optimization does **not** prune legal actions and does not introduce top-k or approximate candidate selection.

## Fixed-checkpoint semantic check

TSP50, 10,000 instances, test seed `20260904`, AM `native_conditional_free`, seed-1234 checkpoint:

- baseline vs previously saved costs, max abs diff: `0.0`
- optimized vs baseline, max abs diff: `0.0`
- optimized exact per-instance equality fraction: `1.0`
- baseline mean cost: `6.1457518886566165`
- optimized mean cost: `6.1457518886566165`

Thus the optimized inference path is exactly identical on all 10,000 evaluated instances for this benchmark.

## Efficiency

| Metric | Canonical | efficiency-v1 | Change |
|---|---:|---:|---:|
| 10k-instance eval wall time | 24.47 s | 22.48 s | ~8.1% less time |
| 300-step training wall time | 107.45 s | 99.57 s | ~7.3% less time |
| training peak GPU memory | 16.682 GB | 8.831 GB | ~47.1% less memory |

The training runs use the same seed and configuration, but stochastic training trajectories are not claimed to be bit-identical; the fixed-checkpoint inference equivalence above is the strict semantic check.

## Layout

- `project/`: runnable source snapshot (`src/`, `experiments/`, `tests/`, `pyproject.toml`).
- `ascc-efficiency-v1.patch`: exact local delta against the canonical GroupOpt source base commit.
- `source_base_commit.txt`: canonical source commit used as the base.
- `benchmark/`: timing logs, short train logs, summaries, and raw fixed-checkpoint costs.

The original `Ptr-net-GT` main branch is intentionally untouched; this material lives only on `ascc-efficiency-v1`.
