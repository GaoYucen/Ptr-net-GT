# Pair-ASCC validation protocol

`single_naive` retains the historical fixed-source implementation.  `single_opt`
skips only source projections and logits that cannot affect a fixed source action;
the exact-output unit test is required before its use as the latency baseline.
`seq2` is the same sequential policy with the optimized path, retaining the state
update between every edge.  `pair_add` jointly selects two edges using additive
scores and a joint legality mask.  `pair_interaction` adds the learned low-rank
pair term.

Run a smoke benchmark before a full sweep:

```bash
PYTHONPATH=src python experiments/benchmark_pair_validation.py \
  --out results/pair-validation-smoke --nodes 50 --batches 1 \
  --trajectories 1 --warmup 2 --repeats 3 --runs 1
```

The full latency sweep writes one record for every timing repetition to
`latency_raw.csv`; it never stores only an already-averaged latency.  Greedy uses
one trajectory.  Values above one trajectory are sampling budgets.

On the current 4090 host the kernel module is 535 while the container's default
`libcuda.so` symlink points at 550.  Use `LD_LIBRARY_PATH=/cuda_fix` so CUDA loads
the matching 535 user-space driver.  The queue script applies this automatically;
the path can be overridden with `PAIR_VALIDATION_CUDA_DRIVER_PATH`.

The ten-seed screen is queued with `run_pair_validation_queue.sh`.  After all
runs finish, aggregate only seed-level repeats for the primary non-inferiority
claim:

```bash
PYTHONPATH=src python experiments/summarize_pair_validation.py \
  --runs results/pair-validation-screen --out results/pair-validation-summary
```

AM and POMO are deliberate adapter slots.  They fail explicitly until their
official checkpoint and preprocessing adapter are supplied; they must not be
replaced by an internal ASCC model.
