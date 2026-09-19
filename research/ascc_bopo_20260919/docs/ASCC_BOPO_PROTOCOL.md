# ASCC + BOPO joint-training screening protocol, 2026-09-19

This is a new source-first model on a separate branch. Historical canonical and
standalone adapters are preserved. No positive outcome is assumed.

## Method

- POMO/BOPO-compatible encoder and endpoint decoder, loaded strictly by name and
  shape. All encoder and decoder parameters are trainable during the screening.
- Every construction step, including the first, may choose any unresolved source.
  No continuation gate, deviation quota, fixed-source fallback, or hard top-k.
- Endpoint query uses the selected source path's actual first node. Cached node
  projections, path endpoints, and sizes avoid all-source endpoint proposals.
- A trainable head residual sees source-path start, source, candidate head, the
  candidate path's other end, and size. It starts at zero. Route mode recovers the
  upstream checkpoint at initialization; adaptive source starts untrained.
- Complete-tour costs rank trajectory preferences. The score is mean per-edge
  joint source + endpoint log probability, not marginal permutation probability.
- Explicit hybrid rollout: column0 is greedy even in training mode; other columns
  sample both learned source and endpoint. BOPO uses best-anchored rank filtering
  and objective-ratio scaling. Ties within relative tolerance 1e-6 are skipped.
- REINFORCE control scores stochastic columns only and uses a leave-one-out
  baseline including the greedy trajectory. It does not pretend greedy is sampled.
- Source-first/no-summary and the new endpoint interface are method changes; this
  is not a numerically equivalent rewrite of canonical ASCC.

## First screening budget

Four primary arms: route/learned source × REINFORCE/BOPO. One seed, 1234. TSP100,
batch4 instances, 16 trajectories, K8 preference samples; 200 optimizer steps per
arm, capped at 600 seconds of train/validation loop per arm. Same official
BOPO TSP100 checkpoint and RNG initialization; learning rates 1e-5 backbone and
1e-4 new modules. Same generated training coordinates independently of action RNG.
Validation128, test256, best of8 anchor contexts, no augmentation or local search.
Test coordinates are fixed and never used for checkpoint selection. Initial,
best-validation and final checkpoints are distinct and retained, including step0.

These are a bounded implementation/learning screen, NOT a converged comparison,
an ICLR main table, or evidence across training seeds. Historical native BOPO
all-starts×8 augmentation is a different decoding budget and is not spliced into
the table. Adaptive source may have a worse initial score because it is new;
best-step0 does not demonstrate a successful learned adaptive method.

After this screen, the decision must consider all arms and full curves, not only
the best positive seed. Additional capacity-matched route and random-source
controls are implemented. Formal work requires these controls, at least5 seeds,
convergence and equal-time comparisons. TSP500/1000 training is not authorized by
this screening file and is not launched automatically.

## Resource and provenance rules

- Use one visible GPU; cap this process's allocator at22% of its total memory.
  Do not terminate other projects or modify system drivers. If sharing causes OOM
  or CUDA errors, record failure and stop the queue.
- The container currently points libcuda.so.1 to550 while its kernel module is535.
  A project-local library selection directory uses the existing535 libraries via
  LD_LIBRARY_PATH; no system symlink is changed.
- Source snapshots/hashes, git commit, checkpoint hash, test/validation hashes,
  separate data/action RNG states and full config are stored with each run.
- Never overwrite a run directory. Failures, prototype smokes, and the first
  anchor-context smoke remain separate from the final path-start implementation.
- Report paired fixed-model differences with instance uncertainty separately from
  cross-seed uncertainty. One seed cannot establish cross-training reliability.

## Falsification gates

1. If route recovery or completion tests fail, do not train.
2. If source or endpoint receives no gradient, do not interpret performance.
3. If ASCC+BOPO only beats ASCC+RL but not route+BOPO, training improved; an ASCC
   advantage has not been established.
4. If extra-capacity route or trained random/heuristic source explains the gain,
   narrow the claim accordingly.
5. Large-scale extrapolation and clustered geometry are hypotheses, not a promise
   that those distributions will produce positive results.
