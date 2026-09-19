# ASCC efficiency optimization snapshot

`base_v1/` is the archived canonical efficiency-v1 source tree used by these
experiments.  Each directory under `variants/` contains only files that differ
from that base; copy a variant over `base_v1/` to reconstruct the corresponding
worktree.  The matching aggregate outputs are under `results/`.

- `v2`: endpoint-computation optimization in `src/groupopt/models/am.py`.
- `v3`: a more aggressive optimization retained for diagnosis.  Its gradient
  path is known to be incorrect and this variant must not be used as the final
  implementation.
- `v31`: repairs the v3 gradient-equivalence defect.  See
  `results/v31/gradient_equivalence.json` and `grad_probe.py`.
- `v4-lite`: a separate lite intervention affecting the AM model and the train
  and evaluation entry points.  It changes the method/compute tradeoff and is
  not asserted to be numerically identical to full ASCC.

The branch excludes fitted checkpoints and tensor archives.  Configs, metric
histories, semantic checks, inference sweeps, and aggregate summaries are
retained.  Historical command paths inside configs are provenance fields and
may need adjustment before rerunning on another machine.
