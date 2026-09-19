# Frozen ICAM component-state pilot, 2026-09-19

This protocol is written before development or test measurements. This is an
endpoint-interface pilot, not sufficient training of the full ASCC policy.

The official ICAM checkpoint and all host weights are frozen. Four arms share
exactly the same endpoint residual architecture and parameter count (the actual
count is programmatically recorded): route/blind, route/aware, random/blind,
random/aware. Blind inputs contain only static embeddings, source path start
and tail, legal head mask and functions of their coordinates. Aware inputs add
component mean embedding, size and target component's other endpoint. The
The shared attention layer pools all legal target-component representations
into the source query, making each endpoint score depend on the other paths.
Equal parameter count does not imply equal effective capacity: some blind
feature slots are constant or redundant. No universal-policy representation
theorem is claimed for the finite residual network. The residual is zero at
initialization, so route initially matches native ICAM.
No learned source selector is claimed in this study.

The training data are an equal mixture of uniform and five-cluster TSP100;
2048 independent instances. Teachers are official native greedy tours improved
with deterministic best-improvement 2-opt (not exact-optimal labels). Each arm
uses the same 2048 teachers, minibatch sequence and sampled depths for paired
seeds 12031, 12037, 12041. Each seed independently initializes the residual.
Each update samples a depth uniformly from 0 to N-2. Route removes a contiguous
prefix of the teacher tour; random removes a uniform subset of teacher edges.
Endpoint supervision does not train a source policy. Adam, learning rate
3e-4, batch size 64, gradient clipping 1.0; planned 12000 updates per arm.

Development: 64 uniform + 64 clustered TSP100 instances, disjoint from training.
Evaluate full greedy tours every 1000 updates; select the checkpoint with the
lowest mean normalized greedy cost over those two equal-size sets. Conditional
NLL/accuracy is separately evaluated on pre-fixed teacher partial states at
depth fractions 0.25, 0.50 and 0.75. Any budget change is recorded prospectively.

Final test is generated from separate fixed seeds only after model selection:
256 uniform TSP100, 128 clustered TSP100, 128 uniform TSP200, 128 clustered
TSP200. Test is inspected once for the frozen selected checkpoints. No model
selection or hyperparameter change uses final test. Native baseline, paired
per-instance costs, greedy-tour validity and latency are retained. Conditional
test labels are the same teacher recipe; NLL improvement is not a tour benefit.

Primary comparisons: aware versus blind within random (state repair), aware
random versus aware route (forest value), each versus native. Report per-seed
means, seed-average paired per-instance bootstrap confidence intervals and
win fractions, with seed and instance variability distinguished. Three seeds
are a pilot, not evidence of universal robustness or publication readiness.

Prospective budget amendment before any development results: the isolated smoke
profile measured 100 batch-64 updates in 1.30 seconds. Consequently the original
3000-update plan was increased to 12000 updates per arm, with validation every
1000 updates. The architecture was frozen before this smoke and remains fixed.

Second-host replication registered before any development or test outcomes:
repeat the complete same four-arm, three-seed protocol on official AM, provided
the zero-residual route passes exact action equivalence to native AM. Each host
uses its own native-plus-2-opt teachers, the same coordinate seeds, and the same
architecture and hyperparameters. ICAM is the primary host. No hyperparameter
changes will be made after either host's final-test inspection.
