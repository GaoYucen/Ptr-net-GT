# Official AM / ICAM TSP200 adapter study

Registered before reading study outcomes, 2026-09-19.

Primary question: does a canonical-style adaptive forest selector improve an
official pretrained host after both route and forest policies receive TSP200
adaptation? Secondary: can the same implementation improve ICAM?

Host AM uses official TSP100 epoch-99 weights. ICAM uses official icam_tsp.pt.
Strict load and native action equivalence precede training. No upstream code is
modified. Ports preserve native source-conditioned endpoint queries; selector
uses canonical-style path start/mean/size and detached normalized native head
summaries. Initial native node is a latent anchor/context; learned source remains
free at every merge. This is a declared new host port, NOT the older AM-style
checkpoint or the previous source-first BOPO experiment.

Primary size TSP200 uniform. Predetermined secondary cells: TSP200 clustered and
TSP500 uniform. No task selection based on observed positive results. No CVRP in
this experiment; complete-graph feasibility proofs are not extended to capacity
or time windows. No claim that native out-of-distribution performance equals a
model trained to convergence on TSP200.

Both hosts: 2,048 native greedy training tours, CPU best-improvement 2-opt (at
most 200 moves); 64 separate teacher-validation instances. Within each host all
arms share the SAME teacher tours. Track data-generation cost. Teachers are
feasible references, not certified OPT. Train/validation/test coordinate seeds
and tensor hashes are recorded. All earlier project test sets are excluded.

Warmup: 2,500 updates, batch32, official host weights trainable at 1e-5. One
sampled partial-state depth per minibatch. Route and random-order forests receive
the same teacher/data/update budget. Host normalization stays in official eval
mode for all arms. Select warmup checkpoint on independent endpoint NLL. Source
selector gets no arbitrary teacher order labels. Warmup is shared across joint
training seeds, therefore reported seed variability is conditional on this
common warmup, not complete end-to-end retraining variability.

Joint stage: route, trained random, learned, each seeds1234/4321/2468; 300 updates
each, batch4 independent instances times4 stochastic rollouts, leave-one-out
REINFORCE baseline, host LR1e-5 and selector LR1e-4, norm clipping1. All arms use
the same training coordinate sequence per seed. Validation64 every50 updates;
select checkpoint including step0. Test256 TSP200, secondary cells64 each.
Single greedy deployment and final test200 eight geometric augmentations are
reported separately. Random source action RNG fixed and documented. Native
route deployment uses official forward, not a slow forest wrapper. No 2-opt or
other local search at deployment.

All arms get equal samples/updates, NOT equal wall time. Record timing and curves
to avoid claiming compute efficiency from sample-matched results. Pretrained
step0/native scores remain visible. If learned exceeds the adapted route, add
the prespecified capacity control before interpreting a parameter-independent
benefit. Otherwise no positive mechanism claim. Fixed/heuristic controls and a
fully matched inference-time frontier remain future requirements, not completed
evidence. Train curves determine whether this run is converged or still only an
adaptation screen; 300 RL updates do not establish ultimate method limits.

GPU resources: one host on each 4090, per-process allocation limit32% (about
7.6GB), CPU threads2 each and2 CPU teacher workers each. Existing device users
are untouched. Full gradient-vs-checkpoint recomputation parity is tested to
avoid the prior mutable-state checkpoint bug. Approximate expected runtime is
hours, from measured full updates around2.3seconds at batch4/TSP200.

Final outputs: native/route/random/learned table, all seed values, paired
instance intervals conditional on trained models, training-seed summaries,
training curves, size/distribution heatmap, and deployment time with measured
hardware limitations. Preserve negative results and report remaining gaps.
