# Supplemental source-order variability

Registered before final test inspection. Primary random-source tables retain
their original fixed RNG seed 9417319. After checkpoint freezing, additionally
evaluate each selected random-source checkpoint with source RNG seeds 9527319,
9637319 and 9747319. Report all three, their average and spread; never select
the best source RNG. The same seed list and batch size 32 are used for blind
and aware. These are uniform random source choices, not learned order policies.
This supplemental inference-variability table does not redefine the primary
single-RNG test protocol or tune checkpoints.
