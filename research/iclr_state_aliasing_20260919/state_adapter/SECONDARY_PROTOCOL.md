# Secondary decoder-plus-residual adaptation, registered before primary test

Motivation: a frozen pretrained decoder may impose a separate adaptation
constraint even when the residual receives additional component information.
This study updates the native decoder as well as the residual while retaining
the frozen encoder and cached node embeddings. It does not train a learned
source policy, and is not fully optimized ASCC.

This plan was fixed using implementation throughput and primary development
curves only, before any primary final-test dataset was generated or inspected.
ICAM is required. AM is also prespecified if the same complete 12-arm suite
fits before 16:45 local time; do not launch partial AM arms based on test results.
If AM is omitted for time, record it as unexecuted, not a negative result.

Four arms: route/blind, route/aware, random/blind, random/aware; independent
seeds 12031, 12037, 12041. Each gets 12000 updates, batch 64; residual Adam LR
3e-4, decoder LR 1e-5; global trainable-gradient clip 1.0. ICAM trains every
parameter under native `net.decoder`. AM trains every native host parameter
outside `init_embed` and `embedder`; actual parameter names/counts are logged.
Encoder parameters stay frozen. The host wrapper recomputes decoder key/value
and query projections in every logits call, so no stale decoder-weight cache
is reused. Only the frozen encoder representations are cached.

Coordinate training/development seeds and native-plus-2-opt teacher recipe are
identical to the frozen study for direct adaptation comparisons. Every residual
and decoder begins independently from the pretrained checkpoint/paired random
initialization, never from a primary study fitted model. Development selection
is again full-tour normalized cost every 1000 updates, separately for each arm.

Final-test seeds are new: 73019211 through 73019214 for uniform100 (256),
cluster100 (128), uniform200 (128), cluster200 (128), respectively. Test is
created only after all 12 checkpoints for this host are frozen. No primary or
secondary test result changes architecture, optimizer, arm inclusion, steps,
or model selection. Report all four arms and all three seeds, regardless of
outcome. Shared architecture does not imply identical effective capacity.
