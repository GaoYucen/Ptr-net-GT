# Controlled conditional endpoint experiment

This experiment studies a selected source endpoint. It does not evaluate the complete source–target policy or claim full-tour TSP competitiveness.

## Data-generating distribution

Draw independent Euclidean coordinates in the unit square. For `(n,k)`, the source is singleton node `0`. Heads `1,...,k` connect to a uniform random permutation of tails `k+1,...,2k`; nodes `2k+1,...,n-1` are singleton components. A pair consists of this forest and a second forest obtained by swapping the partners of two uniformly chosen distinct non-singleton heads. Thus the two forests share the complete coordinate array, source head/tail, active head set, source legality mask, node roles, and component sizes, but differ in head–tail association. Every generated state is retained; no filter chooses cases with differing optimal actions.

The training shape is `(9,3)`. Transfer shapes are `(7,2)`, `(11,4)`, `(13,5)`. The clustered test distribution draws three centers uniformly in `[0.15,0.85]^2`, assigns each point independently to a center, adds Gaussian noise with standard deviation `0.06`, and clips coordinates to the unit square. All coordinates are rounded to float32 before computing float64 labels, so stored coordinates and labels agree.

Training has 40,000 geometries (80,000 forests), with seed `190919101`. Validation has 3,000 geometries, with seed `190919201`. Each test distribution has 1,500 geometries for each of seeds `190919301`, `190919302`, and `190919303`, hence 4,500 geometries / 9,000 forests. All members of a geometry stay in one split. Verification hashes all coordinate arrays and checks disjoint splits.

## Exact action values

Contract each directed path to its ordered head and tail. Completing the forest is equivalent to choosing a cyclic permutation of the contracted components. For each possible first head `a`, enumerate all permutations starting with the source component and `a`; minimize the sum of Euclidean connector lengths. Existing internal path lengths are action-independent within a forest and are omitted from the stored connector `q` array. The stored `internal` array supplies their sum when full-tour lengths are needed.

The primary outcome is `R_F(a) = Q_F(a) - min_b Q_F(b)`, in unit-square Euclidean length. A selected action is scored with its best exact continuation, rather than the cost of a learned or heuristic continuation. Accuracy is secondary and allows all actions within `1e-8` of zero regret.

For a two-state pair, the optimistic pair-aware lower bound is `min_a (R_F(a) + R_F'(a))/2`, minimized over **all** legal heads. This oracle knows which two latent forests form the pair. It is not necessarily the Bayes risk of a blind learner that only observes the geometry and endpoint-visible fields.

For the n=9 tests we additionally enumerate all `3! = 6` partner permutations, and calculate `min_a mean_pi R_pi(a)`. This is the exact conditional Bayes risk for the enriched blind observation, which includes the identity of non-singleton heads and all node coordinates/roles. Individual states from the paired generator have the same uniform marginal distribution over these six permutations. Randomized blind decisions cannot improve the risk because conditional expected regret is linear in their action distribution.

## Matched learned controls

All controls use three pre-norm Transformer encoder layers, width 128, eight attention heads, feedforward width 256, GELU activations, and no dropout or positional embeddings. A linear input embedding is followed by the Transformer, layer norm, a width-128 GELU MLP, and one scalar per node. The selected legal head minimizes its predicted regret.

Each node has eight input fields: its two coordinates relative to the source, head-role flag, consumed-tail-role flag, source flag, two partner-displacement coordinates, and a non-singleton-component flag. **All controls receive identical roles and non-singleton flags.** The blind control receives zero partner displacements. The aware control receives the displacement to the true paired endpoint (zero for singleton components). The separate random-pairing control receives a fresh uniform matching between the known non-singleton heads and tails, drawn independently of the label-generating matching on every minibatch. Both directions of its association are updated consistently.

Each model uses 10,000 AdamW steps, learning rate `3e-4`, weight decay `1e-5`, batch size 512, and global gradient clipping at 1. Training minimizes mean squared error over all legal actions' exact regret values. Training seeds are `19091941`, `19091942`, and `19091943`. The same seeds produce the same initialization and minibatch index sequences across controls; the random-pairing control uses a separate corruption generator so corruption sampling does not alter minibatch sampling. Each run retains the checkpoint with lowest validation mean regret, checked every 500 steps (plus step 1). All test evaluations happen after this selection. CUDA TF32 matmul is disabled.

The random-pairing control and stronger heuristic were added after early pilot validation results. They are labeled supplementary mechanism controls, not preregistered analyses. The initial pilot omitted the common non-singleton flag from blind inputs; this confound was corrected, its runs were discarded from the formal results, and all reported blind/aware models were retrained under the fair inputs.

## Heuristics and intervention

- Nearest head selects the legal head closest to the source.
- Greedy-completion lookahead considers every first head, greedily finishes the component ordering, and selects the head with the cheapest such completion. Its reported score still uses exact continuation regret.
- Assignment-relaxation lookahead fixes each candidate first edge and solves the remaining minimum-cost assignment with component self-loops forbidden and nontrivial subtours allowed. It picks the candidate with the smallest relaxation value. Every relaxation value is checked to be no greater than its corresponding exact completion value.
- Wrong-pairing evaluation uses each aware model with the two members' association maps exchanged within each test pair. Its weights and all other features stay fixed. It is a test-time intervention, not a retrained control.
- The zero-training information curve reveals the partners of heads `1,...,r`, partitions all six states by the revealed relationships, and computes the Bayes risk within each observation class, averaged by class probability. It must be pointwise nonincreasing in `r`; revealing two partners determines the third by exclusion.

## Statistical unit and checks

Average the two states before computing geometry-level statistics. For learned controls, report the mean of the three training seeds and the SD of their test means. Geometry-bootstrap intervals resample geometries after averaging seeds; hierarchical intervals additionally resample the three training seeds. These are empirical uncertainty summaries with only three independent training runs; do not treat 27,000 predictions as independent observations.

Independent validation includes full node-tour enumeration on n=7 instances, a separately implemented memoized Held–Karp recurrence on 170 saved forests, the project's existing Forest mask/context implementation, exact equality of paired blind features and predictions, and a numerical permutation-equivariance check. The true-association representation contains the full component graph, but finite trained networks are not asserted to solve it optimally.
