# Two-hour research protocol

Start: 2026-09-19 07:12 UTC (15:12 China standard time). Target delivery: 09:12 UTC.
Registered before new experiment outcomes. Existing evidence remains historical and
is not relabeled as a fresh test. The original project directories are preserved.

## Research question

Does a source-conditioned neural endpoint interface discard information needed for
optimal completion of directed path forests, how large is that loss under specified
state distributions, and does restoring component information improve learned
conditional decisions or complete tours?

This is an information/representation study. It does not assume that adaptive
source ordering is useful, that group-theoretic notation creates a new algorithm,
or that conditional prediction gains imply routing gains.

## Work packages

1. **Exact paired-state diagnostic (A40).** Geometries are the sampling units.
   Construct legal forests with identical static coordinates, selected-source
   context and legal-head masks but different head-to-tail associations. Compute
   exact completion costs for every legal action. Evaluate the Bayes-optimal
   action that is constrained to be identical within each observation class,
   including compromise actions. Compare identical-capacity blind and aware
   neural models with three training seeds, cheap heuristics, independent geometry
   splits, and predetermined size/distribution shifts. Detailed generator settings
   and update budgets are frozen in that package before test evaluation.
2. **Official host repair pilot (4090).** Use official ICAM and train an explicit
   component-aware residual endpoint, with same-capacity state-blind and route
   controls. Frozen-host adaptation is a limited pilot, not convergence evidence.
   Select checkpoints by validation only. Report conditional loss and full-tour
   quality separately, all training seeds, data budget, runtime and hardware use.
3. **Larger-instance counterfactual stress test (local CPU).** Generate independent
   uniform/clustered Euclidean TSP50/100/200 geometries (256 per cell). Form a
   nearest-neighbor tour improved by deterministic 2-opt, cut it into six paths,
   and swap suffixes of two non-source paths. This creates an observation-matched
   counterfactual forest; it is not an on-policy visitation sample. Enumerate all
   5! directed component cycles to measure irreducible conditional regret.
   No claim of a full-solver performance lower bound follows from these pairs.
   Also include an unchanged-pair control and independent full-tour enumeration
   of the original 7-node witness.
4. **Theory, prior work and paper.** State exact assumptions and proofs; compare
   DRHG, insertion, order-invariant learning and state abstraction. Use official
   ICLR 2027 style and report AI assistance accurately. Preserve negative results.

## Reporting rules

- Primary exact statistic: mean pairwise minimum expected excess completion cost,
  min_a mean_s[Q_s(a)-min_b Q_s(b)]. A fixed source and specified conditional state
  distribution are essential assumptions. This is not a bound for a joint source
  policy, nor for the unrestricted optimum TSP tour.
- Bootstrap geometries, keeping all paired states and model seeds within a unit;
  report seed variability separately. Do not treat paired states as independent.
- Training/validation/test instances are distinct; test metrics do not select
  checkpoints. Any post-hoc experiments are labeled explicitly.
- No OPT-gap language for uncertified LKH/2-opt references. No SOTA claim without
  direct strong comparisons and matching inference compute.
- A submission-formatted PDF is not evidence of acceptance readiness. The final
  decision must explicitly identify unmet scientific requirements.

## Deliverables

English manuscript source and compiled PDF; substantive title/abstract; exact
proofs; reproducible scripts; machine-readable metrics and dataset/config hashes;
Chinese assessment of current evidence and the shortest credible submission path.
No conference submission is performed as part of this research run.
