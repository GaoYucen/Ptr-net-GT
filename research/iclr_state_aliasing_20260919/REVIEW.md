# Independent scientific and implementation review

Reviewer: parallel theory/literature agent. Review date: 2026-09-19. This review is independent of the implementation agents but uses the same project workspace; it is not an external peer review.

## Overall assessment

The present work supports a concrete information audit of a particular neural routing interface. It does not yet establish a new state-of-the-art solver or a benefit from learning construction order. The strongest claim is a certified counterexample and a systematic conditional-decision diagnostic, accompanied by an intervention that supplies the missing endpoint correspondence.

A credible submission should make this limited contribution clear, disclose earlier negative results, and distinguish exact information loss from approximation and optimization error. The draft can be scientifically correct without being competitive for ICLR acceptance. The principal unresolved concern is significance and novelty, not the elementary correctness of the conditional risk identity.

## What was independently checked

- Read `strong_hosts_20260919/NEXT_STEPS.md`, the original endpoint witness, and the native-host adapter interface; reviewed the local literature assessment.
- Verified the selected-source counterexample by independently enumerating all 720 anchored tours and enclosing every Euclidean distance with rational arithmetic. The uniform two-state conditional regret lies in [0.058876330550917, 0.058876330550931]. The best common target is head 6. Artifact: `verify_witness_interval.py` and `witness_interval_certificate.json`.
- Read the complete `counterfactual_stress.py` and protocol, and inspected all six summary cells. Component-cycle enumeration, conditional minimization over every legal target, suffix-swap validity, and geometry bootstrap are consistent with the stated setup. No correctness blocker was found by inspection.
- Read `exact_diagnostic/run.py`, `extend.py`, and the initial `shuffled_control.py`. Feature construction keeps the matched blind inputs identical across paired states; the model has no positional encoding. Exact labels are generated from the stored coordinates, and full node-level enumeration checks the component reduction on independent small examples. The all-pairing extension appropriately distinguishes a true observation-class Bayes floor from a more optimistic two-state floor.
- Read `state_adapter/component_adapter.py`, `run_pilot.py`, and its protocol. Pointer doubling reconstructs teacher components; updates preserve path endpoints; the endpoint mask and global pooling are consistent with legal forest merges. The frozen host has no optimized parameters. Validation selects checkpoints without final-test inspection. The built-in smoke checks cover route equivalence at zero residual, forest features, gradients, and permutation equivariance. No correctness blocker was found by inspection; this is not a proof that every execution is bug-free.
- Checked primary papers including DRHG, BQ-NCO, L2C-Insert, MACSIM, and the updated ICAM record; retrieved actual ICLR 2027 style files and submission requirements.

## Claims that need precise wording

1. **Selected-source mask, not all feasibility masks.** With at least two components, the entire open-tail by head legality matrix encodes the component pairing: the unique forbidden head in each tail's row is its own component head. The audited interface only exposes the selected row. A model consuming the complete matrix need not suffer this information loss.
2. **Conditional lower bound, not full-policy lower bound.** The source selector can distinguish the witness states and alter which states are encountered; its selected source also communicates information. Do not infer a global impossibility theorem for arbitrary joint source-endpoint policies.
3. **Constructed distribution, not observed frequency in deployment.** The larger stress test holds a route fragment fixed and swaps suffixes elsewhere. These are legitimate forests, but one partner can have unnatural fixed edges. The measured approximately 21%–30% positive-regret rate applies to this registered stress distribution. It does not show that that fraction of actual solver decisions is irreducibly wrong.
4. **Cost regret, not accuracy alone.** Different optimal labels do not imply a large deployment loss. Compute the minimum average regret over *all* legal actions; a compromise target can dominate both individual optima in the aliased problem.
5. **Information versus effective capacity.** Both diagnostic networks and residual arms have equal parameter counts, but the aware input makes additional parameters informative. This is an intended representation intervention. Separately trained shuffled correspondence controls help distinguish meaningful information from extra active input channels; equal parameter count alone does not settle every capacity issue.
6. **The residual is a bundle of state features.** The pilot aware arm provides source/target mean embeddings, component sizes, and target endpoint correspondence. Therefore its improvement, if present, belongs to the full component-state intervention. A route-only improvement cannot be attributed to nontrivial target pairing, because all unvisited route targets are singleton components.
7. **Cached decoding time is not end-to-end latency.** The current pilot evaluation consumes cached embeddings and includes tour-validation assertions. A deployment timing claim needs separate encoding and decoding measurements, synchronized GPU execution, and inference runs without correctness checks in the timed block.
8. **Novelty must be narrow.** DRHG already represents a path by both endpoints; BQ-NCO already studies information-preserving state quotients; state-abstraction theory already explains loss from merging states with incompatible optimal actions. The potential contribution is a useful quantitative audit and empirically predictive failure mechanism in a relevant interface, not those ingredients independently.

## Requested remedies communicated to implementation agents

The exact-diagnostic agent was asked to train three independently seeded shuffled-correspondence controls, with the same minibatch-index stream, steps, optimizer, and validation schedule as the real-information arms. This is explicitly an additional, post-pilot mechanism control. The agent was also encouraged to retain a cheap component-aware assignment-relaxation comparator even if it outperforms the learned model.

The host-adapter agent was asked to distinguish target-pair corruption from the other dynamic features, preserve the original teacher/validation protocol, and measure deployment latency separately from cached, assertion-heavy evaluation. These requests do not justify changing the main experiment after viewing its final test.

## ICLR competitiveness and required next evidence

The one-example theorem plus small supervised task is not a sufficient main-conference contribution by itself: it risks being read as deliberately withholding information already supplied by known fragment solvers. Two-host repair results would strengthen practical relevance, especially if the gain persists against same-state route baselines and inexpensive component-aware policies. A failure to beat the enhanced route remains a valid negative finding and should redirect the claim toward interface diagnosis.

Before claiming a mature submission, the authors need to assess whether the audit exposes a broad issue, whether it predicts intervention benefits on states that actual solvers visit, and whether the learned correction has value after its training and inference cost. Direct DRHG or an explicitly DRHG-style state baseline is especially important if fragment representation or repair performance is central. A state-information contribution does not need to be sold as a new group-theoretic method.

The best use of a two-hour deliverable is a reproducible research package and an honest, submission-formatted manuscript for author review. It cannot replace the authors' literature judgment, scientific ownership, or careful decision about whether the result merits submission. No acceptance probability is claimed.

## Submission checks

The genuine ICLR 2027 abstract deadline is 2026-09-19 19:59 China standard time; full paper is due 2026-09-26 19:59 China standard time. The author list freezes at the abstract deadline. The initial main text is limited to nine pages, and the manuscript and supplemental package must preserve anonymity. A separate AI-use statement is mandatory. See the [official author guidelines](https://iclr.cc/Conferences/2027/AuthorGuidelines) and [author AI policy](https://iclr.cc/Conferences/2027/AIPolicyForAuthors).

No submission action has been taken in this review. The research package includes absolute local paths and historical provenance that must be checked for anonymity before public submission. Any statement that human authors have checked AI-assisted work must describe checks that they actually completed.

## Manuscript claim audit, first completed draft

Read `paper/main.tex` and `paper/appendix.tex` after their initial creation. The proofs, witness interval, distinction between pair-identity information and the full observation class, and the scope restriction away from joint policies are scientifically consistent. No theorem or reported numerical value required correction in that draft.

The authoring agent received specific requests to add the full-matrix versus selected-row distinction, restrict factorial counts to m>=2, remove ambiguous sign wording for learned-versus-route cost differences, cite a classical multi-fragment primary source, and state the stress-test tolerance for positive regret. These are clarity and scope fixes, not evidence of an invalid theorem.

When incorporating the learned diagnostic, explicitly describe the enriched blind control: it receives static roles and non-singleton membership information in addition to the usual selected-source context. Its full-pairing Bayes floor is for that well-defined controlled observation, not a measured Bayes risk of the native host's on-policy states. The residual-host intervention additionally supplies component means and sizes, so its results concern the combined component-state intervention.

The implementation agent has agreed to keep target-tail corruption as a post-hoc sensitivity measurement on frozen checkpoints and to collect separate end-to-end and cached-encoding timing runs outside validity assertions. A secondary decoder-plus-residual adaptation study is being specified before primary final-test inspection; it must remain separately labeled from the main frozen-host study, with all arms retained.
