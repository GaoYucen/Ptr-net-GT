# Paired-forest conditional endpoint diagnostic

This is an exact, reproducible mechanism experiment. It is not a full-tour TSP benchmark.

All learned controls use 415,489 parameters, 10,000 AdamW steps, batch 512, 40,000 training geometries, and the same three training seeds. Validation contains 3,000 independent geometries. Each test distribution contains 4,500 geometries from three new data seeds (9,000 forests).

Mean exact completion regret (unit-square Euclidean length; lower is better). Learned values are means across three seeds.

| Test | Pair oracle lower bound | Blind | Random pairing | True pairing | Assignment | Nearest |
|---|---:|---:|---:|---:|---:|---:|
| Uniform n=9 | 0.015999 | 0.034215 | 0.033743 | 0.006448 | 0.036292 | 0.077438 |
| Clustered n=9 | 0.006414 | 0.017263 | 0.017779 | 0.006486 | 0.016674 | 0.033536 |
| Uniform n=7 | 0.024366 | 0.033377 | 0.035022 | 0.006933 | 0.041536 | 0.081481 |
| Uniform n=11 | 0.010508 | 0.033919 | 0.033890 | 0.012270 | 0.030450 | 0.070807 |
| Uniform n=13 | 0.007914 | 0.033320 | 0.032951 | 0.018868 | 0.026256 | 0.067036 |

Enumerating all six possible pairings for the same 4,500 uniform n=9 geometries gives an exact blind conditional Bayes risk of **0.028314** (geometry-bootstrap 95% CI 0.026926–0.029574). The true-pairing model obtains **0.006330** on this full family. This Bayes risk includes the same component-size side information given to both learned models.

On paired IID tests, true pairing reduces mean regret by **81.2%** versus blind. The paired reduction is 0.027767 (geometry-and-seed bootstrap 95% CI 0.025720–0.029810). Swapping the association at test time increases regret by 0.056257 (95% CI 0.052556–0.059909).

Independent Held–Karp DP checked 170 saved forests against enumeration: maximum absolute discrepancy 8.9e-16. Geometry hashes confirm disjoint train/validation/test sets. The existing Forest implementation verifies identical source context and legality masks in paired states. Blind features and predictions are exactly equal within each pair. Permutation equivariance maximum numerical error is 3e-07.

All confidence intervals and per-seed values are in summary.json. Raw coordinates, pairings, exact Q values, predictions, checkpoints, and training logs are in results/.

Limits and protocol chronology:
- All source choices are fixed to singleton node 0; the joint source-target policy is outside this bound.
- All learned controls have identical architecture/parameter count and common geometry, masks, node roles, component-size information. Only true versus absent versus independent random head-tail association changes.
- The pair-aware oracle lower bound knows which two latent pairings form the pair. The all-six Bayes risk conditions only on the enriched blind observation and averages every permitted pairing.
- Random-pairing training, all-pairing evaluation, and assignment-relaxation heuristic are supplementary controls designed after pilot validation was observed.
- Wrong-pairing evaluation swaps the two pairings at test time, with the model weights fixed. It is an intervention/distribution change, not a separately trained model.
- No benchmark SOTA, neural runtime advantage, or submission acceptance claim is supported.

Reproduce:

```bash
python run.py --output results --device cuda --steps 10000 --validate-every 500
python extend.py --output results --device cuda
python shuffled_control.py --output results --device cuda
python assignment_baseline.py --output results
python information_curve.py --output results
python verify.py --output results
python make_report.py --output results
```
