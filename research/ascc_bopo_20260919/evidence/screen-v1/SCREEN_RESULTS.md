# ASCC × BOPO: one-seed screening results

200-step pilot, not a converged or cross-seed result. Positive delta favors ASCC.

| Source | Objective | Initial val | Best step | Best test | Final test | Train s |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| learned | bopo | 9.754332 | 200 | 8.939863 | 8.939863 | 191.7 |
| learned | reinforce | 9.754332 | 200 | 8.640332 | 8.640332 | 189.4 |
| route | bopo | 7.857608 | 0 | 7.807222 | 7.805747 | 151.0 |
| route | reinforce | 7.857608 | 100 | 7.811115 | 7.809347 | 145.1 |

| Objective | Route−ASCC | Relative improvement | Conditional instance CI |
| --- | ---: | ---: | --- |
| bopo | -1.132641 | -14.508% | [-1.203142, -1.062140] |
| reinforce | -0.829217 | -10.616% | [-0.881121, -0.777313] |

The interval conditions on these fitted models. It is not training-seed uncertainty.
Best step0 means the selected model is the initialization; do not call it learned gain.
All checkpoints, final results, configs and histories remain available.
