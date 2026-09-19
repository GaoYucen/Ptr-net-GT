| Dataset | Arm | Mean cost (seed SD) | NLL | Accuracy |
|---|---|---:|---:|---:|
| uniform100 | random-aware | 11.26824 (0.03085) | 1.6563 | 0.6801 |
| uniform100 | random-blind | 11.67118 (0.09453) | 1.7821 | 0.6619 |
| uniform100 | route-aware | 8.14784 (0.01533) | 0.2390 | 0.9301 |
| uniform100 | route-blind | 8.15549 (0.02507) | 0.2408 | 0.9310 |
| uniform100 | native | 8.08901 | — | — |
| cluster100 | random-aware | 5.12508 (0.05586) | 1.5386 | 0.5877 |
| cluster100 | random-blind | 5.33105 (0.15731) | 1.5873 | 0.5642 |
| cluster100 | route-aware | 3.79080 (0.02737) | 0.7317 | 0.7951 |
| cluster100 | route-blind | 3.78205 (0.01377) | 0.7808 | 0.7812 |
| cluster100 | native | 4.72676 | — | — |
| uniform200 | random-aware | 16.87742 (0.07273) | 1.6384 | 0.6085 |
| uniform200 | random-blind | 17.32302 (0.35237) | 1.7306 | 0.5981 |
| uniform200 | route-aware | 11.56295 (0.02016) | 0.3517 | 0.8906 |
| uniform200 | route-blind | 11.59990 (0.02039) | 0.3656 | 0.8906 |
| uniform200 | native | 11.62107 | — | — |
| cluster200 | random-aware | 7.78122 (0.12051) | 1.6180 | 0.5113 |
| cluster200 | random-blind | 8.15251 (0.28847) | 1.7027 | 0.4896 |
| cluster200 | route-aware | 4.98957 (0.00731) | 0.8193 | 0.7891 |
| cluster200 | route-blind | 5.01079 (0.01024) | 0.8392 | 0.7891 |
| cluster200 | native | 7.14688 | — | — |

Confidence intervals below bootstrap paired test instances after averaging the three training seeds. They do not quantify uncertainty over all possible training seeds.

| Dataset | Comparison | Improvement % [95% CI] | Win fraction |
|---|---|---:|---:|
| uniform100 | random-aware vs random-blind | 3.048 [2.444, 3.669] | 0.727 |
| uniform100 | random-aware vs route-aware | -38.431 [-39.372, -37.540] | 0.000 |
| uniform100 | route-aware vs route-blind | 0.044 [-0.169, 0.257] | 0.523 |
| uniform100 | random-aware vs native | -39.362 [-40.278, -38.509] | 0.000 |
| uniform100 | route-aware vs native | -0.734 [-0.995, -0.489] | 0.348 |
| cluster100 | random-aware vs random-blind | 2.920 [1.613, 4.165] | 0.742 |
| cluster100 | random-aware vs route-aware | -35.630 [-37.934, -33.330] | 0.000 |
| cluster100 | route-aware vs route-blind | -0.365 [-1.063, 0.312] | 0.516 |
| cluster100 | random-aware vs native | -15.751 [-20.352, -11.005] | 0.234 |
| cluster100 | route-aware vs native | 14.671 [11.713, 17.832] | 0.883 |
| uniform200 | random-aware vs random-blind | 2.308 [1.666, 2.953] | 0.773 |
| uniform200 | random-aware vs route-aware | -46.053 [-47.064, -45.070] | 0.000 |
| uniform200 | route-aware vs route-blind | 0.256 [-0.057, 0.582] | 0.562 |
| uniform200 | random-aware vs native | -45.267 [-46.207, -44.306] | 0.000 |
| uniform200 | route-aware vs native | 0.488 [0.179, 0.783] | 0.672 |
| cluster200 | random-aware vs random-blind | 3.590 [2.531, 4.698] | 0.781 |
| cluster200 | random-aware vs route-aware | -56.259 [-59.196, -53.446] | 0.000 |
| cluster200 | route-aware vs route-blind | 0.309 [-0.118, 0.748] | 0.539 |
| cluster200 | random-aware vs native | -26.152 [-31.823, -20.421] | 0.195 |
| cluster200 | route-aware vs native | 19.388 [16.432, 22.614] | 0.961 |
