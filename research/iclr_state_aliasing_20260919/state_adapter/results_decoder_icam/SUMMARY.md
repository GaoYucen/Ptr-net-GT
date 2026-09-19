| Dataset | Arm | Mean cost (seed SD) | NLL | Accuracy |
|---|---|---:|---:|---:|
| uniform100 | random-aware | 9.66071 (0.02520) | 0.5391 | 0.8390 |
| uniform100 | random-blind | 9.04932 (0.09882) | 1.1828 | 0.8533 |
| uniform100 | route-aware | 7.83631 (0.00783) | 0.6177 | 0.9531 |
| uniform100 | route-blind | 7.86799 (0.03039) | 0.5550 | 0.9501 |
| uniform100 | native | 7.80561 | — | — |
| cluster100 | random-aware | 4.73104 (0.02697) | 1.5928 | 0.6250 |
| cluster100 | random-blind | 5.14309 (0.12037) | 2.9901 | 0.5373 |
| cluster100 | route-aware | 3.95882 (0.02663) | 2.3566 | 0.7405 |
| cluster100 | route-blind | 3.94607 (0.02439) | 2.2741 | 0.7543 |
| cluster100 | native | 4.07384 | — | — |
| uniform200 | random-aware | 13.84792 (0.19378) | 0.5882 | 0.8220 |
| uniform200 | random-blind | 12.93943 (0.11067) | 1.6624 | 0.8203 |
| uniform200 | route-aware | 10.82647 (0.02630) | 0.4876 | 0.9418 |
| uniform200 | route-blind | 10.89987 (0.07390) | 0.4453 | 0.9375 |
| uniform200 | native | 10.78664 | — | — |
| cluster200 | random-aware | 6.43505 (0.07610) | 1.4128 | 0.6267 |
| cluster200 | random-blind | 7.14676 (0.22086) | 2.2981 | 0.5660 |
| cluster200 | route-aware | 5.19430 (0.02670) | 1.5171 | 0.7448 |
| cluster200 | route-blind | 5.18422 (0.02754) | 1.4735 | 0.7465 |
| cluster200 | native | 5.28372 | — | — |

Confidence intervals below bootstrap paired test instances after averaging the three training seeds. They do not quantify uncertainty over all possible training seeds.

| Dataset | Comparison | Improvement % [95% CI] | Win fraction |
|---|---|---:|---:|
| uniform100 | random-aware vs random-blind | -7.465 [-8.488, -6.429] | 0.207 |
| uniform100 | random-aware vs route-aware | -23.306 [-24.199, -22.448] | 0.000 |
| uniform100 | route-aware vs route-blind | 0.379 [0.267, 0.505] | 0.664 |
| uniform100 | random-aware vs native | -23.766 [-24.628, -22.884] | 0.000 |
| uniform100 | route-aware vs native | -0.394 [-0.571, -0.239] | 0.293 |
| cluster100 | random-aware vs random-blind | 7.534 [6.167, 8.877] | 0.844 |
| cluster100 | random-aware vs route-aware | -19.902 [-21.594, -18.209] | 0.016 |
| cluster100 | route-aware vs route-blind | -0.490 [-1.064, 0.071] | 0.438 |
| cluster100 | random-aware vs native | -16.607 [-18.670, -14.462] | 0.094 |
| cluster100 | route-aware vs native | 2.513 [1.119, 3.985] | 0.609 |
| uniform200 | random-aware vs random-blind | -7.436 [-8.533, -6.327] | 0.102 |
| uniform200 | random-aware vs route-aware | -27.925 [-28.952, -26.933] | 0.000 |
| uniform200 | route-aware vs route-blind | 0.637 [0.450, 0.854] | 0.820 |
| uniform200 | random-aware vs native | -28.386 [-29.391, -27.335] | 0.000 |
| uniform200 | route-aware vs native | -0.370 [-0.524, -0.232] | 0.258 |
| cluster200 | random-aware vs random-blind | 9.184 [7.931, 10.464] | 0.914 |
| cluster200 | random-aware vs route-aware | -24.333 [-25.739, -22.965] | 0.000 |
| cluster200 | route-aware vs route-blind | -0.316 [-0.754, 0.124] | 0.461 |
| cluster200 | random-aware vs native | -22.097 [-23.555, -20.640] | 0.008 |
| cluster200 | route-aware vs native | 1.545 [0.447, 2.658] | 0.633 |
