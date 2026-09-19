| Dataset | Arm | Mean cost (seed SD) | NLL | Accuracy |
|---|---|---:|---:|---:|
| uniform100 | random-aware | 10.49267 (0.05087) | 0.7859 | 0.7279 |
| uniform100 | random-blind | 10.85099 (0.08921) | 0.8489 | 0.7174 |
| uniform100 | route-aware | 8.13863 (0.01568) | 0.2388 | 0.9297 |
| uniform100 | route-blind | 8.17844 (0.02486) | 0.2330 | 0.9301 |
| uniform100 | native | 8.07669 | — | — |
| cluster100 | random-aware | 4.69799 (0.02910) | 1.3289 | 0.6415 |
| cluster100 | random-blind | 4.92494 (0.03276) | 1.4119 | 0.6458 |
| cluster100 | route-aware | 3.75769 (0.01634) | 0.7270 | 0.7760 |
| cluster100 | route-blind | 3.76163 (0.03284) | 0.7252 | 0.7734 |
| cluster100 | native | 4.81241 | — | — |
| uniform200 | random-aware | 15.13291 (0.06153) | 0.8973 | 0.7179 |
| uniform200 | random-blind | 15.54622 (0.11973) | 0.9257 | 0.7179 |
| uniform200 | route-aware | 11.50825 (0.03401) | 0.3256 | 0.9002 |
| uniform200 | route-blind | 11.51524 (0.01257) | 0.3232 | 0.8967 |
| uniform200 | native | 11.55860 | — | — |
| cluster200 | random-aware | 6.61991 (0.06568) | 1.4207 | 0.6207 |
| cluster200 | random-blind | 6.94274 (0.01068) | 1.4600 | 0.6128 |
| cluster200 | route-aware | 4.95814 (0.01555) | 0.8187 | 0.7569 |
| cluster200 | route-blind | 4.97989 (0.04752) | 0.8383 | 0.7509 |
| cluster200 | native | 7.26866 | — | — |

Confidence intervals below bootstrap paired test instances after averaging the three training seeds. They do not quantify uncertainty over all possible training seeds.

| Dataset | Comparison | Improvement % [95% CI] | Win fraction |
|---|---|---:|---:|
| uniform100 | random-aware vs random-blind | 2.899 [2.238, 3.538] | 0.727 |
| uniform100 | random-aware vs route-aware | -29.061 [-29.907, -28.197] | 0.000 |
| uniform100 | route-aware vs route-blind | 0.404 [0.181, 0.640] | 0.566 |
| uniform100 | random-aware vs native | -29.949 [-30.779, -29.168] | 0.000 |
| uniform100 | route-aware vs native | -0.773 [-1.085, -0.478] | 0.395 |
| cluster100 | random-aware vs random-blind | 3.808 [2.582, 5.055] | 0.711 |
| cluster100 | random-aware vs route-aware | -25.243 [-27.064, -23.440] | 0.000 |
| cluster100 | route-aware vs route-blind | 0.014 [-0.515, 0.544] | 0.523 |
| cluster100 | random-aware vs native | -6.275 [-10.339, -2.087] | 0.273 |
| cluster100 | route-aware vs native | 14.907 [11.872, 18.170] | 0.930 |
| uniform200 | random-aware vs random-blind | 2.424 [1.730, 3.096] | 0.719 |
| uniform200 | random-aware vs route-aware | -31.574 [-32.423, -30.702] | 0.000 |
| uniform200 | route-aware vs route-blind | 0.019 [-0.310, 0.347] | 0.516 |
| uniform200 | random-aware vs native | -30.940 [-31.812, -30.108] | 0.000 |
| uniform200 | route-aware vs native | 0.420 [-0.023, 0.807] | 0.719 |
| cluster200 | random-aware vs random-blind | 3.862 [2.745, 4.965] | 0.805 |
| cluster200 | random-aware vs route-aware | -33.622 [-35.616, -31.589] | 0.000 |
| cluster200 | route-aware vs route-blind | 0.355 [-0.117, 0.827] | 0.625 |
| cluster200 | random-aware vs native | -4.138 [-8.563, 0.515] | 0.289 |
| cluster200 | route-aware vs native | 21.908 [18.765, 25.160] | 1.000 |
