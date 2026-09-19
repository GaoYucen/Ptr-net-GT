| Dataset | Arm | Mean cost (seed SD) | NLL | Accuracy |
|---|---|---:|---:|---:|
| uniform100 | random-aware | 9.09543 (0.07100) | 2.7167 | 0.8711 |
| uniform100 | random-blind | 9.29023 (0.04843) | 2.8976 | 0.8720 |
| uniform100 | route-aware | 7.85124 (0.00453) | 0.5670 | 0.9457 |
| uniform100 | route-blind | 7.85987 (0.00508) | 0.6332 | 0.9488 |
| uniform100 | native | 7.82815 | — | — |
| cluster100 | random-aware | 5.53620 (0.09232) | 5.0851 | 0.5521 |
| cluster100 | random-blind | 5.57916 (0.02614) | 5.2657 | 0.5547 |
| cluster100 | route-aware | 3.91657 (0.01150) | 2.0974 | 0.7396 |
| cluster100 | route-blind | 3.93844 (0.01605) | 2.1833 | 0.7248 |
| cluster100 | native | 4.01236 | — | — |
| uniform200 | random-aware | 13.20913 (0.21269) | 3.0137 | 0.8047 |
| uniform200 | random-blind | 13.39669 (0.06190) | 3.0465 | 0.8108 |
| uniform200 | route-aware | 10.86064 (0.00523) | 0.3751 | 0.9601 |
| uniform200 | route-blind | 10.87084 (0.01594) | 0.3967 | 0.9592 |
| uniform200 | native | 10.83731 | — | — |
| cluster200 | random-aware | 8.27192 (0.21466) | 3.9250 | 0.5460 |
| cluster200 | random-blind | 8.17531 (0.07689) | 4.0354 | 0.5608 |
| cluster200 | route-aware | 5.14975 (0.01864) | 1.5914 | 0.7248 |
| cluster200 | route-blind | 5.16754 (0.01233) | 1.6663 | 0.7101 |
| cluster200 | native | 5.25627 | — | — |

Confidence intervals below bootstrap paired test instances after averaging the three training seeds. They do not quantify uncertainty over all possible training seeds.

| Dataset | Comparison | Improvement % [95% CI] | Win fraction |
|---|---|---:|---:|
| uniform100 | random-aware vs random-blind | 1.400 [0.552, 2.224] | 0.625 |
| uniform100 | random-aware vs route-aware | -15.833 [-17.056, -14.615] | 0.012 |
| uniform100 | route-aware vs route-blind | 0.100 [-0.009, 0.216] | 0.387 |
| uniform100 | random-aware vs native | -16.168 [-17.400, -15.003] | 0.000 |
| uniform100 | route-aware vs native | -0.302 [-0.462, -0.129] | 0.340 |
| cluster100 | random-aware vs random-blind | -0.169 [-1.406, 1.000] | 0.570 |
| cluster100 | random-aware vs route-aware | -42.871 [-46.958, -39.069] | 0.016 |
| cluster100 | route-aware vs route-blind | 0.388 [-0.223, 0.981] | 0.562 |
| cluster100 | random-aware vs native | -39.093 [-42.782, -35.562] | 0.008 |
| cluster100 | route-aware vs native | 2.162 [0.999, 3.365] | 0.609 |
| uniform200 | random-aware vs random-blind | 0.783 [-0.250, 1.781] | 0.594 |
| uniform200 | random-aware vs route-aware | -21.650 [-22.974, -20.270] | 0.000 |
| uniform200 | route-aware vs route-blind | 0.083 [-0.055, 0.222] | 0.469 |
| uniform200 | random-aware vs native | -21.897 [-23.322, -20.542] | 0.000 |
| uniform200 | route-aware vs native | -0.216 [-0.372, -0.092] | 0.367 |
| cluster200 | random-aware vs random-blind | -2.377 [-3.687, -1.072] | 0.391 |
| cluster200 | random-aware vs route-aware | -62.102 [-67.089, -57.341] | 0.000 |
| cluster200 | route-aware vs route-blind | 0.205 [-0.288, 0.706] | 0.539 |
| cluster200 | random-aware vs native | -58.068 [-62.468, -53.937] | 0.000 |
| cluster200 | route-aware vs native | 1.861 [0.805, 2.944] | 0.570 |
