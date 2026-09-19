# ASCC source ordering：精确后悔值与端到端验证

日期：2026-09-19。代码分支：`codex/ascc-bopo-joint-20260919`。本实验检验的不是“ASCC 是否已经优于 baseline”，而是两个更基础的问题：同一路径森林中，source 的选择是否真的影响后续解质量；当前 learned source 是否能利用这种差异。

## 实验一：精确 source regret

使用现有 learned-REINFORCE 和 learned-BOPO checkpoint。在 TSP20/TSP50 上分别产生 learned-prefix 和 route-prefix 路径森林，在剩余 4、6、8、10 个分量时截取状态。每个配置 96 个固定 uniform 实例，共 1,536 个森林状态。

对每个 unresolved source：

1. 由同一个 fitted endpoint policy 贪心选择合法 endpoint；
2. 固定这条边；
3. 用 Held--Karp 动态规划精确连接剩余定向路径分量；
4. 将完成成本与该状态下最好的 source 比较。

这里的 oracle 只选择 source，不替换 endpoint policy。动态规划在随机小实例上与全排列穷举逐项一致；提取出的 learned/route 下一动作也与原 rollout 逐项一致。

![精确 source regret](RESEARCH_ROOT/implementation_20260919/evidence/oracle-regret-v1/source_regret_main.png)

主图使用 learned-REINFORCE checkpoint 的 learned-prefix 状态，并按独立实例聚合四个 construction stage 后计算区间：

| Source policy | 平均 regret（占 oracle completion cost） | Oracle source 命中率 |
| --- | ---: | ---: |
| Learned | 4.82% | 69.9% |
| Random expectation | 4.61% | 66.6% |
| Continue route | 4.33% | 71.1% |
| Min endpoint entropy | 1.92% | 82.0% |
| Max endpoint margin | 2.04% | 82.9% |
| Shortest proposed edge | **1.35%** | 75.5% |
| Shortest component | 4.39% | 64.3% |
| Fixed index | 5.17% | 64.5% |

同一状态最差与最好 source 的平均差距为 oracle completion cost 的 18.72%；按分量数和规模拆分后约 12.7%–22.9%。因此 source ordering 不是完全无效的自由度。

Learned 相对 random 的平均 regret 差为 +0.21 percentage point，按 192 个独立 instance-size 单元计算的条件 95% 区间为 [-0.67, 1.09]，不能认为优于 random。Learned 分别比 min-entropy、max-margin、shortest-edge 多 2.89、2.77、3.46 percentage points regret；对应条件区间均不含 0。

该结论在 objective、prefix 来源、规模和剩余分量数之间并不完全稳定。以下热图中，learned 相对 random 在 32 个 cell 中只在 15 个 cell 更好，取值范围为 -2.00 到 +3.36 percentage points。

![learned 对 random 的稳健性](RESEARCH_ROOT/implementation_20260919/evidence/oracle-regret-v1/learned_vs_random_heatmap.png)

## 实验二：端到端 greedy construction

把实验一表现较好的 min-entropy、max-margin、shortest-proposed-edge source rule 接入完整 rollout。所有 policy 使用同一个 learned-REINFORCE checkpoint、相同 128 个固定实例、8 个 multi-anchor rollouts、greedy endpoint，无 augmentation 和局部搜索。Random 对三个 action seed 分别运行，图中使用逐实例平均。

![端到端 source ordering](RESEARCH_ROOT/implementation_20260919/evidence/heuristic-rollout-v1/heuristic_rollout_main.png)

| Policy | TSP20 cost | 相对 route | TSP50 cost | 相对 route |
| --- | ---: | ---: | ---: | ---: |
| Route | **3.8855** | 0.00% | **5.7185** | 0.00% |
| Learned | 4.1362 | -6.45% | 6.0509 | -5.81% |
| Random（3 seeds 平均） | 4.1139 | -5.88% | 6.1563 | -7.66% |
| Min entropy | 4.3631 | -12.29% | 6.2491 | -9.28% |
| Max margin | 4.3839 | -12.83% | 6.1208 | -7.03% |
| Shortest proposed edge | 4.1861 | -7.74% | 6.2974 | -10.12% |
| Shortest component | 5.1412 | -32.32% | 7.4309 | -29.94% |
| Fixed index | 5.1379 | -32.23% | 7.1449 | -24.94% |

所有 route-vs-policy 的逐实例区间均保存在 `heuristic-rollout-v1/summary.json`。上表中的负数表示比 route 更差。

局部精确 regret 很低的三个 heuristic 在端到端 rollout 中仍比 route 差 7%–13%。它们反复改变状态分布，后续 endpoint policy 进入训练不足的森林状态，局部一步优势没有累积成 tour 优势。这个结果否定了“直接用局部 confidence 或 one-step oracle 标签替代 source policy即可”的简单方案。

## 研究判断

本轮支持：

- source 选择会显著改变 fitted endpoint policy 作出决定后的最佳可完成成本；研究这个自由度并非没有信号；
- endpoint uncertainty 与短边 heuristic 能识别一部分局部好 source；
- 当前 learned source 没有稳定超过 random/route，并明显落后于局部 heuristic；
- 局部 source regret 不能直接预测重复使用该规则后的完整 tour 质量。

本轮不支持：

- ASCC 已经优于 route construction；
- learned source 已经学到有效的 adaptive ordering；
- 训练一个 one-step regret classifier 就足以获得端到端收益；
- BOPO 是当前主要瓶颈或必要解法。

下一步若继续，应研究非短视的 candidate value：预测一次 `(source, endpoint)` 决策在当前 downstream policy 下的 rollout-to-go，而不是预测由精确 solver 接管后的成本。应先在 late-stage（剩余 4–10 个分量）做单次 intervention，比较 route、learned、one-step oracle 和 rollout-value oracle；只有 rollout-value oracle 在真实后续 policy 下仍有明显收益，才值得训练 value-guided source policy。

## 范围与复现

- 训练 checkpoint 只有一个 seed；区间刻画固定模型的实例差异，不包含训练 seed 方差。
- checkpoint 在 TSP100 训练，本实验的 TSP20/TSP50 是 size transfer 诊断。
- exact-regret 状态来自当前 policy trajectory，并非所有可能森林的均匀样本。
- confidence heuristic 需要为所有 source 计算 endpoint 分布，TSP50 的本次实现约为 route 推理时间的 1.6–1.9 倍；质量已经更差，因此未做等时间补偿。
- 原始记录、逐实例 cost、坐标 seed、checkpoint hash、图表和 PDF 均保存在 `results/ascc-bopo-joint-20260919/oracle-regret-v1` 与 `heuristic-rollout-v1`。
