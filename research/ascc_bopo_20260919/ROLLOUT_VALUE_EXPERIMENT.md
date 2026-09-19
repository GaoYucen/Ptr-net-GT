# ASCC 长期 source value：oracle 上限与学习验证

日期：2026-09-19。代码分支 `codex/ascc-bopo-joint-20260919`，最新实验提交 `ed445fc`。

## 问题

上一轮实验发现 source 选择存在局部差异，但 one-step completion regret、entropy、margin 和短边规则不能改善完整 tour。本轮直接回答更强的问题：如果 source selector 能准确预测一个动作在**当前真实 downstream policy**下的最终 tour cost，这块自由度是否能产生最终性能优势？

所有实验固定同一个 learned-REINFORCE checkpoint。route-prefix 产生路径森林；对每个 source 使用 fitted greedy endpoint，再用真实 route policy 完成 tour。Oracle 枚举 source，但不替换 endpoint 或 downstream policy。

## 1. Rollout-value oracle

TSP20/TSP50 各 64 个固定实例。首先在剩余 10 个分量时比较一次干预和每步重新干预，并同时运行 endpoint oracle。

| 方法 | TSP20 improvement | TSP50 improvement |
| --- | ---: | ---: |
| Source oracle，一次 | +2.97% | +0.14%（区间含 0） |
| Source oracle，反复 | +3.04% | +0.16%（区间含 0） |
| Endpoint oracle，一次 | +1.05% | 0.00% |
| Endpoint oracle，反复 | +2.71% | +0.20% |

TSP50 的一次干预进一步扫描剩余 4、6、8、10、15、20、30 个分量的位置。m=20/30 使用 64 个实例确认：

- m=20 source oracle：+0.675%，route−oracle cost 的条件 95% 区间 `[0.0046, 0.0738]`；
- m=30 source oracle：**+0.836%**，区间 `[0.0142, 0.0829]`；
- m=30 endpoint oracle：+0.270%，区间包含 0；
- m=30 有 53% 的实例被 source oracle 改善，endpoint oracle 为 12%。

![Rollout-value oracle](RESEARCH_ROOT/implementation_20260919/evidence/rollout-stage-confirm-v1/rollout_value_oracle.png)

结论：对当前 endpoint 和 downstream route，source 自由度确实能产生最终 tour 优势。价值主要出现在较早的路径森林阶段；只优化最后几个分量低估了 source ordering 的空间。

## 2. 独立划分上的 learned selector

在 TSP50、剩余 30 个分量时生成 candidate rollout-value 数据：训练 80、validation 24、test 48，坐标 seed 分别为 2026100101/102/103。每个状态包含全部 30 个 source 的真实 downstream cost。

第一组共享 MLP 使用 14 个局部森林特征；比较线性、32/64 hidden width，classification/regression 和三个初始化，共 18 个配置，只按 validation cost 选模型。第二组比较 ExtraTrees、RandomForest、Histogram Gradient Boosting 的回归/分类及 validation abstention。模型和阈值不直接使用 test 标签选择。

| Test 方法 | 相对 route | 改善实例比例 | Oracle source 命中率 |
| --- | ---: | ---: | ---: |
| Rollout-value oracle | **+0.845%** | 60.4% | 100% |
| 原 learned source | -3.11% | 16.7% | — |
| Min entropy | -3.01% | 20.8% | — |
| Shortest proposed edge | -5.06% | 12.5% | — |
| MLP value selector | -0.74% | 16.7% | 35.4% |
| Tabular value selector | -0.09% | 2.1% | 35.4% |

![Value selector](RESEARCH_ROOT/implementation_20260919/evidence/value-selector-v1/value_selector_result.png)

新 test 上的 oracle 为 +0.845%，与前一批 64 个实例的 +0.836% 几乎一致，说明约 0.8% 的 headroom 不是单批实例偶然产生的。当前学习器没有兑现该上限：MLP 会做过多错误偏离；树模型通过 validation gate 大多退回 route，因而接近 0 而非取得正收益。

虽然单个模型和阈值未直接用 test 标签选择，但在看到 MLP test 失败后才追加树模型，因此这个 test 已属于开发过程，不能作为未来论文的最终 held-out test。

## 3. 回答与后续门槛

本轮对“学好是否能更好”的回答是：

> **是，存在可重复的最终性能上限；但当前 ASCC 还没有学好，尚无可部署的性能优势。**

这个上限不是“所有 source 顺序都更好”，而是少数状态下选择正确 source 可避免较大的后续损失。候选之间的区别需要组件间全局交互和 downstream value；局部 entropy、margin、边长、组件大小以及独立 candidate MLP 都不够。

下一版方法应使用 set/graph value model，同时编码全部路径分量及候选连接，预测相对 route 的 rollout advantage；训练需要更多 on-policy forest states，并用 DAgger 式迭代重新标注 selector 实际访问的状态。首先应设定明确门槛：

1. 在新的 validation 上学到正收益并选择 checkpoint/abstention；
2. 冻结所有设计后，在从未参与开发的新 test 上超过 route；
3. 至少 5 个 selector 训练 seed；
4. 报告模型推理开销，并与相同时间的 multi-start/beam control 比较；
5. 若只能取得 oracle 的很小比例或不超过更大 decoding budget，停止将 source ordering 作为主贡献。

## 4. 复现与限制

- Oracle 使用真实未来 cost，是诊断上限，不是可部署算法；其额外 rollout 计算不能与 route 推理时间直接比较。
- 所有结果仍基于一个 endpoint checkpoint；没有训练 seed 方差。
- TSP50 实例为 uniform 分布；尚未验证 clustered、TSPLIB 和更大规模。
- 原始逐实例数据、checkpoint hash、训练/验证/测试 seeds、selector 权重和图表保存在服务器 `results/ascc-bopo-joint-20260919/rollout-oracle-v1`、`rollout-stage-*` 和 `value-selector-v1`。
