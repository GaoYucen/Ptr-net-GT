# AM TSP50 正式消融实验备份

- 实验编号：`paper_ablation_am_tsp50_v1`
- 宿主模型：AM
- 问题：Uniform Euclidean TSP50
- 训练种子：1234、4321、2468
- 每组训练：10,000 steps
- 独立测试：固定 seed 20260904 的 10,000 个相同实例
- 完整方法 `native_conditional_free` 复用正式主实验结果
- 新训练模式：Random Tail、No Head Summary、No Path State

## 三种子平均 cost

| 模式 | Mean cost |
|---|---:|
| Full GroupOpt | 6.102593 |
| Random Tail | 7.240256 |
| No Head Summary | 6.080290 |
| No Path State | 6.291540 |

## 核心结论

- Full 相对 Random Tail 改善 15.713%，三种子方向一致，配对 95% CI 为 [1.131279, 1.144047]。
- Full 相对 No Path State 改善 3.003%，三种子方向一致，配对 95% CI 为 [0.185448, 0.192444]。
- No Head Summary 反而比 Full 平均低 0.3668%；三种子中两个种子支持移除该信息，说明当前 head summary 不是有效贡献项，后续应将精简版本作为候选默认方法重新验证，而不能宣称该模块有效。

备份包含配置、训练指标、日志、逐实例 costs、评估 summary、CSV/JSON 汇总和实现验证；大型 checkpoint 已排除。
