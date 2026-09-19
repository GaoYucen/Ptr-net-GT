# TSP50 主实验 v1

本目录保存已经完成的三种子短程主实验的小体积审计记录。

- 方法：`native_original` 与 `native_conditional_free`
- 训练种子：1234、4321、2468
- 测试：uniform TSP50，seed 20260902，10,000 个实例
- `summary.json`：四个宿主的 macro 与 pooled 配对统计
- `per_seed.csv`：每个训练种子的均值、差值、95% CI 和胜率

原始逐实例 `costs.pt` 与日志保存在被 Git 忽略的 `artifacts/` 中。
