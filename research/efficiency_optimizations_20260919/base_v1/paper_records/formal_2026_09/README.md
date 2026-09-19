# 论文正式实验结果归档（2026 年 9 月）

本目录保存论文当前正式实验的结果数据，按实验阶段分为四组：

| 目录 | 内容 | 原始本地备份 |
|---|---|---|
| `main_tsp50/` | AM-style、PtrNet、GPN、POMO 的 TSP50 主对比，三训练种子 | `artifacts/server-backup-2026-09-04/paper-final-tsp50-v1/` |
| `scale_tsp/` | AM、POMO 在 TSP20/50/100 的各规模重新训练对比；TSP50 复用主实验 | `artifacts/server-backup-2026-09-04/paper-scale-effect-tsp-v1/` |
| `capacity_am/` | AM 的参数量匹配单链对照；Original 和 Full 复用主实验 | `artifacts/server-backup-2026-09-05/paper-capacity-control-am-tsp50-v1/` |
| `ablation_am/` | AM 的 Random Tail、No Head Summary、No Path State 消融；Full 复用主实验 | `artifacts/server-backup-2026-09-05/paper-ablation-am-tsp50-v1/` |

每组保留原有相对目录结构：`summary/` 是汇总 JSON、CSV 和验证结果；`eval/` 中的 `summary.json` 与 `costs.pt` 分别是每组评估摘要和逐实例代价；`train/` 中的 `config.json` 与 `metrics.jsonl` 记录训练配置与过程指标。另保留冻结协议 `protocol_snapshot.json` 和已有的来源说明。各方法使用固定独立测试集，组内同一实例进行配对评估。

`costs.pt` 虽使用 PyTorch 的 `.pt` 扩展名，但内容是评估得到的逐实例代价数据，**不是模型 checkpoint**。本目录不包含 `step-*.pt`、`latest.pt` 等训练权重，也不收录历史失败试验、短程 pilot、服务器日志或生成的图表。需要续训时，仍须另行保存服务器上的 checkpoint。

请从每组的 `summary/` 阅读结果；复核统计量时使用相应 `eval/` 中的 `costs.pt`。原备份保留于本地 `artifacts/`，此处为用于 GitHub 共享的小体积正式证据副本。
