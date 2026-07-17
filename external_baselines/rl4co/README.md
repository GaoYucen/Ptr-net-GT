# RL4CO 外部集成说明

本目录用于接入 `RL4CO`，以便在不污染当前 `ptrnet_gt/` 主包的前提下，复现更现代的 TSP baseline，如：

- Attention Model (AM)
- POMO

## 设计原则

- 不把 RL4CO 代码直接并入 `ptrnet_gt/`
- 尽量通过独立脚本完成训练与评估
- 与当前仓库统一：
  - TSP20 / TSP50 / TSP100
  - 固定随机种子
  - 统一结果导出 JSON

## 计划文件

- `train_am.py`: 用 RL4CO 训练 Attention Model
- `train_pomo.py`: 用 RL4CO 训练 POMO
- `evaluate.py`: 评估 RL4CO checkpoint，并导出统一格式结果

## 安装建议

建议先在当前环境中安装：

```bash
pip install rl4co lightning torchrl tensordict
```

如需固定版本，建议在实际成功跑通后，把版本记录到本文档中。

## 统一输出目标

评估输出建议写入：

```text
outputs/baseline_comparison/
```

并包含字段：

- `method`
- `avg_cost`
- `std_cost`
- `feasible_tour_rate`
- `time_per_instance_sec`
- `num_instances`
- `size`
- `checkpoint`
