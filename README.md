# Ptr-net-GT

一个面向 **TSP（Traveling Salesman Problem）** 的精简研究代码库，核心方法是：

- `ComponentMergeState`
- `ComponentMergeDecoder`
- 基于置换对称性的群论分析模块

当前仓库只维护这一条主线，不再包含其它问题类型或旧实验入口。

---

## 项目结构

```text
Ptr-net-GT/
├── ptrnet_gt/                 # 主包
│   ├── config.py              # 配置加载与 override
│   ├── group_theory/          # 置换、群作用、轨道、稳定子、规范化
│   ├── models/                # 图编码器与 component-merge 解码器
│   ├── problems/              # TSP 定义
│   ├── states/                # ComponentMergeState
│   ├── training/              # 训练循环与 baseline 逻辑
│   └── utils/                 # 张量、mask、日志、tour 指标
├── configs/
│   └── component_merge/       # TSP20 / 50 / 100 配置
├── scripts/
│   ├── train.py               # 训练入口
│   ├── evaluate.py            # 评估入口
│   ├── evaluate_baselines.py  # 传统 / 仓库内 baseline 评估
│   ├── solve_reference.py     # Concorde / LKH reference 统一入口
│   └── inspect_checkpoint.py  # checkpoint 结构查看
├── external_baselines/        # 外部 baseline 适配脚本
│   ├── concorde/
│   ├── lkh/
│   ├── rl4co/
│   └── sym_nco/
├── tests/                     # 单元测试与等变性测试
├── checkpoints/               # 本地权重目录
├── outputs/                   # 训练与评估输出
├── environment.yml
└── README.md
```

---

## 核心模块

### 状态
- `ptrnet_gt/states/component_merge.py`

### 模型
- `ptrnet_gt/models/component_merge_decoder.py`
- `ptrnet_gt/models/graph_encoder.py`

### 问题定义
- `ptrnet_gt/problems/tsp/problem.py`

### 训练与评估
- `scripts/train.py`
- `scripts/evaluate.py`
- `ptrnet_gt/training/trainer.py`

### 群论相关
- `ptrnet_gt/group_theory/permutation.py`
- `ptrnet_gt/group_theory/group_action.py`
- `ptrnet_gt/group_theory/orbit.py`
- `ptrnet_gt/group_theory/stabilizer.py`
- `ptrnet_gt/group_theory/canonicalization.py`

---

## 环境安装

建议使用：

```bash
conda env create -f environment.yml
conda activate ptr-net-gt
```

---

## macOS + PyTorch 运行说明

在当前 macOS 环境下，导入 `torch` 时可能出现 OpenMP 冲突。运行相关命令时建议统一添加：

```bash
KMP_DUPLICATE_LIB_OK=TRUE
```

例如：

```bash
KMP_DUPLICATE_LIB_OK=TRUE python scripts/train.py --config configs/component_merge/tsp20.yaml
KMP_DUPLICATE_LIB_OK=TRUE python scripts/evaluate.py --config configs/component_merge/tsp20.yaml --checkpoint outputs/component_merge_tsp20/model.pt
KMP_DUPLICATE_LIB_OK=TRUE python -m pytest tests/test_component_merge_state.py -q
```

---

## 训练

### 默认训练

```bash
KMP_DUPLICATE_LIB_OK=TRUE python scripts/train.py --config configs/component_merge/tsp20.yaml
```

### 最小调试训练

```bash
KMP_DUPLICATE_LIB_OK=TRUE python scripts/train.py \
  --config configs/component_merge/tsp20.yaml \
  --override training.epochs=1 \
  --override training.epoch_size=32 \
  --override training.batch_size=8 \
  --override training.val_size=16 \
  --override training.eval_batch_size=8
```

训练完成后会在：

```text
outputs/<experiment_name>/model.pt
```

生成 checkpoint。

---

## 评估

```bash
KMP_DUPLICATE_LIB_OK=TRUE python scripts/evaluate.py \
  --config configs/component_merge/tsp20.yaml \
  --checkpoint outputs/component_merge_tsp20/model.pt
```

调试示例：

```bash
KMP_DUPLICATE_LIB_OK=TRUE python scripts/evaluate.py \
  --config configs/component_merge/tsp20.yaml \
  --checkpoint outputs/component_merge_tsp20/model.pt \
  --override evaluation.num_instances=16 \
  --override evaluation.batch_size=8
```

评估输出包含：
- 平均 tour cost
- cost 标准差
- 可行 tour 比例
- cost 一致性误差

---

## Orbit-sum 快速单 seed 验证

如果你只想快速检查 `fixed_order` 与 `orbit_sum` 在**小预算**下是否已经出现方向性差异，可以直接运行：

```bash
KMP_DUPLICATE_LIB_OK=TRUE /opt/conda/envs/py11/bin/python scripts/run_orbit_sum_quick_curve.py \
  --config configs/component_merge/tsp20.yaml \
  --samples 2048 4096 8192 16384 \
  --batch-size 64 \
  --val-size 128 \
  --seed 1234 \
  --skip-permutation-probe
```

这个脚本会：

1. 对每个预算点分别训练 `fixed_order` 与 `orbit_sum`；
2. 默认只跑 **1 epoch / 1 seed**，用于快速 smoke 验证；
3. 自动调用 `scripts/evaluate.py` 做 greedy 评估；
4. 将汇总结果写入：

```text
outputs/orbit_sum_quick_curve/results.json
```

建议优先关注每条记录中的：

- `objective`
- `train_samples_budget`
- `mean_tour_length`
- `training_samples`
- `training_updates`

如果你只想先做更短的试跑，可以把预算进一步缩小，例如：

```bash
KMP_DUPLICATE_LIB_OK=TRUE /opt/conda/envs/py11/bin/python scripts/run_orbit_sum_quick_curve.py \
  --config configs/component_merge/tsp20.yaml \
  --samples 1024 2048 \
  --batch-size 64 \
  --val-size 64 \
  --seed 1234 \
  --skip-permutation-probe
```

---

## Baseline 与 Reference 工作流

当前仓库已支持一条统一的 baseline / reference 对比链路：

1. 生成固定测试集；
2. 用 Concorde 或 LKH-3 生成参考 `costs`；
3. 用 `scripts/evaluate_baselines.py` 评估传统方法、仓库内神经方法；
4. 用 `scripts/aggregate_results.py` 汇总结果。

### 1. 生成固定测试集

```bash
KMP_DUPLICATE_LIB_OK=TRUE /opt/conda/envs/py11/bin/python scripts/generate_tsp_dataset.py \
  --size 20 \
  --num-instances 10000 \
  --seed 1234 \
  --output data/tsp_uniform/tsp20_test_10000.pt
```

### 2. 生成 reference 解

#### 小规模：Concorde（默认只建议 `TSP50 以下`）

```bash
KMP_DUPLICATE_LIB_OK=TRUE /opt/conda/envs/py11/bin/python scripts/solve_reference.py \
  --solver concorde \
  --dataset data/tsp_uniform/tsp20_test_10000.pt \
  --output data/tsp_uniform/reference/tsp20_concorde.pt
```

> `Concorde` 默认只用于小规模；当 `size >= 50` 时，脚本会默认拒绝运行，避免把高耗时设置误当成常规 baseline。

#### 中大规模：LKH-3

```bash
KMP_DUPLICATE_LIB_OK=TRUE /opt/conda/envs/py11/bin/python scripts/solve_reference.py \
  --solver lkh \
  --dataset data/tsp_uniform/tsp100_test_10000.pt \
  --lkh-executable /path/to/LKH \
  --output data/tsp_uniform/reference/tsp100_lkh.pt \
  --runs 1 \
  --max-trials 1000 \
  --seed 1234
```

推荐协议：

- `TSP20`: Concorde 或 LKH-3
- `TSP50`: 优先 LKH-3；Concorde 仅在你明确接受耗时时才手动放开
- `TSP100+`: 使用 LKH-3，不默认使用 Concorde

### 3. 评估 baseline

传统 baseline：

```bash
KMP_DUPLICATE_LIB_OK=TRUE /opt/conda/envs/py11/bin/python scripts/evaluate_baselines.py \
  --dataset data/tsp_uniform/tsp20_test_10000.pt \
  --reference data/tsp_uniform/reference/tsp20_concorde.pt \
  --methods random,nearest_neighbor,nearest_neighbor_multistart,nn_two_opt \
  --output outputs/baseline_comparison/tsp20_traditional.json
```

加入仓库内模型（如 Component Merge / Pointer Network / Attention Model）时，可继续传对应 config 和 checkpoint 参数。

### 4. 外部 baseline

#### RL4CO

`external_baselines/rl4co/` 中已包含：

- `train_am.py`
- `train_pomo.py`
- `evaluate.py`

用于快速复现 AM / POMO，并导出统一 JSON 结果。

#### Sym-NCO

`external_baselines/sym_nco/evaluate.py` 采用“外部命令适配”模式，不强依赖官方仓库内部 API。你需要提供：

- `--symnco-root`
- `--checkpoint`
- `--dataset`
- `--command`

例如：

```bash
KMP_DUPLICATE_LIB_OK=TRUE /opt/conda/envs/py11/bin/python external_baselines/sym_nco/evaluate.py \
  --symnco-root /path/to/Sym-NCO \
  --checkpoint /path/to/checkpoint.pt \
  --dataset data/tsp_uniform/tsp20_test_10000.pt \
  --command 'python eval.py --checkpoint {checkpoint} --dataset {dataset} --output {raw_output}' \
  --output outputs/baseline_comparison/symnco_tsp20.json
```

### 5. 汇总结果

```bash
KMP_DUPLICATE_LIB_OK=TRUE /opt/conda/envs/py11/bin/python scripts/aggregate_results.py \
  outputs/baseline_comparison/tsp20_traditional.json \
  outputs/baseline_comparison/symnco_tsp20.json \
  --output-dir outputs/baseline_comparison
```

`scripts/aggregate_results.py` 可直接消费：

- `scripts/evaluate_baselines.py` 输出的 `results` 列表
- 外部 baseline 输出的单条 JSON 记录

---

## 测试

核心测试：

```bash
KMP_DUPLICATE_LIB_OK=TRUE python -m pytest \
  tests/test_component_merge_state.py \
  tests/test_component_merge_masks.py \
  tests/test_component_merge_decoder.py -q
```

群论与等变性测试：

```bash
KMP_DUPLICATE_LIB_OK=TRUE python -m pytest \
  tests/test_group_action.py \
  tests/test_state_equivariance.py \
  tests/test_transition_equivariance.py \
  tests/test_cost_invariance.py -q
```

---

## 当前支持范围

当前仓库支持：

- TSP 数据生成与训练
- `ComponentMergeState` 状态更新与掩码约束
- `ComponentMergeDecoder` 解码
- 训练 checkpoint 保存与评估
- 节点重标号下的群作用、等变性与不变性分析

不维护其它问题类型，也不提供旧版 baseline 训练入口。
