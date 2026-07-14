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
│   └── inspect_checkpoint.py  # checkpoint 结构查看
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
