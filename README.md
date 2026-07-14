# Ptr-net-GT

本项目研究 **基于连通分量合并状态（Component Merge State）与置换群对称性（Permutation Group Symmetry）的 TSP 双端神经构造方法**，并保留标准 **Attention Model** 与 **Pointer Network** 作为对比基线。

当前仓库已从历史的多问题、多脚本实验集合，**收缩为 TSP-only 的研究代码库**，主线聚焦于 `ComponentMergeState + ComponentMergeDecoder`。

---

## 研究主线

建议将当前 `dev-3` / 清理分支的核心研究流程理解为：

```text
TSP 实例
  ↓
图节点编码
  ↓
ComponentMergeState 初始化
  ↓
双端边选择与连通分量合并
  ↓
可行性掩码与防止提前成环
  ↓
完整 Hamilton 回路
  ↓
训练 / 评估 / 对称性分析
```

核心研究关注点包括：

- 图编码器
- 连通分量合并状态
- 双端解码模型
- 可行性掩码
- 节点重标号下的不变性 / 等变性
- 轨道、稳定子与规范化等群论模块
- 与标准 Attention / Pointer 基线的对比

---

## 当前目录结构

```text
Ptr-net-GT/
├── ptrnet_gt/                 # 正式 Python 包（整理中）
├── scripts/                   # 统一命令行入口（整理中）
├── configs/                   # 训练/实验配置（整理中）
├── experiments/               # 实验分析脚本、可视化、notebook
├── tests/                     # 单元测试与集成测试
├── nets/                      # 历史基线模型实现
├── problems/tsp/              # TSP 问题定义
├── checkpoints/               # 本地权重存放目录
├── outputs/                   # 运行输出目录
├── assets/figures/            # 可视化图像资源
└── data/                      # 数据说明与样例目录
```

---

## 已整理的内容

当前已完成的第一阶段整理包括：

- 建立 `ptrnet_gt/` 包结构；
- 新建 `group_theory/` 占位模块；
- 建立 **TSP-only** 的统一训练/评估入口；
- 新建 `configs/component_merge/tsp20.yaml`；
- 将评估脚本迁移到 `experiments/evaluation/`；
- 将可视化脚本迁移到 `experiments/visualization/`；
- 将 notebook 迁移到 `experiments/notebooks/`；
- 将 GIF 资源迁移到 `assets/figures/`；
- 删除旧双端模型、旧状态和非 TSP 问题代码；
- 清理非 TSP pretrained，仅保留 TSP 历史权重；
- 将历史参数与命令文件归档；
- 更新 `.gitignore`，避免继续提交训练产物。

---

## 核心代码位置

### 当前核心状态

- `ptrnet_gt/states/component_merge.py`

### 当前测试

- `tests/test_component_merge_state.py`
- `tests/test_component_merge_masks.py`

### 当前核心/基线模型

- `ptrnet_gt/models/component_merge_decoder.py`
- `nets/graph_encoder.py`
- `nets/attention_model.py`
- `nets/pointer_network.py`
- `nets/critic_network.py`

### 新包入口（阶段性兼容）

- `ptrnet_gt/models/graph_encoder.py`
- `ptrnet_gt/problems/tsp.py`
- `ptrnet_gt/training/`
- `ptrnet_gt/group_theory/`

---

## 环境安装

建议优先使用已有环境文件：

```bash
conda env create -f environment.yml
conda activate ptr-net-gt
```

如果环境名与文件中定义不一致，请按 `environment.yml` 实际内容为准。

### OpenMP 说明（当前 macOS 环境必读）

当前环境下 `torch` 导入会触发 `libomp.dylib already initialized`。临时运行方式是：

```bash
export KMP_DUPLICATE_LIB_OK=TRUE
```

新的 `scripts/train.py` 和 `scripts/evaluate.py` 已默认设置该环境变量，但如果你直接运行测试或手动导入 `torch`，建议先执行上面的命令。

---

## 训练

### 统一训练入口

当前推荐使用新的配置驱动入口：

```bash
python scripts/train.py --config configs/component_merge/tsp20.yaml
```

调试时可以覆盖配置：

```bash
python scripts/train.py \
  --config configs/component_merge/tsp20.yaml \
  --override training.epochs=1 \
  --override training.epoch_size=32 \
  --override training.batch_size=8 \
  --override training.val_size=16
```

---

## 评估

### 统一评估入口

```bash
python scripts/evaluate.py \
  --config configs/component_merge/tsp20.yaml \
  --checkpoint outputs/component_merge_tsp20/model.pt
```

调试示例：

```bash
python scripts/evaluate.py \
  --config configs/component_merge/tsp20.yaml \
  --checkpoint outputs/debug_train/component_merge_tsp20/model.pt \
  --override evaluation.num_instances=16 \
  --override evaluation.batch_size=8
```

---

## 数据生成

```bash
python scripts/generate_data.py --problem tsp --name validation --seed 4321
python scripts/generate_data.py --problem tsp --name test --seed 1234
```

---

## 测试

当前最关键的是状态、模型和对称性测试：

```bash
KMP_DUPLICATE_LIB_OK=TRUE python -m pytest tests/test_component_merge_state.py tests/test_component_merge_masks.py tests/test_component_merge_decoder.py -q
KMP_DUPLICATE_LIB_OK=TRUE python -m pytest tests/test_group_action.py tests/test_state_equivariance.py tests/test_transition_equivariance.py tests/test_cost_invariance.py -q
```

---

## 计算群论模块

目前已建立以下占位模块，后续会逐步实现并接入测试：

```text
ptrnet_gt/group_theory/
├── permutation.py
├── group_action.py
├── orbit.py
├── stabilizer.py
└── canonicalization.py
```

当前已具备最基础的接口：

- 逆置换
- 置换复合
- 节点 / tour / edge 置换
- 实例、状态、mask、logits 的群作用接口雏形

---

## 下一阶段计划

接下来建议按以下顺序继续推进：

1. 继续把训练/评估细节从旧 `train.py` / `reinforce_baselines.py` 迁入 `ptrnet_gt/training/`；
2. 将 `ptrnet_gt/problems/tsp.py` 和 `ptrnet_gt/problems/tsp/problem.py` 去桥接化；
3. 继续加强群作用、轨道、稳定子、规范化与等变性测试；
4. 持续清理剩余历史根目录入口代码。

---

## 说明

当前仓库处于“**结构整理优先**”阶段：

- 第一阶段目标是让项目结构更清晰；
- 尽量不立即破坏历史训练流程；
- 新包与历史代码暂时并行存在；
- 后续再逐步完成核心逻辑迁移和清理删除。

如果你要继续推进到下一阶段，建议优先处理：

1. `dual_state_2.py` 迁移；
2. 双端模型合并；
3. 单一训练/评估入口落地；
4. 群论模块与等变性测试接入。
