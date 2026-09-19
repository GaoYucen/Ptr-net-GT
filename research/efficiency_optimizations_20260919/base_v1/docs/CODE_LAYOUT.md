# Canonical GroupOpt 代码说明

本文档只描述当前论文方法。历史 Adapter 和失败 pilot 不属于下列调用链。

## 1. 核心范式

### `src/groupopt/framework/forest_decoder.py`

这是模型无关的 GroupOpt 执行器，也是新宿主唯一需要依赖的核心文件。

- `ForestHeadProposal` 保存所有合法 `tail -> head` 的条件对数概率，以及供 Tail Selector 使用的摘要特征。
- `ForestAwareDecoder` 规定 `score_heads`、`score_tails`、`update_context` 三个钩子。
- `decode_forest_edges` 统一执行先选 tail、再选 head、更新 Forest 的 rollout，并累计联合 log-likelihood。
- 执行器会检查 tail/head 的形状和合法性，模型不能绕过问题层的防提前成环约束。

### `src/groupopt/framework/neural.py`

定义张量化问题插件和模型输出协议。`ConstructionOutput` 中保存 tour cost、联合似然、每一步的 tail/head、最终 successor、tail entropy 和联合动作 entropy。

## 2. TSP 状态机

### `src/groupopt/problems/tsp_tensor.py`

维护多条不相交局部路径组成的 Forest：

- 顶点度数；
- 每个连通分量及其开放端点；
- 合法 `tail -> head` mask；
- 合并两条局部路径的状态转移；
- 只在最后一步允许闭合 Hamiltonian 环；
- 供 Tail Selector 使用的路径端点、平均表示和规模特征。

这部分不包含任何宿主神经网络参数，因此所有模型共享完全相同的可行性定义。

### `src/groupopt/problems/distributions.py`

生成固定 seed 的 TSP 坐标，以及跨问题 pilot 的 CVRP 输入。论文当前四组正式实验只使用 uniform TSP；跨问题 pilot 不进入本版正式结论。

### `src/groupopt/problems/cvrp_tensor.py`、`m_cycle_cover.py`

提供 CVRP 和指定环数覆盖问题的构造状态机，用于后续跨问题实验。它们虽由统一训练/评估入口导入，但目前不对应已归档的四组正式论文数值。

## 3. 宿主模型

### `src/groupopt/models/am.py`

AM-style self-attention 编码器和 Attention decoder。`native_original` 执行原生单链构造；`native_conditional_free` 将 AM 风格的 head attention 提升到全部合法 tail，并用 AM 风格 Tail Selector 比较局部路径。

### `src/groupopt/models/ptrnet.py`

Pointer Network 接入。保留循环 decoder hidden/cell 和 Pointer scorer，用同一种 Pointer 风格分别实现条件 head 分布与 Tail Selector。

### `src/groupopt/models/gpn.py`

Graph Pointer Network 接入。除节点表示外，还保留相对坐标上下文；Forest 分支会为每个候选 tail 计算相应的相对 head 提案。

### `src/groupopt/models/pomo.py`

与官方 POMO 参数布局兼容的宿主。保留多起点 POMO 训练和 best-of-POMO 评估，同时接入统一 Forest rollout。当前正式容量匹配和信息消融实验在 AM 宿主上进行。

### `src/groupopt/models/native_conditional.py`

共享概率工具：合法条件分布、熵、预期边长和最大 head 概率摘要。它不决定 tail，只为不同宿主避免重复数值代码。

### `src/groupopt/adapters/registry.py`

根据 checkpoint 的 `config.json` 构建 AM、PtrNet、GPN 或 POMO。核心框架不反向依赖注册表。

## 4. 实验代码

### `experiments/train.py`

统一训练循环，负责固定随机数生成器、验证集、优化器、指标、checkpoint 和精确续训状态。只接受 canonical 模式及论文必要对照。

### `experiments/train_ptrnet.py`、`train_gpn.py`、`train_pomo.py`

只负责构造对应宿主，然后复用统一训练循环，不复制实验逻辑。

### `experiments/evaluate.py`

在与训练 seed 不同的固定独立测试集上评估 checkpoint，保存 `summary.json` 和逐实例 `costs.pt`。逐实例数据用于 Original/GroupOpt 的配对置信区间和胜率。

### `experiments/validate_native_original.py`

验证 AM、PtrNet、GPN 的 `native_original` 与固定起点单链表达逐动作等价，并检查 Original 反向传播不会激活 GroupOpt-only 参数。

### `experiments/validate_pomo_official.py`

在提供官方 POMO 源码与 checkpoint 时，检查参数布局、逐动作 tour 和 cost 与官方实现完全兼容。

### `experiments/paper/`

- `README.md`：四组已完成正式实验的复现步骤、结果位置和边界。
- `protocol_paper_suite_v1.json`：当时更大的实验计划；未完成条目不属于当前正式结果。
- `run_final_tsp50_multiseed.sh`：四宿主长程三种子主表，可选复用已有收敛 checkpoint；从头运行时不设置 `REFERENCE_ROOT`。
- `run_scale_tsp.sh`、`summarize_scale_tsp.py`：在 TSP20/100 分别训练并与主表 TSP50 一起汇总。
- `run_capacity_control_am_tsp50.sh`、`summarize_capacity_control_am.py`：AM 等活跃参数单链对照。
- `run_ablation_am_tsp50.sh`、`summarize_ablation_am.py`：AM 的 Random Tail、No Head Summary、No Path State 消融。
- `protocol_*_v1.json`：与归档中四份 `protocol_snapshot.json` 完全一致的冻结协议。
- `protocol_tsp50.json`：已完成短程主实验的不可含糊配置。
- `run_main_tsp50.sh`：从头运行四宿主、两方法、三训练种子。
- `summarize_main_tsp50.py`：检查测试集身份并输出逐种子、macro 和 pooled 配对统计。

## 5. 不属于当前方法的内容

下列路线不应再以 GroupOpt 主方法出现在代码、图表或论文主结果中：

- official AM decoder 前的小型 Adapter；
- `official_amstyle_*`、`official_edge_native_*` 等历史模式；
- joint gate、joint scorer 和单链 bridge；
- SYM-NCO 训练目标；
- 分布漂移 pilot。

它们保存在历史分支 `codex/pre-paper-canonical-snapshot`，仅供追溯失败原因。
