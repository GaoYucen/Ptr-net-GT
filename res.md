# TSP20 模型与 Baseline 性能结果

## 结果来源

本文档基于仓库内已有统一评测结果整理，未额外重跑训练。

- 结果文件：`outputs/baseline_comparison/tsp20_test10000_all_models.json`
- 固定测试集：`data/tsp_uniform/tsp20_test_10000.pt`
- 问题规模：`TSP20`
- 测试实例数：`10000`
- 评测 seed：`1234`
- 神经模型解码方式：`greedy`

其中三类神经模型使用的训练预算一致：

- `epochs = 10`
- `epoch_size = 65536`
- `batch_size = 512`
- `training_updates = 1280`
- `training_samples = 655360`

当前未提供 Concorde / LKH 参考最优值，因此 **optimality gap 暂无法报告**。

---

## 主结果表

| Method | Mean Tour Length | Std | Feasible Rate | Time / Instance (s) | Throughput (inst/s) | Params | Training Updates | Training Samples |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Random Tour | 10.4043 | 1.2187 | 100% | 0.00000220 | 453833.76 | - | - | - |
| Nearest Neighbor | 4.4933 | 0.5395 | 100% | 0.00000243 | 411489.48 | - | - | - |
| Multi-start Nearest Neighbor | 4.0442 | 0.3924 | 100% | 0.00004065 | 24599.81 | - | - | - |
| NN + 2-opt | 4.2017 | 0.4397 | 100% | 0.01452296 | 68.86 | - | - | - |
| Component Merge | 4.2548 | 0.4474 | 100% | 0.00045919 | 2177.77 | 559872 | 1280 | 655360 |
| Pointer Network | 4.1000 | 0.3831 | 100% | 0.00022697 | 4405.81 | 527872 | 1280 | 655360 |
| Attention Model | **3.9946** | 0.3899 | 100% | 0.00029468 | 3393.49 | 510720 | 1280 | 655360 |

---

## 排名与观察

按平均路径长度从优到劣排序：

1. **Attention Model**: `3.9946`
2. **Multi-start Nearest Neighbor**: `4.0442`
3. **Pointer Network**: `4.1000`
4. **NN + 2-opt**: `4.2017`
5. **Component Merge**: `4.2548`
6. **Nearest Neighbor**: `4.4933`
7. **Random Tour**: `10.4043`

### 1. 当前最优结果

在当前仓库已有结果中，**Attention Model** 表现最好，平均 tour length 为 **3.9946**。在统一训练预算下，它优于当前的 `Component Merge` 和 `Pointer Network`。

### 2. 当前 Component Merge 的位置

`Component Merge` 当前结果为 **4.2548**：

- 明显优于 `Random Tour`
- 优于单起点 `Nearest Neighbor`
- 但落后于：
  - `Attention Model`
  - `Pointer Network`
  - `Multi-start Nearest Neighbor`
  - `NN + 2-opt`

这说明当前 `Component Merge` **已经学到有效策略**，但从现有结果看，**尚未超过更强 baseline**。

### 3. Baseline 的强弱关系

一个值得注意的现象是：**Multi-start Nearest Neighbor** 的结果已经达到 **4.0442**，非常接近甚至优于当前部分神经方法。这意味着：

- 起点对称性/多起点策略对 TSP20 很重要；
- 如果后续要突出群论或结构设计收益，至少需要稳定超过这个 baseline；
- 仅优于单起点贪心还不够说明模型结构本身有明显优势。

### 4. 关于 NN + 2-opt 的结果

当前 `NN + 2-opt` 的结果为 **4.2017**，比 `Multi-start Nearest Neighbor` 还差，同时推理耗时显著更高：

- `NN + 2-opt`：约 `0.0145 s / instance`
- `Multi-start NN`：约 `0.0000407 s / instance`

这与常见直觉不完全一致，因此建议后续复核：

- 当前 2-opt 实现是否为完整改进版；
- 起始路径是否固定为单起点 NN；
- stopping criterion 是否过早；
- 是否需要比较 `Multi-start NN + 2-opt`。

### 5. 速度与质量权衡

- 传统启发式中，`Nearest Neighbor` 最快且明显强于随机。
- `Multi-start NN` 以很低额外代价带来明显质量提升。
- 神经模型中，`Pointer Network` 推理略快于 `Attention Model`，但解质量稍差。
- `Component Merge` 当前推理最慢的神经模型之一，且质量尚未占优，因此后续需要重点关注其 **解码效率** 和 **解质量提升空间**。

---

## 面向后续实验的结论

基于当前已有结果，可以先给出一个阶段性判断：

> 在固定 TSP20 测试集、统一训练预算、greedy 解码下，当前 `Component Merge` 尚未超过 `Attention Model`，也未超过 `Pointer Network` 和 `Multi-start Nearest Neighbor`。

因此，如果后续工作目标是支撑群论相关设计的有效性，建议优先推进：

1. 对 `Component Merge` 做进一步训练与调参；
2. 检查其状态表示、动作空间与 credit assignment 是否限制了性能；
3. 增加更公平/更强的对比，如：
   - `Multi-start NN + 2-opt`
   - Concorde / LKH 参考值
   - 多 seed 汇总结果
   - permutation / equivariance 相关鲁棒性实验

---

## 备注

- `outputs/baseline_comparison/tsp20_full_comparison_with_rl4co.json` 中包含 RL4CO smoke 结果，但该文件仅使用 `128` 个实例，且部分 checkpoint 为 smoke 训练结果，**不应与本页 10000 实例正式结果直接混合比较**。
- 因此本页仅采用 `tsp20_test10000_all_models.json` 作为主结论依据。