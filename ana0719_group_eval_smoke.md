# ana0719：群作用评估入口与 beam dedup smoke 对比记录

## 1. 本轮目的

这一轮不是继续改训练目标，而是把前面讨论的“群作用 / 轨道压缩 / 等变一致性”整理成**可以直接观测的评估指标与 smoke 实验闭环**。

围绕用户前面提出的四个方向，这一轮优先落地的是：

1. **群作用降低搜索空间**；
2. **群不变 / 等变一致性可测化**；
3. **减少枚举冗余的推理侧证据**。

其中第 3 点（group-aware attention）本轮还没有改模型结构，而是先把评估入口铺好。

---

## 2. 本轮代码改动

涉及文件：

- `ptrnet_gt/models/component_merge_search.py`
- `scripts/evaluate.py`
- `tests/test_component_merge_beam_dedup.py`
- `tests/test_evaluate_metrics.py`

---

## 3. Beam dedup 统计增强

### 3.1 新增统计项

在 `ptrnet_gt/models/component_merge_search.py` 中，beam search 现在除了已有的：

- `duplicate_edge_state_rate`
- `duplicate_component_state_rate`
- `unique_edge_states_per_step`
- `unique_component_states_per_step`

还会进一步返回：

- `expanded_candidates_per_step`
- `kept_candidates_per_step`
- `mean_expanded_candidates_per_step`
- `mean_kept_candidates_per_step`
- `dedup_retention_rate`

这样可以更直接回答：

> beam 的预算里，到底有多少比例浪费在等价或重复 partial states 上？

### 3.2 解释方式

- `expanded_candidates_per_step`：每一步实际展开了多少候选；
- `kept_candidates_per_step`：排序/去重后真正保留下来的候选；
- `dedup_retention_rate`：保留比例，越低说明原始展开里的冗余越大。

注意：当前实现里即使 `dedup=False`，也仍然会统计 canonical duplicate 情况，因此它可以用来**诊断未去重时的冗余严重程度**。

---

## 4. permutation consistency probe 整理

在 `scripts/evaluate.py` 中，本轮把已有 permutation probe 整理为统一输出：

- `permutation_exact_successor_consistency`
- `permutation_edge_jaccard`
- `permutation_edge_recall`
- `permutation_relative_length_diff`
- `permutation_action_prob_equiv_error`

同时新增：

```bash
--skip-permutation-probe
```

方便在只想测 decode 性能时跳过 probe。

### 指标含义

#### 1. `permutation_exact_successor_consistency`

把输入实例做随机节点置换，分别 decode，再把置换结果映回原编号后比较 successor 结构是否完全一致。

#### 2. `permutation_edge_jaccard`

比较两次 decode 后 edge set 的 Jaccard 相似度。

#### 3. `permutation_relative_length_diff`

比较原输入与置换输入输出 tour length 的相对差异。

#### 4. `permutation_action_prob_equiv_error`

对 joint-edge action logits 做回置换后比较概率分布误差，作为一步动作层面的等变误差 probe。

---

## 5. 单元测试

### 5.1 更新 beam dedup 测试

`tests/test_component_merge_beam_dedup.py` 现在检查：

- `dedup_retention_rate` 字段存在；
- `expanded_candidates_per_step` / `kept_candidates_per_step` 长度正确；
- retention rate 落在 `[0, 1]`。

### 5.2 新增 evaluate helper 测试

`tests/test_evaluate_metrics.py` 新增了一个轻量测试，固定：

> 当 `enabled=False` 时，`_maybe_compute_permutation_metrics()` 返回全 `None` 结构，并且不尝试执行 probe。

### 5.3 测试命令

```bash
KMP_DUPLICATE_LIB_OK=TRUE /opt/conda/envs/py11/bin/python -m unittest \
  tests.test_component_merge_beam_dedup \
  tests.test_evaluate_metrics
```

结果：

```text
Ran 3 tests in 0.014s
OK
```

---

## 6. 评估实验设置

### 6.1 为什么要切 smoke 子集

原本我尝试直接在：

```text
data/tsp_uniform/tsp20_test_10000.pt
```

上做 beam 对比，但当前 `scripts/evaluate.py` 在指定固定 dataset 时，会直接载入该文件中全部实例，因此即使写了：

```text
--override evaluation.num_instances=128
```

也不会把固定数据文件实际裁小。

因此为了快速得到可用结果，我额外生成了两个 smoke 子集：

- `data/tsp_uniform/tsp20_test_64_smoke.pt`
- `data/tsp_uniform/tsp20_test_8_smoke.pt`

其中这轮最终使用的是 64 样本子集。

### 6.2 使用的 checkpoint

本轮选择：

```text
outputs/component_merge_tsp20_orbit_sum_smoke/model.pt
```

它对应前面已经验证过更贴合轨道监督思想的 `orbit_sum` 训练目标。

### 6.3 对比命令

#### 不开 dedup

```bash
KMP_DUPLICATE_LIB_OK=TRUE /opt/conda/envs/py11/bin/python scripts/evaluate.py \
  --config configs/component_merge/tsp20.yaml \
  --checkpoint outputs/component_merge_tsp20_orbit_sum_smoke/model.pt \
  --dataset data/tsp_uniform/tsp20_test_64_smoke.pt \
  --override evaluation.batch_size=64 \
  --decode beam \
  --beam-size 4
```

#### 开 dedup

```bash
KMP_DUPLICATE_LIB_OK=TRUE /opt/conda/envs/py11/bin/python scripts/evaluate.py \
  --config configs/component_merge/tsp20.yaml \
  --checkpoint outputs/component_merge_tsp20_orbit_sum_smoke/model.pt \
  --dataset data/tsp_uniform/tsp20_test_64_smoke.pt \
  --override evaluation.batch_size=64 \
  --decode beam \
  --beam-size 4 \
  --beam-dedup
```

---

## 7. 64 样本 smoke 结果

### 7.1 不开 dedup

关键结果：

| 指标 | 数值 |
|---|---:|
| avg_cost | 5.253851 |
| duplicate_edge_state_rate | 0.356174 |
| duplicate_component_state_rate | 0.447971 |
| beam_dedup_retention_rate | 0.277778 |
| mean_expanded_candidates_per_step | 14.40 |
| mean_kept_candidates_per_step | 4.00 |
| permutation_consistency | 1.0 |
| permutation_edge_jaccard | 1.0 |
| permutation_relative_length_diff | 0.0 |
| permutation_action_prob_equiv_error | 1.06e-09 |

### 7.2 开 dedup

关键结果：

| 指标 | 数值 |
|---|---:|
| avg_cost | 5.204905 |
| duplicate_edge_state_rate | 0.034397 |
| duplicate_component_state_rate | 0.236111 |
| beam_dedup_retention_rate | 0.277127 |
| mean_expanded_candidates_per_step | 14.40 |
| mean_kept_candidates_per_step | 3.990625 |
| permutation_consistency | 1.0 |
| permutation_edge_jaccard | 1.0 |
| permutation_relative_length_diff | 0.0 |
| permutation_action_prob_equiv_error | 1.10e-09 |

### 7.3 并列观察

#### 1. dedup 明显压低了 edge-state 重复率

从：

```text
0.356174 -> 0.034397
```

这说明：

> 当前 beam 扩展中，的确存在大量仅仅因为构造顺序不同而重复的 partial edge states；canonical dedup 可以显著消除这种冗余。

#### 2. component-level 重复也下降，但没有 edge-state 那么极端

从：

```text
0.447971 -> 0.236111
```

说明 component canonicalization 视角下也有冗余，但当前最直接受益的仍是 selected-edge state quotienting。

#### 3. 在相同 beam size 下，cost 有小幅改善

从：

```text
5.253851 -> 5.204905
```

这是一个积极信号：

> dedup 不只是“统计上减少重复”，而是已经把本来浪费在等价状态上的 beam 预算，部分释放给了更有价值的候选。

#### 4. permutation probe 在该 checkpoint 上表现非常稳定

两组结果中都接近：

- exact consistency = 1.0
- edge jaccard = 1.0
- relative length diff = 0.0
- action prob equiv error ≈ 1e-9

这说明对这个小规模 smoke checkpoint 而言：

> 其节点重编号等变性已经相当稳定，至少在当前 probe 方式下没有明显破坏。

---

## 8. 对前面四个研究点的支撑程度

### 8.1 点 1：群作用降低搜索空间

**已经得到直接支撑。**

因为现在我们不仅能从理论上说“存在等价轨道”，而且能从实验上给出：

- duplicate state rate；
- kept / expanded ratio；
- dedup 后 cost 改善。

### 8.2 点 2：群不变 / 等变一致性

**已经有最小 probe 闭环。**

虽然目前还不是完整论文级大实验，但至少：

- 指标已经接入统一 evaluate；
- 可以对不同 objective / checkpoint 横向比较。

### 8.3 点 3：group-aware attention

**本轮尚未实现。**

但这一轮完成的评估接口将非常适合作为后续实验支撑：

- 如果后面加 decoder-level group/state-aware bias；
- 就可以直接观察它对 cost、duplicate ratio、permutation action equiv error 的影响。

### 8.4 点 4：减少枚举复杂度

**已经拿到有效的推理侧证据，但要谨慎表述。**

更合理的表述仍然是：

> 通过 canonical quotienting 减少有效搜索冗余、降低 beam 预算浪费，而不是宣称把 TSP 的最坏复杂度降成多项式。

---

## 9. 当前发现的工程问题

### 9.1 固定 dataset 路径下，`evaluation.num_instances` 不会裁切文件内容

这导致 smoke evaluate 时需要手动准备小数据子集。

后续可以考虑直接在 `scripts/evaluate.py` 中补一个行为，例如：

- 当传入固定 dataset 且设置了 `evaluation.num_instances`；
- 只取前 `N` 条样本。

这会让 smoke 实验更方便。

### 9.2 dedup 并没有显著提升 wall-clock 时间

本轮 64 样本结果里：

- 不开 dedup：`51.29s`
- 开 dedup：`51.44s`

说明当前版本的 dedup 主要体现为：

1. **减少搜索冗余**；
2. **改善 beam 预算利用率**；
3. **小幅改善 cost**；

但**还没有体现 runtime 加速**。

这是合理的，因为当前 dedup 本身也需要 canonicalization 和哈希开销。

因此论文里更稳妥的卖点应是：

> improved effective beam efficiency / reduced redundant state exploration

而不是直接写成 speedup。

---

## 10. 当前阶段结论

这一轮已经把“群作用优势”从概念层推进到了**可测指标 + smoke 实验结果**。

可以明确说：

1. 当前 `component_merge` beam search 中，确实存在显著的等价 partial-state 冗余；
2. canonical dedup 能显著压低 duplicate state rate；
3. 在相同 beam size 下，dedup 已经表现出小幅 cost 改善；
4. permutation consistency probe 已经成为可重复使用的统一评估接口；
5. 这些结果足以支撑“群作用压缩搜索空间”和“等变一致性可测化”作为论文主线的一部分。

---

## 11. 建议的下一步

### 优先级 A：补更系统的实验表

建议直接比较：

1. `fixed_order`
2. `orbit_sum`
3. `orbit_sum + consistency_weight`

对每个 checkpoint 统一跑：

- greedy
- beam
- beam + dedup
- permutation probe

### 优先级 B：改 evaluate 支持固定数据集裁切

这样可以更方便做：

```bash
--override evaluation.num_instances=64
```

而不需要额外存 smoke 子集文件。

### 优先级 C：实现 decoder-level group/state-aware bias

比直接改 encoder attention 更稳，推荐放在：

- `ptrnet_gt/models/component_merge_decoder.py`

然后继续观察：

- `avg_cost`
- `duplicate_*`
- `permutation_action_prob_equiv_error`

---

## 12. 一句话总结

本轮已经为“群作用降低搜索空间”和“群等变一致性”补齐了可量化评估入口，并通过 64 样本 beam smoke 对比确认：**canonical dedup 能显著减少等价状态冗余，并在固定 beam 预算下带来小幅质量提升。**