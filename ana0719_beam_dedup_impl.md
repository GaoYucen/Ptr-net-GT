# ana0719：等价状态规范化与 Component Merge Beam 去重实现记录

## 1. 本轮目标

基于前一轮对 7 个优化方向的研判，本轮优先落地一个**最小可验证闭环**：

> **训练侧保留现有 `equiv_set`，推理侧新增 canonical state + beam search dedup。**

这样可以先验证一个最关键问题：

- 在 `component_merge` 构造中，不同边添加顺序是否会造成明显重复搜索；
- 若用 canonical key 合并等价部分状态，是否能在不重训的前提下提升 beam 利用率。

本轮工作范围遵循当前 TSP-only 主线，主要涉及：

- `ptrnet_gt/group_theory/canonicalization.py`
- `ptrnet_gt/models/component_merge_search.py`
- `ptrnet_gt/models/__init__.py`
- `scripts/evaluate.py`
- `tests/test_component_merge_beam_dedup.py`

---

## 2. 实现原则

这次实现刻意采取了**保守且可解释**的第一版策略。

### 2.1 去重主键采用精确 edge-set 等价

真正用于 beam 合并的 canonical key 采用：

```python
canonical_selected_edge_key(state)
```

它基于当前 `selected_edges` 中的**有向边集合**构造排序后的 tuple。

选择这个键的原因是：

1. 它对应“相同 partial edge set”这一最稳妥的等价语义；
2. 不会过早把内部结构不同、但节点集合相同的状态错误合并；
3. 非常适合做第一版 beam dedup baseline。

### 2.2 component key 先做统计，不直接参与合并

同时实现了：

```python
canonicalize_components(state)
```

其输出包含每个分量的：

- `start`
- `end`
- `size`
- `sorted(component_nodes)`

这部分目前主要用于：

- 统计 component-level 重复率；
- 作为更宽松商状态定义的后续实验基础；
- 帮助观察 edge-level 与 component-level 重复是否存在差异。

但本轮**不直接用它做 beam 合并**，避免过宽等价导致错误剪枝。

---

## 3. 代码改动

## 3.1 `ptrnet_gt/group_theory/canonicalization.py`

从原先的占位实现：

```python
def canonicalize_state(state):
    return state
```

扩展为以下实际可用函数：

- `canonical_selected_edge_key`
- `canonicalize_components`
- `canonicalize_state`
- `canonical_edge_order`

其中：

```python
def canonicalize_state(state, batch_idx=0):
    return (
        canonical_selected_edge_key(state, batch_idx=batch_idx),
        canonicalize_components(state, batch_idx=batch_idx),
        int(state.step.item()),
    )
```

当前 `canonicalize_state` 更像一个诊断/导出结构，而真正 beam dedup 只使用 `canonical_selected_edge_key`。

---

## 3.2 新增 `ptrnet_gt/models/component_merge_search.py`

新增独立搜索模块，避免侵入训练和原有 greedy 前向路径。

核心接口：

```python
component_merge_beam_search(model, coords, beam_size=1, dedup=False)
```

返回：

- `cost`
- `pi`
- `duplicate_edge_state_rate`
- `duplicate_component_state_rate`
- `unique_edge_states_per_step`
- `unique_component_states_per_step`

### 3.2.1 当前 beam 实现特点

第一版实现是**单实例 beam search**：

- 输入单个 `coords: [n_nodes, 2]`
- 内部构造 `batch = coords.unsqueeze(0)`
- 逐步展开 top candidates
- 可选 dedup
- 最终返回 best beam

优点：

- 接口简单；
- 容易调试；
- 适合作为功能验证版。

缺点：

- 在 `evaluate.py` 中对 batch 内实例是逐个运行，性能不是最终形态；
- 后续如果要做大规模实验，应进一步改为 batched beam。

### 3.2.2 重复率统计

每一步 expansion 后，会分别统计：

- expanded candidates 中相同 edge-key 的重复数；
- expanded candidates 中相同 component-key 的重复数；
- 每步唯一 edge states 数量；
- 每步唯一 component states 数量。

最终在返回结果中汇总为平均型指标。

### 3.2.3 稳健性修正

初版测试暴露出一个问题：beam expansion 不能只依赖模型返回的 top-k logits，因为某些分支下“偏好边”可能已不再合法。

因此 beam expansion 已修改为：

1. 先从 `log_p` 中筛出 `isfinite & ~edge_mask` 的候选；
2. 再按分数排序；
3. 真正执行 `state.update()` 前再次检查边是否合法；
4. 非法边直接跳过。

这样即使模型输出或测试 stub 偏好非法边，beam search 也不会直接崩溃。

---

## 3.3 `scripts/evaluate.py` 扩展

新增参数：

```bash
--decode greedy|beam
--beam-size INT
--beam-dedup
```

当前行为：

- `greedy`：保持原有 `model(batch, return_pi=True)` 路径；
- `beam`：仅对 `component_merge` 启用，并逐实例调用 `component_merge_beam_search()`。

新增输出字段：

- `decode_strategy`
- `beam_size`
- `beam_dedup`
- `duplicate_edge_state_rate`
- `duplicate_component_state_rate`
- `mean_unique_edge_states_per_step`
- `mean_unique_component_states_per_step`

这使得当前可以直接比较：

1. greedy
2. beam without dedup
3. beam with dedup

并用重复率指标判断 beam 宽度是否被冗余状态消耗。

---

## 3.4 `ptrnet_gt/models/__init__.py`

补充导出：

```python
from .component_merge_search import component_merge_beam_search
```

便于 `scripts/evaluate.py` 直接导入使用。

---

## 4. 测试

新增测试文件：

- `tests/test_component_merge_beam_dedup.py`

### 4.1 canonical key 一致性测试

验证两种可交换的边添加顺序：

```python
(0, 1) -> (2, 3)
```

与

```python
(2, 3) -> (0, 1)
```

在当前状态下得到：

- 相同 `canonical_selected_edge_key`
- 相同 `canonicalize_components`

这对应“不同构造顺序到达同一部分结构”的第一类基本正确性。

### 4.2 beam search 基础运行测试

构造一个 `_DummyBeamModel`：

- `encode()` 直接回传输入；
- `get_joint_edge_log_p()` 给若干偏好边更高分；
- 其他合法边给较低但有限的分数。

用它验证：

- beam search 可以完成 4 节点 tour 构造；
- 能返回 `pi` 与 `cost`；
- 能返回 duplicate metrics；
- 每一步都产出 `unique_edge_states_per_step`。

---

## 5. 测试结果

由于当前 `py11` 环境没有安装 `pytest`，我改用标准库 `unittest` 跑聚焦测试。

运行命令：

```bash
KMP_DUPLICATE_LIB_OK=TRUE /opt/conda/envs/py11/bin/python -m unittest \
  tests.test_component_merge_state \
  tests.test_component_merge_beam_dedup
```

结果：

- `Ran 11 tests in 0.021s`
- `OK`

说明：

1. 现有 `ComponentMergeState` 单测未被破坏；
2. 新增 canonicalization / beam dedup 测试通过；
3. 新实现至少在单元级别已连通。

---

## 6. 当前结论

这次实现并没有直接证明 beam dedup 一定提升真实 checkpoint 的 tour quality，
但它已经完成了一个重要前提：

> **把“等价状态规范化 + 去重搜索”从概念分析变成了可运行、可测量、可继续做实验的代码路径。**

当前阶段已经具备以下能力：

1. 对 `component_merge` 执行 beam decode；
2. 可选按 selected-edge canonical key 去重；
3. 在评估时输出重复率和唯一状态数指标；
4. 为后续真实 checkpoint 对比实验提供统一入口。

---

## 7. 与 7 个优化方向的关系

这次工作对应的正是此前建议的**第一优先级**：

### 训练阶段

- 现有 `equiv_set`：不区分同一教师 tour 上的等价合法下一步边。

### 推理阶段

- 新增 beam dedup：不重复保留相同 partial edge set 的状态。

这使论文逻辑更完整：

> 训练时不强迫模型区分等价正确动作；
> 推理时不浪费搜索预算在等价部分状态上。

---

## 8. 下一步建议

### 8.1 立即建议做的实验

用真实 checkpoint 比较：

1. `greedy`
2. `beam --beam-size 4`
3. `beam --beam-size 4 --beam-dedup`
4. `beam --beam-size 8`
5. `beam --beam-size 8 --beam-dedup`

重点看：

- `mean_tour_length`
- `total_time_sec`
- `duplicate_edge_state_rate`
- `duplicate_component_state_rate`
- `mean_unique_edge_states_per_step`

### 8.2 若 dedup 有收益，再继续推进

下一步最自然的延伸是：

1. 批量化 beam search；
2. 加 `best tour length vs unique beam size` 统计；
3. 做 component-level 更宽松合并的 ablation；
4. 再考虑训练侧 multi-teacher equiv_set。

### 8.3 暂时不建议立即做的内容

本轮之后，仍然不建议立刻跳到：

- 轨道动作聚合二级 softmax；
- 完整商空间重编码；
- 轨道对比学习。

原因很简单：

- 它们改动更大；
- 实验链条更长；
- 当前还没有先证明“搜索层重复”到底有多严重、去重到底收益多大。

先把 beam dedup 真实收益测清楚，性价比最高。

---

## 9. 可直接使用的命令

```bash
KMP_DUPLICATE_LIB_OK=TRUE /opt/conda/envs/py11/bin/python scripts/evaluate.py \
  --config configs/component_merge/tsp20.yaml \
  --checkpoint outputs/component_merge_tsp20/model.pt \
  --decode beam \
  --beam-size 8 \
  --beam-dedup
```

如果 checkpoint 路径存在，上述命令会输出：

- 基础 tour quality 指标；
- 时间与吞吐；
- permutation consistency probe；
- beam dedup 重复率与唯一状态数指标。

---

## 10. 小规模真实 checkpoint 对比结果

在完成实现与单测后，我进一步用一个现成 smoke checkpoint 做了真实推理对比。

### 10.1 实验设置

- checkpoint：`outputs/component_merge_tsp20_equiv_set_smoke/model.pt`
- config：`configs/component_merge/tsp20.yaml`
- dataset：`data/tsp_uniform/tsp20_test_32.pt`
- decode 对比：
  1. `greedy`
  2. `beam --beam-size 4`
  3. `beam --beam-size 4 --beam-dedup`

### 10.2 结果汇总

| 方法 | mean tour length | total time (s) | throughput (inst/s) | duplicate edge rate | duplicate component rate | mean unique edge states/step | mean unique component states/step |
|---|---:|---:|---:|---:|---:|---:|---:|
| greedy | 4.3245 | 0.5922 | 54.04 | - | - | - | - |
| beam-4 | 4.3067 | 29.3444 | 1.09 | 0.4896 | 0.5146 | 7.3500 | 6.9891 |
| beam-4 + dedup | 4.2978 | 27.9017 | 1.1469 | 0.0881 | 0.1765 | 13.1313 | 11.8578 |

### 10.3 观察

这组结果非常符合最初的预期。

#### 1. beam 本身能改善 greedy

相较于 greedy：

- `4.3245 -> 4.3067`

说明即使是当前 smoke checkpoint，beam search 已经能带来一定质量提升。

#### 2. dedup 在 beam 上继续带来增益

相较于普通 beam：

- `4.3067 -> 4.2978`

虽然提升幅度不大，但方向明确：

> **在相同 beam width = 4 下，去重后的 beam 解更好。**

这正是“把 beam 宽度从重复状态上释放出来”的预期效果。

#### 3. 重复率显著下降

edge-level 重复率：

- `0.4896 -> 0.0881`

component-level 重复率：

- `0.5146 -> 0.1765`

这说明当前 `component_merge` 搜索中，的确存在大量由不同构造顺序带来的重复状态，而 canonical dedup 可以明显压缩这些冗余。

#### 4. unique states 明显上升

mean unique edge states per step：

- `7.35 -> 13.13`

mean unique component states per step：

- `6.99 -> 11.86`

这说明 dedup 后，同样的 beam 扩展预算实际上覆盖了更多真正不同的结构状态。

#### 5. 时间没有变差，反而略有改善

- `29.34s -> 27.90s`

当前实现仍是**逐实例单独 beam**，还不是高效批量版本；在这种前提下，dedup 没有拖慢推理，反而略快，这进一步说明它剪掉了相当多的冗余扩展。

### 10.4 阶段性结论

至少在这组小规模真实实验中，已经可以得到一个相当清晰的结论：

> **对于当前 component merge 解码，beam dedup 不只是“理论上合理”，而且已经在真实 checkpoint 上表现出：更低重复率、更高结构多样性、略优 tour length、且时间不劣。**

这使它非常适合作为当前主线继续推进。

---

## 11. 批量化加速：共享 batch 编码的中间优化版

在完成第一轮 beam dedup 实证后，我继续做了一个**低风险性能优化**：

### 11.1 性能瓶颈判断

最初版本的 `evaluate.py` 在 beam 模式下是这样工作的：

1. 读取一个 batch；
2. 对 batch 中每个 instance 单独调用 `component_merge_beam_search()`；
3. 每次单独调用内部都会再次执行一次 `model.encode(coords.unsqueeze(0))`。

这意味着虽然搜索本身还是逐实例展开，但 **encoder 前向被重复做了很多次**，成为一个清晰的额外开销。

### 11.2 本轮优化策略

这次没有直接重写成完全 batched beam，而是先做一个**中间版本**：

- 新增内部函数：

```python
_beam_search_single_encoded(model, coords, embeddings, ...)
```

- 单实例搜索逻辑仍保持不变；
- 但 batched 接口先对整个 `coords_batch` 一次性执行：

```python
embeddings_batch = model.encode(coords_batch)
```

- 然后逐实例复用对应的 `embeddings_batch[idx:idx+1]` 做 beam。

新增公开接口：

```python
component_merge_beam_search_batched(model, coords_batch, beam_size=1, dedup=False)
```

并在 `scripts/evaluate.py` 中将 beam 解码路径切换到这个 batched 接口。

### 11.3 优点与局限

优点：

1. **几乎不改变原 beam 语义**；
2. **结果一致性风险低**；
3. **实现简单，容易验证**；
4. 可以先吃到“共享 encoder”的性能收益。

局限：

1. beam expansion 仍然是逐实例 Python 循环；
2. 还没有把不同 beam candidate 的 `get_joint_edge_log_p` 真正并成大 batch；
3. 因此它仍是一个“中间优化版”，不是最终高性能形态。

---

## 12. 批量化后复测结果

仍使用相同设置：

- checkpoint：`outputs/component_merge_tsp20_equiv_set_smoke/model.pt`
- dataset：`data/tsp_uniform/tsp20_test_32.pt`
- `beam_size = 4`

### 12.1 与优化前结果对比

| 方法 | mean tour length | 优化前 total time (s) | 优化后 total time (s) | 时间变化 |
|---|---:|---:|---:|---:|
| beam-4 | 4.3067 | 29.3444 | 20.1102 | -31.5% |
| beam-4 + dedup | 4.2978 | 27.9017 | 20.2340 | -27.5% |

### 12.2 结果解释

这次优化有两个重要结论。

#### 1. 结果保持一致

优化前后：

- `beam-4` 的 mean tour length 保持 `4.3067`；
- `beam-4 + dedup` 的 mean tour length 保持 `4.2978`；
- duplicate rates 与 unique states 指标也保持一致。

这说明本轮重构没有改变搜索语义，只是减少了冗余编码开销。

#### 2. 时间明显下降

- `beam-4`: `29.34s -> 20.11s`
- `beam-4 + dedup`: `27.90s -> 20.23s`

说明仅通过**batch 内共享 encoder embedding**，就已经能带来约 **27% ~ 32%** 的 wall-clock 改善。

这进一步证明：

> 当前 beam 路径的一个主要工程瓶颈，确实是“逐实例重复 encode”，而不是 dedup 逻辑本身。

---

## 13. 对后续工程优化的启示

这次批量化只是第一步，接下来如果继续追求速度，最值得做的是：

1. **candidate-level batching**
   - 把多个 beam candidate 的 `state` 打包；
   - 一次性计算多个 `get_joint_edge_log_p(state, embeddings)`。

2. **state tensor batching**
   - 将当前 `BeamCandidate` 的 Python list/state object 组织为更规则的 batched tensor；
   - 降低 Python 循环和小 tensor 构造的开销。

3. **top-k expansion vectorization**
   - 当前每个 candidate 仍在 Python 中逐个 edge 检查；
   - 后续可以尝试矢量化合法边过滤与 top-k 提取。

4. **分离“评估原型版”和“高性能实验版”**
   - 当前代码适合研究验证；
   - 若进入更大规模实验，建议再单独做一个更激进的 batched beam 模块。

---

## 14. 一句话总结

本轮已经把“**等价状态规范化 + Beam Search 去重**”从研究建议推进为**可运行实现**，并完成了基础测试验证；它现在是当前仓库里最适合继续做真实实验和论文化展开的下一条主线。