# ana0719：Beam Search 消融对比结果

## 1. 目的

本记录聚焦回答一个非常具体的问题：

> 在当前 `component_merge` TSP-only 主线上，beam search 相比 greedy 是否有收益；而 `beam dedup` 是否能进一步提升 beam 预算利用率与最终解质量。

本次整理以仓库内已经落盘的评估结果为主，主结果来自：

- `outputs/ablations_tsp20/beam_eval/greedy_eval.json`
- `outputs/ablations_tsp20/beam_eval/beam4_eval.json`
- `outputs/ablations_tsp20/beam_eval/beam4_dedup_eval.stdout.json`

并补充引用两份已有分析中的 smoke 观察：

- `ana0719_beam_dedup_impl.md`
- `ana0719_group_eval_smoke.md`

---

## 2. 主结果来源与设置

### 2.1 主对比对象

三组 decode 设置：

1. `greedy`
2. `beam --beam-size 4`
3. `beam --beam-size 4 --beam-dedup`

### 2.2 数据与模型

从结果 JSON 可读出主实验设置为：

- checkpoint：`outputs/component_merge_tsp20_ep3/model.pt`
- dataset：`data/tsp_uniform/tsp20_test_10000.pt`
- problem size：`TSP20`
- 样本数：`10000`

这组结果比 smoke 子集更适合作为正式 beam ablation 结论，因为它覆盖了完整 10k test 集。

---

## 3. 10k 测试集上的 beam ablation 主结果

### 3.1 结果汇总表

| 方法 | mean tour length | total time (s) | throughput (inst/s) | duplicate edge rate | duplicate component rate | mean unique edge states/step | mean unique component states/step |
|---|---:|---:|---:|---:|---:|---:|---:|
| greedy | 4.057396 | 2.0636 | 4845.99 | - | - | - | - |
| beam-4 | 4.054950 | 5945.5006 | 1.6819 | 0.543894 | 0.566037 | 6.567925 | 6.249065 |
| beam-4 + dedup | 4.051333 | 6073.4344 | 1.6465 | 0.124245 | 0.214525 | 12.610865 | 11.310835 |

### 3.2 直接比较

#### 1. beam 相比 greedy 有稳定但不大的质量提升

- `4.057396 -> 4.054950`
- 绝对改进约 `0.002446`

说明当前 `component_merge_tsp20_ep3` checkpoint 上，beam search 本身是有收益的，只是幅度偏小。

#### 2. dedup 在 beam 基础上继续带来额外增益

- `4.054950 -> 4.051333`
- 相对普通 beam 再下降约 `0.003617`

也就是说，**在相同 beam width = 4 下，加入 canonical dedup 后解继续变好**。

但这里要特别注意：

> 这个结论成立的前提是“**已经决定要开 beam-4**”；它并不自动推出“beam-4 + dedup 相对 greedy 在工程上一定划算”。

#### 3. beam 中确实存在大量重复状态

普通 beam 的重复率很高：

- `duplicate_edge_state_rate = 0.543894`
- `duplicate_component_state_rate = 0.566037`

这意味着在当前搜索过程中，超过一半的候选扩展可归因为重复/等价 partial states。

#### 4. dedup 显著降低重复率

加入 dedup 后：

- edge-level：`0.543894 -> 0.124245`
- component-level：`0.566037 -> 0.214525`

这说明 `canonical_selected_edge_key` 驱动的去重并不是“形式上加了个开关”，而是真正在搜索中压掉了大量冗余候选。

#### 5. dedup 显著提高了结构多样性

唯一状态数显著上升：

- `mean_unique_edge_states_per_step: 6.567925 -> 12.610865`
- `mean_unique_component_states_per_step: 6.249065 -> 11.310835`

这表明同样的 beam 扩展预算下，去重后保留下来的候选覆盖了更多真正不同的结构状态，而不是被不同构造顺序的重复状态占满。

---

## 4. 如何解读这些结果

### 4.1 关于“质量提升”

从绝对数值看，这次 beam ablation 的提升幅度不算大，但方向一致：

- `greedy < beam-4 < beam-4 + dedup`

这很重要，因为它说明 dedup 不是只改善中间统计量，而是已经对最终 tour length 产生了正向影响。

### 4.2 关于“搜索空间压缩”

真正更有说服力的是重复率与唯一状态数：

- 普通 beam 中重复 edge/component states 很多；
- dedup 后重复率显著下降；
- 同时 unique states 明显上升。

因此更准确的表述应该是：

> 当前方法并不是把 TSP 搜索复杂度“理论上降成了多项式”，而是**在实际 beam 推理中减少了等价状态冗余，提高了有效 beam 利用率**。

### 4.3 关于“时间代价”

在这组 10k 结果里：

- `beam-4`: `5945.50s`
- `beam-4 + dedup`: `6073.43s`

dedup 略慢，约增加 `127.93s`，相对开销约 `2.15%`。

这个代价相对于：

- 大幅降低重复率；
- 显著提升 unique state coverage；
- 小幅改善 mean tour length；

是可以接受的。换句话说，这里体现的是**“少量额外 bookkeeping 换更有效的 beam 预算利用”**。

不过，如果把参照物换成 greedy，结论就会明显不同。

### 4.4 从工程性价比角度看：相对 greedy，beam 整体代价非常大

这是你指出的关键点。

先看 `beam-4 + dedup` 相对 `greedy`：

- mean tour length：`4.057396 -> 4.051333`
- 绝对改进约：`0.006063`
- 相对改进约：`0.15%`

但时间上：

- total time：`2.0636s -> 6073.4344s`
- 约慢了：`2943x`

throughput 也从：

- `4845.99 inst/s`

降到：

- `1.6465 inst/s`

也就是说，**从“最终效果提升多少”与“多花了多少推理时间”来看，beam-4 + dedup 相对 greedy 的工程性价比确实偏低**。

同理，即使不看 dedup，只看 `beam-4` 相对 `greedy`，也存在同样问题：

- 质量只提升约 `0.06%`
- 但时间已经慢到约 `2881x`

所以当前更准确的判断应该是：

1. `dedup` **相对普通 beam 是有价值的**；
2. 但 `beam` **相对 greedy 是否值得开启**，在当前实现和当前 checkpoint 上，答案偏向 **不太值得**。

---

## 5. 与已有 smoke 结果是否一致

是基本一致的。

### 5.1 32 样本真实 checkpoint smoke

`ana0719_beam_dedup_impl.md` 中已有一组较小规模对比：

| 方法 | mean tour length | total time (s) | duplicate edge rate | duplicate component rate |
|---|---:|---:|---:|---:|
| greedy | 4.3245 | 0.5922 | - | - |
| beam-4 | 4.3067 | 29.3444 | 0.4896 | 0.5146 |
| beam-4 + dedup | 4.2978 | 27.9017 | 0.0881 | 0.1765 |

它同样支持三个结论：

1. beam 优于 greedy；
2. dedup 优于普通 beam；
3. dedup 显著降低重复率。

### 5.2 64 样本 orbit-sum smoke

`ana0719_group_eval_smoke.md` 中的 64 样本对比也显示：

- `avg_cost: 5.253851 -> 5.204905`
- `duplicate_edge_state_rate: 0.356174 -> 0.034397`
- `duplicate_component_state_rate: 0.447971 -> 0.236111`

说明即使更换 checkpoint，dedup 对“减少重复状态”和“小幅改善解质量”的方向仍然一致。

---

## 6. 阶段性结论

基于当前仓库已有的 beam ablation 结果，可以给出比较明确的结论：

1. **beam search 相比 greedy 的质量收益存在，但幅度很小**；
2. **当前 beam 实现相对 greedy 的推理代价极高，因此单看工程部署性价比并不理想**；
3. **beam search 中确实存在大量由等价/重复 partial states 带来的预算浪费**；
4. **canonical beam dedup 能显著压低重复率并提升唯一结构覆盖数**；
5. **在“已经决定使用 beam”的前提下，dedup 是值得保留的，因为它相对普通 beam 有小幅质量增益，额外时间代价也不大。**

换句话说，当前结果更适合支持下面这个更谨慎的结论：

> `beam dedup` 的价值主要在于**改善 beam 内部效率**，而不是证明“当前 beam 推理整体优于 greedy 的工程性价比”。

如果后续要在论文或分析里表述，建议优先使用下面这类措辞：

> canonical dedup improves effective beam efficiency by reducing redundant equivalent partial states during component-merge decoding.

或者中文表述：

> 规范化去重通过减少等价部分状态的重复探索，提高了 beam 搜索的有效预算利用率。

---

## 7. 后续建议

若继续做 beam 方向的实验，建议按优先级推进：

1. 先不要急着扩大 beam size，而是优先优化 beam 实现本身的速度；
2. 在更快的 beam 实现上，再补 `beam-size 8` 与 `beam-size 8 + dedup`；
3. 比较 `best tour length vs unique beam size`，判断 dedup 是否真正提升“有效 beam 宽度”；
4. 若最终目标是工程可用性，应直接比较“greedy baseline vs 优化后 beam”的 wall-clock / quality trade-off。 
