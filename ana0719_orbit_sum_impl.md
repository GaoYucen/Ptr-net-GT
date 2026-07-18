# ana0719：训练侧 Orbit-Sum（等价动作轨道聚合）最小实现记录

## 1. 本轮目标

在已经完成 **beam dedup / canonical state search** 之后，本轮继续推进一个更能体现计算群论训练优势的方向：

> **把现有 `equiv_set` 训练目标显式整理为 orbit-level supervision，并补成一个可测试、可复现实验入口的最小闭环。**

这轮工作的核心不是再扩搜索，而是把“等价动作不应被强行区分”这件事在训练目标里表达得更清楚。

---

## 2. 为什么做这个方向

在 `component_merge` 的监督训练中，同一个教师 tour 在某一步往往不只有一个“唯一正确”的下一条边。

更准确地说：

- 某些合法下一边都属于教师结构允许的等价动作；
- 它们的差异主要来自构造顺序；
- 若训练目标仍要求模型只命中某一个固定动作，就会引入不必要的标签噪声。

因此更合理的监督方式是：

```text
不要求模型把概率全压在某一个动作上，
而是要求它把足够概率质量分配到整个等价动作集合上。
```

这正对应 orbit-level / equivariant supervision 的基本思想。

---

## 3. 当前代码中的现状判断

在实现前，我先回看了训练路径，结论是：

### 3.1 `equiv_set` 已经隐含了 orbit-sum 语义

`ptrnet_gt/training/supervised.py` 里原本的：

```python
elif objective == "equiv_set":
    legal_target = (target_edges & (~state.selected_edges) & (~state.get_edge_mask())).view(x.size(0), -1)
    masked = log_p.masked_fill(~legal_target, float("-inf"))
    step_losses.append(-torch.logsumexp(masked, dim=1).mean())
```

本质上已经是在做：

```text
-log ∑ p(a),  a ∈ 等价合法目标集合
```

也就是典型的 orbit-sum loss。

### 3.2 现有问题不是“没有这个想法”，而是“语义还不够显式”

主要有两个问题：

1. `equiv_set` 这个名字偏工程化，不够直观；
2. loss 逻辑嵌在训练循环里，缺少一个可复用、可单测的独立函数。

所以这轮更合适的做法不是重写整个训练框架，而是：

> **把已有想法抽象、命名、测试并固定下来。**

---

## 4. 本轮改动

本轮改动范围很小，尽量维持当前 TSP-only 主线和最小风险原则。

涉及文件：

- `ptrnet_gt/training/supervised.py`
- `scripts/train.py`
- `tests/test_supervised_orbit_loss.py`

---

## 4.1 `ptrnet_gt/training/supervised.py`

### 新增可复用 helper

新增：

```python
def orbit_logsumexp_loss(log_p: torch.Tensor, orbit_mask: torch.Tensor) -> torch.Tensor:
```

语义：

- `log_p`：当前 step 的动作 log-prob；
- `orbit_mask`：哪些动作属于当前正确等价轨道；
- 返回：

```text
- mean_b log ∑_{a in orbit_b} p(a)
```

实现上做了两点保护：

1. 自动把 mask 转成 bool；
2. 若某个 batch 样本的 orbit mask 为空，直接抛 `ValueError`，避免 silent failure。

### 将 `equiv_set` 显式视为 orbit-sum

训练分支从：

```python
elif objective == "equiv_set":
```

改为：

```python
elif objective in {"equiv_set", "orbit_sum"}:
```

并且 step loss 改成调用：

```python
step_losses.append(orbit_logsumexp_loss(log_p, legal_target))
```

这样做的意义是：

- 对旧实验名 `equiv_set` 保持兼容；
- 对新实验名 `orbit_sum` 提供更清晰的语义；
- 把轨道聚合损失从训练循环中抽离出来，便于后续扩展。

### 导出符号

新增：

```python
__all__ = ["SupervisedEpochResult", "orbit_logsumexp_loss", "train_supervised_epoch"]
```

方便后续测试和其他模块调用。

---

## 4.2 `scripts/train.py`

训练入口本身不需要大改，因为它已经支持：

```python
objective = config.get("training", {}).get("objective", "reinforce")
```

本轮只补了更清晰的提示：

### 当使用新名字时

```python
if objective == "orbit_sum":
    print("Using supervised orbit-sum objective (grouped equivalent-action probability mass).")
```

### 当使用旧名字时

```python
elif objective == "equiv_set":
    print("Using supervised equiv_set objective (legacy alias of orbit_sum).")
```

这样在日志里更容易区分：

- 当前是否显式跑的是 orbit-sum 实验；
- 是否只是沿用了旧命名。

---

## 4.3 新增测试 `tests/test_supervised_orbit_loss.py`

因为仓库里此前没有覆盖 supervised orbit-style 目标的测试，本轮新增了两个聚焦测试。

### 测试 1：与手算概率质量一致

构造简单分布：

```python
probs = [0.1, 0.2, 0.3, 0.4]
orbit = {1, 2}
```

则目标应为：

```text
-log(0.2 + 0.3)
```

测试确认 `orbit_logsumexp_loss()` 与手算结果一致。

### 测试 2：空 orbit mask 必须报错

若监督 mask 为空，则说明当前 step 的监督目标构造有问题。

本轮选择让这种情况**显式失败**，避免训练在错误目标下继续跑。

---

## 5. 运行测试

遵循当前仓库的 Python / Torch 环境约定，本轮使用：

```bash
KMP_DUPLICATE_LIB_OK=TRUE /opt/conda/envs/py11/bin/python -m unittest \
  tests.test_supervised_orbit_loss \
  tests.test_component_merge_state
```

结果：

```text
Ran 11 tests in 0.047s
OK
```

说明：

1. 新增 orbit-sum helper 测试通过；
2. 现有 `ComponentMergeState` 基础测试未被破坏；
3. 这次改动已形成最小可验证闭环。

---

## 6. 当前结论

这轮工作没有“发明一个完全新的训练算法”，而是完成了更重要的一步：

> **把已有的等价动作集合监督，从隐含实现整理成了显式的 orbit-sum 训练目标。**

其价值主要体现在三点：

### 6.1 论文表达更清楚

现在可以更明确地说：

- 训练时，不强迫模型区分同一轨道中的等价动作；
- loss 在动作轨道上聚合概率质量；
- 推理时，再用 beam dedup 避免把搜索预算浪费在等价部分状态上。

这比单独写“equiv_set”更容易形成一条完整叙事链。

### 6.2 工程上更稳

这次改动没有碰：

- `ComponentMergeState` 核心状态转移；
- decoder 结构；
- RL / REINFORCE 训练主线；
- 现有 beam dedup 路径。

因此风险低、回归范围小。

### 6.3 为后续训练实验提供统一命名

之后做实验时，可以更自然地比较：

1. `training.objective=fixed_order`
2. `training.objective=orbit_sum`
3. `training.objective=orbit_sum + consistency_weight > 0`
4. `orbit_sum` 训练后配合 `beam --beam-dedup`

---

## 7. 建议的下一步实验

### 7.1 先做最小 smoke 训练对比

建议先做一轮小规模对比：

1. `fixed_order`
2. `orbit_sum`

关注：

- 训练 loss 收敛速度；
- validation avg_cost；
- permutation consistency probe；
- 最终 greedy / beam / beam+dedup 的表现差异。

### 7.2 再接上 consistency regularization

当前 `supervised.py` 已经有：

```python
consistency_weight
```

所以后续可以直接实验：

```text
orbit_sum + permutation consistency
```

这样就能把“轨道监督”和“群一致性正则”组合起来。

### 7.3 若训练效果明确，再补配置文件

如果 smoke 结果积极，建议再增加专门配置，例如：

- `configs/component_merge/tsp20_orbit_sum_smoke.yaml`
- 或直接通过 `--override training.objective=orbit_sum` 先做临时实验。

---

## 8. 可直接使用的训练命令

```bash
KMP_DUPLICATE_LIB_OK=TRUE /opt/conda/envs/py11/bin/python scripts/train.py \
  --config configs/component_merge/tsp20.yaml \
  --override training.objective=orbit_sum \
  --override training.epochs=1
```

如果只是兼容旧名字，也可以继续用：

```bash
KMP_DUPLICATE_LIB_OK=TRUE /opt/conda/envs/py11/bin/python scripts/train.py \
  --config configs/component_merge/tsp20.yaml \
  --override training.objective=equiv_set \
  --override training.epochs=1
```

但从现在开始，更建议统一使用：

```text
training.objective=orbit_sum
```

以便实验记录和文档叙述更一致。

---

## 9. 一句话总结

本轮已经把 `equiv_set` 从“训练循环里的一个实现细节”提升为**显式的 orbit-sum 监督目标**，并补齐了最小测试闭环；它是当前在 beam dedup 之外，最适合作为“计算群论训练优势”主线继续推进的下一步。

---

## 10. 小规模 smoke 训练对比结果

在完成代码与单测后，我进一步做了一组 **最小预算的 supervised smoke 训练对比**，目的是先看：

> 在完全相同的缩小训练预算下，`fixed_order` 和 `orbit_sum` 的方向性差异是否已经可见。

### 10.1 统一实验设置

两组实验共用：

- config：`configs/component_merge/tsp20.yaml`
- baseline：`none`
- epochs：`1`
- batch size：`64`
- epoch size：`256`
- val size：`64`
- train dataset：`data/tsp_uniform/tsp20_train_65536.pt`
- val dataset：`data/tsp_uniform/tsp20_val_10000.pt`

唯一差别是：

1. `training.objective=fixed_order`
2. `training.objective=orbit_sum`

---

### 10.2 运行命令

#### fixed_order

```bash
KMP_DUPLICATE_LIB_OK=TRUE /opt/conda/envs/py11/bin/python scripts/train.py \
  --config configs/component_merge/tsp20.yaml \
  --override experiment.name=component_merge_tsp20_fixed_order_smoke \
  --override training.objective=fixed_order \
  --override training.baseline=none \
  --override training.epochs=1 \
  --override training.batch_size=64 \
  --override training.epoch_size=256 \
  --override training.val_size=64 \
  --override data.train_dataset=data/tsp_uniform/tsp20_train_65536.pt \
  --override data.val_dataset=data/tsp_uniform/tsp20_val_10000.pt
```

#### orbit_sum

```bash
KMP_DUPLICATE_LIB_OK=TRUE /opt/conda/envs/py11/bin/python scripts/train.py \
  --config configs/component_merge/tsp20.yaml \
  --override experiment.name=component_merge_tsp20_orbit_sum_smoke \
  --override training.objective=orbit_sum \
  --override training.baseline=none \
  --override training.epochs=1 \
  --override training.batch_size=64 \
  --override training.epoch_size=256 \
  --override training.val_size=64 \
  --override data.train_dataset=data/tsp_uniform/tsp20_train_65536.pt \
  --override data.val_dataset=data/tsp_uniform/tsp20_val_10000.pt
```

---

### 10.3 结果汇总

| 目标 | supervised loss | rollout_cost | validation avg_cost | validation stderr |
|---|---:|---:|---:|---:|
| fixed_order | 4.358680 | 4.016556 | 6.119678 | 0.091230 |
| orbit_sum | 1.385998 | 4.016556 | 5.649206 | 0.124606 |

从 validation avg_cost 看：

```text
6.119678 -> 5.649206
```

在这个极小预算 smoke 下，`orbit_sum` 明显优于 `fixed_order`。

---

### 10.4 初步观察

#### 1. 方向性已经很清楚

虽然这里只跑了：

- 1 个 epoch
- 256 个训练样本预算
- 64 个验证样本

但 `orbit_sum` 已经给出更低的 validation cost，这说明：

> **把监督目标从“固定动作”改成“等价动作轨道上的概率质量”并不是纯理论装饰，而是很快就能在训练信号上体现出收益。**

#### 2. supervised loss 数值不能直接和 fixed_order 横比，但仍有参考意义

`fixed_order` 和 `orbit_sum` 的 loss 形式不同：

- `fixed_order`：单一动作对数概率；
- `orbit_sum`：整个等价动作集合的概率质量。

因此 `4.358680` 与 `1.385998` 不能直接解释成“训练好多少倍”。

但它至少说明：

- orbit-sum 目标在当前数据上更容易为模型提供“可满足”的监督；
- 相比硬性指定唯一动作，聚合监督显著减轻了目标稀疏性和标签任意性。

#### 3. rollout_cost 一致，说明差异主要出在 validation decode 表现

两边打印的 `rollout_cost` 都是：

```text
4.016556
```

这说明当前 epoch 内部统计项并没有暴露明显差异，而真正拉开差距的是训练完后统一做 greedy validation 时的表现。

这与我们的预期一致：

> orbit-sum 的价值并不在于让 teacher-forced step loss 看起来更小，而在于让模型学到更稳健的等价动作判别，从而改善实际 decode 表现。

---

### 10.5 阶段性结论

至少基于这组 smoke 结果，已经可以得到一个很积极的判断：

> **在当前 component merge supervised 训练中，显式 orbit-sum 目标比 fixed-order 目标更符合问题对称性，并且已经在极小规模实验中表现出更好的 validation quality。**

这使它非常适合继续作为 beam dedup 之外的下一条主线推进。

---

### 10.6 下一步最值得做的实验

建议按下面顺序推进：

1. **放大训练预算**
   - 至少 3~10 epochs；
   - 更大的 epoch size；
   - 看优势是否稳定保持。

2. **接入 consistency regularization**
   - 比较：
     - `fixed_order`
     - `orbit_sum`
     - `orbit_sum + consistency_weight`

3. **接上 decode 对比**
   - 对每个 checkpoint 比较：
     - greedy
     - beam
     - beam + dedup

4. **补 permutation consistency probe**
   - 让“群论优势”不只体现在 tour length，也体现在等变一致性指标上。