# config_0719：当前最优配置研判

## 1. 结论先行

基于仓库内现有实验记录，**“目前性能最好”的配置需要分成两个层次来看**：

1. **全仓库当前已记录的最优结果**：`Attention Model`（不是 component-merge 路线）
2. **当前 TSP-only / component-merge 主线中已验证最优的配置**：`joint_edge + equiv_set + 关闭动态特征`

如果本次问题的语境遵循当前仓库主线（见 `.clinerules` 中的 TSP-only 与 component-merge 优先规则），那么更值得落档的结论是第 2 条。

---

## 2. 证据来源

本结论来自以下现有文档与配置，而**不是本次重新训练**：

- `ana0718.md`
- `ana0719.md`
- `ana0719_group_eval_smoke.md`
- `res.md`
- `configs/component_merge/tsp20.yaml`

---

## 3. 全仓库范围：当前记录中的最好结果

`res.md` 给出的统一 TSP20 测试集结果显示：

| Method | Mean Tour Length |
|---|---:|
| Attention Model | **3.9946** |
| Multi-start Nearest Neighbor | 4.0442 |
| Pointer Network | 4.1000 |
| NN + 2-opt | 4.2017 |
| Component Merge | 4.2548 |

因此如果问题是：

> **整个仓库里目前哪个模型/配置效果最好？**

那么答案是：

> **Attention Model 最好，TSP20 测试集 mean tour length = 3.9946。**

但这条结论主要对应 baseline 比较，不是当前重点推进的 `component_merge` 研究主线。

---

## 4. component-merge 主线：当前最优配置

结合 `ana0718.md` 与 `ana0719.md`，在当前已做的 component-merge 路线实验中，**最明确、最稳定的正向结果**来自：

> **`action_mode=joint_edge` + `training.objective=equiv_set` + `model.use_dynamic_role_features=false`**

也就是：

- 不再使用旧的 `tail_head` 两阶段动作；
- 使用 `joint_edge` 直接对合法边打分；
- 使用 `equiv_set` 等价动作集合监督；
- 暂时关闭动态角色特征，以隔离主效应；
- 当前任务仍是 TSP20。

### 4.1 为什么不是 `tail_head`

`ana0718.md` 的 smoke test 结果：

| 方法 | mean tour length |
|---|---:|
| tail_head | 6.5649 |
| joint_edge (static) | **4.5626** |

说明 `joint_edge` 明显强于 `tail_head`，因此当前最优配置一定在 `joint_edge` 路线里。

### 4.2 为什么不是 dynamic features

`ana0718.md` 结果：

| 方法 | mean tour length |
|---|---:|
| joint_edge (static) | **4.5626** |
| joint_edge (dynamic) | 4.6949 |

`ana0719.md` 对 raw / projected dynamic 的进一步复测也显示：

| 方法 | mean tour length |
|---|---:|
| dynamic raw | **4.6219** |
| dynamic projected | 4.6721 |

因此截至 0719：

> **动态特征相关改动已经实现，但还没有稳定证据表明它能提升最终路线质量。**

### 4.3 为什么是 `equiv_set`

`ana0718.md` 中，在关闭动态特征的 `joint_edge` 路线下：

| 方法 | validation avg_cost | test mean tour length |
|---|---:|---:|
| fixed_order | 4.8358 | 4.7973 |
| equiv_set | **4.3639** | **4.4136** |

这是当前 component-merge 路线里最强、最干净的正向证据：

> **`equiv_set` 明显优于 `fixed_order`。**

### 4.4 为什么不是再加 consistency regularization

`ana0718.md` 中：

| 方法 | validation avg_cost | test mean tour length |
|---|---:|---:|
| equiv_set | **4.3639** | **4.3606** |
| equiv_set + consistency | 4.3770 | 4.3995 |

说明截至当前记录：

> **consistency regularization 的工程链路已打通，但还没有带来比纯 `equiv_set` 更好的性能。**

---

## 5. 推荐写法：当前最佳 component-merge 配置

如果要给出一个面向当前主线的“最佳配置”表述，建议写成：

```yaml
problem:
  name: tsp
  size: 20

model:
  name: component_merge
  action_mode: joint_edge
  use_dynamic_role_features: false

training:
  objective: equiv_set
  baseline: none
```

更完整地说：

- **任务**：TSP20
- **模型主干**：component_merge
- **动作空间**：`joint_edge`
- **监督目标**：`equiv_set`
- **动态角色特征**：先关闭
- **一致性正则**：先不加，或保持 `consistency_weight=0.0`

---

## 6. 与默认配置文件的关系

当前仓库默认的 `configs/component_merge/tsp20.yaml` 仍然是：

- `action_mode: joint_edge`
- `training.baseline: rollout`
- 主训练方式仍偏 REINFORCE / 默认训练流程

而从 0718/0719 的分析看，**若讨论的是“当前已经被验证在 component-merge 路线里最有效的实验配置”**，那么它并不是简单等同于默认 yaml，而是更接近：

1. 保留 `joint_edge`
2. 切到 supervised 的 `equiv_set`
3. 暂时关闭 dynamic features
4. 暂不依赖 consistency regularization 提升结果

所以：

> **默认配置文件代表当前正式主配置入口；但从已有实验证据看，component-merge 路线下最优实验选择是 `joint_edge + equiv_set + no dynamic features`。**

---

## 7. 最终研判

### 7.1 如果按“整个仓库当前最好结果”回答

最佳配置 / 模型是：

> **Attention Model，mean tour length = 3.9946（TSP20, test10000）**

### 7.2 如果按“当前 TSP-only 的 component-merge 主线最好配置”回答

最佳配置是：

> **`component_merge + joint_edge + equiv_set + 关闭动态角色特征`**

对应当前最关键证据是：

- `joint_edge` 明显优于 `tail_head`
- `equiv_set` 明显优于 `fixed_order`
- dynamic features 暂无稳定收益
- consistency regularization 暂无额外收益

因此，**我建议把“目前性能最好的配置”在当前项目主线语境下，定义为：**

> **TSP20 / component_merge / joint_edge / equiv_set / no dynamic features / no consistency regularization**

---

## 8. 一句话版本

截至 0719，**全仓库总榜最优是 Attention Model；但在当前 TSP-only 的 component-merge 主线中，最优且最有证据支撑的配置是 `joint_edge + equiv_set + 关闭动态特征`。**