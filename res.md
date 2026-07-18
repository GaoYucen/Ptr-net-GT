TSP 存在明显的循环起点对称性；在当前仓库的 TSP20 结果里，多起点启发式依然是非常强的参照。本轮我基于 **py11 环境** 对已有 3 轮训练产物做了复核，并补充了 baseline / 外部依赖接入状态说明。

# TSP20：3 轮训练后的模型与 Baseline 结果整理

## 结果来源与本轮复核范围

本文档本轮优先整理的是 **TSP20、固定 10000 测试集、3 轮训练预算** 下已经可确认的结果。

- 固定测试集：`data/tsp_uniform/tsp20_test_10000.pt`
- 评测环境：`/opt/conda/envs/py11/bin/python`
- 神经模型解码方式：`greedy`
- 神经模型结果目录：`outputs/ep3_comparison/`

本轮已确认可直接读取/复核的 3 个神经模型结果：

- `outputs/ep3_comparison/component_merge_tsp20_ep3_eval.json`
- `outputs/ep3_comparison/attention_model_tsp20_ep3_eval.json`
- `outputs/ep3_comparison/pointer_network_tsp20_ep3_eval.json`

其中 3 个模型的训练预算一致：

- `epochs = 3`
- `training_updates = 384`
- `training_samples = 196608`

---

## 当前已确认主结果（3 轮训练）

| Method | Mean Tour Length | Std | Feasible Rate | Time / Instance (s) | Throughput (inst/s) | Params | Training Updates | Training Samples |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Component Merge | **4.0574** | 0.3781 | 100% | 0.00020844 | 4797.53 | 625793 | 384 | 196608 |
| Attention Model | 4.0697 | 0.3690 | 100% | 0.00008988 | 11125.86 | 510720 | 384 | 196608 |
| Pointer Network | 4.3177 | 0.4223 | 100% | 0.00006839 | 14621.76 | 527872 | 384 | 196608 |

> 注：上表来自 `outputs/ep3_comparison/*.json`。本轮在 `py11` 环境中重新执行 `scripts/evaluate.py` 后，`Component Merge` 的结果与已有文件保持一致，说明当前仓库评估链路在修复后的 `py11` 环境中是可复现的。

---

## 当前排序与观察（仅基于已确认的 3 轮神经结果）

按平均路径长度从优到劣：

1. **Component Merge**: `4.0574`
2. **Attention Model**: `4.0697`
3. **Pointer Network**: `4.3177`

### 1. 3 轮预算下的阶段性结论

和之前 10 轮结果不同，在 **当前这组 3 轮训练产物** 上：

- `Component Merge` 已经 **略优于** `Attention Model`
- `Component Merge` 明显优于 `Pointer Network`

这说明如果把比较预算限制在更小训练轮数，当前 `Component Merge` 至少在这组现成实验上展现出了更好的早期结果。

### 2. 速度代价仍然存在

虽然 `Component Merge` 在这组 3 轮结果里质量最好，但其推理速度仍最慢：

- `Component Merge`: `0.00020844 s / instance`
- `Attention Model`: `0.00008988 s / instance`
- `Pointer Network`: `0.00006839 s / instance`

因此本轮结论更准确的表述应是：

> 在 3 轮训练预算下，`Component Merge` 的解质量略优，但推理效率仍落后于 AM 和 Pointer Network。

---

## 本轮 py11 环境复核情况

本轮中途发生过一次环境污染：安装 `rl4co/lightning` 时把 `py11` 里的 `torch` 从 `2.5.1+cu121` 覆盖为 `2.13.0`，导致：

```text
libtorch_cuda.so: undefined symbol: ncclCommResume
```

随后已按项目约定在 `py11` 中修复回：

- `torch==2.5.1+cu121`
- `torchvision==0.20.1+cu121`
- `torchaudio==2.5.1+cu121`

修复后已确认：

- `torch` 可正常导入
- `scripts/evaluate.py` 可正常运行
- `component_merge_tsp20_ep3` 评估可在 `py11` 下跑通

---

## 传统 baseline 与外部 baseline：本轮状态

### 1. 传统 baseline

当前我已经在 `py11` 环境中重跑：

```bash
KMP_DUPLICATE_LIB_OK=TRUE /opt/conda/envs/py11/bin/python scripts/evaluate_baselines.py \
  --dataset data/tsp_uniform/tsp20_test_10000.pt \
  --methods random,nearest_neighbor,nearest_neighbor_multistart,nn_two_opt \
  --output outputs/ep3_comparison/traditional_tsp20_eval_py11.json
```

由于 `NN + 2-opt` 在 `10000` 个实例上耗时较长，这个重跑过程一度非常慢；不过本轮最终已经在 `py11` 环境中成功完成，并得到正式结果文件：

```text
outputs/ep3_comparison/traditional_tsp20_eval_py11.json
```

本轮正式跑出的传统 baseline 数值如下：

| Traditional Baseline | Mean Tour Length | Std | Feasible Rate | Time / Instance (s) |
|---|---:|---:|---:|---:|
| Random Tour | 10.4043 | 1.2187 | 100% | 0.00000220 |
| Nearest Neighbor | 4.4933 | 0.5395 | 100% | 0.00000243 |
| Multi-start Nearest Neighbor | **4.0442** | 0.3924 | 100% | 0.00004065 |
| NN + 2-opt | 4.2017 | 0.4397 | 100% | 0.01452296 |

这些数值现在已经由本轮 `py11` 重跑正式确认，不再只是历史参考值。

结合这组传统 baseline 参考值与本页 3 轮神经结果，目前可以先看到：

- `Component Merge (4.0574)` **优于** 单起点 `Nearest Neighbor (4.4933)`
- `Component Merge (4.0574)` **优于** `NN + 2-opt (4.2017)`
- `Component Merge (4.0574)` **略逊于** `Multi-start Nearest Neighbor (4.0442)`

把这组传统 baseline 与本页 3 轮神经结果合在一起看，当前排序为：

1. **Multi-start Nearest Neighbor**: `4.0442`
2. **Component Merge**: `4.0574`
3. **Attention Model**: `4.0697`
4. **NN + 2-opt**: `4.2017`
5. **Pointer Network**: `4.3177`
6. **Nearest Neighbor**: `4.4933`
7. **Random Tour**: `10.4043`

因此，基于本轮已确认结果，可以进一步确认：

- `Component Merge` 是否超过 `Multi-start NN`
- `Component Merge` 是否超过 `NN + 2-opt`

对应答案是：

- `Component Merge` **没有超过** `Multi-start NN`
- `Component Merge` **已经超过** `NN + 2-opt`

### 2. Concorde

本轮没有实际跑通 Concorde，原因是：

- `pyconcorde` 在当前 `py11` 环境下没有可用发行版
- 因此虽然仓库中已添加 Concorde 适配脚本，但**本轮未得到可用的 Concorde reference 结果**

### 3. LKH-3

本轮已经实际跑通 LKH-3，完成情况如下：

- 已下载并编译 `LKH-3.0.14`
- 可执行文件路径：`external_baselines/lkh/upstream/LKH-3.0.14/LKH`
- 已成功通过仓库适配脚本完成 smoke 测试：

```bash
KMP_DUPLICATE_LIB_OK=TRUE /opt/conda/envs/py11/bin/python external_baselines/lkh/solve_dataset.py \
  --dataset data/tsp_uniform/tsp20_test_8_smoke.pt \
  --lkh-executable external_baselines/lkh/upstream/LKH-3.0.14/LKH \
  --output outputs/ep3_comparison/lkh_tsp20_test_8_smoke.pt
```

对应 smoke 结果：

- mean cost: `3.9233`
- time / instance: `0.01971 s`

因此当前可以确认：

> **LKH-3 准备工作已经完成，且仓库适配链路已跑通。**

本轮尚未继续跑 `tsp20_test_10000.pt` 的完整 LKH reference，只是先完成了编译与 smoke 验证。

### 4. Sym-NCO

本轮已经实际完成 Sym-NCO 官方代码获取、3 轮训练与测试评估。

#### 已完成事项

- 成功拉取官方仓库到：`external_baselines/sym_nco/upstream/`
- 补充了数据转换脚本：`external_baselines/sym_nco/convert_dataset.py`
- 补充了最小 3 轮运行包装：`external_baselines/sym_nco/run_tsp20_ep3.py`
- 将仓库 `tsp20` 测试集转换为 Sym-NCO 可读 `pkl`
- 跑通 `TSP20`、`3` 轮训练、`10000` 测试集评估

#### 本轮实际结果

结果文件：

- `outputs/ep3_comparison/sym_nco_tsp20_ep3_eval.json`

核心数值：

- `Greedy SCORE`: `4.0136`
- `Sampling SCORE`: `3.8772`

训练配置为：

- `problem_size = 20`
- `epochs = 3`
- `train_episodes = 65536`
- `train_batch_size = 512`
- `test_episodes = 10000`
- `test_batch_size = 1000`

训练日志里可确认的收敛轨迹：

- epoch 1 train score: `4.0871`
- epoch 2 train score: `3.8990`
- epoch 3 train score: `3.8824`

#### 本轮额外修复

为了让官方代码在当前环境中跑通，本轮还做了几项最小兼容修复：

- 安装缺失轻量依赖：`pytz`
- 修复 `copy_all_src()` 对不存在源码路径的拷贝失败
- 修复 `TSPTester.run()` 中 `while` 循环内过早 `return`，否则 10000 测试集只会跑首个 batch 1000 个样本

因此当前可以确认：

> **Sym-NCO 已经在当前仓库环境下完成最小 3 轮训练，并拿到 TSP20 测试结果。**

---

## 当前阶段可得出的最稳妥结论

基于 **本轮实际可确认结果**，目前最稳妥的结论是：

> 在 `TSP20`、固定 `10000` 测试集、`3` 轮训练预算下，`Sym-NCO` 本轮跑出的 `Greedy SCORE = 4.0136`、`Sampling SCORE = 3.8772`；若与当前仓库已有 `greedy` 神经结果相比，`Sym-NCO greedy` 优于 `Component Merge (4.0574)`、`Attention Model (4.0697)` 与 `Pointer Network (4.3177)`。

如果把 `Sym-NCO` 的 sampling 结果也作为参考，则其本轮数值还明显优于当前这组 3 轮 `greedy` 神经模型结果。

这个结论和先前 10 轮版本的 `res.md` 不同，因此建议后续明确区分：

- **3 轮早期训练结果**
- **10 轮统一预算结果**

否则容易把不同训练预算下的排名混在一起。

---

## 下一步建议

建议按下面顺序继续：

1. 用已编译好的 `LKH` 继续跑 `TSP20/TSP50` 完整 reference；
2. 将 `Sym-NCO` 结果正式并入 `summary.md/csv`；
3. 明确区分 `greedy` 与 `sampling` 口径，避免与当前仓库神经模型的 `greedy` 结果混淆；
4. 若要更公平对比，可补：
   - `Component Merge`
   - `Attention Model`
   - `Pointer Network`
   - `Multi-start NN`
   - `NN + 2-opt`
   - `Sym-NCO`
   - `LKH / Concorde reference gap`

---

## 备注

- 本轮严格使用了 `py11` 环境进行修复与复核。
- `pytest` 当前在 `py11` 环境里不可直接使用（`No module named pytest`），因此本轮没有完成额外测试用例复跑。
- 当前最关键的新增价值是：
  1. 外部 baseline 接入骨架已添加；
  2. `py11` 环境已从损坏状态恢复；
  3. 已完成 LKH-3 编译与 smoke 验证；
  4. 已完成 Sym-NCO 3 轮训练与 TSP20 测试评估；
  5. 已确认 3 轮训练下 `Sym-NCO greedy (4.0136)` 优于当前现成 `Component Merge (4.0574)`。
