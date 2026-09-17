# 论文正式实验复现

当前论文使用的正式数值归档在 [`paper_records/formal_2026_09/`](../../paper_records/formal_2026_09/README.md)。本目录提供从头训练、独立评估和重新汇总这些结果所需的代码。四组已完成的实验是：

| 实验 | 从头运行入口 | 结果汇总入口 | 归档目录 |
|---|---|---|---|
| uniform TSP50 四宿主主对比 | `run_final_tsp50_multiseed.sh` | `summarize_main_tsp50.py` | `main_tsp50/` |
| TSP20/50/100 同规模重训对比 | `run_scale_tsp.sh` | `summarize_scale_tsp.py` | `scale_tsp/` |
| AM 等活跃参数单链对照 | `run_capacity_control_am_tsp50.sh` | `summarize_capacity_control_am.py` | `capacity_am/` |
| AM Random Tail / No Head Summary / No Path State 消融 | `run_ablation_am_tsp50.sh` | `summarize_ablation_am.py` | `ablation_am/` |

主对比使用 AM-style、PtrNet、GPN、POMO；每组采用训练种子 1234、4321、2468，同组 Original 与 GroupOpt 使用相同预算。独立测试集为 seed 20260904 的 10,000 个 uniform 实例。AM/PtrNet/GPN 的预定最终训练步数为 10,000，POMO 为 5,000。规模实验在 20 和 100 个节点上重新训练，50 节点直接复用主实验；TSP200 不在当前正式结果中。容量对照只新训练单链组；消融只新训练三个消融组，其余对照直接复用主实验。

## 从干净检出运行

使用 Python 3.11 和与 CUDA 驱动匹配的 PyTorch，在有 NVIDIA GPU 的环境中执行：

```bash
python3 -m pip install -e '.[ml,dev]'
bash experiments/paper/run_final_tsp50_multiseed.sh
TSP50_ROOT="$PWD/artifacts/paper_final_tsp50_v1" bash experiments/paper/run_scale_tsp.sh
bash experiments/paper/run_capacity_control_am_tsp50.sh
bash experiments/paper/run_ablation_am_tsp50.sh
```

运行器的默认输出写在 `artifacts/`，可用 `OUTPUT_ROOT` 指定新的空目录；非主实验用 `TSP50_ROOT` 或 `REFERENCE_ROOT` 指向已完成的主实验。主表运行器可选 `REFERENCE_ROOT` 导入已有最终 checkpoint；**从头复现实验时不要设置它**。各入口可用 `PYTHON_BIN` 指定解释器。`*_server.sh` 只是当时 SeetaCloud 的后台启动与 exit 状态包装，不改变科学配置，换服务器时应覆盖其默认路径。

每个入口会复制对应的冻结协议文件为 `protocol_snapshot.json`。四份已归档的快照分别与当前协议文件一致。`protocol_paper_suite_v1.json` 仍保留当时更大的计划矩阵，其中跨问题、zero-shot、效率及 TSP200 等条目**不属于已完成的四组正式结果**，不能当作已发表证据。

`artifacts/` 不进入 Git。仓库保留了逐实例 `costs.pt`、评估摘要、训练配置和指标，但**不包含模型权重**；因此可以直接复算论文统计量，却不能仅凭归档重放原 checkpoint 的推理。要重新得到 checkpoint，需按上述入口完成训练。重新训练在不同硬件、CUDA/PyTorch 版本下不承诺逐位相同，论文比较应使用相同预算和预定最终步数，而不是测试集选出的最佳 checkpoint。

## 直接复算已归档结果

无需 GPU 或 checkpoint，汇总脚本可读取 [`paper_records/formal_2026_09/`](../../paper_records/formal_2026_09/README.md) 的 `eval/**/costs.pt`。以主表为例：

```bash
python3 experiments/paper/summarize_main_tsp50.py \
  --root paper_records/formal_2026_09/main_tsp50/eval \
  --output-dir /tmp/groupopt-main-summary \
  --families am ptrnet gpn pomo \
  --seeds 1234 4321 2468 \
  --test-seed 20260904 \
  --experiment-id paper_final_tsp50_v1
```

其余三组直接复算：

```bash
python3 experiments/paper/summarize_scale_tsp.py \
  --scale-root paper_records/formal_2026_09/scale_tsp \
  --tsp50-root paper_records/formal_2026_09/main_tsp50 \
  --output-dir /tmp/groupopt-scale-summary \
  --sizes 20 50 100 --families am pomo \
  --seeds 1234 4321 2468 --test-seed 20260904
python3 experiments/paper/summarize_capacity_control_am.py \
  --capacity-root paper_records/formal_2026_09/capacity_am \
  --reference-root paper_records/formal_2026_09/main_tsp50 \
  --output-dir /tmp/groupopt-capacity-summary --test-seed 20260904
python3 experiments/paper/summarize_ablation_am.py \
  --ablation-root paper_records/formal_2026_09/ablation_am \
  --reference-root paper_records/formal_2026_09/main_tsp50 \
  --output-dir /tmp/groupopt-ablation-summary --test-seed 20260904
```

同实例配对 95% CI、胜率和逐训练种子结果均从 `costs.pt` 重新计算。不同 PyTorch/平台上的浮点归约顺序可能使 JSON 最后几位不同，应按数值容差比较，不应要求逐字节相等。
