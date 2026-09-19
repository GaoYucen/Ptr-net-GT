# Paper Final TSP50 v1 备份说明

- 备份日期：2026-09-04
- 实验标识：`paper_final_tsp50_v1`
- 服务器来源：`/root/autodl-tmp/groupopt-paper-canonical/artifacts/paper_final_tsp50_v1`
- 问题：uniform Euclidean TSP50
- 比较：`native_original` vs `native_conditional_free`
- 宿主：AM-style、PtrNet、GPN、POMO
- 训练种子：1234、4321、2468
- 测试种子：20260904
- 每个训练种子使用同一批 10,000 个独立测试实例配对评估。

## 完整性

- 24 份 `config.json`
- 24 份 `metrics.jsonl`
- 24 份评估 `summary.json`
- 24 份逐实例 `costs.pt`
- 主汇总 `summary/main_tsp50_summary.json`
- 逐种子表 `summary/main_tsp50_per_seed.csv`
- 原生基线验证 `summary/native_original_validation.json`
- 冻结协议 `protocol_snapshot.json`

本地与服务器已通过 `rsync --checksum --dry-run` 比对，无差异。

## 三种子主结果

| 宿主 | Original | GroupOpt | 相对改善 | 各种子均改善 |
|---|---:|---:|---:|---|
| AM-style | 6.4350 | 6.1026 | 5.166% | 是 |
| PtrNet | 6.7418 | 6.2851 | 6.774% | 是 |
| GPN | 6.3025 | 6.2347 | 1.076% | 是 |
| POMO | 6.0373 | 5.9956 | 0.691% | 否；seed 1234 下降 0.886% |

## Checkpoint 去向

本备份故意排除大型 checkpoint，小体积科学结果共约 2.5MB。最终 checkpoint 仍保留在上述服务器目录中：AM/PtrNet/GPN 为 step 10000，POMO 为 step 5000。在删除服务器或换机前，如果需要长期保留可继续训练的模型，还应单独归档这 24 个最终 checkpoint。
