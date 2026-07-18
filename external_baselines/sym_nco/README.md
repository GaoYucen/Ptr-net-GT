# Sym-NCO 外部基线集成说明

本目录用于接入 `Sym-NCO` 作为 **对称学习方法 baseline**。

## 官方资源

- 代码仓库：<https://github.com/alstn12088/Sym-NCO>
- 论文：<https://arxiv.org/abs/2205.13209>
- OpenReview：<https://openreview.net/forum?id=kHrE2vi5Rvs>

## 设计定位

- Sym-NCO 不直接并入 `ptrnet_gt/`
- 建议通过独立 clone / submodule 的方式放在仓库外部或 `external_baselines/` 邻近目录
- 本目录仅提供 **评估适配与统一输出协议**，避免绑定官方仓库某个脆弱内部 API

## 推荐用途

Sym-NCO 更适合作为：

- 第二/第三阶段 baseline
- 对称性学习方向对比
- 与 POMO、Component Merge 的对称性利用方式比较

## 适配思路

由于官方项目结构、依赖、checkpoint 格式可能随版本变化，本目录建议采用“命令适配”而非强依赖导入：

1. 你提供 `--symnco-root` 指向官方仓库；
2. 你提供可运行的评估命令模板；
3. 适配脚本读取固定测试集并调用外部命令；
4. 再把外部结果转换为当前仓库统一 JSON/`.pt` 协议。

## 统一输出目标

建议输出包含：

- `method`
- `checkpoint`
- `dataset`
- `size`
- `num_instances`
- `avg_cost`
- `std_cost`
- `feasible_tour_rate`
- `time_per_instance_sec`
- `total_time_sec`
- `raw_output`

## 示例

```bash
KMP_DUPLICATE_LIB_OK=TRUE /opt/conda/envs/py11/bin/python external_baselines/sym_nco/evaluate.py \
  --symnco-root /path/to/Sym-NCO \
  --checkpoint /path/to/checkpoint.pt \
  --dataset data/tsp_uniform/tsp20_test_10000.pt \
  --command 'python eval.py --checkpoint {checkpoint} --dataset {dataset} --output {raw_output}' \
  --output outputs/baseline_comparison/symnco_tsp20.json
```

其中：

- `{checkpoint}`
- `{dataset}`
- `{raw_output}`

会由适配脚本替换。