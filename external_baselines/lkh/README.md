# LKH-3 外部基线集成说明

本目录用于接入 `LKH-3` 作为 **中大规模 TSP 强参考/强启发式 baseline**。

## 设计定位

- 不把 LKH-3 代码直接并入 `ptrnet_gt/`
- 通过独立脚本适配当前仓库固定 `.pt` 数据集协议
- 主要面向：
  - TSP50
  - TSP100
  - 更大规模实例

## 官方链接

- <http://webhotel4.ruc.dk/~keld/research/LKH-3/>

## 注意事项

LKH-3 是高质量启发式 / strong reference solver；在正式对比中建议将其表述为：

- `strong heuristic`
- `reference solver`

不要在未额外证明时默认称为“精确最优值”。

## 预期准备

1. 下载并编译 LKH-3；
2. 获得可执行文件路径，例如：

```bash
/path/to/LKH
```

3. 用如下命令运行：

```bash
KMP_DUPLICATE_LIB_OK=TRUE /opt/conda/envs/py11/bin/python external_baselines/lkh/solve_dataset.py \
  --dataset data/tsp_uniform/tsp100_test_10000.pt \
  --lkh-executable /path/to/LKH \
  --output data/tsp_uniform/reference/tsp100_lkh.pt
```

## 输出协议

输出 `.pt` 文件，至少包含：

- `method`
- `solver`
- `dataset`
- `size`
- `num_instances`
- `costs`
- `tours`
- `total_time_sec`
- `time_per_instance_sec`
- `lkh_executable`
- `runs`
- `max_trials`

其中 `costs` 可直接作为 `scripts/evaluate_baselines.py --reference` 输入。

## 实现说明

脚本会：

1. 读取统一 `.pt` 数据集；
2. 为每个实例生成 TSPLIB `.tsp` 与 LKH `.par` 文件；
3. 调用外部 LKH 可执行文件；
4. 解析输出 tour；
5. 用仓库内浮点欧氏距离重新计算 `costs`。