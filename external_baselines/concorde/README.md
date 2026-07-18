# Concorde 外部基线集成说明

本目录用于接入 `PyConcorde` / `Concorde` 作为 **小规模 TSP reference baseline**。

## 设计定位

- 不把 Concorde 代码直接并入 `ptrnet_gt/`
- 通过独立脚本读取当前仓库统一数据集协议并导出参考结果
- 默认仅用于 **TSP50 以下**（实现上默认要求 `size < 50`）
- 对于更大规模实例，优先使用 `external_baselines/lkh/`

> 注意：根据当前项目约定，Concorde 只作为 TSP50 以下基线；当大规模太耗时时，不应默认将其作为 baseline。

## 依赖

典型 Python 封装：

- <https://github.com/jvkersch/pyconcorde>

如安装成功，可使用：

```bash
KMP_DUPLICATE_LIB_OK=TRUE /opt/conda/envs/py11/bin/python external_baselines/concorde/solve_dataset.py \
  --dataset data/tsp_uniform/tsp20_test_10000.pt \
  --output data/tsp_uniform/reference/tsp20_concorde.pt
```

## 距离定义注意事项

Concorde / TSPLIB 常见使用 `EUC_2D`，往往涉及坐标缩放与整数化；而当前仓库神经模型与评估默认采用浮点欧氏距离。

因此本目录脚本会：

1. 调用 Concorde 求解；
2. 保存 Concorde 返回的 tour；
3. 使用当前仓库 **浮点欧氏距离** 重新计算 `costs`，以便和 `scripts/evaluate_baselines.py --reference` 对齐。

## 输出协议

脚本输出 `.pt` 文件，至少包含：

- `method`
- `solver`
- `dataset`
- `size`
- `num_instances`
- `costs`
- `tours`
- `total_time_sec`
- `time_per_instance_sec`
- `notes`

其中 `costs` 字段可直接作为：

```bash
scripts/evaluate_baselines.py --reference <reference.pt>
```

## 大规模保护

默认当 `size >= 50` 时脚本会拒绝运行，并提示改用 LKH-3。

如你确实要强行运行，可显式添加：

```bash
--allow-large
```

但这不属于默认推荐协议。