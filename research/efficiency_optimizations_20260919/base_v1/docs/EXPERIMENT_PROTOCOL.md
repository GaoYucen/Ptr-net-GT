# 论文实验边界与证据状态

## 主比较

论文中的效果主张固定为同一宿主内：

```text
native_original  vs  native_conditional_free
```

两组必须使用相同编码器规模、训练 seed、训练数据生成方式、优化器、训练预算、测试坐标和解码预算。Fixed、容量对照和信息删除组只用于解释机制，不能替代 Original。

## 已完成证据

短程主实验覆盖 AM-style、PtrNet、GPN、POMO，训练种子为 1234、4321、2468，测试集为 seed 20260902 生成的 10,000 个 uniform TSP50 实例。四个宿主的三个种子均为正向改善。

机器可读摘要位于 `paper_records/main_tsp50_v1/`。逐实例 costs、训练日志和 checkpoint 不进入 Git，完整本地备份位于 `artifacts/server-backup-2026-08-28/paper-main-v1/`。

严格官方 Kool AM 的 `6.3941 -> 6.3133` 是额外的外部参照，不与 AM-style 主表行混写。

## 尚未完成的论文证据

1. 延长训练直到 Original 和 GroupOpt 均接近收敛，确认提升不是早期学习速度现象。
2. 在 AM 与 POMO 上完成三种子容量匹配、Forest-Fixed 和信息消融。
3. 报告参数量、训练速度、峰值显存和推理吞吐。
4. 接入一个现代 Heavy-Decoder 宿主并先验证其 `native_original`。
5. 统一生成论文表格、学习曲线、消融图和效率—质量折中图。

在第 1 项完成以前，现有结果应表述为“跨宿主一致的短程正向证据”，不能称为最终收敛性能。
