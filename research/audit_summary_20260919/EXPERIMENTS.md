# ASCC 统一实验账本（2026-09-19）

正 Δ = Baseline − ASCC，正 Relative Δ 表示 ASCC 更好。不同协议分别成表。耗时是历史日志，未在本轮 GPU 重测。

## A. 历史正式主实验：四宿主、三训练种子

同一批 10,000 个 uniform TSP50，test seed 20260904。AM/PtrNet/GPN 10,000 steps；POMO 5,000 steps。AM batch512；PtrNet/GPN batch128；POMO batch64×8 rollouts。最终预定步 checkpoint，greedy；POMO 取 8 个起点最好值，其余单次构造。三种子均值不能拿来横向证明某 host 优于另一 host。

| Host model | Baseline | +ASCC | Absolute Δ | Relative Δ | Training cost (B→A) | Inference cost | Seed |
| --- | --- | --- | --- | --- | --- | --- | --- |
| am | 6.435026 | 6.102593 | +0.332433 | +5.166% | 48.67 → 62.94 min | 未记录独立计时 | 1234,4321,2468 |
| ptrnet | 6.741775 | 6.285078 | +0.456697 | +6.774% | 9.85 → 42.73 min | 未记录独立计时 | 1234,4321,2468 |
| gpn | 6.302548 | 6.234738 | +0.067810 | +1.076% | 10.79 → 59.23 min | 未记录独立计时 | 1234,4321,2468 |
| pomo | 6.037294 | 5.995604 | +0.041690 | +0.691% | 8.36 → 30.78 min | 未记录独立计时 | 1234,4321,2468 |


训练时间已将 seed2468 的续训 elapsed 重置分段相加；这是有日志覆盖部分的运行耗时估计，含验证且可能并跑，不能解释为独占 GPU 公平算力预算。保存了 optimizer steps 和训练实例数，但没有等 wall-clock 对照。来源：[main_tsp50_summary.json](RESEARCH_ROOT/audit_20260919/snapshot/groupopt-paper-canonical-repro-20260917/paper_records/formal_2026_09/main_tsp50/summary/main_tsp50_summary.json)、[seed_statistics.json](RESEARCH_ROOT/audit_20260919/evidence/seed_statistics.json)。


### 逐种子与实例配对 CI

| Host | Seed | Baseline | ASCC | Δ | Relative Δ | 固定模型的实例配对 95% CI | 训练日志耗时 B→A |
| --- | --- | --- | --- | --- | --- | --- | --- |
| am | 1234 | 6.429762 | 6.100638 | +0.329125 | +5.119% | [+0.322107, +0.336142] | 48.03 → 62.84 min |
| am | 4321 | 6.431001 | 6.109842 | +0.321158 | +4.994% | [+0.314215, +0.328101] | 48.63 → 63.09 min |
| am | 2468 | 6.444315 | 6.097299 | +0.347015 | +5.385% | [+0.340197, +0.353834] | 49.35 → 62.89 min |
| ptrnet | 1234 | 6.703190 | 6.268559 | +0.434631 | +6.484% | [+0.427215, +0.442047] | 9.95 → 42.90 min |
| ptrnet | 4321 | 6.856543 | 6.309464 | +0.547079 | +7.979% | [+0.539111, +0.555046] | 9.75 → 42.98 min |
| ptrnet | 2468 | 6.665592 | 6.277210 | +0.388382 | +5.827% | [+0.380745, +0.396018] | 9.85 → 42.32 min |
| gpn | 1234 | 6.298819 | 6.196022 | +0.102797 | +1.632% | [+0.096568, +0.109025] | 10.69 → 59.07 min |
| gpn | 4321 | 6.300629 | 6.261102 | +0.039527 | +0.627% | [+0.033013, +0.046042] | 10.81 → 59.30 min |
| gpn | 2468 | 6.308196 | 6.247090 | +0.061105 | +0.969% | [+0.054526, +0.067684] | 10.86 → 59.32 min |
| pomo | 1234 | 5.975256 | 6.028213 | -0.052957 | -0.886% | [-0.056563, -0.049352] | 8.29 → 30.70 min |
| pomo | 4321 | 6.055247 | 5.973662 | +0.081585 | +1.347% | [+0.077932, +0.085238] | 8.45 → 30.86 min |
| pomo | 2468 | 6.081379 | 5.984937 | +0.096442 | +1.586% | [+0.092609, +0.100275] | 8.35 → 30.77 min |


每个 CI 只估计固定训练模型对的测试实例不确定性。三个 seed 共享测试实例，不能把 pooled 30,000 pairs 当作 30,000 次独立训练。

| Host | 跨 seed Δ 的 sample SD | 跨 seed 均值差 t(2) 95% CI | 解释 |
| --- | --- | --- | --- |
| am | 0.013242 | [+0.299537, +0.365328] | 仅 3 seeds；t 区间假设强 |
| gpn | 0.032163 | [-0.012088, +0.147707] | 仅 3 seeds；t 区间假设强 |
| pomo | 0.082302 | [-0.162761, +0.246141] | 仅 3 seeds；t 区间假设强 |
| ptrnet | 0.081617 | [+0.253948, +0.659446] | 仅 3 seeds；t 区间假设强 |


GPN/POMO 的这个区间跨零；不是证明没有收益，而是不能用小得多的 pooled instance CI 代替训练随机性。已重新读取 60 份归档 costs.pt，均值与 summary 最大差 8.23e-7；另重算主表的实例配对 CI。[paired_cpu.json](RESEARCH_ROOT/audit_20260919/evidence/paired_cpu.json)。

## B. 当前服务器重新训练：AM-only 复现

| Host | Seed | Baseline | ASCC | Δ | Relative Δ | Training cost | Inference cost |
| --- | --- | --- | --- | --- | --- | --- | --- |
| AM | 1234 | 6.428387 | 6.145752 | +0.282635 | +4.397% | 51.53 → 67.26 min | 未记录 |
| AM | 2468 | 6.444782 | 6.121466 | +0.323317 | +5.017% | 52.37 → 67.42 min | 未记录 |
| AM | 4321 | 6.416319 | 6.114554 | +0.301766 | +4.703% | 50.24 → 67.61 min | 未记录 |


三种子均值 6.429830 → 6.127257，+4.706%。不能用这行替换四宿主归档表中的 AM 后仍称同一实验。[main_tsp50_summary.json](RESEARCH_ROOT/audit_20260919/snapshot/groupopt-paper-canonical-repro-results/from-scratch-main-tsp50/summary-am-only/main_tsp50_summary.json)

## C. 正式 AM capacity / information ablation

| Mode | Mean cost | Cost − Full ASCC（负值优于 full） | Seeds |
| --- | --- | --- | --- |
| native_original | 6.435026 | +0.332433 | 1234,4321,2468 |
| native_capacity_single_chain | 6.404798 | +0.302205 | 1234,4321,2468 |
| native_conditional_free | 6.102593 | +0.000000 | 1234,4321,2468 |
| native_random_tail | 7.240256 | +1.137663 | 1234,4321,2468 |
| native_free_no_head_summary | 6.080290 | -0.022303 | 1234,4321,2468 |
| native_free_no_path_state | 6.291540 | +0.188946 | 1234,4321,2468 |


Random 在训练、greedy 评估中都真实均匀抽 source；不是 uniform logits 的 argmax 假随机。不过只固定一次评估 action seed，尚未覆盖随机解码方差。native_forest_fixed 实际是“最短分量优先 + 编号打破平局”，不是固定编号顺序。正式归档只覆盖表中三项信息消融，没有 heuristic/order/frozen policy 的完整 factorial training。


证据：[ablation_summary.json](RESEARCH_ROOT/audit_20260919/snapshot/groupopt-paper-canonical-repro-20260917/paper_records/formal_2026_09/ablation_am/summary/ablation_summary.json)，[capacity_control_summary.json](RESEARCH_ROOT/audit_20260919/snapshot/groupopt-paper-canonical-repro-20260917/paper_records/formal_2026_09/capacity_am/summary/capacity_control_summary.json)。

## D. 按规模重新训练（与 zero-shot 分开）

| n | Host | Baseline | ASCC | Δ | Relative Δ | 所有 seeds 正向 |
| --- | --- | --- | --- | --- | --- | --- |
| 20 | am | 3.962789 | 3.935988 | +0.026801 | +0.676% | True |
| 20 | pomo | 3.859847 | 3.877297 | -0.017451 | -0.452% | False |
| 100 | am | 9.713014 | 9.159972 | +0.553042 | +5.694% | True |
| 100 | pomo | 9.293851 | 8.897772 | +0.396079 | +4.262% | True |


TSP50 复用 A 表。TSP200、跨问题与 zero-shot 条目在大 protocol 中出现不等于已经完成。[scale_effect_summary.json](RESEARCH_ROOT/audit_20260919/snapshot/groupopt-paper-canonical-repro-20260917/paper_records/formal_2026_09/scale_tsp/summary/scale_effect_summary.json)

## E. Modern host pilot：冻结／微调与官方 checkpoint

| Host | n / test / decoding | Baseline | ASCC | Δ | Relative Δ | Training cost | Inference cost | Seed | Role |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| LEHD v1 | 50 / 1024 / 8 starts | 5.698751 | 5.707069 | −0.008318 | −0.146% | 463.7 s（ASCC 微调） | 3.52 → 3.60 s | 20260918 | negative pilot |
| LEHD compat-v2 tail | 50 / 1024 / 8 starts | 5.698751 | 5.698751 | 0 | 0% | 220.9 s | 3.52 → 3.62 s | 20260918¹ | best step0 |
| LEHD compat-v2 forest | 同上 | 5.698751 | 5.698751 | 0 | 0% | 408.4 s | 3.52 → 4.21 s | 20260918¹ | best step0 |
| LEHD coupled-v3 | 同上 | 5.698751 | 5.698751 | 0 | 0% | 584.1 s | ASCC 5.18 s | 20260918 | best step0 |
| ICAM frozen | 50 / 1024 / 8 starts | 5.741808 | 5.741808 | ≈0 | ≈0% | 106.4 s | 1.42 → 1.29 s | 20260918 | best step0 |
| BOPO frozen head | 50 / official1000 / 50×8 | 5.694240 | 5.695661 | −0.001421 | −0.02496% | 5000 steps；见 metrics | 4.71 → 56.05 s² | 1234 | matched negative |
| BOPO head residual | 100 / official1000 / 100×8 | 7.761480 | 7.772958 | −0.011478 | −0.14788% | 1500 steps | 21.98 → 647.12 s | 1234 | matched negative |


¹ global model seed 与 data/action seeds 有分离，具体见原脚本，不能把行数视为多个独立训练种子。² 4.71 s 是 bitwise 等价 route wrapper 的 sanity 计时，两个阶段的计时环境未做本轮独占复测。BOPO TSP100 使用后来的逐实例重算 baseline，而非 summary 中四位小数 7.7615；因此 Δ 比早期 repaired summary 略有变化。冻结端点的 BOPO50 只训练 source，不能称为与 canonical joint-training 相同的方法。

- [groupopt-lehd-ascc-e4h1/summary.json](RESEARCH_ROOT/audit_20260919/snapshot/groupopt-lehd-ascc-e4h1/summary.json)

- [groupopt-lehd-ascc-compat-v2/summary.json](RESEARCH_ROOT/audit_20260919/snapshot/groupopt-lehd-ascc-compat-v2/summary.json)

- [groupopt-lehd-ascc-coupled-v3/summary.json](RESEARCH_ROOT/audit_20260919/snapshot/groupopt-lehd-ascc-coupled-v3/summary.json)

- [groupopt-icam-ascc-h2/summary.json](RESEARCH_ROOT/audit_20260919/snapshot/groupopt-icam-ascc-h2/summary.json)

- [groupopt-bopo-h2/full-ascc/seed1234-formal-v1/summary.json](RESEARCH_ROOT/audit_20260919/snapshot/groupopt-bopo-h2/full-ascc/seed1234-formal-v1/summary.json)

- [groupopt-bopo-tsp100/diagnostic-v1/paired_summary.json](RESEARCH_ROOT/audit_20260919/snapshot/groupopt-bopo-tsp100/diagnostic-v1/paired_summary.json)


BOPO50 固定模型配对 CI（Baseline−ASCC）[-0.001772,−0.001088]；BOPO100 [-0.012579,−0.010397]。都排除零且方向不利于当前 adapter，但只覆盖一个 adapter seed，不能否定从头联合训练的另一种 ASCC。

## F. Efficiency / failed runs / exploratory：不得并入主表

| Experiment | Category | Observed | Boundary |
| --- | --- | --- | --- |
| efficiency-v1 | 优化尝试 | 减少 all-source proposal 梯度存储；保留选中行反传 | 与 canonical 区分计时；有导出副本 |
| efficiency-v2 | 数值／效率探索 | 相同 checkpoint 的 greedy cost 有小变动 | 不能称逐动作严格等价 |
| efficiency-v3 | failed run / 确认 bug | checkpoint 闭包绑定错误；训练 step300 val16.1783 | 排除科学性能比较，保留失败证据 |
| efficiency-v3.1 | sanity / bug repair | 本轮 CPU 相同动作/loss，梯度与 v1 相同 | 不能替代最终多种子效果评估 |
| efficiency-v4-lite | 单 seed exploratory retraining | 6.428387→6.128951，+4.658% | 删摘要并只算选中 head；与 archived full 不混写 |
| BOPO100 gated-v3 | exploratory negative | best step0；包含历史 pre-merge continuation repair | 没有建立新的 source 收益 |
| BOPO large source/end-point E2 | counterfactual exploratory | 等候选预算下 endpoint 反事实优于 source 反事实 | 只测选定候选集、单次偏离，非完整 ASCC |
| BOPO large strict E3 | exploratory negative | learned threshold=Infinity，测试全部 abstain | 零伤害不是有效学会 selection |
| TSPLIB oracle 6 instances | exploratory / optimistic upper bound | 样本参与开发，oracle 选择能改善不等于 learned 能改善 | 不可宣称 held-out practical gain |
| official Kool AM evaluation | reference baseline | 官方预训练权重 same test seed mean5.791297 | 没有同训练协议的 +ASCC；非 H4 正证据 |
| BOPO route wrapper | sanity | 400,000 rollouts bitwise equal before ASCC | 支持接入基线一致性，不支持性能改进 |


## G. 本轮 CPU 诊断（不是新增正式实验）

完整 TSP n=2…6 共 4,641 个非终止可完成部分状态、17,538 个合法边检查，mask 与完成枚举零差异。四个 canonical host 的 source 梯度非零且有限；真实 AM checkpoint greedy source 偏离率 56.25%（16 新实例）。仅用于排除管线未接入。

| 同一 AM checkpoint 的 source intervention | 64 新实例 mean cost | Cost − learned |
| --- | --- | --- |
| learned | 6.106561 | +0.000000 |
| fixed_index | 16.618231 | +10.511670 |
| shortest_component | 19.241932 | +13.135371 |
| random | 16.453682 | +10.347122 |
| min_entropy | 9.683888 | +3.577327 |
| max_probability | 9.747701 | +3.641141 |
| min_expected_distance | 9.066748 | +2.960187 |


仅后验干预，endpoint 没有按干预 source 分布重训；巨大退化可以来自 joint-policy 分布失配，不能充当公平 heuristic baseline。详情 [mechanism_cpu.json](RESEARCH_ROOT/audit_20260919/evidence/mechanism_cpu.json)。
