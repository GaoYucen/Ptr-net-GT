# 官方 AM / ICAM 的 ASCC 插件验证

日期：2026-09-19。本报告是有限预算的官方预训练模型适配筛查；不是从头训练到收敛的最终论文主表。

## 本轮结论

已完成两个官方 host、三种训练机制、三个联合训练 seed，共 18 组训练与测试。这是一轮有限预算适配筛查，不是完整论文主实验。

| Uniform TSP200 | Adapted route | Learned ASCC | ASCC 相对改善 |
|---|---:|---:|---:|
| AM | 11.5579 | 14.2735 | -23.50% |
| ICAM | 10.8307 | 11.5086 | -6.26% |

**本轮没有得到 ASCC 对强 route 的平均性能优势。** 这一结论覆盖三个预设场景及 TSP200 aug8 的本轮均值，不意味着已经证明 ASCC 的全局上限。部分学习曲线仍在下降，三个 seed 也共享 warmup。

在两个 host 的 uniform TSP200 上，learned source 都优于 trained random。这说明学习顺序在当前森林模型中有作用，但它尚未补回相对原始 route 的损失。

ICAM 原始 checkpoint 使用 N 起点×aug8 后，uniform TSP200、TSP500、clustered TSP200 的参考差距分别为 0.27%、0.76%、10.62%。更值得验证的是空间结构/分布变化，而不是单纯加大节点数；这些是距可行 LKH 参考的差距，不是已证明的 OPT gap。

当前最具体的技术发现是：endpoint 接口存在不同森林共用相同输入、但最优下一动作不同的情况。下一轮优先补齐目标分量的入口/出口表示，在新的 uniform/clustered TSP200 数据上做同容量 route、random 和 learned 对照；TSP500 作为扩展测试，暂缓 CVRP。补齐状态信息是否带来净收益仍需实验，不能从反例直接推导。

[下一轮可证伪路线](RESEARCH_ROOT/strong_hosts_20260919/NEXT_STEPS.md)

## 1. 实验问题与边界

比较两种官方 host 在 TSP200 适配后，原始 route、训练过的 random-source forest、learned-source forest 的完整 tour 成本。同一份 selector 和森林状态代码接入 AM 与 ICAM，host 各自保留原生 endpoint 计算。这是需要训练的 construction adapter，不是安装后无需训练的插件。两个 host 都属于可按 source 查询 endpoint 的自回归构造模型；本轮没有证明它适用于并行、插入或 improvement-based 模型。

主场景预先定为 uniform TSP200；clustered TSP200 和 uniform TSP500 为预先指定的次级诊断。两者不能因某个结果为正而升级成事先确定的主假设。ICAM 官方训练脚本覆盖 100–500；因此不能把 TSP200/500 统一写成两个 host 的规模外推。

主训练对照使用单起点 greedy 和 8 个几何变换。它们不是 ICAM 官方完整多起点/多采样预算复现；本轮的 gap 不能用来宣称官方论文整体性能下降。

## 2. 强 host 在哪些场景仍有空间

| 场景 | LKH 参考均值 | 官方 AM | AM 相对参考差距 | 官方 ICAM | ICAM 相对参考差距 |
|---|---:|---:|---:|---:|---:|
| Uniform TSP200 | 10.6997 | 11.5814 | 8.24% | 10.8081 | 1.01% |
| Clustered TSP200 | 4.2588 | 6.6990 | 57.30% | 5.2701 | 23.75% |
| Uniform TSP500 | 16.5624 | 20.0304 | 20.94% | 16.8245 | 1.58% |

参考为 elkai 2.0.1 / LKH3、RUNS=3，距离放大 10^6 后取整求解，再用原始坐标的 float64 欧氏距离计分；**没有最优性证书，表中是 gap to reference，不是 optimality gap。** 差距按成本均值之比计算，不是逐实例百分比的均值。测试参考答案完全不参与训练。

[参考元数据](RESEARCH_ROOT/strong_hosts_20260919/evidence/reference/summary.json)

![原生模型的参考差距](RESEARCH_ROOT/strong_hosts_20260919/evidence/native_headroom.png)

## 2.1 单独校准 ICAM 的官方多起点预算

这是看到单起点 headroom 后补做的 post-hoc 评估诊断，**没有额外训练**。用原始官方 checkpoint，N 个起点，以及 N 个起点×8 个几何变换；单独列出，不能与 ASCC 单次解码混成同预算比较。

| 场景 | 单起点参考差距 | N 起点参考差距 | N 起点×aug8 参考差距 | 全部评估秒数 |
|---|---:|---:|---:|---:|
| Uniform TSP200 | 1.01% | 0.59% | 0.27% | 31.4 |
| Clustered TSP200 | 23.75% | 14.11% | 10.62% | 7.9 |
| Uniform TSP500 | 1.58% | 1.12% | 0.76% | 18.8 |

全部评估时间对应实际运行 all-starts×aug8 一次并从中提取子集结果，不能当作单独运行 N 起点的计时。单起点 helper 已与官方原生路径核对；批量形状改变可能带来浮点微差，元数据保存了与先前单起点评估的最大成本差。

[多起点原始结果](RESEARCH_ROOT/strong_hosts_20260919/evidence/icam-multistart/summary.json)

![ICAM 解码预算校准](RESEARCH_ROOT/strong_hosts_20260919/evidence/icam_budget_headroom.png)

## 3. 全部性能结果

相对改善 = (adapted route − ASCC) / adapted route；正数才是改善。表中为联合训练 seeds 1234、4321、2468 的均值，三个 seed 共用各自 host 的 warmup，所以不能当作三次完全独立的端到端训练。

Random 组的 source 随机、endpoint 贪心，动作 RNG 固定为 995731；尚未估计多个动作 seed 的方差。它的 aug8 同时含八条随机 source 轨迹，不能把全部改善归因于几何增强。

| Host / 场景 | 官方原始 | Adapted route | Trained random | Learned ASCC | 绝对改善 | 相对改善 | 已完成 seeds |
|---|---:|---:|---:|---:|---:|---:|---:|
| AM / Uniform TSP200 | 11.5814 | 11.5579 | 18.2188 | 14.2735 | -2.7156 | -23.50% | 3/3 |
| AM / Uniform TSP200 / aug8 | 11.3486 | 11.3229 | 16.5609 | 13.4042 | -2.0813 | -18.38% | 3/3 |
| AM / Clustered TSP200 | 6.6990 | 8.0337 | 17.8553 | 11.9997 | -3.9660 | -49.37% | 3/3 |
| AM / Uniform TSP500 | 20.0304 | 19.7023 | 31.7857 | 24.5729 | -4.8706 | -24.72% | 3/3 |
| ICAM / Uniform TSP200 | 10.8081 | 10.8307 | 12.6999 | 11.5086 | -0.6780 | -6.26% | 3/3 |
| ICAM / Uniform TSP200 / aug8 | 10.7518 | 10.7542 | 11.4916 | 10.9417 | -0.1875 | -1.74% | 3/3 |
| ICAM / Clustered TSP200 | 5.2701 | 5.2792 | 8.1126 | 6.4374 | -1.1582 | -21.94% | 3/3 |
| ICAM / Uniform TSP500 | 16.8245 | 16.9091 | 21.1741 | 18.3969 | -1.4878 | -8.80% | 3/3 |

[机器可读汇总及所有 seed 数值](RESEARCH_ROOT/strong_hosts_20260919/evidence/aggregated.json)

![完整结果](RESEARCH_ROOT/strong_hosts_20260919/evidence/quality.png)

## 4. 统计区间

| Host / 场景 | 每个 seed 的 ASCC 改善 (%) | 训练 seed 的 t 区间 (%) | 配对实例 bootstrap 区间 (%) |
|---|---|---|---|
| AM / Uniform TSP200 | -23.11, -17.45, -29.92 | [-39.02, -7.97] | [-24.01, -22.98] |
| AM / Uniform TSP200 / aug8 | -18.05, -12.98, -24.12 | [-32.24, -4.52] | [-18.62, -18.15] |
| AM / Clustered TSP200 | -110.33, +3.61, -45.26 | [-192.64, +91.33] | [-74.08, -27.38] |
| AM / Uniform TSP500 | -24.44, -16.12, -33.58 | [-46.41, -3.02] | [-25.53, -23.95] |
| ICAM / Uniform TSP200 | -4.69, -11.13, -2.96 | [-16.96, +4.43] | [-6.68, -5.86] |
| ICAM / Uniform TSP200 / aug8 | -1.08, -3.23, -0.92 | [-4.95, +1.46] | [-1.83, -1.66] |
| ICAM / Clustered TSP200 | -17.27, -30.99, -17.57 | [-41.41, -2.47] | [-28.60, -16.39] |
| ICAM / Uniform TSP500 | -7.93, -13.50, -4.97 | [-19.55, +1.95] | [-9.56, -8.09] |

训练 seed 区间以每 seed 相对改善为统计量，使用 df=2 的 t 分布，仅反映共享 warmup 条件下的联合训练变异；表中总体改善为均值之比。实例 bootstrap 先在 seed 间平均每个实例的配对差，再对实例抽样；没有把同一实例的三个 seed 当作独立实例。这两个区间都不是完整端到端不确定性；小样本、有限验证集与多个次级场景也限制了正式显著性主张。

![ASCC 相对改善及区间](RESEARCH_ROOT/strong_hosts_20260919/evidence/improvement.png)

## 5. 训练与推理代价

| Host | Mode | Warmup 分钟 | Joint 分钟/seed | TSP200 greedy ms/instance | TSP200 aug8 ms/instance |
|---|---|---:|---:|---:|---:|
| AM | native | — | — | 24.66 | 180.61 |
| AM | route | 1.80 | 2.76 | 21.97 | 175.06 |
| AM | random | 1.76 | 13.71 | 62.57 | 511.73 |
| AM | learned | 1.76 | 13.85 | 60.05 | 487.61 |
| ICAM | native | — | — | 16.29 | 148.68 |
| ICAM | route | 3.37 | 2.82 | 20.61 | 144.84 |
| ICAM | random | 3.19 | 9.23 | 65.30 | 521.28 |
| ICAM | learned | 3.19 | 10.25 | 60.21 | 483.21 |

额外共有 teacher 生成成本：AM 约 23.38 秒、ICAM 约 14.73 秒；不包括先前官方模型训练。warmup 成本在三个 joint seeds 间复用。表中 joint 是累计训练计算时间，不含验证、测试和 I/O。评估时间包含解码、成本计算、合法性检查与结果传回 CPU，不是纯 forward 时间。两张 GPU 均有其他任务，所以延迟为本次实测，不能当独占硬件标准 benchmark。

所有 arms 匹配数据、更新数、轨迹数，**没有匹配训练 wall time 或推理 wall time**。route 用官方 forward 计时；forest 原型每步为所有 source 算 head 分布，random 中也有不必要计算。因此本轮不能支持计算效率、时间预算下 SOTA 或参数独立贡献。

![学习曲线](RESEARCH_ROOT/strong_hosts_20260919/evidence/learning_curves.png)

![评估耗时与质量](RESEARCH_ROOT/strong_hosts_20260919/evidence/deployment.png)

## 6. 实现与数学检查

- 对两种 host，在 N=10/50/100/200 验证 route adapter 与官方原生动作完全一致，成本误差小于 10^-6。
- route/learned/random/fixed/capacity 生成合法单一 Hamiltonian tour，动作 replay 一致；checkpoint 重计算和普通反传梯度一致。
- learned source 与 host 均有非零梯度；warmup 只训练 endpoint，不给 source 错误的任意顺序标签。
- source 的 head 分布摘要按原方案 detach；这不阻断 selector 自身的策略梯度，但 host 更新忽略了经过这些摘要的导数，不能描述成所有依赖路径上的完整联合梯度。
- 发现一个不影响本轮成本/梯度的诊断指标问题：rollout 在 random/fixed/capacity 模式返回的是未被用于 source 选择的神经分布熵。训练损失没有使用它；本报告的 random 轨迹熵已按实际均匀分布 log(剩余 source 数) 重新计算。后续版本应修正该返回字段。
- CPU 穷举 N=5 的 24 个有向 tour × 120 种 source 顺序，共 2,880 条完整轨迹，全部可达。小规模检查与一般数学论证一致，不替代一般证明。

[AM 验证](RESEARCH_ROOT/strong_hosts_20260919/evidence/am-verification.json)；[ICAM 验证](RESEARCH_ROOT/strong_hosts_20260919/evidence/icam-verification.json)；[穷举结果](RESEARCH_ROOT/strong_hosts_20260919/evidence/forest-exhaustive.json)

完整图中有 m 个路径分量时，每个 tail 有 m−1 个可选 head，固定方向的 Hamiltonian completions 有 (m−1)! 个；合法 assignment 后为 (m−2)! 个。source 选择本身不删除解。群论提供部分置换/稳定子陪集的正确描述，这里没有运行 Schreier–Sims，也没有得到额外的群论剪枝。

## 7. 新发现：endpoint 的森林状态存在混淆

将每条固定方向路径收缩为分量 i，剩余 tour 的连接成本为 w(i,j)=d(tail_i,head_j)，内部长度为已固定常数。该分量间矩阵通常不对称，即使原始 TSP 的距离对称。因而目标 head 所在路径的出口信息会影响后续完成质量。这是标准路径收缩分析，不是新的群论算法贡献。

在固定的 7 点实例上，森林 A 固定 1→3、2→4；森林 B 固定 1→4、2→3。给 source=0 时，两个状态的原生 endpoint 查询起点、tail、mask、静态 embedding 全部一样。穷举所有完成 tour 得到 A 应选 0→2，B 应选 0→5；选对方动作的 regret 分别为 0.15585、0.15896。两种 host 在 A/B 的 endpoint logits 都逐元素相同。

**这确认了当前原生 endpoint 接口不是任意森林的充分状态表示。** source selector 的动态特征确实不同，它可能通过选别的 source 绕开问题，因此反例不是 joint policy 的性能上界，也没有证明这就是测试退化的主因。它支持下一步先实验“给 endpoint 提供目标分量的另一端、大小和汇总特征”，并为 route 做同容量对照。

[精确反例与坐标](RESEARCH_ROOT/strong_hosts_20260919/evidence/endpoint-aliasing.json)

![Endpoint 状态混淆反例](RESEARCH_ROOT/strong_hosts_20260919/evidence/endpoint_aliasing.png)

## 7.1 实际 source 行为

固定读取 joint 验证集的前 16 个实例、每组的 best checkpoint；以下是描述性分析，不参与模型选择。“续接偏离率”是改选刚合并路径以外的 tail 的步数比例，不是与原始 host 的完整节点顺序逐项比较。

| Host / Mode | 续接偏离率 | 前半程选中 source 的 endpoint 熵分位 | 半程时最大路径占比 | Selector 权重偏离初始化的 L2 |
|---|---:|---:|---:|---:|
| AM / random | 96.7% | 0.498 | 4.0% | 0.0000 |
| AM / learned | 53.2% | 0.124 | 29.6% | 0.9714 |
| ICAM / random | 97.1% | 0.497 | 3.8% | 0.0000 |
| ICAM / learned | 31.7% | 0.202 | 42.3% | 0.5906 |

![AM source 轨迹](RESEARCH_ROOT/strong_hosts_20260919/evidence/am-selector-traces.png)


![ICAM source 轨迹](RESEARCH_ROOT/strong_hosts_20260919/evidence/icam-selector-traces.png)

熵分位越低，表示所选 source 的 endpoint 分布在当时的候选 source 中越集中；这不等于预测更正确。更大的主路径与恢复 route-compatible 状态相符，但不能据此断言它造成了性能变化。source 权重相对同一 seed 的完整初始化流程重建后确实改变；random 的未使用 selector 则保持完全不变。

尚缺训练过的置信度启发式、source-frozen 和受控反事实实验，所以不能把这些相关性写成“学习到更优稳定子链”的因果证明。

## 8. 可复现范围与待补实验

每 host 的 teacher 使用 2,048 个训练实例和 64 个独立 teacher 验证实例。训练标签来自 native greedy + 最多 200 次 CPU 2-opt；warmup 各 2,500 updates×32 partial states，joint 各 300 updates×4 新实例×4 rollouts。joint 验证集 64，测试 uniform200 为 256，cluster200 和 uniform500 各 64。checkpoint 只按 uniform200 validation 选，包括 joint step0；测试集不参与选择。

300 RL updates 不证明收敛；warmup 损失下降也没有证明森林状态失配已被充分消除。warmup 的选择标准为 endpoint NLL，与完整 tour 质量不同；训练轨迹仍在改善时不能声称“充分训练也无效”。原生 checkpoint 另列，避免只对比被适配削弱的 route。

固定/启发式 source 的独立训练、source-frozen 与摘要 stop-gradient 消融、无 warmup 对照、完整多起点强 host 预算、同有效参数对照、等推理时间曲线、≥5 个独立端到端 seeds 均不是本轮已经完成的证据。若出现可信正收益，必须补对等有效容量和预算对照后再归因。

暂缓直接迁移 CVRP/CVRPTW：多车 depot 的表示与容量/时间窗的可完成性需要单独建模，不能照搬单一 Hamiltonian cycle 的 mask。更有信息增益的近期扩展是 TSP 中的规模和空间分布变化，同时报告匹配分布适配的 route 对照。

## 9. 文件与版本

- 训练实现冻结 commit：`f396b5c`；分析脚本另行提交，不改变已加载的训练实现。
- AM 官方仓库 commit：`c9abf41ac2f878a55b20dc7e829bc942bb999631`。
- ICAM 官方仓库 commit：`58309bc2fd2e6f3b65ee8ac83151b99edc5f9b59`。
- 远端工作树：`/workspace/计算群论/repos/groupopt-ascc-strong-hosts-20260919`。
- 远端 checkpoint / 原始结果：`/workspace/计算群论/results/ascc-strong-hosts-20260919`。
- 本地 evidence 保存元数据、逐实例成本、图表和训练历史；完整权重保留在远端。

[事先登记的协议](RESEARCH_ROOT/strong_hosts_20260919/PROTOCOL.md)；[训练脚本](RESEARCH_ROOT/strong_hosts_20260919/run_strong.py)；[适配实现](RESEARCH_ROOT/strong_hosts_20260919/strong_adapter.py)

官方来源：[AM](https://github.com/wouterkool/attention-learn-to-route)、[ICAM](https://github.com/CIAM-Group/ICAM)。ICAM 是 IEEE TITS 的方法，本报告没有把它改称 ICLR/ICML/NeurIPS 论文，也没有宣称它是当前统一预算下全领域 SOTA。


[18 组实验完整性校验：代码、数据、checkpoint 与逐实例成本](RESEARCH_ROOT/strong_hosts_20260919/evidence/evidence-integrity.json)