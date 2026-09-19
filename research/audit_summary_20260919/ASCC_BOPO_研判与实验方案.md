# ASCC × BOPO：合作者建议研判与实验方案

2026-09-19。补充上一轮 Research Audit；本轮没有修改训练代码或启动训练。已复核服务器 BOPO trainer、ASCC50 full、ASCC100 full-v2 三个文件哈希，与审阅快照一致。

核心结论：值得测试真正的 BOPO 联合训练 ASCC；当前主要 adapter 实验并未检验这一组合。不能用未完成 forest 的边长直接替代最终目标，也不能将所有历史分支都说成限制了 source 自由度。

## 1. 四条建议分别判断

1. “按路线长度排序，间隔采样”：基本正确但不完整。官方 trainer 在完整 rollout 后，以完整 tour 成本排序，按排名间隔取 K 个，再将最优候选与其他候选配对，用目标值比率缩放偏好 loss。间隔是 rank，不是固定距离差。BOPO 是训练方法，具体 TSP 网络主要采用 POMO 架构，不是独立的 source/endpoint 预测器。
2. “路径森林可以适配”：成立。ASCC 生成完整 tour 后，具有可比较的相同目标函数。中间状态不必是连续路线。
3. “每轮 source/endpoint 都交给 BOPO，按 forest 长度采样”：联合训练合理；按未完成 forest 的当前总长度排序不可靠。应先对完整轨迹排序，再通过包含两个动作分支的 log likelihood 更新网络。逐步偏好是另一个方法，需要 completion value。
4. “现在限制换 source，应该每一步允许换”：对 gated / one-deviation 分支成立；对 full-ASCC 不完全成立。允许每步选择任何合法未决 source，合理；强制每步偏离上一 endpoint，不合理，也会排除 route-following 这个合法特例。

这里的 source 是未分配 successor 的路径尾，不是每条路线的第一个节点；endpoint 是未被用作 image 的路径头。

## 2. 现有实现与真正 BOPO 的区别

| 分支 | source | endpoint | 实际训练 |
| --- | --- | --- | --- |
| BOPO50 full-ascc | 除初始 anchor 外，每步从全部合法 source 中选择 | 冻结 host、argmax | source-only REINFORCE |
| BOPO100 full-ascc-v2 | 每步自由选择合法 source | 冻结 host 基础上加可训练 residual，训练时采样 | source+head 的 REINFORCE |
| BOPO100 gated-v3 | 先决定是否偏离 route-follow default | 冻结 head | 带 gate 的受限接入 |
| large selector | 选择一次偏离机会，部分规则最终全部 abstain | 原 host 补全 | 反事实／监督 selector 探索 |
| 建议的新实验 | 每步自由选择合法 source | 可训练，训练时采样 | 对完整 ASCC 轨迹实施 BOPO preference loss |

证据：官方 `repos/groupopt-modern-hosts/BOPO/TSP/TSPTrainer.py:166` 的 rollout 后排序与偏好 loss；`experiments/groupopt-bopo-h2/full-ascc/train_bopo_full_ascc.py:397` 和 `experiments/groupopt-bopo-tsp100/full-ascc-v2/train_bopo_tsp100_full_ascc_v2.py:456` 均为 `(advantage * grouped_ll).mean()`。路径均以服务器 `/workspace/计算群论/` 为根。

这补充了上一轮审阅：既有负结果约束的是“冻结 BOPO 权重后接 adapter”的方案，不能直接否定“BOPO 训练算法 × ASCC 联合策略”。与此同时，full 分支已有自由 source，因此不能预设只撤 gate 就会改善。

## 3. 为什么不能按 partial forest 边长给偏好标签

记已选边成本为 C(s)，未来补全成本为 R，最终目标为 C(s)+R。不同 forest 的 R 不同；即使边数相同，C(s) 的大小也不能推出最终成本优劣。尚未形成环的路径 forest 本来也不是树状分叉结构。

五节点精确反例：坐标依次为 `(0,0),(1,0),(2,0),(0,2),(2,2)`，编号0–4。

| Partial forest | 固定边 | 当前总长 | 最优 Hamiltonian completion 总长 |
| --- | --- | ---: | ---: |
| A | 1→3、2→1 | 3.236068 | 10.064495 |
| B | 3→4、4→2 | 4.000000 | 8.000000 |

两者均有两条已固定边且可完成，按当前边长会给出相反偏好。已枚举所有以0为首的 Hamiltonian tours 验证，数据见 `partial_cost_counterexample.json`。

若研究逐步偏好，应比较相同初始 prefix 的不同候选动作，用相同补全器、相同 rollout 数与随机性控制，估计最终总成本；小 n 可计算精确 completion cost。当前边长可作网络特征或下界的一部分，不足以作为真实偏好标签。下界本身也不保证两个状态的真实价值次序。

还有一个容易忽略的理论点：在完整图 TSP、completion-preserving mask 下，对任何 source u，总有某个 endpoint 保留最优 completion。因此 `min_v V*(s∪{u→v})=V*(s)` 对所有合法 u 都成立。source 的价值必须体现在有限能力的 endpoint/后续策略、探索或计算预算下，而不是 source 本身改变了最优可完成值。

## 4. 第一版建议：完整轨迹 BOPO，保持自由 source

令轨迹 τ=((u₁,v₁),…,(uₙ,vₙ))，每步先选择 source，再选择 endpoint：

\[
P_\theta(\tau\mid X)=\prod_t \pi_\theta(u_t\mid s_t)\rho_\theta(v_t\mid u_t,s_t),\qquad
\ell_\theta(\tau)=\frac1n\sum_t[\log\pi_\theta(u_t\mid s_t)+\log\rho_\theta(v_t\mid u_t,s_t)].
\]

一个联合 source–endpoint 边动作计一步；确定性／强制动作的 log probability 为0。基线 source 是确定的，可用同一定义使其项为0，不要无说明地在ASCC用2n而baseline用n平均。

每个实例生成 B 条完整可行轨迹，按最终 tour 成本排序，选 K 个排名分位，以最优轨迹 w 与其他 l 配对：

\[
\mathcal L=-\frac1{|\mathcal P|}\sum_{(w,l)\in\mathcal P}
\log\sigma\left[\frac{C_l}{C_w}(\ell_\theta(\tau_w)-\ell_\theta(\tau_l))\right].
\]

这是将官方 objective-ratio preference loss 应用到 ASCC 轨迹的建议；尚无新实验结果。应记录 source、head 各自梯度尺度，检查 loss 饱和与熵坍缩。相同成本／同一 tour 不应被强行赋予严格优劣；可按预定容差过滤这类 pair，并保留有效 pair 数。

重要区分：同一 successor permutation 可由多种 source 顺序生成。上式是轨迹概率，不是最终置换的边缘概率；后者需要对所有对应轨迹求和。论文应明确优化的是带最终 tour 成本标签的构造轨迹。

不要长期冻结一个只见过单路径的 endpoint，然后要求 source 独力适配 forest。建议同初始化、同解冻规则对比联合训练，endpoint 显式看到源路径和目标路径的两端、规模及分量状态。是否从头训练或预训练后联合微调应分成不同实验协议。

大 n 优先使用 source-first 的 Lite 设计，只为被选 source 计算 endpoint logits；path 特征需要增量维护。避免每一步所有 source×endpoint 重评分及其与 n 个起点相乘的成本。保留全部合法 endpoint 的支持；若用硬 top-k 剪枝，必须另外声明它会限制实际可达解集合。

官方 TSP 代码还有一个需要预先核查的细节：`TSPModel.py:50` 先判断 `self.training or eval_type=='softmax'`，训练时会先走该分支，后面的 hybrid 分支无法进入。若按论文采用“B−1条采样＋1条greedy”，应显式控制 greedy rollout，而不能只设置配置中的 `eval_type='hybrid'`。本轮没有改动此代码。

## 5. BOPO 在500–1000是否接近OPT

不能这样概括，尤其不适用于本项目采用的 POMO-style BOPO checkpoint。

| 论文设置 | BOPO + augmentation 报告 gap |
| --- | ---: |
| uniform TSP50，同规模训练 | 0.01% |
| uniform TSP100，同规模训练 | 0.04% |
| TSPLIB 200≤n<500，16例 | 10.41% |
| TSPLIB 500≤n<1000，6例 | 22.44% |

前两行来自Table2，后两行来自Table3；后者是泛化设置，不是500／1000同规模充分训练的结果，也不包含n恰为1000的独立估计。BOPO是训练范式，host不同可能有不同gap。

本项目 `experiments/groupopt-bopo-large/evidence-e3-tsplib-selector-v1/summary.json` 的6个574–783节点实例，baseline平均gap为22.436861%；其checkpoint、all starts×8协议与本组结果相符，不能解释为改进空间耗尽。六个实例参与过方法开发，只能作为开发／复核集，后续需要新held-out数据。

正式OPT评估应统一TSPLIB的EDGE_WEIGHT_TYPE／取整与求和口径；同一instance的模型成本和参考值必须使用同一个目标。合成大规模样本若只有LKH等启发式参考，称gap-to-reference，不能称已证OPT gap。

来源：[BOPO论文 §4与Tables2–3](https://arxiv.org/html/2503.07580v3)、[ICML正式出版页面](https://proceedings.mlr.press/v267/liao25a.html)、[官方代码](https://github.com/L-Z-7/BOPO)。

## 6. 应在哪些场景检验优势

首要科学问题是：相同模型容量、训练器和预算下，学习source是否带来独立于训练方法的增益。

先做一个2×2：route-follow / ASCC × REINFORCE / BOPO。四组共享encoder、可比较的decoder容量、相同训练数据、初始化规则、样本数和调参预算；另画等GPU时间曲线。加入最少的Random／heuristic ASCC对照。三个seed可筛查，五个以上用于正式确认。不要把ASCC+BOPO只与route-follow+REINFORCE比较。

| 优先级 | 场景 | 问题与判据 |
| --- | --- | --- |
| 1 | TSP100/200，uniform保留，系统变化cluster数、分离度、密度差、离群比例 | 检验容易局部连接与难桥接决策并存时，source选择是否有用；负结果全部保留 |
| 2 | TSP100训练→200/500/1000测试；与多尺度训练分表 | 区分size泛化、分布泛化与同规模学习；双方采用相同训练范围 |
| 3 | 新的、未参与开发的TSPLIB及合成large held-out集合 | 检验实用收益；以paired成本和质量—时间曲线为主 |
| 4 | 异构硬约束／稀疏有向问题 | 此时source合法候选数可不同，但要重新证明mask可完成性；不是当前TSP结论的直接外推 |

cluster场景是机制假说，不是已知会提升的场景。已有large E2中，等候选预算下改endpoint优于改source，这个反证必须保留；新联合训练应证明能改变该现象，不能只挑新host或新分布追逐正结果。

机制指标建议优先测：同状态下不同source的有限策略completion regret、随后长边／跨簇连接的形成、源／端点的误差传播、主动换source与最终改善的对应关系。候选endpoint数量在完整图TSP各source相同，不能用它说明“优先解决约束更强变量”。

大规模评价同时给固定trajectory数和固定时间两种预算，让baseline能把节省的计算用于更多采样或相同后处理。单次构造、multi-start、augmentation、local-search分开报告。若只胜过不擅长size泛化的BOPO-POMO，论文应定位为该类host的构造机制改进；要声称大规模竞争力，还需与适合大规模的强模型和传统求解器比较。

建议的执行顺序是：先完成100/200的正确联合训练与2×2因果对照，再扩大到500/1000；保留小规模near-OPT结果作为边界。不要直接在全量all-source dense解码器上启动大规模训练。
