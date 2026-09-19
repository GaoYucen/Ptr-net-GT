# 本轮之后的决策规则

这是基于本轮接口诊断制定的下一轮方案，不是已经获得的性能结果。

已完成登记的 AM / ICAM、route / random / learned、三个联合训练 seed，共 18 组。三个预设场景均未取得 learned ASCC 相对 adapted route 的平均收益。300 个 RL updates 和共享 warmup 只构成适配筛查，不构成充分训练；不能据此判定方法的全局上限。

## 场景选择

原生 checkpoint 的补充解码预算校准已经完成：ICAM 使用全部 N 个起点×8 个几何变换后，uniform TSP200/500 距本轮 LKH 参考分别为 0.27%/0.76%，clustered TSP200 仍为 10.62%。这支持优先研究空间分布变化，但聚类结果仅来自一个预先设定的人工分布、64 个测试实例，不能外推到全部非均匀 TSP。LKH 参考没有最优性证书；该补充评估也不是 ASCC 的同预算比较。

近期应保留 TSP200 的 uniform/clustered 对照，并以 TSP500 检查规模迁移。先不扩到 TSP1000，也不立即迁移 CVRP。扩大规模不能解决 endpoint 缺少当前分量关系的问题；CVRP 还会额外改变 depot 表示、车辆分配和容量可完成性。

如果 clustered 场景出现正信号，下一轮须预先登记多个聚类数、聚类紧度和独立新测试集，并加入相同数据/预算的 route 分布适配。不能只击败一个从未适应这种分布的模型。公开非均匀实例可作为后续外部验证，但其距离规则、参考解与采样协议需单独核对。

## P0：先检验接口是否提供足够的森林信息

本轮 7 点精确反例说明：固定 source 后，当前 endpoint 接口不能区分某些应选不同 head 的森林。原生 endpoint 只看到静态节点 embedding、该 source 所在路径的起点/tail 和合法 head mask，缺少目标 head 所在分量的另一端等信息。

更直接的数学描述是：把每条固定方向路径收缩为一个分量 i，剩余连接成本为 w(i,j)=d(tail_i,head_j)，已固定的内部长度为常数。即使原问题是对称 Euclidean TSP，这个分量间连接矩阵通常也不对称。因此模型需要识别“进入这个 head 后，会从哪个 tail 出来”。这是标准的路径收缩分析，不应包装成新的群论定理；与 DRHG 等已有片段表示的关系仍需比较。

可测试一个很小的共享状态模块：每个开放分量包含 head、tail、节点 embedding 汇总、规模；给 source 和 endpoint 都提供这些特征。endpoint 可使用 source/target 分量特征的 residual score。该模块需要验证置换等变性和单链退化情形，但不能因为用了分量表示就宣称新颖性；DRHG 等已有路径压缩表示。

最小下一轮比较：原始 route、同状态/容量增强的 route、state-aware random forest、state-aware learned forest。先固定一种 host 做有明确训练终止条件的开发，再冻结后验证第二个 host。训练预算按学习曲线预设，而不是不断跑短实验后宣布方法无效。

判定顺序：

1. 新 endpoint 是否打破反例中的输入混淆，并能在独立森林状态上改善条件决策？
2. 完整 greedy tour 是否改善？teacher NLL 改善不代替这一步。
3. learned 是否超过训练过的 random/固定/启发式顺序？
4. 是否超过同容量、同适配预算的 route 和原生强 checkpoint？
5. 扣除训练与推理开销后是否仍有实际收益？

若有正信号，再加入“只学习第一个 source，之后固定沿路径续接”的对照，检验收益是否主要来自挑选更好的起点。它不能由 trained-random 或同参数 route 对照完全替代。

反例没有证明当前 joint policy 的全局性能上限。source policy 可以避免某些混淆状态；从零构造也不必访问所有合法 partial forests。因此不能把负结果全归因于这一个接口问题。

## P1：插件的计算成本

当前原型每步为所有 source 算完整 endpoint 分布，且通过 N×N 分量等价矩阵计算路径特征。它适合检查机制，不适合直接宣称大规模高效插件。

可分开测试：维护分量的 head/tail/汇总量，以廉价 source scorer 选择 source，再只调用一次原生 endpoint。若取消原来的 head-distribution summary，必须作为方法改动重新验证，不能声称数值等价优化。增量维护状态和只计算实际需要的 queries 则可先做等价性测试。

历史 AM-style TSP50 的三 seed 摘要中，去掉 head summary 的均值为 6.0803，完整版本为 6.1026；去掉 path state 为 6.2915。这是旧实现的结果，不能转移成当前官方 host 的结论，但为“保留分量信息、检验能否去掉昂贵的全 source head 摘要”提供了消融动机。依据：[旧消融数据](RESEARCH_ROOT/audit_20260919/snapshot/groupopt-paper-canonical-repro-20260917/paper_records/formal_2026_09/ablation_am/summary/ablation_summary.json)。

只限制 source 候选顺序，且任意所选 source 的全部合法 head 仍可选，在完整图 TSP 下不必损害全部 tour 的可达性；截断 endpoint 候选可能损害可达性。这两个优化不能混为一谈。

## ICLR 投稿门槛

“插件接上两个 host”不是充分贡献。可争取的论点应同时包含：可解释的适用条件、可靠的性能收益、学习顺序相对廉价顺序的独立价值，以及可接受的计算成本。群论可用于定义 partial permutation 与可完成性，不能代替上述证据。

若只有 state-aware random 改善，应将贡献重新解释为森林状态适配，而不是 learned stabilizer-chain 的价值。若增强 route 达到相同效果，应保留这个反证。若公平适配后仍无可复现收益，暂不把当前版本作为 ICLR 主方法，保留结果并停止围绕更大节点数寻找偶然正例。
