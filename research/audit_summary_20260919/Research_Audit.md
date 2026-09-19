# ASCC Research Audit

审阅日期：2026-09-19。对象：4090 `/workspace` 中的 GroupOpt / ASCC 系列项目；以实际源码、逐实例结果、训练历史为依据。**本轮没有修改原始研究代码、没有重训、没有启动 GPU 实验。** 只做只读检查、有限 CPU 诊断、统计重算和目录整理。本文的“成立”有明确任务和版本边界，不等于对所有 host 或所有 permutation-constrained 问题成立。

## 1. 当前项目状态

**项目已具备可工作的、从多条有向路径中自适应选择 source 并合并路径的神经构造机制，在若干自行实现的 host 上有可靠的有限训练预算收益；尚未证明收益来自群论所特有的机制，也尚未证明对现代强 host 的通用收益或同算力优势。**

主线为 commit `81400c2d68d783cd9ccaaa3bf46909555ad272ba` 的 `groupopt-paper-canonical-repro-20260917`。前两条历史提交为 `2fe4acd`（归档正式结果）、`7438417`（论文模板）。efficiency v1/v2/v3/v3.1/v4-lite 是同一基线上的有未提交改动的 Git worktree；不是五个已冻结独立算法版本。现代 LEHD/ICAM/BOPO 是另外一组 standalone adapter/pilot，与 canonical 从头联合训练不是同一实验干预。旧 `Ptr-net-GT` 的 component merge／对称性／轨道分析和效率导出副本应保留为历史，不能拿来补当前方法缺失的实验证据。

发现并保留的关键版本差异：

| 证据层 | 实际内容 | 可用于什么 |
| --- | --- | --- |
| `paper_records/main_tsp50_v1` | 更早的短程主表，test seed20260902 | 历史探索，不能冒充正式终点 |
| `paper_records/formal_2026_09` | 四宿主三 seed、规模、AM capacity/ablation，60 份 costs.pt | 当前最完整的正式归档 |
| `groupopt-paper-canonical-repro-results` | 归档复算 + 4090 上 AM-only 重训 | 对 AM 的独立运行复现；不是四宿主全部重跑 |
| `groupopt-ascc-efficiency-*` | 效率变体、梯度错误及修复、Lite | 工程与方法变体，分别评价 |
| `groupopt-lehd-* / icam / bopo-*` | 冻结／微调现代 host | 适用边界、负结果、机制探索 |
| `.server-control/jobs` 相关记录 | 运行脚本、修复记录、状态摘要 | 查实际执行路径；GREEN 不是科学结论 |

正式归档的 24 个最终训练 checkpoint **没有随该归档保存在这台服务器的 canonical 目录**；README 明确说明保留在原 `/root/autodl-tmp/...` 服务器。这里保存了逐实例 costs，可以重算统计；本机 AM 重训 checkpoint 则实际存在，且本轮成功加载。不能把“有 cost 向量”写成“所有历史模型都已独立重新推理验证”。[正式 checkpoint 归档缺口](RESEARCH_ROOT/audit_20260919/snapshot/groupopt-paper-canonical-repro-20260917/paper_records/formal_2026_09/main_tsp50/BACKUP_README.md:35) [当前服务器 AM 复现](RESEARCH_ROOT/audit_20260919/snapshot/groupopt-paper-canonical-repro-results/from-scratch-main-tsp50/summary-am-only/main_tsp50_summary.json:1) [本轮枚举／梯度／checkpoint CPU 诊断](RESEARCH_ROOT/audit_20260919/evidence/cpu_diagnostics.json:1)

论文状态也比文件夹名保守：canonical 的 `paper/iclr2027/main.tex` 仍是模板，method/experiments 是 TODO，没有可以逐定理核对的完整 ASCC 论文正文。实质方法说明在 README、源码和早期三页思路 PDF 中。旧实验协议还写“四宿主所有 seeds 都改善”，但正式 POMO seed1234 为负，应纠正论文叙述；本轮保留原件。[论文 method 占位稿](RESEARCH_ROOT/audit_20260919/snapshot/groupopt-paper-canonical-repro-20260917/paper/iclr2027/sections/03_method.tex:1) [过时实验协议](RESEARCH_ROOT/audit_20260919/snapshot/groupopt-paper-canonical-repro-20260917/docs/EXPERIMENT_PROTOCOL.md:15) [正式主表逐种子结果](RESEARCH_ROOT/audit_20260919/snapshot/groupopt-paper-canonical-repro-20260917/paper_records/formal_2026_09/main_tsp50/summary/main_tsp50_per_seed.csv:1)

目录整理已完成到 `/workspace/计算群论/`，分类为 `repos / experiments / results / data / legacy / environments / job_records`。移动 110 个条目（含 82 个 job 目录、3 个环境及一个 AppleDouble 伴随文件）；验证 44,746 个文件元数据，其中 19,375 个源码／checkpoint 文件哈希无变化。原路径根目录没有遗留 GroupOpt/Ptr-net 项目或兼容链接。只在共享控制器原有隐藏 jobs 目录保留记录链接；Git 的 5 个定位文件与环境路径元数据已修复。11 个硬编码路径入口另存为 `portable_entrypoints`，原脚本未改。迁移后 7 个 CPU 测试通过。[目录迁移校验](RESEARCH_ROOT/audit_20260919/evidence/migration_verified.json:1) [原路径→新路径完整清单](RESEARCH_ROOT/audit_20260919/evidence/migration_plan.json:1) [迁移后 7 个 CPU 测试](RESEARCH_ROOT/audit_20260919/evidence/post_move_tests.txt:1)

## 2. 当前 ASCC 的准确技术定义

**canonical ASCC = 带可行性约束的有向 path-forest 构造 + 学习 source/tail 选择 + 给定 source 的 endpoint/head 策略。**

状态 $s_t=(X,\mathrm{succ},\mathrm{pred},\mathrm{component},t)$ 保存已固定的映射、已用 image、路径分量和已加边数。未决 source 是 `successor < 0` 的路径尾；未用 image 是 `predecessor < 0` 的路径头。未到最后一步只能连接不同分量，最后闭成一个 Hamiltonian cycle。cost 为实际选中 source/head 的欧氏边长和。[TSP 状态与完整 mask](RESEARCH_ROOT/audit_20260919/snapshot/groupopt-paper-canonical-repro-20260917/src/groupopt/problems/tsp_tensor.py:57) [successor/predecessor/component 更新](RESEARCH_ROOT/audit_20260919/snapshot/groupopt-paper-canonical-repro-20260917/src/groupopt/problems/tsp_tensor.py:186)

一步实现为

\[
q_\theta(u,v\mid s_t,h_t)=\pi_\theta(u\mid s_t,h_t,\mathrm{summary})\,
\rho_\theta(v\mid u,s_t,h_t).
\]

先为所有 source 算条件 endpoint 分布，再把归一化熵、期望边长、最大概率作为 source 特征；source 接收节点 embedding、路径均值/起点/规模、上一 head 等信息。选出 source 后从对应真实行取 endpoint；其 log probability 与 source log probability 相加进入 REINFORCE。训练采样两级动作，测试逐级 greedy；逐级 greedy 不等于在联合 $(u,v)$ 概率上取一次 argmax。[统一 rollout 与联合 log likelihood](RESEARCH_ROOT/audit_20260919/snapshot/groupopt-paper-canonical-repro-20260917/src/groupopt/framework/forest_decoder.py:84) [训练 RNG、loss 与 checkpoint](RESEARCH_ROOT/audit_20260919/snapshot/groupopt-paper-canonical-repro-20260917/experiments/train.py:135)

必须细化 README 所谓 “Forest-aware endpoint”：**canonical AM endpoint 本身没有显式输入路径均值、路径规模或候选路径的另一端**，主要输入 graph embedding、候选 source embedding 和动态 mask；路径特征显式用于 source policy。PtrNet/GPN 还使用随构造历史更新的 LSTM context；GPN 有相对坐标。POMO endpoint 保留初始 anchor embedding，并把每个候选 source 当作 last-node 查询，第一步 source 按 POMO 起点强制指定。不能宣传所有 host 已实现完整的双端 path-aware endpoint。[AM endpoint 与 source 实际输入](RESEARCH_ROOT/audit_20260919/snapshot/groupopt-paper-canonical-repro-20260917/src/groupopt/models/am.py:304) [PtrNet Forest 实现](RESEARCH_ROOT/audit_20260919/snapshot/groupopt-paper-canonical-repro-20260917/src/groupopt/models/ptrnet.py:248) [GPN Forest 实现](RESEARCH_ROOT/audit_20260919/snapshot/groupopt-paper-canonical-repro-20260917/src/groupopt/models/gpn.py:285) [POMO Forest 实现](RESEARCH_ROOT/audit_20260919/snapshot/groupopt-paper-canonical-repro-20260917/src/groupopt/models/pomo.py:287)

这既不是纯粹在输入端给节点排一次序，也不是显式计算一个非平凡群的 BSGS/Schreier–Sims。canonical `src/groupopt` 中没有稳定子生成元维护、陪集代表元乘法或轨道计算模块；它用 successor/predecessor 和路径分量实现组合状态机。数学群论解释可以严格补上，但不等于代码已经获得群算法收益。

现代分支必须另定义：BOPO50 冻结 encoder 和 endpoint，endpoint 全程 argmax，只训练 source；BOPO100 又加入 head residual；LEHD coupled 冻结 host、训练 source/pair/residual，并有强 continuation bias。它们不能共用“Full ASCC”一个名称后汇总。尤其 BOPO50 的确定性 endpoint 只保留每个 source 的一个值，实际运行 policy 的 solution support 不可能等于理论全 completion space。[BOPO50 确定性 frozen endpoint](RESEARCH_ROOT/audit_20260919/snapshot/groupopt-bopo-h2/full-ascc/train_bopo_full_ascc.py:238) [LEHD best-step 恢复路径](RESEARCH_ROOT/audit_20260919/snapshot/groupopt-lehd-ascc-coupled-v3/lehd_ascc_coupled_v3.py:191)

## 3. 数学正确性

### 3.1 可以严格成立的部分

令 $\Omega=\{1,\ldots,n\}$，部分单射 $p_t:D_t\to I_t$，任取扩展它的置换 $g_t\in S_n$。采用普通函数复合 $(gh)(x)=g(h(x))$，定义逐点稳定子

\[
H_t=\operatorname{Stab}_{S_n}(D_t)=\{h:h(u)=u,\ \forall u\in D_t\}.
\]

则所有满足已定映射的置换精确为

\[
C_t=\{\sigma\in S_n:\sigma|_{D_t}=p_t\}=g_tH_t,
\qquad F_t=C_t\cap\mathcal F(X).
\]

证明很短：若 $\sigma=g_th$，则固定域上的像等于 $g_t$；反向取 $h=g_t^{-1}\sigma$，它固定 $D_t$。这里是明确约定后的左陪集；若论文采用右作用，要同步调整乘法方向。

选 $u\notin D_t$ 是选下一层稳定子的基点，令 $H_{t+1}=\operatorname{Stab}_{H_t}(u)$。**只选 source 尚未增加映射约束，所以并未缩小 $C_t$ 或 $F_t$**；给定 endpoint $v$ 后，才选择该划分的一个子陪集，并加入 $\sigma(u)=v$。未决 source 在 $H_t$ 下的轨道是 $\Omega\setminus D_t$，经 $g_t$ 映射对应未用 images。代码的可行 mask 对这个候选集合再加问题约束。[TSP 状态与完整 mask](RESEARCH_ROOT/audit_20260919/snapshot/groupopt-paper-canonical-repro-20260917/src/groupopt/problems/tsp_tensor.py:57) [统一 rollout 与联合 log likelihood](RESEARCH_ROOT/audit_20260919/snapshot/groupopt-paper-canonical-repro-20260917/src/groupopt/framework/forest_decoder.py:84)

每一步群可以嵌套，但最后 $S_1=S_0=\{e\}$，最后强制映射不构成严格变小的子群；“每步都严格缩小 stabilizer”不准确。这里应区别 **选择划分变量** 与 **选择划分分支**。

### 3.2 Completion-preserving：对完整图 TSP 成立

一个无 premature cycle 的部分解是 $m=n-t$ 条有向路径（含单点）。每条路径可缩为一个分量；任意排列这些分量成有向环，都给出 Hamiltonian completion。因而当 $m\ge2$：

\[
|F_t|=(m-1)!,\qquad |A_t(u)|=m-1\quad\text{对每个 unresolved }u,
\]

每个合法 endpoint 分支对应 $(m-2)!$ 个 completions。$m=1$ 时仅剩唯一闭环。

代码 mask 正好排除已经使用的 image 和同分量路径头，直到最后一步。这既必要又充分。对任何目标可行 tour、任意自适应 source 次序，每次选该 tour 中此 source 的后继，该边都不会被 mask 掉，因此目标 tour 仍可达。这里的保证是**动作空间层面的可达性**，不是 greedy 推理会访问所有解，也不是每次确定 endpoint 后仍保留不满足这个 endpoint 的原解。[TSP 状态与完整 mask](RESEARCH_ROOT/audit_20260919/snapshot/groupopt-paper-canonical-repro-20260917/src/groupopt/problems/tsp_tensor.py:57)

本轮独立枚举 n=2…6 的全部 4,641 个可完成非终止部分状态，检查 17,538 个合法 edge transition：mask 与精确 completion union 零差异，completion 计数公式也零差异。这是对代码的有限验证，通用 n 的依据是以上路径压缩证明。[本轮枚举／梯度／checkpoint CPU 诊断](RESEARCH_ROOT/audit_20260919/evidence/cpu_diagnostics.json:1)

直接推论：**完整图 TSP 中，B“优先选合法 endpoint 更少的 source”和 E“选择 source 使后续可行 action/completion 数下降更快”在当前硬约束计数意义下没有区分度。** 所有 source 同样有 $m-1$ 个候选；同深度所有 forest 的可完成解数相同。若所指是“高质量 completion 的质量分布”或“神经策略的有效支持”，必须另外定义、测量，不能偷换成全部 feasible set 的 cardinality。

### 3.3 严格成立与研究贡献是两回事

| 主张 | 判断 | 原因 |
| --- | --- | --- |
| successor permutation/partial injection | 已成立 | 每个 source/image 各使用一次 |
| 部分映射集合为 coset | 已成立，论文需写明作用/复合约定 | 如上构造，代码隐式表示 |
| 动态选 source 对应选择下一 stabilizer | 已成立 | 子群索引的选择；source 单独不消去解 |
| state + endpoint 等于 coset∩feasible set | 对 canonical 完整图 TSP 成立 | 精确 mask，状态与证明一致 |
| 所有 source/refinement 都有不同剪枝强度 | 不成立（当前 TSP） | 分支数与 completion 数相同 |
| route-following 是可行构造机制的特例 | 已成立 | 限制 source 为当前路径尾即可 |
| 原官方神经模型完整参数化也是严格特例 | 基本成立但需逐 host 证明 | 还要初始 context、decoder hidden、logits、training objective 一致 |
| 任意图/任意 permutation 约束都 completion-preserving | 未成立 | 稀疏图可因缺边卡死；一般需 extension oracle |
| ASCC 已有独立的 computational group theory 算法贡献 | 证据不足 | 当前没有群算法驱动的压缩、搜索或复杂度改善 |

它与 classical CSP 的 dynamic variable ordering 在 source 选择这个层面没有已证明的本质区别；有潜力的新内容是**将变量选择与 source-conditioned neural construction、path 状态和端到端训练具体结合**。B&B 的 variable selection 影响整棵搜索树，而这里通常只生成一条 construction trajectory；也不能直接借用 B&B 的节点数减少结论。[Learning to Branch, ICML 2018](https://proceedings.mlr.press/v80/balcan18a.html)、[Gasse et al., NeurIPS 2019](https://proceedings.neurips.cc/paper/2019/hash/d14c2267d848abeb81fd590f371d39bd-Abstract.html) 已研究学习变量选择。

另须区别：稳定子链中的置换作用不是任意加权 TSP instance 的目标值对称群。重新排列 labels 时连同坐标一起变换是重标号等价；固定坐标直接改 successor 是另一个 tour，通常不同 cost。不能从 $S_n$ 的 coset 表示推出目标值对称性或安全 orbit pruning。

### 3.4 已找到的数学边界和历史不一致

- **旧 PDF 的“每条 source→image 边直接视为 transposition，所有代表元乘积即最终 successor”缺少必要的坐标搬运，按字面理解不成立。** 例如 3-cycle 的三条非自环边对应三个 transpositions，其乘积是奇置换，而 3-cycle 是偶置换。正确实现必须用累计 $g_t$ 将目标 image 拉回 $g_t^{-1}(v)$，选取固定旧 domain 的代表元，再更新 $g_t$；目前 canonical 直接写 successor，避免了这个乘积错误，但没有实现旧 PDF 的乘法算法。来源为早期 PDF 第1页 §2–3；第2页的起点/终点路径合并思想则与当前状态机一致。:codex-file-citation{path="RESEARCH_ROOT/audit_20260919/evidence/legacy_research_idea.pdf" purpose="source"}
- **CVRP extension 不保留所有合法 CVRP solutions。** 只要两条路线还能满足容量合并，程序就继续合并，没有 stop/return-depot 动作。两个客户各 demand0.4：两条 depot 单客路线合法，却不可能作为最终输出；程序必合并为一条。此反例已执行。它可能在欧氏、车辆无额外限制时保留某些最优解存在性，但那是另一个待证明命题，不能称所有 feasible solutions 可达；也不是客户集上的完整 successor permutation。此问题不影响本次正式 TSP 主表。[CVRP 强制合并终止规则](RESEARCH_ROOT/audit_20260919/snapshot/groupopt-paper-canonical-repro-20260917/src/groupopt/problems/cvrp_tensor.py:95) [v3 梯度反例与 CVRP 可达性反例](RESEARCH_ROOT/audit_20260919/evidence/variant_gradient_cpu.json:1)
- `m_cycle_cover` 规定每环至少3个节点、n≥3m，是限制每环最小长度的 cycle cover；不是允许自环/2-cycle 的一般 directed cycle cover，也不同于旧 PDF 的部分设想。当前最小测试只验证输出有效，不等于证明每个完成都仍可达；需为该精确定义补充独立穷举。[定环数 cover 的最小环长定义](RESEARCH_ROOT/audit_20260919/snapshot/groupopt-paper-canonical-repro-20260917/src/groupopt/problems/m_cycle_cover.py:29)

## 4. 实现正确性

### 已确认正常

1. canonical source 在训练、推理都进入真正的选择路径。选中 source 后用 `[batch_index, selected_tail]` 抽取对应 endpoint 分布，不是先排序节点再忘记反映到 tensor 索引。successor/predecessor 的 scatter 与分量合并一致；小规模精确检查通过。[统一 rollout 与联合 log likelihood](RESEARCH_ROOT/audit_20260919/snapshot/groupopt-paper-canonical-repro-20260917/src/groupopt/framework/forest_decoder.py:84) [successor/predecessor/component 更新](RESEARCH_ROOT/audit_20260919/snapshot/groupopt-paper-canonical-repro-20260917/src/groupopt/problems/tsp_tensor.py:186) [本轮枚举／梯度／checkpoint CPU 诊断](RESEARCH_ROOT/audit_20260919/evidence/cpu_diagnostics.json:1)
2. 四个 host 的小型 sampling/backward 测试中，source 参数梯度非零且有限。`native_head_summary(...).detach()` 切断摘要回到 endpoint 的通路，但没有切断 source log-probability 的 REINFORCE。[本轮枚举／梯度／checkpoint CPU 诊断](RESEARCH_ROOT/audit_20260919/evidence/cpu_diagnostics.json:1) [AM endpoint 与 source 实际输入](RESEARCH_ROOT/audit_20260919/snapshot/groupopt-paper-canonical-repro-20260917/src/groupopt/models/am.py:304)
3. 实际服务器 AM baseline/ASCC step10000 checkpoint 可严格加载，无 missing/unexpected keys，文件哈希不同、config mode 正确。16 个新 TSP50 的 greedy source 偏离率为 baseline0、ASCC56.25%；证明这个 checkpoint 的 source 没有被 mask/indexing 抵消。不把这16例成本当主实验。[本轮枚举／梯度／checkpoint CPU 诊断](RESEARCH_ROOT/audit_20260919/evidence/cpu_diagnostics.json:1)
4. 数据、validation、action RNG 在 canonical trainer 中分离；相同 host 同 seed 同 steps 的数据生成序列有结构保证，不会因为 ASCC 多抽一个动作而直接推进训练数据 RNG。独立 CPU test set 由独立 seed 生成。[训练 RNG、loss 与 checkpoint](RESEARCH_ROOT/audit_20260919/snapshot/groupopt-paper-canonical-repro-20260917/experiments/train.py:135) [checkpoint 加载与独立测试](RESEARCH_ROOT/audit_20260919/snapshot/groupopt-paper-canonical-repro-20260917/experiments/evaluate.py:48)
5. Random ablation 确实调用 `torch.multinomial` 选 source，再返回该动作的 one-hot distribution；测试 greedy 也是真随机选择，固定 action seed 可复现。没有“uniform logits argmax 退化为最小编号”的 bug。[Random source 实现](RESEARCH_ROOT/audit_20260919/snapshot/groupopt-paper-canonical-repro-20260917/src/groupopt/models/am.py:597)

### 明确 bug／不能用于相应结论

- **efficiency-v3 backward 状态捕获 bug。** loop 中 `state_for_tail`/`tail_mask_for_tail` 被闭包共享，延迟重算时看到别的时间步状态。本轮相同小型初始化、数据、采样随机数：v1/v3/v3.1 的 source、head、loss 完全相同；v3 gradient relative L2=5.8000，v3.1=0。v3 训练质量退化不能说明方法无效；它也不能证明“仅压缩内存完全等价”。已有 v3.1 通过闭包 factory 修复；本轮没有修改任一版本。[v3 checkpoint 闭包 bug](RESEARCH_ROOT/audit_20260919/snapshot/groupopt-ascc-efficiency-v3/src/groupopt/models/am.py:387) [v3.1 已有修复](RESEARCH_ROOT/audit_20260919/snapshot/groupopt-ascc-efficiency-v31/src/groupopt/models/am.py:390) [v3 梯度反例与 CVRP 可达性反例](RESEARCH_ROOT/audit_20260919/evidence/variant_gradient_cpu.json:1)
- **历史 BOPO gated continuation 在合并之后查原 head 分量，可能选错 continuation tail。** 已有修复明确改为 PRE-MERGE 查询。修复前 run 应保留为 failed/debug，不和修复后“one deviation”结果混用；无法恢复同一状态定义的旧 oracle 不作为正式证据。[历史 continuation 修复证据](RESEARCH_ROOT/audit_20260919/jobs_snapshot/.server-control/jobs/20260918-180400-bopo-tsp100-gated-v3-premerge-route-fix/task.sh:1)
- **随机 source 的 entropy 指标不是真实策略熵。** 抽样发生在 helper 内，输出是 one-hot，公共 entropy 会记0，实际均匀 source 熵应为 log(合法 source 数)。因为其 log-probability 与待训练参数无关，省略该常量不导致 endpoint REINFORCE 错误；但 Random 的 entropy 不能与 learned 的 entropy直接比较。[Random source 实现](RESEARCH_ROOT/audit_20260919/snapshot/groupopt-paper-canonical-repro-20260917/src/groupopt/models/am.py:597) [统一 rollout 与联合 log likelihood](RESEARCH_ROOT/audit_20260919/snapshot/groupopt-paper-canonical-repro-20260917/src/groupopt/framework/forest_decoder.py:84)
- **efficiency-v1 的 entropy 返回值 detach。** 如果将 `tail_entropy_coefficient` 设为非零，原 trainer 的 entropy bonus 在该变体不再对参数施加熵梯度。已审的正式 AM 配置默认没有非零熵正则，因此不是当前主表失效证据，却是未来开此 ablation 前必须修的语义差异。[efficiency-v1 entropy detach](RESEARCH_ROOT/audit_20260919/snapshot/groupopt-ascc-efficiency-v1/src/groupopt/models/am.py:415) [训练 RNG、loss 与 checkpoint](RESEARCH_ROOT/audit_20260919/snapshot/groupopt-paper-canonical-repro-20260917/experiments/train.py:135)

### 可疑／证据尚不完整

- **Stop-gradient 是方法定义的一部分。** endpoint summary 依赖共享 encoder/head 参数，detach 后优化使用的不是完整联合 policy 对所有参数的精确 score gradient；是刻意忽略该依赖的半梯度训练。它可能合理，但需要声明并用“有/无 stop-gradient”验证，不能写成完全端到端无偏 REINFORCE 后略去这个细节。[AM endpoint 与 source 实际输入](RESEARCH_ROOT/audit_20260919/snapshot/groupopt-paper-canonical-repro-20260917/src/groupopt/models/am.py:304)
- `validate_native_original.py` 是内部原生/固定分支比较；AM 甚至是两个名称共用同一函数。它验证接口一致性，**没有证明 AM-style 与官方 Kool AM 等价**。PtrNet/GPN 则有不同内部函数，但依然不能据此声称忠实复现外部论文的完整 encoder、训练与解码协议。POMO 有另一个官方验证脚本，应独立给出执行证据。[native_original 验证范围](RESEARCH_ROOT/audit_20260919/snapshot/groupopt-paper-canonical-repro-20260917/experiments/validate_native_original.py:16) [AM original/fixed 共用实现](RESEARCH_ROOT/audit_20260919/snapshot/groupopt-paper-canonical-repro-20260917/src/groupopt/models/am.py:219)
- canonical 评估会按参数名检查缺失和非白名单多余参数，做得较严谨；但接受独立的 `config.json`，没有强制与 checkpoint 内 config 所有实验字段完全一致，也没有保存本次解码代码 hash/测试坐标 hash。相同 shape 的语义错配仍有可能。当前已抽查 AM checkpoint 没有发现该错误，其他正式历史 checkpoint 未在本轮检索的材料中找到，未做推理复验。[checkpoint 加载与独立测试](RESEARCH_ROOT/audit_20260919/snapshot/groupopt-paper-canonical-repro-20260917/experiments/evaluate.py:48) [本轮枚举／梯度／checkpoint CPU 诊断](RESEARCH_ROOT/audit_20260919/evidence/cpu_diagnostics.json:1)
- 本轮 CPU 运行发出了 CUDA804 初始化警告；CPU 检查成功。没有测试当前 GPU 可用性或修复驱动环境，不能将历史 GPU 时延当作本轮复测性能。[迁移后 7 个 CPU 测试](RESEARCH_ROOT/audit_20260919/evidence/post_move_tests.txt:1)

**LEHD 不变结果的诊断：** coupled-v3 的初始 continuation logit=4；训练 sampling source deviation 曾达到约0.54，参数与 residual 随训练改变，到step1200验证 greedy deviation约0.03465且cost变差。代码用 strict validation improvement 选 best，最后恢复step0，故最终 deviation=0、测试cost与baseline完全一致。compat-v2和ICAM同样选step0。说明“尝试学习过，但训练/贪心/选择机制返回安全的原策略”，不是已证明模块未接入，也不是证明学到了有效 adaptive chain。下一步应比较 step0/best/final 的权重差、同一实例逐步 logits margin/source trace，而非继续给几乎相同门控模块换名字。[LEHD coupled 完整训练历史](RESEARCH_ROOT/audit_20260919/snapshot/groupopt-lehd-ascc-coupled-v3/summary.json:1) [LEHD best-step 恢复路径](RESEARCH_ROOT/audit_20260919/snapshot/groupopt-lehd-ascc-coupled-v3/lehd_ascc_coupled_v3.py:191) [LEHD continuation bias](RESEARCH_ROOT/audit_20260919/snapshot/groupopt-lehd-ascc-compat-v2/lehd_ascc_compat_v2.py:98) [ICAM 零偏离结果](RESEARCH_ROOT/audit_20260919/snapshot/groupopt-icam-ascc-h2/summary.json:1)

## 5. 当前实验真正支持的结论

完整数值、逐种子 CI、训练成本和不同协议分表见 [完整统一实验账本](RESEARCH_ROOT/audit_20260919/EXPERIMENTS.md:1)。这里列核心结果；cost 越低越好，Δ=Baseline−ASCC。

| Host | 正式 Baseline | 正式 +ASCC | Absolute Δ | Relative Δ | Training cost，B→A | Inference cost | Seed |
| --- | ---: | ---: | ---: | ---: | --- | --- | --- |
| AM-style | 6.435026 | 6.102593 | +0.332433 | +5.166% | 48.67→62.94 min | 正式主表未独立记录 | 3 |
| PtrNet | 6.741775 | 6.285078 | +0.456697 | +6.774% | 9.85→42.73 min | 同上 | 3 |
| GPN | 6.302548 | 6.234738 | +0.067810 | +1.076% | 10.79→59.23 min | 同上 | 3 |
| POMO | 6.037294 | 5.995604 | +0.041690 | +0.691% | 8.36→30.78 min | 同上 | 3 |

训练时间是从历史日志分段重建的每 seed 平均运行时间，含验证、可能共享 GPU；不是纯训练独占计时。种子1234/4321/2468，测试1万例；POMO8起点、其他single construction。当前服务器 AM-only 重训为 **6.429830→6.127257（4.706%）**，与旧归档5.166%分开。[正式主表逐种子结果](RESEARCH_ROOT/audit_20260919/snapshot/groupopt-paper-canonical-repro-20260917/paper_records/formal_2026_09/main_tsp50/summary/main_tsp50_per_seed.csv:1) [跨 seed 统计与分段耗时](RESEARCH_ROOT/audit_20260919/evidence/seed_statistics.json:1) [当前服务器 AM 复现](RESEARCH_ROOT/audit_20260919/snapshot/groupopt-paper-canonical-repro-results/from-scratch-main-tsp50/summary-am-only/main_tsp50_summary.json:1)

支持的科学结论只有以下这些：

- 在已执行的 host/训练长度/数据分布/decoding 设置下，完整 canonical construction package 的三 seed 平均收益存在；AM 和 PtrNet 更稳定，GPN 小，POMO不稳定。POMO seed1234 退化0.886%；TSP20 的POMO三个 seeds 全退化，平均约0.452%。不能描述为所有种子、所有规模均提升。[正式主表逐种子结果](RESEARCH_ROOT/audit_20260919/snapshot/groupopt-paper-canonical-repro-20260917/paper_records/formal_2026_09/main_tsp50/summary/main_tsp50_per_seed.csv:1) [按规模重训摘要](RESEARCH_ROOT/audit_20260919/snapshot/groupopt-paper-canonical-repro-20260917/paper_records/formal_2026_09/scale_tsp/summary/scale_effect_summary.json:1)
- AM 的等活跃参数 single-chain baseline均值6.404798，仍不如full6.102593，full相对它改善4.718%；两者 active参数均841,600，Original709,888。这反驳了“AM的全部收益只因参数数目增加”这个最简单解释，**不排除不同计算结构、学习动力学、上下文或算力预算效应**，也未推广到其他 host。[活跃参数验证](RESEARCH_ROOT/audit_20260919/snapshot/groupopt-paper-canonical-repro-20260917/paper_records/formal_2026_09/capacity_am/summary/parameter_validation.json:1) [AM 等活跃参数单链对照](RESEARCH_ROOT/audit_20260919/snapshot/groupopt-paper-canonical-repro-20260917/src/groupopt/models/am.py:385)
- AM 训练 Random Order 得到7.240256，明显差于learned6.102593，不支持“随机source已获得大部分收益”。去掉path-state退化到6.291540，支持该信息对此模型有用。去掉head-summary反而平均6.080290，两个 seeds 的成本低于 full；证据不支持“native proposal entropy/confidence是必要机制”。[AM 消融原始摘要](RESEARCH_ROOT/audit_20260919/snapshot/groupopt-paper-canonical-repro-20260917/paper_records/formal_2026_09/ablation_am/summary/ablation_summary.json:1)
- 按规模重新训练到TSP100时，AM/POMO平均均提升；这支持有限规模下的可训练性，不等于TSP50→100/1000 zero-shot泛化。[按规模重训摘要](RESEARCH_ROOT/audit_20260919/snapshot/groupopt-paper-canonical-repro-20260917/paper_records/formal_2026_09/scale_tsp/summary/scale_effect_summary.json:1)
- 现代host已有重要负结果：BOPO50 **5.694240→5.695661（−0.02496%）**，BOPO100 **7.761480→7.772958（−0.14788%）**；逐实例CI排除0且方向不利。BOPO50官方route wrapper在400,000个rollout成本上bitwise相等，因此简单的baseline wrapper成本错位解释得到一定排除。[BOPO50 负结果及 checkpoint 哈希](RESEARCH_ROOT/audit_20260919/snapshot/groupopt-bopo-h2/full-ascc/seed1234-formal-v1/summary.json:1) [BOPO100 逐实例配对负结果](RESEARCH_ROOT/audit_20260919/snapshot/groupopt-bopo-tsp100/diagnostic-v1/paired_summary.json:1) [BOPO bitwise wrapper sanity](RESEARCH_ROOT/audit_20260919/snapshot/groupopt-bopo-h2/route-follow-sanity/summary.json:1)

统计边界：已重读60份cost向量，主表配对统计复算；每个训练seed的instance CI可用来评估固定模型对，但不能把共享test instances的30,000个seed-instance pairs当独立重复。按3个训练seed重算t(2)区间，AM差值[0.2995,0.3653]、PtrNet[0.2539,0.6594]；GPN[−0.0121,0.1477]、POMO[−0.1628,0.2461]跨0。n=3且t假设较强，因此后两者不能声称跨训练随机性已经稳健显著。[本轮配对统计重算](RESEARCH_ROOT/audit_20260919/evidence/paired_cpu.json:1) [跨 seed 统计与分段耗时](RESEARCH_ROOT/audit_20260919/evidence/seed_statistics.json:1)

## 6. 当前不能支持的结论

| 假设 | 审阅判断 |
| --- | --- |
| H1 固定构造顺序是限制 | 有实证动机，但不是可行解表达能力不足；route-following本来就能表示所有TSP tours。有限容量、有限训练和探索偏置才是待检验的限制 |
| H2 learned source选择更合适trajectory | 确认顺序会改变、AM learned优于训练Random；尚未建立“更合适”的独立可测指标及跨host因果解释 |
| H3排除参数/action space/网络/训练预算解释 | 仅AM等活跃参数对照排除最简单的参数数量解释；wall-clock、FLOPs、source/head耦合和算力匹配仍未排除 |
| H4相对通用construction mechanism | 有多个autoregressive风格的实现可接入；没有现代强host通用正结果，也没有parallel/insertion/improvement全覆盖 |
| H5价值来自feasible-completion-space refinement | coset解释严格可写，但TSP可行空间计数不区分source；目前没有群论独特机制证据 |

还不能声称：达到最终收敛性能；达到SOTA；提升最优性gap而非仅baseline-relative cost；所有强baseline都从同起点公平重训；cross-problem理论自动成立；全程严格端到端gradient；论文主张已经完成数学证明；“oracle可改善”即“learned selector已改善”。

特别注意所谓“收敛主表”：PtrNet Original末2000步validation仍改善约0.215–0.374，固定step10000不能作为收敛证明。官方Kool AM已有同test seed预训练baseline约5.7913，比AM-style主表绝对cost低，但训练、架构、checkpoint来源不同，只能提醒baseline强度，不能直接把两者拼成公平排名。[PtrNet 最后2000步验证成本](RESEARCH_ROOT/audit_20260919/evidence/convergence_snapshot.json:1) [官方 AM 预训练测试日志](RESEARCH_ROOT/audit_20260919/snapshot/groupopt-official-am-kool-data/official_tsp50_seed20260904.log:1)

## 7. 最危险的 alternative explanation 与 mechanism 审阅

目前最危险的解释依次为：**有限训练下的学习速度／baseline较弱；不同有效计算与状态表示；source与endpoint共同适应后的耦合；只在某些host归纳偏置下有效。** 纯参数数量解释已被AM控制部分削弱，随机order解释在已训练AM中不占优势，但仍不能一并宣布其他解释排除。

归档 AM 效率 benchmark 中，original→canonical 的评估耗时为11.96→25.30秒（2.12倍），300步训练耗时51.42→108.49秒（2.11倍），峰值显存1.935→16.682 GB（8.62倍）。v1显存降至8.831 GB，仍有明显额外成本。这组短程benchmark与正式主表分开，不能混作其推理计时；但足以说明必须排除计算预算解释。本轮没有GPU重测。[AM 同轮效率 benchmark](RESEARCH_ROOT/audit_20260919/snapshot/groupopt-ascc-efficiency-v3-results/benchmark/efficiency_summary.json:1)

| 候选机制 | 现有证据与判断 | 最小证伪方式 |
| --- | --- | --- |
| A 优先高置信度source | 小型已训AM轨迹呈低entropy/高maxprob相关性，但删summary不降反升 | 同状态source选择与confidence条件相关、重训heuristic对照、反事实value |
| B 优先约束更强source | 当前完整图TSP硬mask候选数完全相等 | 先画每步候选数验证；另在非均匀约束问题检验 |
| C 推迟模糊决策 | 合理假说，尚无defer后regret降低证据 | 记录同变量等待过程的entropy与真实completion regret |
| D 改变error propagation | 有机会但无因果证据 | 固定同一prefix，强制一处source/endpoint偏离，比较后续regret与失误传播 |
| E 更快缩小feasible space | 当前TSP的cardinality解释不成立 | 改测ε-optimal completions质量/概率质量，而非合法数量 |
| F 更好partial structure | path-state消融支持信息有用；仍非结构因果解释 | 相同t比较长短分量分布、边交叉、可完成最优成本、后续regret |
| G learned variable ordering | 与代码最吻合的保守解释 | 与固定编号、Random、最短分量、entropy/regret启发式公平重训 |
| H 额外计算/参数 | capacity control削弱纯参数解释；计算明显增加 | 等FLOPs/时间与同endpoint source对照 |

本轮64个新uniform TSP50、一个真实AM checkpoint的后验干预：learned cost6.10656；最小entropy9.68389、最大prob9.74770、最小expected-distance9.06675、Random16.45368、固定编号16.61823、最短分量19.24193。learned所选source的平均entropy rank约0.218、expected-distance rank0.187、maxprob rank0.776。这表明source确实参与决策，并与confidence/距离有关；**绝不能把这些冻结endpoint后的巨大退化当作公平heuristic胜利**，因为endpoint在learned source的state分布上训练，与新source策略严重失配。训练Random的7.2403与后验Random的16.4537也正说明不能混淆两类实验。[本轮同 checkpoint source 干预](RESEARCH_ROOT/audit_20260919/evidence/mechanism_cpu.json:1) [AM 消融原始摘要](RESEARCH_ROOT/audit_20260919/snapshot/groupopt-paper-canonical-repro-20260917/paper_records/formal_2026_09/ablation_am/summary/ablation_summary.json:1)

现代BOPO轨迹更能界定边界：约10.48%的source决策不等于previous endpoint，但当previous endpoint仍合法时，主动换source仅约5.99%；其余包含必须跳到其他开放尾的情况。**偏离率不等于有效探索率，更不等于收益。** [BOPO50 source trajectory 诊断](RESEARCH_ROOT/audit_20260919/snapshot/groupopt-bopo-h2/full-ascc/source-order-diagnostic/summary.json:1)

大规模E2已有比“继续找能提升的host”更有价值的反证：在18个uniform/clustered实例、90个选定state上，匹配候选预算的endpoint改动，比source改动有更好的平均best gain；uniform source−endpoint约−0.5208，clustered约−0.2664。不能推广到全部候选空间，但目前不支持“source是更值得花预算的自由度”。后续E3 learned selector在验证约束下threshold=Infinity，全部abstain；TSPLIB6例还参与了开发，是exploratory，不是held-out。某些控制记录虽标GREEN/claim strengthened，只说明曾找到oracle机会，不说明当前学习方法有效。[等候选预算 source/endpoint 反事实比较](RESEARCH_ROOT/audit_20260919/snapshot/groupopt-bopo-large/evidence-e2-source-endpoint-v1/summary.json:1) [learned selector 全 abstain](RESEARCH_ROOT/audit_20260919/snapshot/groupopt-bopo-large/evidence-e3-structured-selector-v1/summary.json:1) [TSPLIB 开发样本与零触发](RESEARCH_ROOT/audit_20260919/snapshot/groupopt-bopo-large/evidence-e3-tsplib-selector-v1/summary.json:1) [GREEN/oracle 状态标签原记录](RESEARCH_ROOT/audit_20260919/jobs_snapshot/.server-control/jobs/20260918-233000-bopo-large-cf-round-a-terminal/result_summary.json:1)

适用边界判断：

| Host架构 | ASCC适配意义 | 目前结论 |
| --- | --- | --- |
| AR route-following / pointer / light decoder | 最直接，可复用source-conditioned scoring | canonical有效，但强checkpoint adapter未必获益 |
| Heavy decoder | 改source同时改变重编码子问题，成本高且context容易失配 | LEHD已测配置没有建立收益；不宜据此否定所有joint-training设计 |
| Heatmap/parallel/non-autoregressive | 可将ASCC作为约束解码器，但这是新decoding pipeline | 没有当前项目中的公平实证；需和已有自适应扩展decoder比 |
| Insertion | 原本就自适应选位置，不能直接等同successor逐点固定 | 插入会替换既有边，常破坏单调coset refinement定义，需另建模 |
| Improvement-based / local search | 完整解上撤销和交换约束 | 不直接属于当前单调construction；可探索局部destroy-repair，不能称原样通用 |

## 8. 最值得补的实验：按科学信息增益排序

### P0：不做就无法判断方法核心主张

1. **冻结一个方法定义和完整复现清单。** 指定canonical／Lite／frozen-host哪一个是论文方法；code+diff+checkpoint+训练config+测试坐标hash绑定。找回24个历史final checkpoint，先在小batch重放动作与成本，再完整重算。禁止继续把后验oracle、wrapper、joint-training叫同一个ASCC。
2. **真正隔离source选择的factorial实验。** 保持同一可表达任意source的endpoint架构、encoder、数据流、训练步数和调参预算；分别从头训练固定编号、route-following、uniform Random、最短分量、最小entropy／最大regret启发式、learned。Random需多个evaluation action seeds。并做 learned/frozen随机source网络和等活跃参数control。使用已训checkpoint干预只作另一个“策略依赖性”实验。至少AM和一个已经较强的host，预先固定判据：learned若不优于最强heuristic，则贡献改为construction framework而非学习ordering。
3. **收敛与等算力检验。** 原训练步数为共同起点，给baseline与ASCC相同GPU时间/样本预算两条轴，记录quality–time曲线。先看PtrNet原模型是否仍快速追赶，再决定是否重跑全host。若budget匹配后收益消失，报告样本效率/训练速度差异，撤回最终性能优势。
4. **LEHD/BOPO falsification gate。** 同一validation实例比较step0/best/final、权重变化、每步合法source、source logit margin、主动/强制偏离与endpoints；做一次强制合法换source/改source权重的可追踪干预。对于BOPO，先把等预算endpoint-only反事实作为竞争解释。只有出现足够且可预测的正completion gain，才追加训练；不能由oracle存在性直接批准大训练。
5. **理论精确定义与反例闭环。** 写完整图TSP的coset与completion-preserving证明，纠正source-only refinement/计数剪枝叙述；单独确定cycle-cover/CVRP的feasible set。对任意图或额外约束，先用小n穷举判定mask完备性。现有CVRP反例必须在论文边界中处理。

### P1：论文主实验必要

- 至少5个预定训练seeds，报告每个seed、paired instance差、跨seed不确定性；conditional instance bootstrap与training-seed统计分开。test set冻结，调参只用validation。
- 明确架构/训练范式的现代基线矩阵：优先复用现有BOPO/LEHD以形成正负边界，再加一个可与source扩展正交的强light-decoder对照；不要把换host当作寻找正结果。
- TSP50/100/200分别训练与50→100/200 transfer分表；加入clustered/非均匀分布和真正未参与开发的TSPLIB实例。训练数据、augmentation、多起点/采样/后处理预算逐项对齐。
- 同硬件、预热、CUDA同步、相同batch记录端到端时间、峰值显存、参数、模型调用数；用质量—时延曲线比较，允许baseline把ASCC增加的时间用于更多samples。
- 正式negative table必须包含POMO20、POMO50负seed、LEHD/ICAM无收益、BOPO50/100退化；明确哪些是错误实现、哪些是有效的负结果。

### P2：增强论文、解释为什么有效

- **质量敏感的completion-space分析。** 对n≤10–12用exact DP/枚举计算 $V^*(s)=\min_{\sigma\in F(s)}c(\sigma)$，比较所选edge的 $V^*(s')-V^*(s)$、保留ε-optimal completions比例、confidence calibration；避免用恒定completion cardinality讲机制。
- source entropy、主动偏离、选中path长度、最小/平均endpoint距离的分阶段图；控制step和候选数，再比较instance难度。通过相同prefix干预区分关联与因果。
- source frozen、去path-state、去last-head、去summary、stop-gradient on/off、source+original endpoint、random+ASCC endpoint的可解释矩阵；不再只删除一个信息通道就声称“adaptive refinement已证实”。
- Lite是有根据的改进方向：摘要消融未见必要性，可以source先行、只算选中endpoint，避免每步全部source×endpoint重评分；需重新训练与多seed确认，不用切换checkpoint模式的即时退化否定Lite。
- 对heavy/frozen host探索与真实残余子问题一致的path表示，以及用counterfactual advantage训练source；这是待测方法建议，本轮未实现。

### P3：可选

- 与目标函数真正相容的非平凡群作用、轨道去重、代表元计算能否降低计算；只有实现且有独立收益才提高群论贡献力度。
- insertion／improvement／diffusion host的扩展；先重写状态语义，不要求所有host必须提升。
- 动态一次resolve多个source、束搜索和多trajectory采样；这些会改变预算或支持集，应放在单独研究线。

所要求的九项ablation当前状态：Fixed编号完整重训缺；Random已做AM；heuristic已有最短分量代码但正式同预算重训结果缺；learned已做；learned+original/frozen endpoint有BOPO50但属于另一训练范式；random+canonical endpoint有AM训练组；same-parameter有AM；source frozen正式对照缺；entropy日志和BOPO轨迹诊断存在，但跨seed、分阶段、因果机制分析不足。[名为 fixed 的最短分量启发式](RESEARCH_ROOT/audit_20260919/snapshot/groupopt-paper-canonical-repro-20260917/src/groupopt/models/am.py:577) [AM 消融原始摘要](RESEARCH_ROOT/audit_20260919/snapshot/groupopt-paper-canonical-repro-20260917/paper_records/formal_2026_09/ablation_am/summary/ablation_summary.json:1) [活跃参数验证](RESEARCH_ROOT/audit_20260919/snapshot/groupopt-paper-canonical-repro-20260917/paper_records/formal_2026_09/capacity_am/summary/parameter_validation.json:1) [BOPO50 确定性 frozen endpoint](RESEARCH_ROOT/audit_20260919/snapshot/groupopt-bopo-h2/full-ascc/train_bopo_full_ascc.py:238)

## 9. 当前论文最可能成立的核心 contribution

建议采用以下保守表述：

> 我们研究神经TSP构造中的变量选择策略，将传统单路径扩展推广为在多个有向路径分量之间自适应选择下一条source–endpoint连接。该构造可用部分置换及逐点稳定子陪集表示；在完整图TSP上，所用可行性规则保证任意source选择不会排除给定部分解的可行完成。我们在多个autoregressive风格host的匹配训练步数实验中观察到收益，并通过随机source与AM容量对照研究其来源，同时刻画现代冻结host接入中的无收益和负收益边界。

其中“新的群论算法”“更快缩小feasible set”“通用SOTA提升”“不增加有效计算”不应出现。“Adaptive Stabilizer-Chain Construction”可作形式化名称，但贡献中心应落在**学习构造变量选择及其神经状态接口**。若公平heuristic实验不支持learned优越，论文应进一步收缩到路径forest construction的经验与适用性研究。

创新性还必须面对近邻：2025年的学习decoder容量工作直接针对light-decoder瓶颈，可用于检验H3；2026年的NEXCO和其引用的adaptive expansion路线已围绕有意义的partial solutions、confidence驱动变量确定和feasibility projection展开。ASCC的一次source后一次image、部分置换/coset描述与它们不完全相同，但“自适应逐步确定可行解”本身已经不足以作为新颖性主张。[ReLD，ICLR 2025](https://openreview.net/pdf?id=4pRwkYpa2u)、[NEXCO，ICLR 2026](https://proceedings.iclr.cc/paper_files/paper/2026/hash/3d681cc4487b97c08e5aa67224dd74f2-Abstract-Conference.html)。

## 10. 距 ICLR / ICML / NeurIPS 完整论文还缺什么

| 维度 | 现状 | 主要缺口 |
| --- | --- | --- |
| Theory | 完整图TSP可行性有简洁正确证明，coset表示可严格成立 | 正式写出；明确source/endpoint量词与作用约定；去掉不成立的计数机制；cycle-cover/CVRP单独处理 |
| Method | canonical机制可工作，多host接口存在 | 冻结一个最终版本；说清摘要detach、endpoint状态信息、联合训练与frozen adapter区别 |
| Experiment | 有三seed归档、cost向量、AM重训 | 强baseline、收敛/等算力、更多seed、完整checkpoint provenance |
| Mechanism | Random与path-state消融有价值，confidence必要性遭反证 | heuristic/冻结source/factorial控制、质量敏感completion measure、同状态反事实 |
| Baseline | 老host风格实现；已接BOPO/LEHD/ICAM | 不再将内部等价验证当官方复现；现代host改善尚未建立；加入直接创新近邻 |
| Scalability | TSP20/50/100重训与一些500–1000 oracle探索 | dense canonical每步O(n²)评分、总约O(n³)，成本随n扩大；需要同时间规模曲线及held-out大规模效果 |

推荐的现代对照顺序是：**现有BOPO/LEHD作为边界与反证 → ReLD作为decoder-capacity竞争解释 → adaptive expansion/NEXCO作为相关构造机制对照**。BOPO是ICML2025的训练范式，不是一个与POMO完全独立的新架构，不能仅因用了BOPO checkpoint就把host多样性加一；LEHD确为NeurIPS2023。ICAM公开记录为IEEE TITS，不应标成ICLR/ICML/NeurIPS主会baseline。[BOPO](https://proceedings.mlr.press/v267/liao25a.html)、[LEHD](https://proceedings.neurips.cc/paper_files/paper/2023/hash/1c10d0c087c14689628124bbc8fa69f6-Abstract-Conference.html)、[ICAM](https://arxiv.org/abs/2405.01906)。本轮只核验来源与适配意义，没有下载新host、没有跑新baseline。

**当前最优先的科研决策不是继续扩大正结果列表，而是用公平训练的heuristic/source对照和等算力曲线，判断收益究竟是学习变量选择、路径状态表示、还是更有利的训练与计算分配。** 在这两类P0检验完成前，现有结果适合支撑一个有边界的研究假说，尚不足以支撑强版本的通用ASCC论文。
