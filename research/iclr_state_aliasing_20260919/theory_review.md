# Theory, novelty, and submission audit

Prepared 2026-09-19. This document separates proven statements, historical experiments, and untested hypotheses. It is a research review, not evidence of an accepted paper.

## 1. Immediate assessment / 中文判断

当前最可信的主线是**森林构造接口的状态混淆诊断、不可约决策代价，以及针对该障碍的受控修复**。已有结果不支持“学习构造顺序普遍提升强求解器”；也不支持“群论自动带来更优算法”。两小时可以交付可靠诊断、一个小规模受控学习实验、完整可审阅稿件及复现包；目前不能据此保证 ICLR 录用或把初步结果称作充分实证。

最重要的新颖性约束：DRHG 的式 (1)–(3) 已明确把每个片段端点的自身坐标、另一端坐标和是否片段端点标志作为输入，且访问一个端点后强制访问另一端。**给 endpoint 添加 partner tail、路径收缩、片段重连或组件表示都已有直接先例**。BQ-NCO 更早研究了保留最优策略的状态商。这里可争取的是对“把单路径 decoder 接到森林动作空间”的具体失配进行可重复、定量、跨模型的诊断，并证明这种诊断能预测和指导干预；不是发明状态充分性原理。[DRHG §3](https://arxiv.org/html/2502.16170), [BQ-NCO](https://proceedings.neurips.cc/paper_files/paper/2023/hash/f445ba15f0f05c26e1d24f908ea78d60-Abstract.html)

## 2. Verified ICLR 2027 logistics

Official guidelines are published. Abstract deadline: **2026-09-18 23:59 AoE = 2026-09-19 19:59 Asia/Shanghai**. Full paper: **2026-09-25 23:59 AoE = 2026-09-26 19:59 Asia/Shanghai**. The author list cannot add or remove authors after the abstract deadline; abstracts must describe genuine work. The initial main text is limited to nine pages; references and appendices are outside that limit. Submissions are anonymous. A separate AI-use statement is mandatory and outside the main-text limit. These facts were verified against the [official author guidelines](https://iclr.cc/Conferences/2027/AuthorGuidelines) and [call for papers](https://iclr.cc/Conferences/2027/CallForPapers).

The [official style archive](https://media.iclr.cc/Conferences/ICLR2027/iclr-2027-style-files.zip) was downloaded directly; `iclr2027_conference.sty`, `iclr2027_conference.bst`, `natbib.sty`, and `fancyhdr.sty` were extracted unchanged into `iclr_20260919/paper/`. This is the actual 2027 template, not a renamed earlier edition.

The [AI policy](https://iclr.cc/Conferences/2027/AIPolicyForAuthors) requires disclosure covering conceptual/theoretical development, mathematical claims/proofs, hypotheses, experimental methodology, implementation, and result interpretation. This project has used AI assistance in those areas and must disclose that accurately. A manuscript must not claim human checking that has not actually occurred.

## 3. Paper-ready mathematical setup

We consider a complete directed graph on a finite vertex set V with edge costs d(u,v); the Euclidean symmetric TSP is a special case. A legal partial solution F is a vertex-disjoint union of directed paths, including singleton paths. Existing edges are irrevocable: paths cannot be reversed, split, or rerouted. Let the m path components have heads h_i and tails t_i. At m > 1, an action joins t_i to h_j for i != j. At m = 1, the sole remaining action closes the path. There are no capacity, time-window, depot, or forbidden-edge constraints.

Given a selected source tail t_s, define A(F,s) as its legal target heads. Let Q_F(s,a) be the minimum **total final tour cost** among all completions preserving F and the new edge t_s -> a. Define V_F(s) = min_a Q_F(s,a). Because every source has at least one outgoing edge on an optimal completion, V_F(s) equals the unrestricted optimal completion value for any legal source, although the conditional action values can depend strongly on s. The one-decision regret is r_F(s,a) = Q_F(s,a) - V_F(s). Using future cost instead of total cost gives the same regret, since all previously fixed costs cancel.

An endpoint representation is a deterministic map z = phi(F,s,x), where x denotes the fixed problem data. A memoryless endpoint policy receives only z and private randomness independent of the latent forest conditional on z. In the audited adapters, z contains the static node embeddings, source-component head, source tail, and legal-head mask; it does not contain the correspondence between every target head and its component tail. Policies that also receive history, source-selector hidden state, or state-dependent random seeds are outside this definition.

## 4. Proposition: exact regret from aliased observations

**Proposition 1 (irreducible conditional decision regret).** Fix an observation z. Suppose forests and selected sources (F,s) are distributed according to mu_z and all have phi(F,s,x) = z and the same finite action set A_z. Then the minimum expected one-decision regret over all randomized policies measurable only with respect to z is

\[
\Delta(z) = \min_{a\in A_z} \mathbb E_{\mu_z}[Q_F(s,a)]
              - \mathbb E_{\mu_z}[\min_{b\in A_z}Q_F(s,b)]
           = \min_{a\in A_z}\mathbb E_{\mu_z}[r_F(s,a)].
\]

In particular, randomization does not lower this minimum. For a finite-support distribution with strictly positive probability on every state, Delta(z) = 0 if and only if the optimal-action sets have a nonempty intersection. Consequently, disjoint unique optimal actions imply Delta(z) > 0.

**Proof.** Let p(a|z) be any distribution over actions. Its expected regret is sum_a p(a|z) E_mu[r_F(s,a)], a convex combination of nonnegative constants. Its minimum over the probability simplex is the smallest constant, attained by a deterministic action. The minimum is zero precisely when some action has zero regret on every state with positive mass. □

**Status of contribution.** This is a standard finite-action Bayes-risk/value-of-information identity, specialized into an operational diagnostic. The principle that an abstraction can fail to preserve optimal decisions is classical; cite Li, Walsh, and Littman (2006), and BQ-NCO for its neural combinatorial optimization context. Do not advertise Proposition 1 as a new general theorem in reinforcement learning.

**Important scope.** Delta is a bound for the specified conditional state distribution and fixed observation interface. It is not a lower bound on the optimal unrestricted solver, a jointly learned source policy, or full greedy-tour regret from singleton initialization. A source policy may change visited states or encode information in its selected source; a constructed state distribution need not match on-policy occupancy. Positive regret in an exact conditional task establishes representational insufficiency without establishing how much deployment performance is lost.

**Useful corollary (numerical robustness).** If |Qhat_F(s,a) - Q_F(s,a)| <= epsilon for every evaluated state/action, then |Deltahat(z) - Delta(z)| <= 2 epsilon. Each of the two terms in the difference defining Delta changes by at most epsilon. This justifies reporting certified intervals when exact rational distance bounds are available. A heuristic completion oracle generally gives only upper bounds on Q, so its observed action disagreements do not by themselves certify positive irreducible regret.

## 5. Verified seven-point counterexample

The existing evidence file `strong_hosts_20260919/evidence/endpoint-aliasing.json` fixes source 0 and compares F_A = {1->3, 2->4} with F_B = {1->4, 2->3}. Both expose legal heads {1,2,5,6}, the same source context, and the same static node embedding input. The official-host adapters produce exactly identical endpoint probability vectors across these two states; their source selectors can distinguish the states. Exhaustive completion yields:

| Target head | Q_A | Q_B | Regret A | Regret B | Mean regret |
|---|---:|---:|---:|---:|---:|
| 1 | 3.506182239205 | 3.370771868129 | 0.383081515919 | 0.308519303171 | 0.345800409545 |
| 2 | 3.123100723285 | 3.221214019213 | 0 | 0.158961454255 | 0.079480727127 |
| 5 | 3.278947044168 | 3.062252564959 | 0.155846320882 | 0 | 0.077923160441 |
| 6 | 3.186468699578 | 3.116637249768 | 0.063367976293 | 0.054384684809 | **0.058876330551** |

For the uniform two-state distribution, the Bayes-optimal aliased action is **head 6**, not one of the statewise optimal heads. The correct lower bound is 0.058876330551. Averaging cross-optimal regrets, or taking half the smaller cross-optimal regret, is incorrect: the common compromise action performs better.

Independent verification treated the JSON decimal coordinates as exact rationals. Each Euclidean distance was enclosed between adjacent multiples of 10^-15 using integer square roots, then all 6! = 720 anchored directed tours were enumerated. This gives the rigorous interval

\[
0.058876330550917 \leq \Delta \leq 0.058876330550931.
\]

This numerical certificate concerns the decimal instance supplied in the evidence file. Its tiny distinction from the exact binary floating-point coordinates is immaterial to the displayed digits, but should not be hidden if exact arithmetic is claimed.

Reproduction with standard-library Python:

```python
import itertools, json, math
from fractions import Fraction
case = json.load(open('strong_hosts_20260919/evidence/endpoint-aliasing.json'))
x = [[Fraction(str(v)) for v in p] for p in case['coordinates']]
n, scale = len(x), 10**15
lo = [[0] * n for _ in x]
hi = [[0] * n for _ in x]
for i in range(n):
    for j in range(n):
        q = sum((a-b)**2 for a, b in zip(x[i], x[j]))
        k = math.isqrt(q.numerator * scale**2 // q.denominator)
        lo[i][j] = k
        hi[i][j] = k + (k*k*q.denominator != q.numerator*scale**2)
ql, qh = [], []
for edges in case['partial_edges']:
    lower, upper = {}, {}
    for rest in itertools.permutations(range(1, n)):
        tour = (0,) + rest
        succ = {tour[i]: tour[(i+1) % n] for i in range(n)}
        if not all(succ[s] == t for s, t in edges):
            continue
        a = succ[0]
        l = sum(lo[i][j] for i, j in succ.items())
        u = sum(hi[i][j] for i, j in succ.items())
        lower[a] = min(lower.get(a, l), l)
        upper[a] = min(upper.get(a, u), u)
    ql.append(lower)
    qh.append(upper)
actions = sorted(ql[0])
low = min(sum(q[a] for q in ql) for a in actions) - sum(min(q.values()) for q in qh)
high = min(sum(q[a] for q in qh) for a in actions) - sum(min(q.values()) for q in ql)
print(Fraction(low, 2*scale), Fraction(high, 2*scale))
```

## 6. Proposition: sufficient dynamic information

**Proposition 2 (component contraction).** Under the setup in Section 3, let C(F) be the sum of fixed internal edge costs. Define the directed component-level matrix W_ij = d(t_i,h_j) for i != j. Each Hamiltonian completion of F corresponds bijectively to a directed Hamiltonian cycle over the m components, and its cost is

\[
C(F)+\sum_{i=1}^m W_{i,\sigma(i)},
\]

where sigma is a cyclic permutation of the components. Consequently, the collection of paired component endpoints {(h_i,t_i)} together with d is sufficient to determine all future feasible merge actions, immediate costs, successor abstract states, and optimal future values. C(F) is additionally needed to reconstruct the total objective but not to choose an optimal future action.

**Proof.** Existing paths must each be traversed contiguously in their fixed direction. A completed tour therefore orders the components cyclically; conversely any such component order gives a legal completion in a complete graph. The only unfixed costs are edges between a tail and the next head. Merging i->j replaces the two pairs (h_i,t_i),(h_j,t_j) with (h_i,t_j), at cost d(t_i,h_j), leaving all other pairs unchanged. When one pair remains, closure costs d(t_i,h_i). Thus the paired endpoints determine rewards and transitions recursively. □

The contracted problem is generally asymmetric even if d(u,v)=d(v,u), because W_ij = d(t_i,h_j) need not equal W_ji = d(t_j,h_i). A direction-aware component representation is therefore natural. Internal member embeddings and component size may help an approximate learned architecture or other constraints, but they are not mathematically necessary for exact future optimization in this unconstrained fixed-direction problem.

This is a standard contraction/sufficient-statistic observation, directly related to the representations already used by DRHG. It should be a transparent justification of the intervention, not a claimed novel theorem. It is not a proof that an arbitrary neural component encoder is injective or that a local pair-scoring MLP can implement every optimal policy. To claim formal information sufficiency, the policy must have access to the paired endpoints of **all** components, or an equivalent cost-preserving encoding.

**Counting corollary.** For m>=2 fixed-direction components, there are (m-1)! directed cyclic completion orders. Fixing any legal source-target merge leaves (m-2)! completions. Every source has exactly m-1 legal target heads. Source choice alone does not shrink the set of tours reachable under unrestricted later endpoints. These statements do not establish equal cost distributions, learning difficulty, or neural accuracy among sources. They do rule out a smallest-legal-domain explanation for source selection in this particular complete-graph mask.

**Equivariance requirement.** A relabeling rho of vertices maps each pair (h_i,t_i) to (rho(h_i),rho(t_i)); actions must relabel accordingly. A shared component encoder followed by permutation-equivariant attention and shared candidate scoring satisfies this structurally, subject to correct masks and no index-specific features. In route-only states, all unvisited components are singleton pairs, so each target tail equals its head; this is a useful implementation check, not proof of numerical equivalence to an unmodified pretrained decoder.

**Single-row mask versus the full mask.** The insufficiency claim is about the selected source's legal-head mask, not every possible mask-based representation. For m>=2, the entire tail-by-head feasibility matrix determines the head-tail pairing: within the head set, the unique forbidden head in each open-tail row is that tail's own component head. Rows for closed sources are entirely forbidden; columns for non-head nodes are entirely forbidden. Thus an encoder receiving the full matrix can recover component connectivity. A source policy receiving that matrix or dynamic features can also distinguish the witness forests. This precludes applying the conditional theorem to the joint policy without a separate argument.

## 7. Related work: paper-ready text

**State abstractions and representation sufficiency.** State abstractions can accelerate decision making only when the information they remove is irrelevant to the desired policy or value guarantees (Li et al., 2006). In neural combinatorial optimization, BQ-NCO uses bisimulation quotienting to obtain reduced constructive MDPs while preserving an optimal-solution guarantee (Drakulic et al., 2023). Our question is complementary: when a decoder interface designed for a single path is reused under a more flexible construction process, can it merge states requiring different endpoint decisions? We instantiate the standard conditional decision-risk identity with exhaustive TSP completions and test the resulting diagnosis through controlled representation interventions.

**Fragment and insertion representations.** Building a solution through multiple partial paths is a classical TSP construction strategy; it is not itself a new group-theoretic algorithm. DRHG reduces retained path segments to paired endpoint features and learns to reconnect them during destroy-and-repair search (Li et al., 2025). L2C-Insert learns insertion locations for constructive routing rather than restricting every action to path appending (Luo et al., 2025). Learning to Segment identifies stable portions of routes and aggregates them to accelerate iterative solvers (Ouyang et al., 2026). We do not claim endpoint pairing, contraction, or flexible construction as inventions. Our experiments isolate the decision information lost when route-trained endpoint interfaces are exposed to directed forest states.

A primary reference for classical multi-fragment construction is [Mamano et al. (2019)](https://arxiv.org/abs/1902.06875), which gives geometric algorithms for the greedy multi-fragment TSP tour. Its presence does not substitute for a direct experimental comparator if a new solver advantage is claimed.

**Order and equivalent trajectories.** Learning which variable to assign next predates neural routing and has been studied for constraint-satisfaction search (Song et al., 2021/2022; cite the verified version used). Order-invariant reinforcement learning trains autoregressive black-box optimizers under randomized variable orders (Goudet et al., 2026). MACSIM combines joint multi-agent assignments with set-based supervision to exploit action-sequence symmetries (Luttmann and Xie, 2026). These approaches concern learned order, randomized factorization, or equivalent action histories; none makes missing component connectivity available automatically. Conversely, repairing a state representation does not establish a benefit from learning source order. Our source-order comparisons therefore require identical endpoint information and matched training and inference budgets.

**Backbone scope.** The Attention Model (Kool et al., 2019) and ICAM (Zhou et al., 2026) are route-construction backbones. The failure claim in this study concerns our forest adapters' observation interface, not the correctness of either native solver. ICAM's current paper title is *Instance-Conditioned Adaptation for Large-scale Generalization of Neural Routing Solver* (IEEE T-ITS, DOI 10.1109/TITS.2026.3674538), updated from its older preprint title.

## 8. Novelty and reviewer risk / 中文审稿判断

| 可能写进论文的主张 | 当前状态 | 推荐写法 |
|---|---|---|
| endpoint 接口能混淆最优动作不同的森林 | 已有具体反例，独立枚举与区间算术复核 | 陈述接口限定的representational counterexample |
| 混淆导致正的不可约条件regret | 标准决策论应用，数值可证 | 把贡献放在可操作诊断和实证，不包装基础定理 |
| 两端配对表示可以修复信息缺失 | 形式上充分；DRHG已有直接表示先例 | 作为机制干预和设计原则，并明确引用 |
| 修复提升完整tour质量 | 必须等待真实实验；条件NLL不够 | 报告正负结果，分别呈现条件任务与部署任务 |
| 学习source有独立价值 | 现有强host筛查为负 | 需要同表示、同训练的random/fixed/heuristic/route比较 |
| 群论带来新算法优势 | 未建立 | 仅作为partial permutation的解释背景，避免标题主张 |
| 能达到SOTA或保证中稿 | 不成立 | 不作保证，不选择性丢掉失败结果 |

最高审稿风险是“人为删掉DRHG已知的重要信息，然后展示补回去有用”。缓解路径：证明该接口迁移确实出现在一个有代表性的研究问题；给出跨host、跨分布/规模的预先指定诊断；展示表示regret能够预测实际修复收益；加入信息相同的廉价对照；保留不提升的结果。仅一个手造7点例子和一个小MLP分类实验，更接近机制说明或工作坊成果，通常不足以支撑强ICLR主会结论。

第二风险是有限算力筛查被误读成理论上限。历史18组强host训练只有300 RL updates及共享warmup，适合立项判断，不能声称所有learned-order策略都失败。历史结果与本轮新实验必须在正文中明确区分开发数据和确认数据。

第三风险是伪重复、数据泄漏和自适应筛选。一个geometry的多个森林、多个等价置换、两个alias pair成员高度相关。训练/验证/测试必须按geometry分组；统计区间应按geometry或训练seed聚合。不能把在test上选出的“最难pair”再当作未经筛选的总体发生率。

## 9. Experimental design criteria

1. Separate a **constructed stress test** from **on-policy prevalence**. In the stress test, hold coordinates/source/mask fixed and vary only hidden head-tail pairing. Report disagreement frequency and the exact Bayes regret over all pairings in the registered family; retain zero-regret families. An on-policy claim needs independently sampled actual rollout states, not only adversarial pairings.
2. Split by coordinate instance before deriving forests, augmentations, or pairings. All states from a coordinate instance belong to one split. A random per-state split leaks the geometry and encourages memorizing a finite pairing family.
3. Exact completion labels enable a clean conditional task: report mean regret, not only top-1 accuracy or NLL. Use the same coordinate encoder, model size, training data, seed, optimizer steps, checkpoint-selection rule, and objective for blind and paired-endpoint models. A larger model cannot establish the source of improvement by itself.
4. Include a wrong/shuffled-pairing intervention. Keep per-state endpoint marginals and extra feature dimensions the same while scrambling their correspondence. If paired features help and scrambled ones do not, this supports the role of connectivity. A test-time corruption alone is also a distribution shift; label it as a sensitivity intervention, and prefer a separately trained control when feasible.
5. Include a cheap component-aware heuristic or residual scorer to reveal whether most benefit is from direct exit geometry rather than learned deep reasoning. Any deterministic score used to fabricate labels must not be called an exact oracle.
6. For full construction, compare endpoint-blind and endpoint-aware versions under the same fixed or random source first. Only then compare learned source with trained random, fixed, heuristic, and information/capacity-matched route. Native route is an external anchor, not the only fair ablation.
7. Report independent seeds and per-instance paired differences, hardware, wall time, exact sample/update counts, parameter counts, start/augmentation counts, teacher-generation cost, and failed or unfinished runs. Never turn missing results into zeros. Multiple training seeds quantify training variability; many test instances do not substitute for them.
8. Do not transfer a fixed-source conditional lower bound into a full-policy optimality guarantee. If claiming the repair explains a deployment gain, quantify where aliased families overlap actual visited states and whether measured diagnosis correlates with paired intervention benefit.

## 10. Accurate draft AI-use disclosure

Generative AI tools assisted with developing the state-aliasing formulation and proof drafts, designing diagnostic experiments, implementing and checking experiment and analysis code, surveying related work, interpreting results, and drafting the manuscript and figures. The accompanying artifacts identify the executed experiments and preserve their outputs. [Before submission, the authors must replace this sentence with a truthful account of the checks they personally completed and accept responsibility for the final claims and artifacts.]

The bracketed instruction belongs in the preparation notes, not a submitted manuscript. Do not automatically replace it with a claim that all authors have checked everything.

## 11. Reference verification notes

- `li2006abstraction`: author's hosted original PDF, https://thomasjwalsh.net/pub/aima06Towards.pdf.
- `drakulic2023bqnco`: NeurIPS 2023 proceedings page, https://proceedings.neurips.cc/paper_files/paper/2023/hash/f445ba15f0f05c26e1d24f908ea78d60-Abstract.html.
- `li2025drhg`: AAAI 2025 proceedings, https://ojs.aaai.org/index.php/AAAI/article/view/34018; methodology checked against https://arxiv.org/html/2502.16170. Author name follows the official record, not an inferred spelling correction.
- `luo2025insert`: NeurIPS 2025 proceedings, https://papers.nips.cc/paper_files/paper/2025/hash/7e3192a54b4ce5855a90dc182eac2036-Abstract-Conference.html.
- `ouyang2026segment`: title located in official ICLR 2026 download list at https://iclr.cc/Downloads/2026, linking to poster 10007337; authors/method checked at https://arxiv.org/abs/2507.01037. OpenReview forum pN261iTKvr required a browser challenge in this session.
- `luttmann2026macsim`: ICLR 2026 proceedings, https://proceedings.iclr.cc/paper_files/paper/2026/hash/1bf4cad47f5a54c98fbe7d10516ebf77-Abstract-Conference.html.
- `goudet2026order`: title located in official ICML 2026 download list at https://icml.cc/Downloads/2026, linking to poster 62945; authors/method checked at https://arxiv.org/abs/2510.01824. No invented PMLR volume/pages.
- `zhou2026icam`: current authors/title/DOI verified at https://arxiv.org/abs/2405.01906 (last revised 2026-06-28). No invented journal volume/pages.
- `kool2019attention`: author repository https://github.com/wouterkool/attention-learn-to-route and OpenReview https://openreview.net/forum?id=ByxBFsRqYm.
- `liao2025bopo`: official PMLR entry https://proceedings.mlr.press/v267/liao25a.html.
- `song2019ordering`: cited as the arXiv preprint to avoid assigning an unverified journal year. The current arXiv metadata gives related journal DOI 10.1016/j.engappai.2021.104603.

No negative literature search establishes that nobody has previously observed the specific aliasing phenomenon. Novelty language should remain narrow until a fuller expert literature review is completed.
