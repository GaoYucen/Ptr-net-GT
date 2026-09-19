# 计算群论 / ASCC 项目目录索引

整理日期：2026-09-19。服务器：4090。项目根目录：`/workspace/计算群论/`。

## 首先阅读

- `audit_20260919/Research_Audit.md`：十部分研究审阅，含数学证明、实现问题、H1–H5、机制、P0–P3 和论文贡献边界。
- `audit_20260919/EXPERIMENTS.md`：按协议分表的主实验、复现、消融、现代 host 负结果、失败与探索记录。
- `audit_20260919/evidence/`：本轮 CPU 诊断、统计重算、旧思路 PDF 和迁移校验。
- `audit_20260919/migration/plan.json`：110 个迁移条目的完整原路径／新路径映射。

## 目录与研究角色

| 目录 | 内容 | 使用注意 |
| --- | --- | --- |
| `repos/groupopt-paper-canonical-repro-20260917` | 正式 canonical 源码、配置、paper_records、论文模板 | 当前审阅主线；不是已经完成的论文正文 |
| `repos/groupopt-ascc-efficiency-*` | 效率 Git worktree，名称表示五个分别存在的目录 | 保留原未提交改动；v3 有已确认梯度 bug，v31 是修复分支 |
| `repos/groupopt-modern-hosts` | BOPO、ICAM、NCO_code 上游源码与权重 | 区分上游 host 与本地 adapter |
| `repos/groupopt-official-am-kool` | 官方 AM 参考实现 | 官方 checkpoint 测试不是同预算 +ASCC 对照 |
| `experiments/` | LEHD、ICAM、BOPO50/100/large 实验脚本、日志、权重、结果 | standalone adapter，不能与 canonical joint-training 混表 |
| `results/` | canonical 复现与 efficiency 系列输出 | 保留负结果，不将调试运行升格为主实验 |
| `data/` | 官方 AM 测试数据与相关记录 | 历史数据不重生成覆盖 |
| `legacy/` | Ptr-net-GT、旧理论材料、效率导出副本 | 历史来源；不能代替当前实现证据 |
| `environments/` | 三个已有 Python 虚拟环境 | 已修复激活脚本、shebang 和 editable install 路径；未重装依赖 |
| `job_records/` | 82 个相关控制任务的原始运行记录 | 历史 task.sh 不应直接重跑；GREEN 不是科学结论 |
| `portable_entrypoints/` | 11 个仅替换绝对路径及入口依赖位置的脚本副本 | 原研究脚本未改；通过语法检查，未启动训练或完整 GPU 推理 |

## 从新路径使用 canonical

```bash
cd /workspace/计算群论/repos/groupopt-paper-canonical-repro-20260917
source /workspace/计算群论/environments/groupopt-paper-canonical-repro-20260917/bin/activate
```

本轮已验证 `import groupopt` 定位到新目录；Git 主仓库与五个 worktree 能正常定位。迁移后运行了 `tests/test_am_ablations.py`、`tests/test_problem_extensions.py`，共 7 个 CPU 测试通过。没有启动训练，GPU 可用性与完整 GPU 推理未在本轮复测。

历史脚本和 JSON 中的旧绝对路径作为原始实验记录保留，不能保证原命令直接执行。对于 11 个硬编码 Python 入口，应查 `audit_20260919/migration/portable_entrypoints.json` 使用对应的新副本；显式指定一个新的输出目录，避免覆盖历史实验。其余启动命令请按迁移映射更新输入、输出和工作目录。

## 保留与校验

本轮仅移动目录、修复 Git／环境定位元数据、生成审阅材料和路径副本，没有修改原始研究代码、checkpoint、结果文件。验证了 44,746 个文件的迁移元数据及 19,375 个源码／checkpoint 文件哈希。Git worktree 的 5 个 `.git` 定位文件是预期变更，单独记录。

`/workspace` 根目录不留原 GroupOpt／Ptr-net 项目目录或兼容链接。共享 `.server-control/jobs` 下仅保留指向这里 `job_records` 的链接，以保持控制器记录可读。其他研究项目、共享数据与共享服务未移动。

迁移没有删除历史目录内容。若未来需要调整目录布局，应以 `migration/plan.json` 为依据，先确认无活动进程，再迁移并重新修复 worktree 和环境定位；不要直接重新执行本轮的迁移脚本。
