# Recent GroupOpt / ASCC experiment index (2026-09-19)

This index accompanies the `codex/iclr-state-aliasing-20260919` research
branch.  The branch is based on `ascc-efficiency-v1` and collects the recent
research lines that were executed separately during the September 2026 audit.

## Uploaded research lines

- `research/iclr_state_aliasing_20260919/`: exact state-aliasing witness,
  counterfactual stress tests, controlled conditional predictors, 48 official
  AM/ICAM adapter arms, manuscript source, figures, aggregate outputs, and
  verification reports.
- `research/ascc_bopo_20260919/`: ASCC × BOPO implementation, BOPO versus
  REINFORCE screen, source-regret and rollout-value diagnostics, heuristic
  source policies, value selectors, fragment reconnection/destruction tests,
  transfer tests, plots, protocols, and aggregate records.
- `research/strong_hosts_20260919/`: the earlier official AM/ICAM adapter code,
  protocols, verification scripts, aggregate summaries, reference-budget
  calibration, and non-tensor evidence.
- `research/modern_host_pilots_20260919/`: the preceding LEHD, ICAM, BOPO50,
  BOPO100, and large-instance compatibility pilots, including their scripts and
  aggregate summaries. These are distinct bounded interventions and should not
  be pooled with the later controlled studies.
- `research/efficiency_optimizations_20260919/`: the exact changed source files
  for efficiency v2, v3, v3.1, and v4-lite relative to the branch base, plus
  their non-tensor benchmark summaries and gradient checks.
- `research/audit_summary_20260919/`: consolidated experiment ledger,
  scientific audit, literature route, and selected CPU/statistical checks.

## Interpretation warnings

- Efficiency v3 contains the documented gradient-equivalence defect.  It is
  retained as evidence, not as the recommended implementation.  v3.1 is the
  corresponding repaired gradient path.  v4-lite is a separate compute-saving
  intervention and must not be presented as numerically equivalent to full
  ASCC without its recorded checks.
- The BOPO screen has one training seed and 200 optimizer steps per arm.  Its
  negative result is a bounded screen, not a convergence theorem.
- The newer AM/ICAM state-adapter study has 48 completed arms, but it trains
  route/random-source controls rather than a fully optimized learned source
  policy.  Conditional information gains did not produce a competitive
  complete-tour forest solver.
- Results from canonical training, BOPO, frozen official hosts, and decoder
  adaptation use different protocols and must not be merged into one pooled
  performance table.

## Intentionally omitted artifacts

Virtual environments, upstream third-party repositories, external official
checkpoints, fitted model weights, `.pt` instance tensors, job-control records,
runtime logs, archive files, and machine-specific handoff notes are not tracked
on this branch.  JSON/CSV/JSONL summaries, figures, source code, protocols, and
validation reports are retained.  Omission of binary tensors means the public
branch supports inspection and regeneration, but not bit-for-bit replay of
every stored prediction without retraining or obtaining the private artifacts.
