# Public branch scope

This directory is the source-and-summary snapshot prepared on 2026-09-19 for
continued research.  It includes experiment implementations, protocols,
aggregate JSON/CSV summaries, manuscript sources, generated figures, and the
independent verification report.

The Git branch intentionally omits virtual environments, official third-party
checkpoints, fitted `.pt` checkpoints, per-instance tensor archives, generated
training caches, and execution logs.  Those files are not required to inspect
the method or regenerate the synthetic diagnostics, but they are required to
replay every recorded neural prediction without retraining.  The verification
report was produced against the complete private workspace before this reduced
snapshot was created; it must not be interpreted as a fresh verification run
over the reduced branch alone.

The manuscript is a research draft.  It reports negative and mixed outcomes as
well as positive conditional diagnostics, and it does not claim a state-of-the-
art routing solver or conference acceptance readiness.  No conference
submission is performed by pushing this branch.

Start with `README.md`, `REVIEW.md`, `exact_diagnostic/RESULTS.md`, and
`ARTIFACT_VERIFICATION.md`.  The official-backbone experiments depend on the
external AM and ICAM repositories and checkpoints described in
`state_adapter/README.md`.
