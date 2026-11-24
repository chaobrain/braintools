# Artifact retention

Documentation (`docs-html`) keeps its separately managed policy.
Distribution packages are kept for 7 days if publishing fails or is cancelled.
Successful publication removes only that new run's distribution artifact.

Raw test footprints and coverage data: 2 days. Diagnostics: 7 days.
Test/impact maps and nightly baselines: 14 days; partial impact maps: 5 days.
Plans, host holds and other short-lived coordination data: 3 days.
Only applicable artifact types are configured in this repository.

`artifact-cleanup.yml` runs after all source workflow jobs finish successfully.
It uses an explicit allowlist and never deletes mappings, final nightly reports,
documentation, failed-run evidence or artifacts from older runs. The fixed
2026-11-24T04:10:58Z rollout boundary also protects all pre-existing runs,
including a later retry of one. Cancelled/failed or foreign-source runs are kept.
The privileged workflow never checks out or executes source-run code.

Brainspike row bundles remain available for 7 days even after successful runs:
a successful coordinator can still report a failing test status. Mapping rotation
across historical runs is intentionally excluded; each baseline remains available
until its configured expiration. A rerun after successful cleanup may need to
rerun producers; failed publications retain their original build for recovery.

Validation: `node --test .github/artifact-cleanup.test.mjs`.
