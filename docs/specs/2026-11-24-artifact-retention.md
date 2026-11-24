# Artifact retention across braintools, braincell and brainspike

Reduce future artifact storage without touching existing artifacts or the
separately configured documentation deployment policy.

Each upload gets a bounded retention period: distributions and diagnostics
7 days, raw footprints and coverage inputs 2 days, coordination inputs 3 days,
and complete cross-run mapping baselines 14 days. Keep the existing 5-day
partial-map window. Failed publications are not actively cleaned and remain
recoverable within the 7-day retention period.

A default-branch `workflow_run` consumer cleans only allowlisted intermediates
of successful, completed, same-repository runs created after the fixed rollout
boundary. It never checks out source code, sweeps historical runs or deletes
mapping/final-report artifacts. Recheck the source attempt before deletion.
Brainspike row bundles stay available for diagnostics because a successful
coordinator can still record test failures. No historical mapping rotation is
introduced in this change.

Implement the explicit upload policies and isolated cleanup workflow in each
repository. Verify YAML, unchanged documentation workflow bytes, and mocked API
tests for success, failure, old runs, forks, stale retries, pagination and API
errors. Do not run a live deletion test or trigger existing workflows.
