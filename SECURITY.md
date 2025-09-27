# Security Policy

## Supported Versions

We provide security fixes for the most recent minor release. Older releases may
receive fixes on a best-effort basis when patches are low risk.

| Version     | Supported          |
|-------------|--------------------|
| `main`      | Always              |
| Latest tag  | Yes                 |
| Older tags  | No, please upgrade |

## Reporting a Vulnerability

Please email `security@braintools.dev` (or `chao.brain@qq.com` if the security mailbox is unavailable) with the subject line `SECURITY` and a
description of the issue. Include the following details when possible:

- Affected versions and environment information
- Steps to reproduce or proof-of-concept code
- Expected impact and any suggested mitigations

If encrypted communication is required, request our PGP key in the initial
message. We currently acknowledge reports within **5 business days** and aim to
provide a remediation plan within **10 business days**.

Avoid opening public GitHub issues for security concerns until we have
coordinated a fix and disclosure timeline.

## Coordinated Disclosure

We follow a coordinated disclosure model:

1. Validate the report and reproduce the issue.
2. Develop and test a fix, preparing regression tests when practical.
3. Share a release plan with the reporter, including the public disclosure date.
4. Publish the fix and changelog. Optionally credit reporters who request it.

We may ask reporters to test candidate patches or confirm the resolution before
public release.

## Third-Party Dependencies

If the vulnerability originates in a third-party project that BrainTools uses,
we will forward the report to the appropriate maintainers and coordinate follow
up where possible. Please mention any upstream components involved in your
report.

## Additional Help

For non-security questions or bug reports, file an issue on GitHub or contact
us via conduct@braintools.dev. Responsible disclosure keeps the community safe
and is greatly appreciated.
