# Security Policy

## Supported versions

This project has no release branches. Security fixes land on `main`, and the production server
deploys `main` automatically once its CI passes ([docs/deployment.md](docs/deployment.md)), so
the only supported version is the current `main` (and the commit the server has deployed from
it). Older commits and forks do not receive fixes; update to the latest `main`.

| Version                            | Supported |
| ---------------------------------- | --------- |
| `main` (latest commit)             | Yes       |
| The commit deployed from `main`    | Yes       |
| Anything older, forks, local edits | No        |

## Reporting a vulnerability

**Please do not report security vulnerabilities through public GitHub issues, pull requests or
discussions.** A public report tells attackers about the problem before a fix is deployed.

Report it privately through GitHub's private vulnerability reporting instead:

**[Report a vulnerability](https://github.com/PredictionExplorer/CS-Image-Generation/security/advisories/new)**
(repository **Security** tab, then **Report a vulnerability**).

Only you and the maintainers can see the report. It helps if it includes:

- the component and file(s) affected, and the commit you tested (`git rev-parse HEAD`);
- the kind of issue (for example command injection, path traversal, credential exposure,
  workflow token abuse, supply-chain tampering) and its impact;
- step-by-step instructions or a proof of concept that reproduces it, with the exact command
  line and environment (OS, CPU architecture, Rust, Python and FFmpeg versions);
- any suggested fix or mitigation.

The form needs only a GitHub account, the same as opening an issue. The maintainers may invite
you to a temporary private fork to collaborate on the fix.

## What to expect

This is a small, maintainer-run project, so these are targets, not contractual guarantees:

| Step                                                | Target                         |
| --------------------------------------------------- | ------------------------------ |
| Acknowledge the report                              | within 3 business days         |
| Initial assessment (confirmed or not, severity)     | within 7 days                  |
| Status updates while a fix is in progress           | at least every 7 days          |
| Fix merged to `main` and deployed (critical, high)  | within 14 days of confirmation |
| Fix merged to `main` and deployed (medium, low)     | within 90 days of confirmation |

Severity is rated with CVSS through the GitHub security advisory. Once a fix is deployed, the
maintainers publish the advisory (requesting a CVE when warranted) and credit the reporter
unless they prefer to stay anonymous. Please keep the details private until the advisory is
published; if we cannot meet the targets above we will agree a disclosure date with you.

## Scope

In scope: everything in this repository, in particular

- **the generator**: the `three_body_problem` Rust crate and binary (simulation, rendering,
  video encoding through FFmpeg, the ember edition and its determinism certificate, and the
  package and NFT trait metadata it writes);
- **the sync script**: `run.py` and `_utils.py` (fetching token seeds from the CosmicGame API
  and the Arbitrum RPC, SSH/SCP uploads to the asset host, handling of the `.env` settings,
  the generation log and failure ledgers);
- **the deployment agent**: `ops/deploy/`, the systemd units in `ops/systemd/` and the one-time
  bootstrap in `ops/server/` (how the server pulls, verifies, builds and switches commits);
- **the CI and supply chain**: the GitHub Actions workflows in `.github/workflows/`, Dependabot
  configuration, the repository settings applied by `ops/github/`, and the pinned toolchains
  and dependencies (`Cargo.lock`, `rust-toolchain.toml`, `pyproject.toml`).

Out of scope:

- vulnerabilities in third-party dependencies, FFmpeg, Rust, Python or GitHub itself: report
  them upstream (tell us too if this project's use of them makes the problem exploitable);
- the CosmicSignature smart contracts, website, CosmicGame API server and asset host, which
  live outside this repository;
- attacks that require an already compromised maintainer account, production server or
  developer machine, and social engineering of the maintainers;
- denial of service by sheer volume, and reports from automated scanners without a
  demonstrated impact.

## Safe harbor

We will not pursue or support legal action against anyone who researches and reports a
vulnerability in good faith under this policy: avoid privacy violations, data destruction and
service disruption, test only against your own copy of the code (never the production server
or the CosmicSignature infrastructure), and give us reasonable time to fix the issue before
any disclosure.

## How this repository protects itself

For context, the controls already in place (see [`ops/github/README.md`](ops/github/README.md)):
changes reach `main` only through pull requests that pass CI; GitHub Actions are pinned to
full commit SHAs and run with a read-only token by default; Dependabot alerts and security
updates, secret scanning with push protection, CodeQL code scanning and OpenSSF Scorecard are
enabled; and the production server pulls `main` from GitHub itself (GitHub holds no credentials
for the server and never pushes to it) and deploys a commit only after its `CI passed` check
succeeds.
