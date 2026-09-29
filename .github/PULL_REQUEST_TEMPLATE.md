<!--
Title: a Conventional Commit subject, e.g. `fix(ember): clamp the ink remap at the grid edge`.
Pull requests are squash- or rebase-merged, and a squash merge uses the title as the commit
subject on `main`, so write it for the history. The commit body starts empty rather than as this
description: a `[skip ci]` quoted here would otherwise stop CI, and so the deploy, on `main`.

Merging into `main` DEPLOYS TO PRODUCTION automatically once the `CI passed` check is green:
the server builds and tests the commit, switches to it between sync runs and starts a sync run
(docs/deployment.md). Treat the merge button as the deploy button.

Security vulnerabilities: do not describe them here; see SECURITY.md.
-->

## Summary

<!-- What changes, in a few sentences or bullets. Link issues with "Closes #123". -->

## Why

<!-- The problem this solves or the motivation, and any alternatives you rejected. -->

## Testing

<!-- Tick what you ran and add anything else (commands, seeds, screenshots, contact sheets). -->

- [ ] Hooks pass: `pre-commit run --all-files` (format, lint, typing, workflow and spelling checks)
- [ ] Rust: `cargo test --release --locked` (with FFmpeg built with libwebp on `PATH`)
- [ ] Python: `python3 -m unittest discover -s tests/python`
- [ ] Determinism goldens pass unchanged: `tests/ember_determinism.rs` and the ember unit goldens
- [ ] Visual check for rendering changes (`just contact-sheet` / `just golden-gallery`)
- [ ] Not applicable (docs-only or metadata-only change)

## Risk and deployment impact

<!--
Merging deploys automatically. Answer each point, "none" is a fine answer.
- Rendered output: does this change pixels, videos, the ember certificate, package layout or
  NFT trait metadata? For new tokens only, or would already-published tokens be regenerated
  or backfilled by run.py?
- Runtime: build time, render time per package, disk use, memory.
- Operations: run.py behaviour, .env settings, systemd units, the deploy agent, the server.
- Rollback: is `python3 ops/deploy/cosmicsig_deploy.py rollback` enough, or does undoing this
  need more (published files, remote assets, state files)?
-->

- **Risk:** low / medium / high
- **Rendered output:**
- **Operations:**
- **Rollback:**

## Checklist

- [ ] The title is a Conventional Commit subject (`feat`, `fix`, `docs`, `ci`, `build`, `chore`, `refactor`, `perf`, `test`, `style`, `revert`, `deps`)
- [ ] Documentation is updated where behaviour changed (README, `docs/`, `CONTRIBUTING.md`, `docs/deployment.md`)
- [ ] Determinism goldens are untouched, **or** re-blessed deliberately: `ALGORITHM_VERSION` bumped and every affected digest re-blessed and checked on x86_64 and aarch64 (docs/ember-design.md §0.2); the `ci/reference` baseline regenerated if the main render changed (ci/README.md)
- [ ] Dependency changes are deliberate (`Cargo.lock` committed; `libm` stays pinned unless the goldens are re-blessed)
- [ ] No secrets, `.env`, `output/`, generated media or other large binaries are committed
- [ ] Safe to deploy to production as soon as CI passes
