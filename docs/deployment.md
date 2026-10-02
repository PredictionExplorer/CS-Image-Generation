# Deployment: continuous delivery of `main` to the generator host

Every commit that lands on GitHub `main` and passes CI is deployed to the generator host
automatically: the host pulls it, builds and tests it natively, switches to it between two sync
runs, and starts a sync run, which regenerates whatever the new version needs. Nobody logs in to
deploy. This guide is for the operator: how it works, how to set it up, and what to do when
something needs a human.

- [Architecture](#architecture)
- [Safety guarantees](#safety-guarantees)
- [First-time setup](#first-time-setup)
- [How a deploy works](#how-a-deploy-works)
- [The CI gate](#the-ci-gate)
- [When a deploy changes the ember look](#when-a-deploy-changes-the-ember-look): the old
  editions are withdrawn, or kept online until replaced, and rendered again
- [Operating it](#operating-it): status, logs, pause and resume, retry, rollback, manual runs
- [Troubleshooting](#troubleshooting)
- [Security model](#security-model)
- [Reference](#reference): files, environment, state, exit statuses
- [Changing the deployment](#changing-the-deployment)

## Architecture

```text
 GitHub                                  Generator host: user "user", systemd --user, lingering
┌───────────────────────┐  git fetch   ┌─────────────────────────────────────────────────────────┐
│ main  ◄── PR merges   │ ◄─────────── │ cosmicsig-deploy.timer   2 min after each tick           │
│                       │              │  └► cosmicsig-deploy.service                            │
│ Actions: "CI passed"  │  REST API    │      ops/deploy/cosmicsig_deploy.py run                  │
│ check run per commit  │ ◄─────────── │       1. fetch origin/main                               │
└───────────────────────┘ (read-only)  │       2. CI gate: "CI passed" succeeded for the commit?  │
                                       │       3. build + test in a staging worktree              │
                                       │       4. wait for the running sync; take run.lock        │
                                       │       5. fast-forward the checkout, install the binary   │
                                       │          and the units, smoke test (undo on failure)     │
                                       │       6. enable the sync timer, start a sync run         │
                                       │                                                         │
                                       │ cosmicsig-sync.timer     5 min after each run            │
                                       │  └► cosmicsig-sync.service                              │
                                       │      run.py (holds run.lock) ──► generator ──► scp ─────┼──► asset host
                                       └─────────────────────────────────────────────────────────┘
```

Everything runs unprivileged, as the user that owns the checkout, under that user's systemd
manager. Lingering keeps the manager (and the timers) running from boot, without a login session.
All four units are rendered from templates in [`ops/systemd/`](../ops/systemd), so a change to
them is deployed like any other change. The only root step is a one-time bootstrap.

| Piece | What it is |
|-------|------------|
| [`ops/deploy/cosmicsig_deploy.py`](../ops/deploy/cosmicsig_deploy.py) | The deploy agent (standard library only). `run` is one tick; `status`, `install`, `pause`, `resume`, `retry` and `rollback` are for the operator. |
| `cosmicsig-deploy.timer` / `.service` | One tick 3 minutes after boot, then 2 minutes after each tick ends. |
| `cosmicsig-sync.timer` / `.service` | `run.py`: 2 minutes after boot, then 5 minutes after each run ends (as before). |
| [`ops/server/bootstrap-root.sh`](../ops/server/bootstrap-root.sh) | The one-time root step: retire the legacy system units, enable lingering. |
| `run.lock` in the checkout | `run.py`'s single-instance lock. The agent takes it too while it switches. |

## Safety guarantees

The agent runs unattended on the production host, so it is built around these invariants. Each
one has tests in [`tests/python/test_deploy.py`](../tests/python/test_deploy.py).

1. **Only CI-passed commits.** A commit is deployed only if GitHub reports a completed,
   successful `CI passed` check run from GitHub Actions for exactly that commit, and only after
   `cargo build --release --locked`, `cargo test --release --locked` and the Python unit tests
   passed for it **on this host**.
2. **A render is never interrupted.** The switch happens only between sync runs: the agent stops
   the sync timer (so no new run starts), waits without a time limit for a running sync to
   finish, and then holds `run.lock`, so a `run.py` started by hand is waited for too.
3. **The checkout only moves forward along `main`**, and only from a clean tracked tree on branch
   `main`. A rewritten `main` (a force-push) or local commits are refused: a human decides.
4. **Production state is never touched.** `.env`, `output/`, the logs and the ledgers are
   untracked files; the switch refuses a commit that would write a tracked file where one of
   them lives (git would silently overwrite an ignored file).
5. **A switch is all or nothing.** Code, binary and units change together, and a failed check
   (units that systemd would not load, or a failed smoke test) undoes all three. The binary is
   replaced atomically (temp file, fsync, rename), and only when its bytes change, so `run.py`'s
   generator identity (which resets its backfill ledger) changes only with the binary. If the
   undo fails too, or the agent dies half way through a switch (killed, a power loss),
   auto-deploy pauses and the sync timer is disabled until a human has put the checkout back.
6. **Failures are final, not retried in a loop.** A failed build, test or switch is recorded and
   that commit is not tried again until `retry` or a newer commit. A CI failure is re-checked
   every 15 minutes, since a re-run on GitHub can turn it green.
7. **A quiet no-op.** A tick with nothing to do runs one `git fetch` and a few local checks: no
   GitHub API call, no log line, no write.
8. **A pause is honoured to the last moment.** `pause` (and `rollback`, which pauses first)
   stops any switch that has not started changing the checkout, even when the tick was already
   building or waiting for the sync.

## First-time setup

This is a one-time migration from the legacy setup (a system unit that ran `run.py` from a
checkout that was pulled and built by hand). The steps are safe to repeat.

**Prerequisites** on the host (the existing generator host has all of them): the checkout,
cloned from `https://github.com/PredictionExplorer/CS-Image-Generation.git` with its `.env`;
rustup in `~/.cargo/bin` (it installs the toolchain `rust-toolchain.toml` pins); FFmpeg with
libwebp, libx264 and libx265; Python 3.10+ at `/usr/bin/python3`; git; passwordless SSH from the
service user to the asset host. See the README's *Setting Up from Scratch (Ubuntu)*.

**Prerequisites on GitHub**, settled by an organization owner before `install`: once auto-deploy
runs, every account that can merge into `main` can run code on this host (see
[Security model](#security-model)). Require two-factor authentication for the
`PredictionExplorer` organization, lower its base permission to Read, and grant write access to
this repository only to the people allowed to deploy to this host. See
[ops/github/README.md](../ops/github/README.md#beyond-this-repository).

**1. Root, once: retire the legacy units and enable lingering.** The script is not in the
legacy checkout yet, so take it from `origin/main` without touching the checkout:

```bash
cd ~/Dev/CS-Image-Generation
git fetch origin
script=$(mktemp) && git show origin/main:ops/server/bootstrap-root.sh > "$script" &&
  less "$script" &&                          # read what you run as root
  sudo bash "$script" "$USER" "$PWD"
rm -f -- "$script"
```

`mktemp` makes a new file that only you can write. A fixed name in the shared `/tmp` is not
safe for a script run as root: another local account could create that file first, keep it
writable, and change it after you have read it.

The bootstrap stops and disables the system `cosmicsig-sync.timer`, **waits** for a sync run in
flight to finish (it never interrupts one; progress every minute), removes
`/etc/systemd/system/cosmicsig-sync.{service,timer}`, runs `loginctl enable-linger`, and prints
the next commands. It does not touch the checkout or any state.

**2. As the service user, in a login session** (`ssh user@host`; `sudo -u user` lacks the session
`systemctl --user` needs): fast-forward the checkout and install the user units.

```bash
cd ~/Dev/CS-Image-Generation
git status --short --untracked-files=no   # must print nothing
git switch main
git merge --ff-only origin/main
python3 ops/deploy/cosmicsig_deploy.py install
```

`install` renders the four units into `~/.config/systemd/user/`, reloads systemd, checks that
systemd loads them (and, where installed, that `systemd-analyze --user verify` accepts them) and
enables `cosmicsig-deploy.timer`, whose first tick starts at once. It warns if lingering is off.
It does **not** enable the sync timer: the first successful deploy does, so the sync never runs a
binary the agent has not built and tested.

**3. Watch the first deploy.** With no deploy recorded yet, the first tick goes through every
step even though the checkout already is at `origin/main`: it builds and tests (a first build
takes several minutes), installs the binary, enables `cosmicsig-sync.timer` and starts a sync
run.

```bash
journalctl --user -u cosmicsig-deploy -u cosmicsig-sync -f
python3 ops/deploy/cosmicsig_deploy.py status
```

**Optional: a GitHub token.** The public repository needs no credentials. The agent calls the
API only while a new commit waits for CI (at most one request per tick), far below the 60
requests an hour GitHub allows without a token. If the host shares its IP with other API users,
create a fine-grained token with read-only access to public repositories, no other permission
and an expiry date, and put it in the agent's environment file (mode 600 keeps other users out;
it does not hide the token from the code the host builds and tests, which runs as the service
user: see [Security model](#security-model)):

```bash
install -m 600 /dev/null ~/.config/cosmicsig-deploy.env
echo 'COSMICSIG_DEPLOY_GITHUB_TOKEN=github_pat_...' >> ~/.config/cosmicsig-deploy.env
```

## How a deploy works

One tick of `cosmicsig_deploy.py run`:

1. **Lock.** Take `deploy.lock` (non-blocking). If another tick, `install` or `rollback` holds
   it, exit 0 at once.
2. **Pause.** If auto-deploy is paused, log the reason and exit 0. If the legacy system units
   are installed, refuse (ERROR, exit 1): two schedulers would start the sync. Then two things an
   earlier run may have left behind: a switch that died half way (its record is still in
   `state.json`) pauses auto-deploy for a human (ERROR, exit 1), unless the checkout is back at
   the commit that switch started from (see
   [Troubleshooting](#a-switch-could-not-be-undone-or-was-interrupted)); and a sync timer that
   the last switch could not enable again is enabled now (retried every tick, ERROR and exit 1
   until it works).
3. **Fetch.** `git fetch --prune origin +refs/heads/main:refs/remotes/origin/main`. The target
   is `origin/main`. A fetch that fails (network) is a WARNING; the next tick retries.
4. **Anything to do?** If `state.json` records the target as deployed, the checkout's HEAD is the
   target and the installed binary's SHA-256 matches the recorded one, exit 0 quietly. If the
   target failed before (build, tests, switch or rollback), log one INFO line and exit 0. If it
   failed CI less than 15 minutes ago, likewise.
5. **Safety checks** (each an ERROR and exit 1, recorded as `last_error`): the tracked tree
   is clean; the checkout is on `main`; HEAD is an ancestor of the target (a fast-forward); the
   target contains the unit templates, the agent and `run.py` (a commit without them would stop
   auto-deploy for good, so it is refused).
6. **CI gate** (see [below](#the-ci-gate)). Waiting for CI: INFO, exit 0. CI failed: recorded,
   one ERROR, exit 0. The API unreachable or rate-limited: WARNING, exit 0. Nothing is deployed
   in any of these cases.
7. **Build and test** in the staging worktree `~/.local/share/cosmicsig-deploy/stage`, a git
   worktree of the checkout (shared objects, separate files), checked out clean at the target:
   `cargo build --release --locked`, `cargo test --release --locked`, then
   `python3 -m unittest discover -s tests/python`. Each runs under `nice -n 10` with a timeout
   and `CI=true` (so FFmpeg-dependent tests fail instead of skipping), streams its output to the
   journal, and shares the persistent `CARGO_TARGET_DIR` `~/.local/share/cosmicsig-deploy/target`
   (incremental builds). A failure is recorded as final for the commit (ERROR, exit 1). The
   binary `cargo build` produced (kept aside before `cargo test`, which may rebuild that path
   with the dev-dependencies' features) is stored with its SHA-256 as
   `releases/<sha>/three_body_problem` once all tests pass; a release that already exists is
   reused without rebuilding.
8. **Switch**, between sync runs. A `pause` (or a `rollback`, which pauses first) that arrives
   at any point before step 4 gives the switch up with nothing changed; the tested release is
   kept, so the tick after `resume` switches without rebuilding.
   1. stop `cosmicsig-sync.timer`, so no new sync run starts;
   2. wait while `cosmicsig-sync.service` is active (poll every 30 s, INFO every 10 min, no time
      limit);
   3. take `run.lock` (a `run.py` started by hand is waited for up to an hour, then the switch
      is given up until the next tick), check the pause flag a last time, check the checkout
      again, and record the switch in `state.json` (so a switch that dies half way is noticed);
   4. refuse if the target adds a tracked file where an untracked one exists (what the current
      commit tracks is git's to replace, for example a tracked file that becomes a directory);
      then `git merge --ff-only --no-overwrite-ignore <target>`;
   5. install the binary into `target/release/three_body_problem` atomically, unless the bytes
      are identical; the replaced one is kept as `three_body_problem.previous`;
   6. render the unit templates (`@REPO@`, `@PYTHON@`) and install those that changed into
      `~/.config/systemd/user/`, then `systemctl --user daemon-reload`, and check that systemd
      loads every unit (`LoadState=loaded`) and, where `systemd-analyze` is installed, that
      `systemd-analyze --user verify` accepts them: a deploy unit that did not load would stop
      auto-deploy for good;
   7. smoke test: `three_body_problem --version`, `run.py --help` and the new agent's `--help`
      must succeed;
   8. on any failure in 4 to 7 (an unexpected error included): undo all of it
      (`git reset --hard` to the previous HEAD, which is safe because the tree was clean; the
      previous binary; the previous units), record the commit as failed (ERROR, exit 1). If the
      undo fails too, auto-deploy pauses itself and the sync timer is stopped and disabled, so
      not even a reboot starts a sync on that checkout: see
      [Troubleshooting](#a-switch-could-not-be-undone-or-was-interrupted);
   9. release `run.lock`; enable and start the sync timer (after a failure, restore it as it
      was) and, after a success, start a sync run at once: `systemctl --user start --no-block
      cosmicsig-sync.service`. A timer that cannot be enabled is recorded in `state.json`, and
      every tick retries it (step 2) until it runs.
9. **Record** the deploy in `state.json` (atomically; the same update drops the switch record),
   prune releases (the newest 5, plus the deployed and the previous one) and log
   `deployed <sha> (<subject>) in <duration>`.

**Latency.** Merge to production takes the CI run (the slowest job), up to 2 minutes until the
next tick, the build and tests on the host (a few minutes, incremental), plus the rest of a sync
run that is in progress (a backfill package is a full render, which takes hours), because a
render is never interrupted.

## The CI gate

The CI workflow ends with an aggregate job, `CI passed`, that succeeds only if every other job
succeeded; the `main` ruleset requires it for every pull request. For the target commit the
agent asks

```text
GET https://api.github.com/repos/PredictionExplorer/CS-Image-Generation/commits/<sha>/check-runs
    ?check_name=CI%20passed&filter=latest
```

and considers only check runs named `CI passed`, created by the GitHub Actions app
(`app.slug == "github-actions"`), for exactly that commit. The newest one decides:

| Newest `CI passed` run | Result |
|------------------------|--------|
| completed, `success` | deploy |
| completed, anything else (`failure`, `cancelled`, `timed_out`, ...) | recorded as a CI failure; asked again every 15 minutes, so a re-run on GitHub can still turn it green |
| queued, in progress, or none yet | `waiting for CI` (INFO); the next tick asks again |
| API error, timeout, rate limit | WARNING; the next tick asks again; never deploys |
| HTTP 401 to a request with the token | ERROR naming the token's variable; asked again at once without the token, and that answer decides |

## When a deploy changes the ember look

Every package's `metadata/ember.json` records the ember algorithm that rendered it
(`"algorithm": "ember-v2"`), and the generator reports its own:
`three_body_problem --ember-algorithm` prints, for example, `ember-v3`. A change that alters the
ember edition's rendered bits bumps that id. The sync run the agent starts right after deploying
such a change takes every edition of the older look off the asset host, and the ember backfill
renders them again in the new one. Nothing needs doing by hand after the merge; this section
says what to settle before it, what to expect and how to steer it.

**Before merging the change.** Merging to `main` deploys, and the first sync run acts at once,
so settle these on the generator host first, in the checkout's `.env`:

1. Decide whether the old editions are withdrawn at once (the default, described first) or
   [stay online until each is replaced](#keeping-the-old-editions-online)
   (`COSMICSIG_KEEP_STALE_EMBER=yes`).
2. Check `COSMICSIG_MAX_BACKFILL`. If the backfill is paused (`COSMICSIG_MAX_BACKFILL=0`) and
   the switch is not set, the first run after the deploy withdraws every older edition and
   renders none again: every listed token loses its ember edition until someone removes the
   pause. Before the new look is deployed, either remove the pause or set
   `COSMICSIG_KEEP_STALE_EMBER=yes` (which, with the pause, holds every live edition as it is).
3. Check the room on the asset host (see *How long it takes, and how much room it needs*).

**What the first run does.** Before it plans, `run.py`:

1. reads the algorithm of every live certificate, with one SSH call (`sh` and `sed` on the asset
   host);
2. withdraws each edition whose algorithm is older than the generator's, for every seed in the
   current seed list. Per package, in this order: it deletes `metadata/ember.json`, replaces
   `metadata/assets.json` with the same manifest without its `ember_*` entries (uploaded as
   `assets.json.part` and renamed into place; every other entry and field is kept as it was),
   and deletes the edition's media files: those of the current look (absent ones are fine) and
   any other ember file that the withdrawn manifest listed. The main art, the spectral files
   and the other metadata are never touched;
3. plans as usual. The withdrawn packages now lack only the ember edition, so they are ember
   backfill seeds: regenerated at `--max-backfill` per run (default 1), after any new mint,
   with the [orbit and view check](#the-orbit-and-view-check) below.

The journal shows one line per package,
`WITHDRAWN  seed=0x…  its ember-v2 ember edition is off the asset host`, and one summary,
`Withdrew N stale ember editions (ember-v2 -> ember-v3): the ember backfill renders them again`.
Every later run finds nothing stale: a re-rendered package's certificate records the new id.

**Tokens have no ember edition until they are re-rendered.** This is intended. The artist
retired the old look, so it must not stay online, not even next to the new one while the
backfill works through the collection. A withdrawn package is a valid package without the
edition (its manifest lists no `ember_*` role), exactly like one uploaded before the edition
existed, which consumers already handle; each re-render adds the new edition back.

**What a re-rendered package holds.** An `ember-v3` edition is seven files: the still
(`images/source/ember.png`), its two WebP derivatives, the film (`videos/web/ember.mp4`), the
slow film (`videos/web/ember_slow.mp4`, the same film ten times slower), the archival film
(`videos/hq/ember.mp4`) and the certificate (`metadata/ember.json`, layout 4). The manifest
lists the six media under the roles `ember_source_master`, `ember_web_full`,
`ember_web_preview`, `ember_web`, `ember_slow_web` and `ember_hq`. An `ember-v2` edition had
six files and five roles: everything but the slow film. A package is complete only with all
seven, so an `ember-v2` package also reads as *missing only the ember edition*, whatever its
certificate says.

**How long it takes, and how much room it needs.** Every listed token is rendered again once,
one per run by default: a full package render, then its upload and the timer's 5-minute pause.
An `ember-v3` package takes about 3 hours on this host (its ember stage alone 2 h 06 min
for one token on the otherwise idle host), so a pass over 48 tokens takes about 6 days. Its
ember files hold 0.4 to 0.6 GB per token, the slow film 176 to 294 MB of it (three tokens
measured on 2026-10-02).

- Time: check it against the first `OK  seed=0x…  (total …)  ember edition uploaded` line in the
  journal: the orbit, and so the cost, differs from token to token.
- Size: read each package's sizes from its `UPLOAD … (N MB, timeout Ns)` lines, and check that
  the asset host has room for every token (48 × 0.6 GB is about 29 GB) *before* the pass is far
  along. An edition is staged beside the live files before it is swapped in, so the host also
  needs room for one whole new edition on top of what is online. A full disk fails the transfer
  that hits it: `UPLOAD FAILED`, nothing of the live package is changed, the staged files are
  deleted, and the seed is tried again by a later run (and fails again, after another full
  render, until there is room).

At the default `--max-backfill` of 1, a new mint waits for at most one backfill package (the
rest of the run in progress). The per-seed timeout is 10 hours (`run.py --timeout`), well below
the sync unit's 24-hour limit; it only has to catch a render that hangs. An scp transfer may take
15 minutes, or one second per MB if that is longer.

**Watching progress.**

```bash
journalctl --user -u cosmicsig-sync -f                                      # live
grep -E 'WITHDRAWN|Withdrew|Kept|ember edition uploaded' imgcheck.log         # the whole history
grep -E 'IDENTITY MISMATCH|EMBER FAILED|given up' imgcheck.log                # tokens left behind
python3 run.py --dry-run   # between runs: "... N missing only the ember edition"
```

`--dry-run` exits at once while a sync run holds `run.lock`. On the asset host, in the asset
directory (`COSMICSIG_REMOTE_DIR`), count the live editions per algorithm and list the tokens
still waiting:

```bash
grep -h '^  "algorithm"' 0x*/metadata/ember.json | sort | uniq -c
for d in 0x*/; do [ -f "${d}metadata/ember.json" ] || echo "$d"; done
```

**Safety.** A mass withdrawal needs an explicit, readable signal; a failure or an unexpected file
never starts one.

- Nothing is withdrawn unless the generator reports its id. A binary without
  `--ember-algorithm` (built before the flag existed) logs a WARNING on every run and withdraws
  nothing.
- A certificate whose algorithm cannot be read is kept, with a WARNING naming it. If the
  certificates cannot be read at all (SSH fails), nothing is withdrawn that run: an ERROR, the
  run exits `1`, and the next run tries again.
- Only an older id is stale. After a rollback to a generator with an older id, the newer live
  editions are kept (one WARNING per run: `… are newer than this generator's …`) and new mints
  get the older look; deploying the newer generator again re-renders only those.
- Only seeds in the current seed list are touched, because `run.py` regenerates no other
  package: a withdrawn edition there would never come back. Stale editions of unlisted packages
  are kept and named in a WARNING on every run; remove such packages by hand if they are
  obsolete.
- A package that lacks a core file is left alone: the same run regenerates and uploads it in
  full, new ember edition included.
- The certificate goes first, so a withdrawal that is interrupted (a lost connection, a stop)
  already leaves a backfill seed, and its re-render replaces the manifest entries and media that
  were left. A live `metadata/assets.json` that cannot be read or parsed leaves the package
  untouched, with an ERROR on every run until it is repaired on the asset host. A failed
  withdrawal makes the run exit `1`.
- A re-render whose orbit or view differs from the live package's uploads nothing and is given
  up with that binary (the [check](#the-orbit-and-view-check) below). That token then stays
  without an ember edition until someone decides: a rebuilt generator, or
  `--backfill-mode full`.

### The orbit and view check

An ember edition is uploaded next to main art that stays as published, so it must draw the same
orbit, and since `ember-v3` its bodies follow the main edition's view: the same projection,
viewing rotation, drift and framing. Before `run.py` uploads the ember edition of a re-rendered
package (`--backfill-mode ember`, the default), it compares ten fields of the regenerated
`metadata/nft_traits.json` with the live one's, as exact JSON values (numbers by their exact
decimal value, never as floats):

| What | Fields |
|------|--------|
| the orbit | `simulation.masses`, `generation.borda.selected_index`, `generation.borda.retry_count` |
| the view | `generation.structure.stack_label`, `generation.projection`, `generation.symmetry`, `generation.drift.mode`, `generation.drift.scale`, `generation.drift.arc_fraction`, `generation.drift.orbit_eccentricity` |

- **All equal.** The log shows `same orbit and view as the live package, by every recorded field
  (its viewing rotation and frame are not recorded, and are assumed to match); uploading only
  its ember edition`, then `OK  seed=0x…  (total …)  ember edition uploaded`.
- **One differs, or the regenerated file lacks one.** Nothing is uploaded. An ERROR names the
  seed and each differing field with both values (`… shows a DIFFERENT ORBIT OR VIEW than the
  live one (generation.drift.scale: live 1.16…, regenerated 1.2…)`), and the seed's last line is
  `IDENTITY MISMATCH  seed=0x…  (total …)  nothing uploaded; given up with this binary`. The
  generator is deterministic, so the same binary would render the same package again: the seed
  is given up at once, listed under `identity_mismatches` in `backfill_failures.json`, and every
  later run logs `ember backfill given up: this generator binary regenerates a different orbit
  or view than the live package`. A rebuilt generator tries it again.
- **The live file lacks one.** Every generator that wrote `nft_traits.json` wrote all ten (its
  schema requires them), so such a file is damaged and its orbit or view is not guessed:
  nothing is rendered or uploaded, and an ERROR (`the live package cannot be used for an ember
  backfill: metadata/nft_traits.json lacks …`) repeats on every run until the file is repaired
  on the asset host. This is checked before the render, so it costs none, and it never gives
  the seed up.

**What the check does not verify.** The viewing rotation and the frame themselves are not
recorded in the live package, so `run.py` cannot compare them: they are *assumed* to match once
the ten fields do. The generator derives them while it renders. The rotation is the best of
four by a score that depends on the layer stack, computed with platform floating point, which
is why the stack (`generation.structure.stack_label`) is compared as its recorded proxy. Look at
the first re-rendered token next to its main art before trusting the rest of the pass.

### Keeping the old editions online

By default an older edition is withdrawn at once, and its token has no ember edition until its
turn in the backfill. To have no such gap, set `COSMICSIG_KEEP_STALE_EMBER=yes` in the
checkout's `.env` (or pass `--keep-stale-ember` to a manual run) *before* the change is
deployed: the sync run the agent starts right after the switch otherwise withdraws every stale
edition at once.

With the switch no run withdraws anything. Each run still reads the certificates, logs
`Kept N stale ember editions online (ember-v2 -> ember-v3): the ember backfill replaces each in
place`, and plans those packages as ember backfill seeds, `--max-backfill` per run as usual.
When a package's turn comes, its re-render passes the [check](#the-orbit-and-view-check) above
and replaces the old edition **in place**, staged and then swapped (every ember-mode backfill
uploads this way, whether or not an edition is live):

1. every file of the new edition (the six media, the merged manifest, the certificate) is
   uploaded as `<name>.part` beside its destination. Nothing live changes meanwhile;
2. once all have landed, one SSH call swaps the edition in: it deletes the old certificate (and
   any ember file the old manifest listed that the new edition does not have; `ember-v2` has
   none), renames each medium into place, then the manifest, then the certificate, last.

- The price of no gap is that both looks are online until the pass is over.
- A transfer that fails (a lost connection, a full asset host) changes nothing: the old edition
  stays online byte for byte, the staged `.part` files are deleted, and a later run renders the
  package again. Staging needs room for the old and the new media at once.
- Only the swap itself has in-between states, and it transfers nothing, so it lasts a moment:
  the package has no certificate, and its manifest lists the old edition's entries (with the
  old checksums) over media that are each still the old file or already the new one, then the
  new entries over the new media. No file is ever truncated under its real name. A swap that
  is cut off leaves the package in one of those states, as a backfill seed, until a later run
  renders it again and repeats the upload.
- A re-render that fails, or whose orbit or view differs, uploads nothing, so that token keeps
  its old edition: after a retry cap or an identity mismatch, until the generator binary
  changes.
- Removing the switch, or setting `no`, has the next run withdraw every old edition that is
  still waiting.

| Settings | Older editions | Tokens meanwhile |
|----------|----------------|------------------|
| default | withdrawn at once, by the first run; re-rendered `--max-backfill` per run | no ember edition until re-rendered |
| `COSMICSIG_KEEP_STALE_EMBER=yes` | left online; replaced in place, `--max-backfill` per run | the old look until replaced |
| `COSMICSIG_KEEP_STALE_EMBER=yes` and `COSMICSIG_MAX_BACKFILL=0` | left online; nothing is re-rendered | every live edition exactly as it is |
| `COSMICSIG_MAX_BACKFILL=0` alone | withdrawn at once; nothing is re-rendered | no ember edition until the backfill resumes |

So `COSMICSIG_MAX_BACKFILL=0` (`--max-backfill 0`), together with the switch, is what holds the
live editions exactly as they are; it pauses the backfill of packages that have no ember edition
yet as well, while new mints are still generated, in the current look. The switch alone does not
hold anything: before `ember-v3` it kept the older editions for good (their packages were
complete, so nothing re-rendered them), and that meaning is gone.

## Operating it

All commands run as the service user from the checkout (`cd ~/Dev/CS-Image-Generation`).

### Status

```bash
python3 ops/deploy/cosmicsig_deploy.py status          # human-readable
python3 ops/deploy/cosmicsig_deploy.py status --json   # for scripts and monitoring
systemctl --user list-timers 'cosmicsig-*'
```

`status` shows the deployed commit and when, whether the installed binary is the tested one,
`origin/main` and why it is not deployed yet (if it is not), the rollback target, the pause flag,
a switch that is running (or was interrupted), a sync timer still waiting to be restarted, the
last error, recorded failures, and the state of the four units.

### Logs

```bash
journalctl --user -u cosmicsig-deploy -u cosmicsig-sync -f     # follow both
journalctl --user -u cosmicsig-deploy -p warning --since today # warnings and errors only
tail -f imgcheck.log                                           # run.py's detailed log
```

The agent writes plain lines; under journald each carries its syslog priority, so `-p warning`
and `-p err` work. A tick with nothing to do logs nothing (systemd itself still logs each
start). A tick logs INFO for every decision, streams build and test output, and ends a deploy
with `deployed <sha> (<subject>) in <duration>`.

### Pause and resume

```bash
python3 ops/deploy/cosmicsig_deploy.py pause --reason "investigating render artefacts"
python3 ops/deploy/cosmicsig_deploy.py resume
```

While paused, every tick exits after one INFO line and the sync keeps running on the deployed
version. A tick that is waiting for a sync run when you pause gives its switch up at once; one
that is building finishes the build (its release is kept) and then switches nothing.

To stop the sync itself, pause first and then
`systemctl --user disable --now cosmicsig-sync.timer`: every successful deploy enables and
starts the sync timer again (and so does a tick that retries a restart a switch could not do),
so a disabled timer alone does not survive the next merge to `main`. Undo both with
`systemctl --user enable --now cosmicsig-sync.timer` and `resume`.

### Retry

```bash
python3 ops/deploy/cosmicsig_deploy.py retry            # origin/main
python3 ops/deploy/cosmicsig_deploy.py retry 1a2b3c4d   # a specific commit
```

Forgets a recorded failure (for example a build that failed because the disk was full), so the
next tick tries the commit again. A new commit on `main` needs no `retry`.

### Rollback

```bash
python3 ops/deploy/cosmicsig_deploy.py rollback
```

Pauses auto-deploy, then switches the checkout (`git reset --hard`) and the binary back to the
previously deployed commit with the same switch procedure (it waits for a running sync; stop the
sync first with `systemctl --user stop cosmicsig-sync.service` if the running version itself is
the problem). Run it inside `tmux` or `screen` (or under `nohup`, whose ignored hangup the agent
keeps ignoring): the wait can take hours, and otherwise, if your SSH session drops, the hangup
stops the rollback like `Ctrl-C` does. Nothing is switched then and the sync timer is put back,
but auto-deploy stays paused and the rollback has not happened.

If a tick is building the next commit, the rollback waits for that build (its output is in the
journal); the tick then switches nothing. The rollback only ever leaves the commit that was
deployed when you ran it: if a tick finished deploying a newer commit in the seconds before the
pause took effect, the rollback logs an ERROR naming both commits and changes nothing
(auto-deploy stays paused; check `status`, then `resume`, or run `rollback` again).

The rolled-back commit is recorded as failed, so `resume` does not deploy it again; the next
commit on `main` is deployed normally. Rollback is one step: afterwards there is no previous
deployment until the next deploy. The first deploy records none either, because the agent never
built or tested the commit the checkout was at before it (you switched it by hand). Fix `main`
(revert the bad change in a pull request), then `resume`.

### Running a sync by hand

```bash
systemctl --user start cosmicsig-sync.service     # preferred: journald, the usual limits
python3 run.py --dry-run                          # what a run would do
```

`run.py` holds `run.lock` for its whole run, whatever the mode: while the service runs, a manual
`run.py` (including `--preflight` and `--dry-run`) exits at once with
`Another run holds the single-instance lock .../run.lock (pid N)`. `--help` does not take it.

## Troubleshooting

### A commit is not deployed

Run `status`. The `Pending` line names the reason; the journal has the details
(`journalctl --user -u cosmicsig-deploy -p warning`).

- **`waiting for CI`**: nothing to do. After 45 minutes without a result it becomes a WARNING,
  `still waiting for CI …`, which `status` shows too. If CI never reports, check the Actions run on GitHub. If the commit has no Actions run at all (GitHub skips the push run when the commit message contains `[skip ci]` or a similar instruction), start one with `gh workflow run ci.yml --ref main`.
- **`ci failed`**: fix `main`, or re-run the failed jobs on GitHub (picked up within 15 minutes).
- **`build failed` / `tests failed`**: the commit does not build or pass its tests on this host.
  The journal holds the output. Push a fix; for a host problem (disk full, network while
  rustup installed a toolchain), fix the host and `retry`.
- **`switch failed`**: systemd would not load a changed unit, the smoke test failed, or the
  commit would have overwritten an untracked file (the message names it). Everything was rolled
  back. Fix and push, or move the file away and `retry`.
- **`rolled back by the operator`**: `rollback` left this commit. Fix `main` (revert the bad
  change in a pull request), then `resume`. To deploy the commit after all, `retry` it and
  `resume`.

### main was rewritten, or the checkout has local changes

The agent refuses (ERROR every tick) until a human reconciles the checkout. Inspect first:

```bash
git status --short --untracked-files=no
git log --oneline --graph --decorate -15 HEAD origin/main
```

To discard local changes and follow `origin/main` (no sync run may be active; `.env`, `output/`
and the logs are untracked and survive):

```bash
python3 ops/deploy/cosmicsig_deploy.py pause --reason "reconciling the checkout"
systemctl --user disable --now cosmicsig-sync.timer   # off until the new HEAD is deployed
systemctl --user is-active cosmicsig-sync.service   # repeat until not active/activating
git diff --name-only --diff-filter=A HEAD origin/main   # none of these may exist untracked
git switch main && git reset --hard origin/main
python3 ops/deploy/cosmicsig_deploy.py resume
```

The sync stays off until the agent has deployed the new HEAD: once its `CI passed` check has
succeeded and it has built and passed its tests on this host, the switch enables and starts the
sync timer (`status` then shows `origin/main` deployed). If its CI failed or it does not build,
push a fix to `main` first. To run the sync meanwhile on the new code with the previous binary,
enable the timer deliberately with `systemctl --user enable --now cosmicsig-sync.timer`
(`run.py` copes with an older binary, see the README's *Automation*). If `status` already showed
`origin/main` deployed before the reset (only the checkout had local commits), there is nothing
to deploy: enable the timer again yourself the same way.

### A switch could not be undone, or was interrupted

Rare. Either undoing a failed switch failed too (for example on a full disk; the agent logs
CRITICAL), or the agent died in the middle of a switch (killed, out of memory, a power loss): the
next tick or `rollback` finds the switch's record in `state.json`, sees that the checkout moved,
and logs an ERROR naming the commit the switch started from and the one the checkout is at.
Either way the agent pauses itself with the reason and stops and disables the sync timer: the
checkout is in an unknown state, and not even a reboot may start a sync on it.

Check `status`, `git status`, and `sha256sum target/release/three_body_problem*` against
`~/.local/share/cosmicsig-deploy/releases/*/release.json`. Put the checkout back on the commit
the switch started from, which the message names (normally the deployed commit):
`git reset --hard <sha>`. Put its binary back too: copy `releases/<sha>/three_body_problem` (or
`three_body_problem.previous`, the binary the switch replaced) over
`target/release/three_body_problem`. Then put the units back (`install` renders them from the
restored checkout and reloads systemd), enable the sync timer again, and resume:

```bash
python3 ops/deploy/cosmicsig_deploy.py install
systemctl --user enable --now cosmicsig-sync.timer
python3 ops/deploy/cosmicsig_deploy.py resume
```

The next tick sees the checkout back where the switch started, drops the record, and deploys
`origin/main` as usual. The commit of a failed undo stays failed until `retry`.

### The sync timer did not restart after a switch

`status` shows `NOT restarted after the last switch` on its `Sync timer` line, and every tick
logs an ERROR and exits 1: systemd refused `systemctl --user enable --now cosmicsig-sync.timer`
after a switch (a D-Bus timeout under load, say, or a timer unit it rejects). Every tick that is
not paused retries it; once it works, the error clears.
`systemctl --user status cosmicsig-sync.timer` and `journalctl --user -u cosmicsig-sync.timer`
say why it fails.

### Rate limits and the optional token

`GitHub API rate limit exceeded` warnings mean other clients on the host's IP use up the
unauthenticated quota. Nothing is deployed until the API answers again. Add the read-only token
(see [First-time setup](#first-time-setup)); it only raises the limit to 5,000 requests an hour.

Tokens expire. GitHub answers a request that carries an expired or revoked token with HTTP 401,
even for a public repository, so a `... was rejected by GitHub (HTTP 401)` ERROR means the token
needs replacing (or removing). Meanwhile the agent asks again without it and carries on with that
answer, and `status` shows the rejection as the last error after every tick that asked GitHub
(a tick that asks nothing, such as one with nothing to deploy, clears it).

### Disk usage of the build cache

| Path | Size | Safe to delete? |
|------|------|-----------------|
| `~/.local/share/cosmicsig-deploy/target` | several GB (release and test builds) | yes, when no tick runs (`pause` first); the next build is a full one |
| `~/.local/share/cosmicsig-deploy/stage` | the source tree | yes, it is recreated |
| `~/.local/share/cosmicsig-deploy/releases` | a few binaries | not the deployed or previous one (rollback needs them) |
| `~/.cargo/registry` | crate sources | yes, `cargo` downloads them again |

### Timers do not run after a logout or reboot

Lingering is off: `loginctl show-user "$USER" --property=Linger` prints `Linger=no`. Run the
bootstrap again, or `sudo loginctl enable-linger user`. `install` warns about it.

### `systemctl --user` says "Failed to connect to bus"

The shell has no user session (for example `sudo -u user -i`). Log in over SSH as the user, or
`export XDG_RUNTIME_DIR=/run/user/$(id -u)`.

### The legacy units are back

`run`, `install` and `rollback` refuse while `/etc/systemd/system/cosmicsig-sync.{service,timer}`
exist, because two schedulers would start the sync; `status` lists them. Run the bootstrap again.

## Security model

- **Pull-based.** The host opens every connection: `git fetch` and the check-run query, both
  HTTPS to GitHub. There are no inbound ports, webhooks or deploy keys, and GitHub holds no
  credential to the host; nothing on GitHub can push code or commands to it.
- **No credentials for a public repository.** The optional token is read-only, lives in a mode
  600 file, is sent only to the API (never along a redirect), and is removed from the
  environment of every command the agent runs, which keeps it out of their environments and
  logs. It is no secret from the code the host builds and tests, though: build scripts, proc
  macros and the staged tests run as the service user and can read the env file. So use only a
  fine-grained token with read-only access to public repositories and an expiry date. URLs with
  credentials are masked in log lines.
- **Least privilege.** Everything runs as the unprivileged service user under its own systemd
  manager; the services set `NoNewPrivileges=yes`. Root is needed once, for the bootstrap,
  which is short and reviewable.
- **What runs is what CI checked, and whoever can merge decides what that is.** The ruleset on
  `main` requires a pull request and a green `CI passed` and blocks force-pushes, but it requires
  0 approving reviews: every account that can merge into `main` can run code on this host, as the
  service user that holds the asset host's SSH key. As of September 2026 that is 3 admins and
  2 writers, and the organization does not require two-factor authentication. Keep that set
  small and protected: see the GitHub prerequisites in [First-time setup](#first-time-setup) and
  [ops/github/README.md](../ops/github/README.md#beyond-this-repository). For each commit the
  agent checks `CI passed` again (from the GitHub Actions app only), moves only forward, builds
  with `--locked` (the committed `Cargo.lock`, which cargo-deny audits in CI) and tests on the
  host before switching.
- **Unattended means predictable.** The checkout's git hooks never run during a deploy, git
  never prompts, and every external command has a timeout.

## Reference

### Files

| Path | Content |
|------|---------|
| checkout (`~/Dev/CS-Image-Generation`) | the deployed commit, `.env`, `output/`, run.py's logs and ledgers, `run.lock` |
| `target/release/three_body_problem` (+ `.previous`) | the deployed generator (and the one it replaced) |
| `~/.config/systemd/user/cosmicsig-*.{service,timer}` | the rendered units (never edit these; edit `ops/systemd/`) |
| `~/.config/cosmicsig-deploy.env` | optional agent settings (the token) |
| `~/.local/state/cosmicsig-deploy/state.json` | the deployment record |
| `~/.local/state/cosmicsig-deploy/paused` | exists while auto-deploy is paused (JSON: reason, time) |
| `~/.local/state/cosmicsig-deploy/{deploy,state}.lock` | the agent's locks |
| `~/.local/share/cosmicsig-deploy/{stage,target,releases}` | the staging worktree, the cargo build cache, the tested binaries |
| `~/.local/share/cosmicsig-deploy/built/three_body_problem` | the binary `cargo build` made, kept aside while its tests run |

`XDG_STATE_HOME`, `XDG_DATA_HOME` and `XDG_CONFIG_HOME` move these as usual.

### Environment

| Variable | Default | Purpose |
|----------|---------|---------|
| `COSMICSIG_DEPLOY_GITHUB_TOKEN` (or `GITHUB_TOKEN`) | none | optional read-only token for the API |
| `COSMICSIG_DEPLOY_REPO` | the checkout that holds the agent | the production checkout |
| `COSMICSIG_DEPLOY_GITHUB_API` | `https://api.github.com` | the API base URL |
| `COSMICSIG_DEPLOY_GITHUB_REPO` | parsed from `origin`'s URL | `OWNER/REPO` for the API |
| `COSMICSIG_DEPLOY_PYTHON` | `/usr/bin/python3` | the interpreter in the units and for the staged tests |

### state.json

| Field | Meaning |
|-------|---------|
| `deployed_sha`, `deployed_subject`, `deployed_at` | what is deployed, and since when |
| `binary_sha256` | SHA-256 of the installed generator (a tick compares it) |
| `previous_sha`, `previous_binary_sha256` | what `rollback` switches back to: the commit deployed before (none after the first deploy or a rollback) |
| `rolled_back_from` | the commit the last rollback left |
| `failed_shas` | `{sha: {reason, detail, at, checked_at, seq}}`; reasons `ci`, `build`, `tests`, `switch`, `rollback` (the 20 most recently recorded; `seq` keeps the order they were recorded in, which the file's sorted keys lose; records without `seq`, written by an older agent, count as older than any with it, ordered by `at`) |
| `last_error`, `last_error_at` | the error of the last tick that had one (cleared by a tick without) |
| `sync_timer_restart_pending` | `true` while the sync timer could not be enabled again after a switch; every tick that is not paused retries (dropped when a switch leaves the checkout to a human) |
| `switch_in_progress` | `{target, old_head, old_binary_sha256, mode, started_at}` while a switch changes the checkout; one left behind means the switch died half way |

### Exit statuses of `run`

`0`: nothing to do, deployed, waiting (for CI or a sync run), paused, a CI failure recorded, or
a transient problem (fetch or API unreachable; a rejected token, which is recorded as the last
error). `1`: refused (an unsafe checkout, legacy units, a commit without the agent, an
interrupted switch), a build, test or switch failure (rolled back), a sync timer that still
cannot be enabled, a stop signal, or an unexpected error; the unit shows `failed` until the next
tick.

## Changing the deployment

The agent, the unit templates and `run.py` deploy themselves: a change merged to `main` is built,
tested (the staged suite includes `tests/python/test_deploy.py`, which exercises the new agent),
smoke-tested (`cosmicsig_deploy.py --help` must start) and switched like any other. Changed unit
files are installed and `systemctl --user daemon-reload`ed during the switch. Keep in mind:

- Edit `ops/systemd/*`, never the installed copies; the next deploy overwrites them.
- Changed units must load: a unit systemd refuses (or that `systemd-analyze --user verify`
  rejects, where it is installed) fails the switch, which puts the previous units back. Nothing
  short of starting a unit notices a missing `EnvironmentFile=`, though.
- A commit that removes the agent or a unit template is refused; retire auto-deploy by hand
  (`systemctl --user disable --now cosmicsig-deploy.timer`) before merging one.
- Test locally: `python3 -m unittest tests/python/test_deploy.py -v` (real git, fake cargo,
  systemctl, systemd-analyze and GitHub API; nothing on the machine is touched).
