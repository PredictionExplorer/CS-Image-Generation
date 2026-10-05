# Required changes in `augur-explorer`

Companion to [`augur-explorer-integration.md`](augur-explorer-integration.md).
That document is the full contract; this one is the short, concrete work list
for the Go repository, written after auditing the code at
`PredictionExplorer/augur-explorer` against the **live** asset host on
2026-08-30.

---

## TL;DR

The Go side is essentially **already built** — ingester, migration, handler,
`Allocation`, owner removal, seed-attribute removal are all done and tested.

There is **one blocking defect**: the ingester fetches URLs that do not exist,
so it will `404` on every seed and never ingest a single trait row.

```go
// internal/api/cosmicgame/traits/fetch.go:73,78 — both 404 in production
f.base + "/traits/" + seed + ".json"
f.base + "/asset-manifests/" + seed + ".json"
```

The asset host publishes those files *inside each package directory*, not at
dedicated top-level routes. Fixing this is a small, mechanical change in three
production files plus four test files.

There is also **one new item for the ember edition** (generator 1.1.0,
[§7](#7-ember-edition-generator-110)). It does not block the URL fix. Until
it ships, tokens (new mints included) are served without the `ember_*` keys.
Ship its two parts together:

- the `ember_*` media keys, gated on the manifest roles;
- a periodic re-check of **every** row's manifest, which fetches
  `assets.json` independently of the trait file. Its first pass picks up
  every token backfilled so far, so §7 does not gate the generator rollout.
  It must also cover rows whose stored manifest already lists
  `ember_source_master`: a change of the ember look withdraws published
  editions, and their manifests lose the `ember_*` roles until the edition
  is rendered again ([§7.1](#71-what-changes-on-the-asset-host)).

One check is time-sensitive: if the manifest decoder rejects unknown fields,
make it accept `color_space` ([§7.2](#72-manifest-parsing)). The generation
host has published manifests with it since the ember edition was deployed
(2026-09-29).

§7 was added with the ember edition. It is written against the contract in
[`augur-explorer-integration.md`](augur-explorer-integration.md), not against
a new audit of the Go code.

---

## 1. Evidence

Measured against `https://nfts.cosmicsignature.com` for a backfilled seed
(`0x38b23b92766de4c8989e563ea11b3b3aab1965a546a84c6780a715df3b3df0b1`):

| URL | Status |
|---|---|
| `/traits/0x<seed>.json` | **404** — what the Go fetcher builds today |
| `/asset-manifests/0x<seed>.json` | **404** — what the Go fetcher builds today |
| `/images/new/cosmicsignature/0x<seed>/metadata/nft_traits.json` | **200** `application/json` |
| `/images/new/cosmicsignature/0x<seed>/metadata/assets.json` | **200** `application/json` |
| `/images/new/cosmicsignature/0x<seed>.png` | 200 (flat alias, `image`) |
| `/images/new/cosmicsignature/0x<seed>.mp4` | 200 (flat alias, `animation_url`) |
| `/images/new/cosmicsignature/0x<seed>/videos/hq/main.mp4` | 200 |
| `/images/new/cosmicsignature/0x<seed>/spectral/00_382nm.png` | 200 |

Why: `nfts.cosmicsignature.com` reverse-proxies to a static file service that
exposes the whole package tree. Nothing is configured per-file, so anything the
generator uploads is served automatically. The `/traits/...` shape was proposed
in an early draft of the contract document but never implemented, because it
needs root on the asset host and adds nothing.

Response headers to note:

```text
content-type: application/json
access-control-allow-origin: *
cache-control: public, max-age=3600, must-revalidate
last-modified: Sun, 30 Aug 2026 13:56:18 GMT
```

**There is no `ETag` header.**

---

## 2. Already done — do not redo

Verified present in the current code:

| Item | Location |
|---|---|
| Metadata layout v2.0.0 | `internal/api/cosmicgame/metadata_assembly.go` |
| `properties.owner` removed | enforced by `TestMetadataNeverCarriesOwner` |
| No `seed` in `attributes[]` | enforced by `TestMetadataHasNoSeedAttribute` |
| `Allocation` attribute | `internal/store/cosmicgame/token-traits.go` (`CosmicSignatureMintSource`) |
| Traits tables | `db/migrations/00029_cg_token_traits.sql` |
| Ingester worker + backoff | `internal/api/cosmicgame/traits/ingester.go` |
| Config plumbing | `internal/config/services.go` (`NFT_TRAITS_*`) |

---

## 2b. Data contract compatibility — verified, no changes needed

Checked the live generator output against the Go types on 2026-08-30 so this
does not have to be re-derived:

| Check | Result |
|---|---|
| `nftraits.File` fields vs live `nft_traits.json` | All 8 top-level keys match exactly (`schema_version`, `seed`, `pipeline_version`, `generated_at`, `attributes`, `description_art`, `simulation`, `generation`) |
| `File.validate()` required fields | All present and non-empty; `simulation`/`generation` are objects |
| `Gate()` schema major | Live file is `1.0.0`; `SupportedSchemaMajor = 1` |
| `nftraits.Manifest` vs live `assets.json` | `schema_version` is `2`, matching `SupportedManifestSchema` |
| `ManifestEntry` fields | Live entries carry `path`, `kind`, `role`, `format`, `width`, `height`, `pixel_format`, `bytes`, `sha256` |
| `masterImagePath` / `mainVideoPath` | `images/source/master.png` and `videos/web/main.mp4` both present in the manifest, so `image_details` and `animation_details` will populate |
| Seed form used to build the URL | `traitFetchCandidateSelectSQL` emits `'0x' || lower(m.seed)` from a uint256-padded value, which is exactly the package directory name (`0x` + 64 lowercase hex). Leading zeros are preserved — `CanonicalSeed` does not trim them, and only `SeedsEquivalent` does, for comparison. Seeds such as `0x0031…` therefore resolve correctly. |

So the parsing, gating and merge layers need no changes. The defect is
confined to URL construction.

---

## 3. Blocking changes

### 3.1 `internal/api/cosmicgame/traits/fetch.go`

Replace the two URL builders (lines 71-79):

```go
// trait contract, published inside the seed's package directory
func (f *fetcher) traitsURL(seed string) string {
	return f.base + "/" + seed + "/metadata/nft_traits.json"
}

// asset manifest, published alongside it
func (f *fetcher) manifestURL(seed string) string {
	return f.base + "/" + seed + "/metadata/assets.json"
}
```

Also correct the `newFetcher` doc comment (lines 49-52), which currently states
the base is the host root "not the /images mount" and that files live at
`/traits` and `/asset-manifests`. Both statements are now wrong. The base is
the **collection package root**, e.g.
`https://nfts.cosmicsignature.com/images/new/cosmicsignature`.

### 3.2 `internal/config/services.go`

`TraitsSourceBase()` (lines 180-189) currently derives the base by *stripping*
`/images`, which produces the host root. It must resolve to the collection root
instead:

```go
func (c *APIServer) TraitsSourceBase() string {
	if base := strings.TrimRight(strings.TrimSpace(c.NFTTraitsSourceBase), "/"); base != "" {
		return base
	}
	assets := strings.TrimRight(strings.TrimSpace(c.NFTAssetsPublicBase), "/")
	if assets == "" {
		return ""
	}
	return assets + "/new/cosmicsignature"
}
```

This makes the derived default consistent with `buildMedia`, which already
composes the correct package root as
`in.AssetBase + "/new/cosmicsignature/" + in.Seed`.

Update the struct comment at lines 93-95 and the function comment at lines
176-179, both of which describe the old `/traits/...` scheme.

**Deployment note:** the meaning of `NFT_TRAITS_SOURCE_BASE` changes from
"host root" to "collection package root". Any environment that sets it
explicitly to `https://nfts.cosmicsignature.com` must be updated to
`https://nfts.cosmicsignature.com/images/new/cosmicsignature`, or unset so the
derivation applies.

### 3.3 `internal/api/cosmicgame/metadata_assembly.go`

`buildMedia` lines 205-208 emit the two dead URLs into served metadata. Since
`pkg` (line 194) is already the correct package root, use it and drop the
`SourceBase` dependency:

```go
media["asset_manifest"] = pkg + "/metadata/assets.json"
media["trait_source"] = pkg + "/metadata/nft_traits.json"
```

These no longer need the `if in.SourceBase != ""` guard — they resolve from the
same base as every other media URL. Update the comment at lines 190-192, which
explains the now-removed distinction.

### 3.4 Stale comments

Same wrong `/traits/{seed}.json` scheme described in:

- `internal/api/cosmicgame/traits/ingester.go` lines 65-66
- `internal/api/cosmicgame/cosmicgame.go` lines 154-155

---

## 4. Tests to update

All assert the old URL shape and will fail after the fix:

| File | Lines |
|---|---|
| `internal/api/cosmicgame/traits/fetch_test.go` | 54, 57 (exact URL equality) |
| `internal/api/cosmicgame/traits/ingester_test.go` | 192, 197, 666 (stub server path prefixes) |
| `internal/api/apitest/metadata_traits_test.go` | 146, 149, 434 |
| `internal/api/cosmicgame/metadata_assembly_test.go` | 394, 395 |

The stub servers in `ingester_test.go` switch on `strings.HasPrefix(r.URL.Path,
"/traits/")`; they should match the new suffix form instead, e.g.
`strings.HasSuffix(r.URL.Path, "/metadata/nft_traits.json")`.

Worth adding: a regression test asserting the built URL contains
`/metadata/nft_traits.json`, so this class of mismatch fails loudly rather than
silently degrading to a permanent 404.

---

## 5. Optional: conditional requests

Not a correctness bug — the ingester works without it — but currently dead code
in practice.

`fetch.go` sends `If-None-Match` (line 90) and reads `resp.Header.Get("ETag")`
(lines 103, 111). The asset host sends **no** `ETag`, so `SourceETag` and
`ManifestETag` are always stored empty, the request header is never set, and a
`304` never occurs. Every recheck re-downloads the body.

The practical cost is negligible: 48 tokens times a few KB per recheck
interval, and `ContentHash` already suppresses redundant DB writes. Two options:

1. **Leave it.** Harmless, and it starts working for free if an `ETag` is ever
   added upstream.
2. **Switch to `Last-Modified`/`If-Modified-Since`**, which the host does send.
   Requires renaming `SourceETag`/`ManifestETag` through
   `fetchResult`, `buildUpsert`, `TokenTraitsUpsert`, and the
   `cg_token_traits_fetch` columns — a wider change than the blocking fix, so
   it is better done separately.

Recommendation: ship the URL fix first, treat this as follow-up.

---

## 6. Verification

After the change, against a backfilled seed:

1. `TraitsSourceBase()` resolves to
   `https://nfts.cosmicsignature.com/images/new/cosmicsignature`.
2. `traitsURL(seed)` returns
   `…/images/new/cosmicsignature/0x<seed>/metadata/nft_traits.json`, and that
   URL returns `200` with `application/json`.
3. The ingester promotes at least one token from fallback to enriched: the
   served `/metadata/{tokenID}` gains `properties.simulation`,
   `properties.generation`, `image_details`, and art attributes such as
   `Structure`, `Palette`, `Fate`, `Chaos`.
4. `properties.media.trait_source` and `properties.media.asset_manifest` both
   resolve to `200`.

---

## 7. Ember edition (generator 1.1.0)

Contract: [`augur-explorer-integration.md`](augur-explorer-integration.md)
§1, §2.1, §2.2, §4 step 1 and §5.3 item 6. Generator side:
[`ember-edition.md`](ember-edition.md).

### 7.1 What changes on the asset host

- **New packages** carry eight more files. Seven media files are listed in
  `assets.json` under the roles `ember_source_master`, `ember_web_full`,
  `ember_web_preview`, `ember_web`, `ember_medium_web`, `ember_slow_web` and
  `ember_hq`, each with `"color_space": "srgb"`. The certificate,
  `metadata/ember.json`, has no manifest entry. The manifest stays at
  `schema_version` 2. `ember_medium_web` (`videos/web/ember_medium.mp4`, the
  ember video four times slower) is new with the look `ember-v4`, and
  `ember_slow_web` (`videos/web/ember_slow.mp4`, ten times slower) came with
  `ember-v3`. An `ember-v3` edition has six roles, the editions of the older
  looks five, and §7.3 defines no key for either slow film.
- **The 48 existing packages** get the edition from the ember backfill, one
  per sync run; it started when the edition was deployed (2026-09-29). In
  its default mode the backfill uploads only the ember files and a merged
  `assets.json`: the live non-ember entries are kept verbatim, the `ember_*`
  entries are added, and `generated_at` changes. `nft_traits.json`,
  `generation.json` and the main art are untouched.
- **A new ember look withdraws the published editions, then renders them
  again.** The certificate's `algorithm` names the look (`ember-v1`, then
  `ember-v2`, `ember-v3` and `ember-v4`). The first sync run after a deploy
  that changes it withdraws every edition of the older look: it deletes
  `metadata/ember.json`, rewrites `assets.json` without the `ember_*`
  entries (every other entry and field, `generated_at` included, unchanged)
  and deletes the edition's media files. `nft_traits.json` keeps its bytes
  and `Last-Modified`. The backfill then renders each edition again, one
  package per sync run (3 to 5½ hours each on the generation host with
  `ember-v3`, so about 8 days for all 48, and an estimated 1 to 2 hours more
  each with `ember-v4`, about 11 to 12 days; see [§8](#8-rollout-notes)),
  and the `ember_*`
  entries return with new `bytes` and `sha256`. Until a token's turn comes,
  it has no ember edition. The `ember-v2` change was the first change of
  look, and it withdrew every `ember-v1` edition; `ember-v3` was the second,
  deployed with the older editions kept online (§8); `ember-v4` is the
  third. Withdrawing first is the sync loop's default. The operator can
  instead keep the older
  editions online (`COSMICSIG_KEEP_STALE_EMBER=yes`): nothing is withdrawn
  then, and the backfill replaces each edition in place when its turn
  comes, so the `ember_*` entries change without disappearing in between.
- **The certificate's layout** is `schema_version` 5 with `ember-v4`:
  `inputs.frames.slow_factors` (a list) replaces `inputs.frames.slow_factor`,
  and `outputs.slow_films`, one record per slow film, replaces
  `derived.slow_first_frame`, `outputs.slow_frames_rgb48le_sha256` and
  `outputs.slow_frames_emitted`. It was 4 with `ember-v3`, where
  `inputs.view` and those slow-film fields were new and `config.projection`
  and `derived.projection` went, and 3 with `ember-v2`, where the cinnabar
  statistics went and `derived.hold_time`, `derived.fade_time`,
  `derived.tidal_reference` and `config.tidal` were new
  ([integration §2.2](augur-explorer-integration.md#22-emberjson-the-certificate)).
  The Go side only links the certificate, so this matters only to code
  that parses it.
- **`pipeline_version`** is `1.1.0` in packages generated by the new binary
  (new mints). Existing tokens keep `1.0.0` after their edition lands,
  because the default backfill leaves `nft_traits.json` alone. (A backfill
  run deliberately with `--backfill-mode full` replaces the whole package,
  trait file included, and its `pipeline_version` becomes `1.1.0`.)
  `pipeline_version` therefore cannot tell whether a token has the edition;
  only the manifest roles can.
- **Some packages never get it.** A package whose ember render failed is
  uploaded without the edition. The backfill retries it, and gives up after
  3 failed attempts with the same generator binary. Serving such a token
  without `ember_*` keys is correct.

### 7.2 Manifest parsing

The ember entries are the first to carry `color_space`. Check that decoding
`nftraits.Manifest` / `ManifestEntry` tolerates the key. If the decoder uses
`DisallowUnknownFields`, add the field:

```go
ColorSpace string `json:"color_space,omitempty"` // "srgb" on ember entries
```

Otherwise every new manifest fails to parse.

### 7.3 `properties.media` keys (`buildMedia`)

Build them from `pkg`, the package root, like every other media URL. Emit
them from the stored manifest row only, and never probe the asset host in
the request path:

| Key | Path under `pkg` | Emit when the stored manifest lists |
|---|---|---|
| `ember_image` | `/images/web/ember_full.webp` | `ember_source_master` |
| `ember_still` | `/images/source/ember.png` | `ember_source_master` |
| `ember_video` | `/videos/web/ember.mp4` | `ember_source_master` and `ember_web` |
| `ember_hq_video` | `/videos/hq/ember.mp4` | `ember_source_master` and `ember_hq` |
| `ember_certificate` | `/metadata/ember.json` | `ember_source_master` |

Never gate them on `pipeline_version`. `image` and `animation_url` stay
unchanged. When a re-checked manifest loses `ember_source_master` (a
withdrawn edition, §7.1), the keys go with it.

### 7.4 Ingester: periodic manifest re-check

1. **Candidates.** At a slower cadence than the main loop (e.g. hourly),
   select **every** row, in addition to the seeds missing from the traits
   table. Do not stop re-checking a row once `ember_source_master` appears:
   a withdrawal removes it again (§7.1). A manifest is a few kilobytes, so
   one GET per token per hour is harmless. (A conditional request needs the
   manifest's own `Last-Modified`; the stored one belongs to the trait
   file.)
2. **Fetch `assets.json` on its own.** After an ember backfill or a
   withdrawal the trait file is byte-identical. A conditional request for it
   answers `304` once §5 option 2 lands, and its body hash does not change.
   Neither may short-circuit the manifest refresh. In particular, if
   `ContentHash` covers only the trait file, the new manifest would never be
   written.
3. **Upsert** the row's manifest whenever the fetched manifest differs from
   the stored one. A changed manifest (`ember_*` entries added or removed,
   new `bytes` and `sha256` on them, a new `generated_at`) is expected, not
   an incident. Only the trait file's art values are frozen.

### 7.5 Tests to add

- `buildMedia`: a manifest with all seven ember roles (`ember_medium_web`
  and `ember_slow_web` included) emits the five keys and nothing for the
  slow films; manifests with the six roles of an `ember-v3` edition and the
  five of an `ember-v2` edition emit the same five keys; a manifest without
  `ember_source_master` emits none; a manifest without `ember_web` omits
  `ember_video`.
- A row with `pipeline_version` `1.0.0` and an ember manifest emits the keys
  (proves there is no version gating).
- Ingester: a stub server whose `nft_traits.json` is unchanged (same body,
  or `304` to `If-Modified-Since`) while `assets.json` gains the ember
  entries. The stored manifest is updated and the served metadata gains the
  keys.
- Ingester, withdrawal: a row whose stored manifest lists the ember roles
  is still re-checked. The stub's `assets.json` loses the `ember_*`
  entries (everything else unchanged): the stored manifest is updated, the
  served metadata loses the keys, and nothing is logged as an incident.
  When the entries return with new `sha256` values, the keys return.
- Manifest decoding accepts an entry with `color_space`.

### 7.6 Verification after the first backfilled token

1. Its `metadata/assets.json` lists the `ember_*` roles, and its
   `metadata/nft_traits.json` still has `pipeline_version` `1.0.0` and an
   unchanged `Last-Modified`.
2. Within one re-check interval the served `/metadata/{tokenID}` gains the
   five `ember_*` keys, and each resolves to `200`.

After the deploy of `ember-v2`: within one re-check interval of the first
sync run, a token whose `ember-v1` edition was withdrawn is served without
the `ember_*` keys. Once its edition has been rendered again (its
`metadata/ember.json` records `"algorithm": "ember-v2"`), the keys return
within one interval, and each resolves to `200`.

---

## 8. Rollout notes

- **Traits backfill: done.** Every existing token has its trait file and
  hashed manifest. A `404` on a trait URL now means a new mint the generator
  has not reached yet. That usually lasts 4 to 6 hours: the sync timer
  starts a run within 5 minutes, the package took 3 to 5½ hours to render
  with the `ember-v3` look (an estimated 4 to 7½ hours with `ember-v4`), and
  its trait file is uploaded after its media. It lasts longer when several
  tokens are minted together, because packages are generated one at a
  time, and up to one more package render when the mint arrives while a
  backfill package is rendering. The ingester already treats `404` as
  `fetchMissing` rather than an error. An alert on a lasting trait `404`
  should allow about 8 hours for each package generated ahead of it (the
  costliest `ember-v4` orbit's estimate), with a margin: the cost differs
  from orbit to orbit, and the generation host may be shared with other
  work
  ([`ember-edition.md`](ember-edition.md#runtime) has the measurements).
- **Ember backfill: running since the edition was deployed (2026-09-29).**
  It adds the edition to one existing token per sync run. Until a token's
  turn comes, its manifest has no `ember_*` roles and it is served without
  the keys. That is expected. §7 does not gate this rollout: until §7
  ships, every token, new mints included, is served without the `ember_*`
  keys. Ship §7.3 and §7.4 together. When §7.4 goes live, its first
  re-check pass picks up every token backfilled so far. The exception is
  §7.2: if the manifest decoder rejects unknown fields, fix it now, because
  the new mints' and the backfilled tokens' manifests carry `color_space`.
- **`ember-v2`: deployed 2026-09-30.** Its first sync run withdrew every
  `ember-v1` edition, and the backfill rendered each token's edition again,
  one per sync run (about 2 hours each); it had reached 15 of the 48 tokens
  when it was paused on 2026-10-01.
- **`ember-v3`: deployed 2026-10-03.** It added the slow film (role
  `ember_slow_web`). The operator kept the `ember-v2` editions online
  (`COSMICSIG_KEEP_STALE_EMBER=yes`), so the backfill replaces each in
  place, one per sync run, and their `ember_*` entries change without
  disappearing in between.
- **`ember-v4`: pending deployment.** It adds the medium film (role
  `ember_medium_web`) and renders every edition again, one per sync run. By
  default each older edition is withdrawn first; in between, a token is
  served without the `ember_*` keys, which is expected. A §7.4 re-check that
  skips rows with the ember roles would instead keep serving the withdrawn
  edition's keys, pointing at deleted files. Once the pass is done, refresh
  the marketplaces again (integration §9, step 6).
