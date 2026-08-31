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

## 7. Rollout note: expect 404s for now

The generator-side backfill is **in progress**. As of 2026-08-30 evening, 14 of
48 tokens have a trait file; the remainder return `404` until the sync loop
reaches them (expected to complete late Monday 2026-08-31).

This is normal, and the ingester already treats `404` as `fetchMissing` rather
than an error. Do not interpret a low ingest count during this window as a bug
— confirm against the completed-token count first. Once the backfill finishes,
all 48 should ingest within a couple of ticks at the default 2-minute interval.
