# Consuming `nft_traits.json` in augur-explorer

This document specifies exactly how the `PredictionExplorer/augur-explorer`
repository must be modified to consume the per-seed trait packages produced by
this repository and serve world-class ERC-721 metadata for Cosmic Signature.

It is the **contract document** between the two repositories. The Rust
generator owns everything in it that is derivable from the seed; the Go
metadata server owns everything that comes from chain events; the HTTP handler
only merges.

> **The one rule:** if a fact can be recomputed from the seed, it is computed
> in Rust and shipped in `nft_traits.json` — the Go side copies it verbatim.
> If a fact comes from an event log (round, mint time, prize track, token
> name, ownership), it is read from the indexer database. The handler never
> computes art facts and the generator never guesses chain facts.

---

## 1. What the generator uploads

`run.py` uploads one package per seed to `COSMICSIG_REMOTE_DIR` on the asset
host. As of generator pipeline `1.0.0+traits` the package contains two files
the metadata server must consume (both are now in `REQUIRED_PACKAGE_FILES`,
so the sync loop automatically regenerates and re-uploads every existing
package that lacks them — that is the backfill mechanism):

```text
0x<seed>/
  images/…                      (unchanged)
  videos/…                      (unchanged)
  spectral/…                    (unchanged)
  metadata/
    generation.json             (internal reproducibility log — not served)
    assets.json                 (schema_version 2: + sha256 per file)
    nft_traits.json             (NEW — the public trait contract, schema 1.x)
```

Determinism guarantee: every field in `nft_traits.json` except `generated_at`
is a pure function of the seed. Re-running the generator for the same seed
reproduces identical values, so anyone can verify served traits against the
CC0 pipeline.

### 1.1 Expose the metadata files over HTTP

The images are already served publicly (e.g.
`https://nfts.cosmicsignature.com/images/new/cosmicsignature/0x<seed>.png`).
Expose the two JSON files the same way with an nginx alias on the asset host:

```nginx
# Trait contract files, one per seed.
location ~ ^/traits/(0x[0-9a-f]+)\.json$ {
    alias /home/frontend/nft-assets/new/cosmicsignature/$1/metadata/nft_traits.json;
    types { } default_type application/json;
    add_header Cache-Control "public, max-age=300";
    etag on;
}
location ~ ^/asset-manifests/(0x[0-9a-f]+)\.json$ {
    alias /home/frontend/nft-assets/new/cosmicsignature/$1/metadata/assets.json;
    types { } default_type application/json;
    add_header Cache-Control "public, max-age=300";
    etag on;
}
```

Publishing these URLs is deliberate: collectors and third parties can fetch
the raw trait contract and verify it against the pipeline.

---

## 2. The `nft_traits.json` contract

- Formal schema: [`docs/nft_traits.schema.json`](nft_traits.schema.json)
  (JSON Schema draft 2020-12).
- Real example produced by the generator:
  [`docs/fixtures/nft_traits.example.json`](fixtures/nft_traits.example.json)
  (seed `0x100033` at reduced `--sims 200 --steps 20000`; the file records its
  own parameters, so it is self-describing).

Top-level layout:

| Field | Type | Use in the served metadata |
|---|---|---|
| `schema_version` | semver string | **Gate on the major component.** Serve the fallback (Section 5.5) and alert if `major != 1`. |
| `seed` | `0x…` string | Sanity-check against the DB row's seed. |
| `pipeline_version` | string | Copy into `properties.generation`. |
| `generated_at` | RFC 3339 | Informational; not deterministic. Do not compare across runs. |
| `attributes` | array | **Copy verbatim**, then append chain attributes (Section 5.3). Never rename, re-bucket, or reorder the art attributes. |
| `description_art` | string | First sentence(s) of the served `description`; append the provenance sentence. |
| `simulation` | object | Copy verbatim into `properties.simulation`. |
| `generation` | object | Copy verbatim into `properties.generation`. |

Art attributes the generator may emit (values are final display strings):

| `trait_type` | Presence | Values |
|---|---|---|
| `Structure` | always | 9 vocabulary names (`Triangle Web` … `Tangent Caustics`) |
| `Underlay` / `Accent` | ~58% / ~10% of seeds | vocabulary names; omitted when absent |
| `Symmetry` | ~15% | `Mirror`, `Mandala ×k`, `Rosette ×k`; omitted when none |
| `Projection` | ~18% | `Phase Portrait`, `Cross Braid`, `Hodograph`; omitted for position space |
| `Finish` | rare, repeatable | one entry per active effect: `Prism` (5%), `Diffraction Spikes` (4%), `Stardust` (10%) |
| `Wildcard` | ~7% | `Yes`; omitted otherwise |
| `Palette` | always | family name, e.g. `Aurora Split`, `Ember Triad` |
| `Spectral Class` | always | `O B A F G K M` |
| `Mass Balance` | always | `Equal Trio`, `Heavy Primary`, `Twin Binary` |
| `Fate` | always | `Eternal Dance`, `Ejection` |
| `Chaos` | always | number 0–100 (`display_type: number`, `max_value: 100`) |
| `Syzygies` | always | number (`display_type: number`) |

Versioning policy: minor/patch bumps only add fields or attribute kinds and
are safe to serve without code changes. A major bump means the layout changed;
the ingester must refuse the file, keep serving the last good version (or the
fallback), and page a human.

### 2.1 `assets.json` (schema_version 2)

Unchanged fields plus `sha256` (lowercase hex, SHA-256) on every single-file
entry. Use it to build `image_details` / `animation_details` (Section 5.3) —
do not hash files on the Go side.

---

## 3. Database migration

One new table keyed by seed (PostgreSQL types; adjust to the store in use):

```sql
CREATE TABLE cs_token_traits (
    seed             TEXT PRIMARY KEY,          -- 0x-prefixed lowercase
    schema_major     INT         NOT NULL,
    pipeline_version TEXT        NOT NULL,
    attributes       JSONB       NOT NULL,      -- verbatim array from the file
    description_art  TEXT        NOT NULL,
    simulation       JSONB       NOT NULL,      -- verbatim block
    generation       JSONB       NOT NULL,      -- verbatim block
    assets           JSONB,                     -- assets.json v2 (dimensions/bytes/sha256)
    source_etag      TEXT,
    fetched_at       TIMESTAMPTZ NOT NULL DEFAULT now()
);
```

Storing the blocks verbatim (rather than exploding them into columns) keeps
the Go side forward-compatible with minor schema additions.

---

## 4. Ingest job

Add a small ingester to the cosmicgame module (cron-style loop, e.g. every
2 minutes, plus an on-demand attempt from the handler with a short negative
cache):

1. Select token seeds present in the token table but missing from
   `cs_token_traits` (or with a stale `pipeline_version` after a announced
   regeneration).
2. For each seed, `GET {ASSETS_BASE}/traits/0x<seed>.json` with
   `If-None-Match: <source_etag>`.
   - `404` → package not rendered yet (new mints lag the generator by
     minutes to hours). Not an error; retry on the next tick.
   - `304` → up to date; skip.
   - `200` → parse, then **gate**: `semver_major(schema_version) == 1` and
     `seed` matches. On mismatch: log at error level, increment a metric,
     do not upsert.
3. Fetch `{ASSETS_BASE}/asset-manifests/0x<seed>.json` the same way (optional
   but recommended; required for `image_details`).
4. Upsert the row in one transaction.

Sketch:

```go
func (ing *TraitsIngester) ingestSeed(ctx context.Context, seed string) error {
    url := ing.assetsBase + "/traits/" + seed + ".json"
    body, etag, status, err := ing.fetch(ctx, url, ing.knownEtag(seed))
    if err != nil || status == http.StatusNotFound || status == http.StatusNotModified {
        return err // 404/304 are not failures; the caller just moves on
    }
    var tf NftTraitsFile
    if err := json.Unmarshal(body, &tf); err != nil {
        return fmt.Errorf("traits %s: malformed JSON: %w", seed, err)
    }
    if major := semverMajor(tf.SchemaVersion); major != 1 {
        ing.metrics.SchemaMismatch.Inc()
        return fmt.Errorf("traits %s: unsupported schema %s", seed, tf.SchemaVersion)
    }
    if !strings.EqualFold(tf.Seed, seed) {
        return fmt.Errorf("traits %s: seed mismatch %s", seed, tf.Seed)
    }
    return ing.store.UpsertTokenTraits(ctx, seed, tf, etag)
}
```

Never fetch the trait file inside the HTTP request path without a cache: the
handler reads only the DB row.

---

## 5. Metadata handler changes

All changes are in `internal/api/cosmicgame/api_cosmicgame_metadata.go`
(`handleCstMetadata`, dispatched from `TokenMetadata` for the bare
`/metadata/{tokenID}` route via `internal/api/common/metadata_host.go`).

### 5.1 Remove `properties.owner` — unconditionally

Ownership changes on every transfer while marketplaces cache metadata
aggressively; a baked-in owner is permanently stale. `ownerOf()` on-chain is
the source of truth. Remove it from the served JSON in **both** the enriched
and fallback paths. This is the one change that must ship regardless of
everything else.

### 5.2 Load the traits row

After `CosmicSignatureTokenInfo`, load `cs_token_traits` by the token's seed
(normalize to `0x` + lowercase — the DB currently stores the seed without a
guaranteed prefix; `handleCstMetadata` already normalizes for image URLs).
Row present → enriched path; row absent → fallback path (5.5).

### 5.3 Enriched response assembly

Merge order and rules:

1. **`attributes`** = trait-file attributes, verbatim and first, then append
   chain attributes:
   - `Round` (number) — as today.
   - `Imprinted` (date) — as today, omitted when the mint timestamp is
     unavailable.
   - `Allocation` (string) — Section 6. Omit until implemented.
   - **Drop the `seed` attribute entirely.** Every value is unique, so it is
     useless as a filter and pollutes the trait panel; the canonical seed
     stays in `properties.seed`.
2. **`description`** = `description_art` + one provenance sentence built from
   chain data, e.g.
   `" Imprinted in Round {N} on {DD Mon YYYY}. Same seed, same pixels — re-render it to verify."`
3. **`name`** — unchanged (custom on-chain names already work).
4. **`image` / `animation_url`** — unchanged URL scheme.
5. **`image_details` / `animation_details`** — from the stored `assets` JSON:
   match `images/source/master.png` and `videos/web/main.mp4`; emit
   `{ width, height, format, bytes, sha256 }` and
   `{ width, height, codec, duration_seconds, bytes, sha256 }`. Omit the
   whole object when the manifest row is missing.
6. **`properties`**:

   ```jsonc
   {
     "seed": "0x…",                 // canonical, 0x-prefixed lowercase
     "token_id": 47,
     "round_num": 1,
     "simulation": { … },           // verbatim from the trait file
     "generation": { … },           // verbatim from the trait file
     "media": {                     // every hosted asset, discoverable at last
       "hq_video":        "{base}/…/videos/hq/main.mp4",
       "spectral_sweep":  "{base}/…/videos/web/spectral_sweep.mp4",
       "spectral_sweep_hq":"{base}/…/videos/hq/spectral_sweep.mp4",
       "spectral_bins":   "{base}/…/spectral/",   // 64 x 16-bit PNG
       "asset_manifest":  "{base}/asset-manifests/0x<seed>.json",
       "trait_source":    "{base}/traits/0x<seed>.json"
     }
   }
   ```

   (Adjust the media URLs to however nginx exposes the package tree; the
   point is that the HQ video, sweep videos, and spectral bins stop being
   invisible.)
7. **`metadata_version`** — add a top-level field, start at `"2.0.0"`, bump
   on layout changes of the *served* JSON (independent of the trait-file
   schema version).
8. `background_color`, `external_url` — unchanged.

### 5.4 Do-not list

- Do **not** recompute, rename, re-bucket, or reorder art attributes.
- Do **not** serve `properties.owner` or any other mutable-by-transfer data.
- Do **not** fetch the asset host synchronously inside the request path.
- Do **not** serve a trait row whose `schema_major != 1`.
- Do **not** mutate a token's art traits after first serve — they are frozen
  at imprint by construction; treat any observed diff between a re-fetched
  trait file and the stored row (other than `generated_at`) as an incident.

### 5.5 Fallback path (traits not ingested yet)

Serve exactly today's minimal JSON minus `properties.owner` and minus the
`seed` attribute. New mints will use this path for the minutes-to-hours until
the generator uploads the package; marketplaces then pick up the enriched
version on their next refresh.

---

## 6. The `Allocation` attribute (chain-owned)

Each Cosmic Signature NFT is minted through a specific prize path that the
indexer already records (prize/raffle tables in `internal/store/cosmicgame/`).
Map the mint event to one of:

`Final Gesture` · `Endurance Champion` · `Chrono-Warrior` ·
`Stellar Selection` · `Anchored Selection` · `Last CST Gesture`

Emit it as a string attribute appended after `Imprinted`. This is provenance
storytelling the trait file cannot know — it is entirely the Go side's.

---

## 7. Collection-level metadata (ERC-7572)

Serve a `contractURI` JSON (static file or handler) with collection `name`,
`description`, `image`, `banner_image`, and `external_link`, and point the
NFT contract's `contractURI()` at it if/when the contract supports it.
Marketplaces read this for collection pages; nothing else in this integration
depends on it.

---

## 8. Caching

- `Cache-Control: public, max-age=300` plus a strong `ETag` (hash of the
  serialized body) on `/metadata/{id}`. Named-token renames and trait
  ingestion both change the ETag naturally.
- Keep the existing `Access-Control-Allow-Origin: *`.

---

## 9. Rollout order (zero downtime)

1. **Go**: ship the migration, the ingester, and the handler changes. With no
   trait files on the asset host yet, every token serves the fallback path —
   behavior is today's, minus `owner`.
2. **Rust** (this repo, already done): generator emits `nft_traits.json` +
   hashed `assets.json`; `run.py` requires the new file.
3. **Backfill**: the `cosmicsig-sync` timer sees every existing package as
   incomplete and regenerates + re-uploads all of them. Art is
   pixel-identical (deterministic pipeline; CI reference hashes prove it).
   Video *bytes* may differ (ffmpeg container metadata) — the manifest hashes
   describe the uploaded files, so consistency holds.
4. **Ingest**: the cron ingests each package as it lands; tokens flip from
   fallback to enriched automatically.
5. **Marketplace refresh**: emit ERC-4906 `BatchMetadataUpdate(0, maxId)` if
   the NFT contract supports it; otherwise trigger OpenSea's per-collection
   metadata refresh. Existing holders wake up to rich traits on unchanged
   art.

---

## 10. Testing

- Vendor [`docs/fixtures/nft_traits.example.json`](fixtures/nft_traits.example.json)
  from this repository into the Go repo's test data (`go:embed`). It is a
  real generator output.
- Contract test: unmarshal the fixture, run the merge with a synthetic token
  row, and golden-compare the served JSON. Assert specifically:
  - art attributes appear verbatim and before chain attributes;
  - `owner` is absent;
  - the `seed` attribute is absent while `properties.seed` is present;
  - `description` starts with `description_art`.
- Schema gate test: bump the fixture's `schema_version` to `2.0.0` and assert
  the ingester refuses it.
- Fallback test: no traits row → served JSON matches today's shape minus
  `owner`.
- When regenerating fixtures, any diff other than `generated_at` for the same
  seed and parameters is a determinism regression in the generator — report
  it here, do not paper over it in Go.

---

## 11. Reference: generator-side sources

| Artifact | Source |
|---|---|
| Trait file writer | `src/nft_traits.rs` (`compute_and_write`) |
| Physics analyses | `src/traits_analysis.rs` (syzygies, braid, fate, chaos index) |
| Palette family naming | `src/render/color.rs` (`palette_family`, `PaletteDetails`) |
| Asset hashes | `src/app.rs` (`write_asset_manifest`, schema_version 2) |
| Fast local iteration | `three_body_problem --metadata-only` (skips all rendering; identical trait values) |
| Package requirements | `run.py` `REQUIRED_PACKAGE_FILES` |
| Formal schema | `docs/nft_traits.schema.json` |
