# Reviewed artwork checkpoints

`color-and-form-v1.json` pins the version accepted for further development on
`codex/estuary-color-and-form`. It records the renderer and source-cohort commits
and hashes, all thirty seed/count/flow recipes, complete material identities,
and image/movie hashes for the fifty published optical views.

All thirty recipes were independently reconstructed from the pinned renderer's
`study_recipe(count, flow, width=2048, film=True)` function, the displayed recipe
name, and the exact seeded background, then compared with the published requests.
Recipe hashes use `tools.estuary_studio.common.encoded` canonical JSON encoding.
The recorded `checkpoint_source_commit` is the code/documentation parent of this
manifest commit; it deliberately does not attempt to hash its own Git commit.

Large trajectory recordings and rendered media remain in their verified archives
outside Git. Their content hashes bind the checkpoint to those exact artifacts.
The existing gallery remains at <http://127.0.0.1:8796/films/>. New interaction
texture experiments should start from this baseline and use separate recipes
and output archives. This checkpoint adds no texture effects or renderer changes.
