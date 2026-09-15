# Six studies of one orbit

Seed: **`0xb7f327f9f722`**

Open **index.html** to review the collection. The gallery provides full-screen
viewing, synchronized comparison, slow motion, an original-video reference, and
private notes with timestamps. Export your notes to keep a portable copy.

## The films

1. **Tidal Calligraphy** — flowing trajectory ribbons and fine translucent fibers.
2. **Gravity Loom** — pairwise relationships expressed through an open woven form.
3. **Aurora Veils** — broad, transparent curtains carrying the motion through light.
4. **Light Cast by Gravity** — projected light patterns driven by the three bodies.
5. **Orbital Engraving** — changing line families with fine engraved detail.
6. **Eclipse Garden** — dark forms shaping luminous openings and negative space.

The gallery marks only fully verified films as complete. Every completed study
contains the full recorded interval: **3840×2160, 60 fps, 1,802 frames**, approximately
**30.03 seconds**. The original reference uses its original 1280×828 presentation.

## Picture quality

- **Highest quality** plays the 10-bit HEVC master when the browser supports it.
- **Compatibility version** uses high-quality H.264 at the same 4K resolution.
- Original RGB16 PNG frames remain archived on the experiment server.

Both movie versions are inside each study folder as `master.mp4` and `web.mp4`.
The poster is an actual 4K frame. `recipe.json`, `render.json`, `assembly.json`,
and `verification.json` record settings, provenance, and complete-video checks.

The **Encounter** button selects the nearest frame to the A–C closest physical
approach in the recorded interval, around **27.8 seconds**. Different artistic
mappings can make the same physical event look very different on screen.

## Source and reproduction

All artwork is generated in Rust on CPUs. The six studies use the same original
physical orbit. Their geometry, materials, history, and graphic or optical
interpretations are designed expressions of that motion.

History-based forms can use verified motion from the original simulation's
warm-up before the first visible frame. This supplies an already formed opening
while keeping the recorded visible body positions and timing intact.

Code: branch `codex/six-art-studies` in the `CS-Tidal-Silk` worktree.
Server archive: `/home/user/tidal-silk/six-studies-b7/`.
Frozen source orbit: `/home/user/tidal-silk/results/0xb7f327f9f722.orbit`.

If a browser restricts local movie playback, run the supplied review server:

```sh
python3 serve_review.py . --port 8767
```

Then open `http://127.0.0.1:8767/`. The server accepts only local connections.
