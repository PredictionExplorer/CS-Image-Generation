#!/usr/bin/env python3
"""Render the same seeds through the v1 (Bruton, ortho) and v2 (CIE, perspective)
pipelines and produce a side-by-side HTML contact sheet for subjective review.

Usage:
    python3 ci/compare_v1_v2.py --seeds 0xdeadbeef,0xcafebabe \\
        --resolution 512x288 --steps 40000 --out out/compare

The script invokes ``cargo run --release --`` twice per seed and collects
``image.png`` from the per-seed subdirectories of ``--out``.

Each row in the generated ``index.html`` shows the v1 and v2 outputs together
with the seed so reviewers can flip defaults after a manual pass.
"""

from __future__ import annotations

import argparse
import html
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path


@dataclass
class RenderPair:
    seed: str
    v1_path: Path | None
    v2_path: Path | None


def run(cmd: list[str]) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def render_for_seed(
    seed: str,
    out_root: Path,
    resolution: str,
    steps: int,
    sims: int,
    color_science: str,
    no_perspective: bool,
    no_director: bool,
    shutter_samples: int,
) -> Path:
    tag = f"{color_science}_{'orth' if no_perspective else 'persp'}_" \
        f"{'uni' if no_director else 'dir'}_sh{shutter_samples}"
    out_dir = out_root / f"{seed}__{tag}"
    out_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        "cargo", "run", "--release", "--",
        "--seed", seed,
        "--output", str(out_dir / "seed"),
        "--resolution", resolution,
        "--steps", str(steps),
        "--sims", str(sims),
        "--color-science", color_science,
        "--shutter-samples", str(shutter_samples),
        "--fast-encode",
        "--log-level", "warn",
    ]
    if no_perspective:
        cmd.append("--no-perspective")
    if no_director:
        cmd.append("--no-director")
    run(cmd)

    generated = out_dir / "seed" / "image.png"
    if not generated.exists():
        # Some variants place output under <tag>/seed/image.png; walk up to find it.
        candidates = list(out_dir.rglob("image.png"))
        if candidates:
            generated = candidates[0]
    final = out_dir / "image.png"
    if generated.exists() and generated != final:
        shutil.copy2(generated, final)
    return final


def write_contact_sheet(pairs: list[RenderPair], out_root: Path) -> Path:
    html_path = out_root / "index.html"
    rows = []
    for pair in pairs:
        v1_src = pair.v1_path.relative_to(out_root) if pair.v1_path else ""
        v2_src = pair.v2_path.relative_to(out_root) if pair.v2_path else ""
        rows.append(
            "<tr>"
            f"<td><code>{html.escape(pair.seed)}</code></td>"
            f"<td>{_img_cell(v1_src, 'v1 (bruton, ortho)')}</td>"
            f"<td>{_img_cell(v2_src, 'v2 (CIE, perspective, director)')}</td>"
            "</tr>"
        )
    body = (
        "<html><head><meta charset=\"utf-8\"><title>v1 vs v2</title>"
        "<style>"
        "body{background:#111;color:#eee;font-family:system-ui;padding:24px;}"
        "table{border-collapse:collapse;width:100%;}"
        "td{border:1px solid #333;padding:8px;vertical-align:top;}"
        "img{max-width:100%;display:block;}"
        ".label{font-size:12px;color:#aaa;margin-top:4px;}"
        "</style>"
        "</head><body>"
        "<h1>three-body v1 vs v2 contact sheet</h1>"
        "<p>Left: current default (Bruton RGB, orthographic, uniform pacing). "
        "Right: v2 defaults (CIE 1931, perspective, multi-act director, shutter blur).</p>"
        "<table><thead><tr><th>seed</th><th>v1</th><th>v2</th></tr></thead>"
        "<tbody>" + "\n".join(rows) + "</tbody></table></body></html>"
    )
    html_path.write_text(body, encoding="utf-8")
    return html_path


def _img_cell(rel: Path | str, label: str) -> str:
    if not rel:
        return f"<div class=\"label\">{label} (missing)</div>"
    src = str(rel).replace("\\", "/")
    return (
        f"<a href=\"{html.escape(src)}\"><img src=\"{html.escape(src)}\"/></a>"
        f"<div class=\"label\">{html.escape(label)}</div>"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument(
        "--seeds",
        required=True,
        help="Comma-separated list of seeds (hex 0x...) to render through both pipelines",
    )
    parser.add_argument("--out", required=True, type=Path, help="Output directory root")
    parser.add_argument("--resolution", default="512x288")
    parser.add_argument("--steps", type=int, default=40_000)
    parser.add_argument("--sims", type=int, default=4000)
    parser.add_argument(
        "--shutter-samples-v2",
        type=int,
        default=4,
        help="Shutter samples per step for v2 (default 4; v1 uses 1)",
    )
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    pairs: list[RenderPair] = []
    for seed in [s.strip() for s in args.seeds.split(",") if s.strip()]:
        v1 = render_for_seed(
            seed, args.out, args.resolution, args.steps, args.sims,
            color_science="bruton", no_perspective=True, no_director=True,
            shutter_samples=1,
        )
        v2 = render_for_seed(
            seed, args.out, args.resolution, args.steps, args.sims,
            color_science="cie", no_perspective=False, no_director=False,
            shutter_samples=args.shutter_samples_v2,
        )
        pairs.append(RenderPair(
            seed=seed,
            v1_path=v1 if v1.exists() else None,
            v2_path=v2 if v2.exists() else None,
        ))

    out_html = write_contact_sheet(pairs, args.out)
    print(f"Wrote contact sheet => {out_html}")


if __name__ == "__main__":
    main()
