#!/usr/bin/env python3
"""Plan, verify and publish complete-history shell/normal comparisons.

plan --output DIR [--samples 1000000] [--normal normal.json]
init --collection DIR --seeds 0x... 0x...
publish-stills --collection DIR --still DIR --geometry DIR --orbit FILE [--normal DIR]
publish --collection DIR --normal DIR --shell DIR --still DIR --orbit FILE

Publish accepts the normal replay directory, shell film archive and final still
render archive. It requires native 3840-pixel media. Originals are copied, never
modified. The 30 fps comparison samples the end of each pair of normal frames;
there is no interpolation, crop or additional hold. Completed output identities
are immutable; incomplete work can be retried. Runtime dependencies: FFmpeg and
ffprobe. All Python operations use the standard library.
"""

import argparse
import fcntl
import hashlib
import json
import math
import re
import shutil
import struct
import subprocess
from fractions import Fraction
from pathlib import Path

NORMAL_PAIR_FILTER = (
    "[0:v]select='mod(n,2)',setpts=N/(30*TB),"
    "colorspace=ispace=bt709:itrc=iec61966-2-1:iprimaries=smpte432:irange=tv:"
    "space=bt709:trc=iec61966-2-1:primaries=bt709:range=tv:format=yuv420p[out]"
)


def encoded(value):
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def read(path):
    return json.loads(path.read_text())


def digest(path):
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            result.update(chunk)
    return result.hexdigest()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".partial")
    partial.write_bytes(encoded(value) if not isinstance(value, str) else value.encode())
    partial.replace(path)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def alignment(samples=1_000_000):
    """Reproduce normal::frame_schedule, then select its odd frame indices."""
    require(type(samples) is int and 2 <= samples <= 100_000_000, "Invalid sample count")
    step = max(samples // 1800, 1)
    checkpoints = list(range(step, samples, step))
    if not checkpoints or checkpoints[-1] != samples - 1:
        checkpoints.append(samples - 1)
    require(len(checkpoints) % 2 == 0, "Paired 30 fps schedule requires an even normal frame count")
    frames = [
        {
            "frame": i // 2,
            "normal_frame": i,
            "checkpoint": point,
            "source_fraction": point / (samples - 1),
        }
        for i, point in enumerate(checkpoints)
        if i % 2
    ]
    return {
        "schema_version": 1,
        "source_sample_count": samples,
        "normal_fps": 60,
        "fps": 30,
        "normal_checkpoints": checkpoints,
        "frames": frames,
        "duration_seconds": len(frames) / 30,
        "semantics": "Inclusive accumulated source history; select normal indices 1,3,...; "
        "pair endpoints are shifted 1/60 second earlier in the 30 fps output.",
    }


def validate_normal(normal, plan):
    require(normal.get("complete") is True, "Normal render is incomplete")
    require(
        normal.get("source_sample_count") == plan["source_sample_count"], "Sample count differs"
    )
    require(
        normal.get("frame_checkpoint_indices") == plan["normal_checkpoints"],
        "Normal checkpoints differ from the planned schedule",
    )
    require(
        normal.get("fps") == 60 and normal.get("frame_count") == len(plan["normal_checkpoints"]),
        "Normal cadence differs",
    )
    require(
        normal.get("source_first_step") == 0
        and normal.get("source_last_step") == plan["source_sample_count"] - 1,
        "Normal recording is cropped",
    )


def inspect_orbit(path):
    """Bind file identity and the production replay's sample-only digest."""
    with path.open("rb") as stream:
        require(stream.read(8) == b"CSORBIT1", "Unrecognized orbit cache")
        length = struct.unpack("<Q", stream.read(8))[0]
        require(length <= 64 * 1024 * 1024, "Oversized orbit header")
        header = json.loads(stream.read(length))
        require(
            type(header["count"]) is int and 2 <= header["count"] <= 100_000_000,
            "Invalid orbit sample count",
        )
        require(
            path.stat().st_size == length + 16 + header["count"] * 72, "Orbit payload size differs"
        )
        values = [header["dt"], *header["masses"]]
        require(
            len(values) == 4 and all(math.isfinite(x) and x > 0 for x in values),
            "Invalid orbit timestep or masses",
        )
        samples = hashlib.sha256(struct.pack("<4d", *values))
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            samples.update(chunk)
    return {**header, "sha256": digest(path), "samples_sha256": samples.hexdigest()}


def validate_shell(film, request, movie, normal, orbit, plan):
    require(
        film.get("complete") is True and movie.get("full_decode_verified") is True,
        "Shell film is incomplete",
    )
    request_hash = hashlib.sha256(encoded(request)).hexdigest()
    require(
        film.get("request_sha256") == movie.get("request_sha256") == request_hash,
        "Shell request identity differs",
    )
    require(request["inputs"]["orbit"]["sha256"] == orbit["sha256"], "Shell orbit hash differs")
    require(
        normal["seed"] == orbit["seed"]
        and normal["source_orbit_provenance"] == orbit["provenance"]
        and normal["source_samples_sha256"] == orbit["samples_sha256"]
        and normal["source_dt"] == orbit["dt"] == 0.001
        and orbit["count"] == plan["source_sample_count"],
        "Physical source identity differs",
    )
    wanted = plan["frames"]
    require(
        request.get("fps") == movie.get("fps") == 30 and movie.get("frames") == len(wanted),
        "Shell cadence differs",
    )
    require(
        len(film["frames"]) == len(request["frames"]) == len(wanted), "Shell frame count differs"
    )
    for actual, planned, expected in zip(film["frames"], request["frames"], wanted, strict=True):
        for frame in (actual, planned):
            require(
                frame["index"] == frame["geometry_index"] == expected["frame"]
                and frame["source_fraction"] == expected["source_fraction"]
                and frame["phase"] == "excavation",
                "Shell checkpoint differs",
            )
        require(actual.get("complete") is True, "Shell frame is incomplete")
    timeline = [
        {"kind": "excavation", "render_frame": i, "slot": i, "time_seconds": i / 30}
        for i in range(len(wanted))
    ]
    require(
        film["encoding_timeline"] == request["encoding_timeline"] == timeline,
        "Shell timeline includes holds, missing frames or camera motion",
    )


def checked_artifact(path, record):
    require(
        path.stat().st_size == record["bytes"] and digest(path) == record["sha256"],
        f"Artifact identity differs: {path}",
    )


def publish_copy(source, target, record):
    """A published media path is immutable, including before films complete."""
    checked_artifact(source, record)
    if target.exists():
        checked_artifact(target, record)
        return
    for name in ("stills.json", "comparison.json"):
        receipt = target.parent / name
        if receipt.exists():
            bound = read(receipt).get("artifacts", {}).get(target.name)
            require(
                bound is None or bound["sha256"] == record["sha256"],
                "Published identity differs; use a new collection",
            )
    partial = target.with_name(target.name + ".partial")
    shutil.copyfile(source, partial)
    checked_artifact(partial, record)
    partial.replace(target)


def png_size(path):
    with path.open("rb") as stream:
        header = stream.read(24)
    require(header[:8] == b"\x89PNG\r\n\x1a\n" and header[12:16] == b"IHDR", "Invalid PNG")
    return list(struct.unpack(">II", header[16:24]))


def run(command):
    return subprocess.run(command, check=True, capture_output=True, text=True).stdout


def verify_video(path, ffmpeg, ffprobe, dimensions, frames, fps):
    info = json.loads(
        run(
            [
                ffprobe,
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-count_frames",
                "-show_streams",
                "-of",
                "json",
                str(path),
            ]
        )
    )["streams"][0]
    require(
        [info["width"], info["height"]] == dimensions
        and int(info["nb_read_frames"]) == frames
        and Fraction(info["avg_frame_rate"]) == fps,
        f"Video dimensions/cadence differ: {path}",
    )
    require(abs(float(info["duration"]) - frames / fps) < 0.001, "Video duration differs")
    progress = run(
        [
            ffmpeg,
            "-nostdin",
            "-v",
            "error",
            "-xerror",
            "-threads",
            "4",
            "-i",
            str(path),
            "-progress",
            "pipe:1",
            "-map",
            "0:v:0",
            "-f",
            "null",
            "-",
        ]
    )
    counts = [int(line[6:]) for line in progress.splitlines() if line.startswith("frame=")]
    require("progress=end" in progress and counts and counts[-1] == frames, "Full decode failed")
    return {
        "sha256": digest(path),
        "bytes": path.stat().st_size,
        "frames": frames,
        "fps": fps,
        "dimensions": dimensions,
        "full_decode_verified": True,
        "probe": info,
    }


def transcode(inputs, graph, target, ffmpeg, frames):
    partial = target.with_name(target.stem + ".partial.mp4")
    command = [ffmpeg, "-nostdin", "-v", "error", "-y", "-filter_complex_threads", "2"]
    for source in inputs:
        command += ["-threads", "2", "-i", str(source)]
    command += [
        "-filter_complex",
        graph,
        "-map",
        "[out]",
        "-frames:v",
        str(frames),
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "slow",
        "-crf",
        "17",
        "-pix_fmt",
        "yuv420p",
        "-threads",
        "4",
        "-r",
        "30",
        "-color_primaries",
        "bt709",
        "-color_trc",
        "iec61966-2-1",
        "-colorspace",
        "bt709",
        "-color_range",
        "tv",
        "-movflags",
        "+faststart",
        "-map_metadata",
        "-1",
        str(partial),
    ]
    run(command)
    return partial, command


def seed_name(seed):
    require(isinstance(seed, str) and re.fullmatch(r"0x[0-9a-fA-F]{2,64}", seed), "Invalid seed")
    return seed


def update_gallery(root, seeds=(), completed=None, stills=None):
    root.mkdir(parents=True, exist_ok=True)
    with (root / ".collection.lock").open("a+b") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        path = root / "collection.json"
        data = read(path) if path.exists() else {"schema_version": 1, "seeds": {}}
        for seed in seeds:
            data["seeds"].setdefault(seed_name(seed), {"status": "pending"})
        if stills:
            seed = seed_name(stills["seed"])
            item = data["seeds"].setdefault(seed, {})
            if item.get("status") != "ready":
                item["status"] = "stills"
            item.update(
                {
                    "shell_still": True,
                    "normal_still": item.get("normal_still", False)
                    or "normal.png" in stills["artifacts"],
                }
            )
        if completed:
            seed_name(completed)
            data["seeds"][completed] = {
                "status": "ready",
                "receipt": f"{completed}/comparison.json",
            }
        write(path, data)
        write(root / "index.html", GALLERY)


def publish_stills(args):
    orbit = inspect_orbit(args.orbit)
    seed = seed_name(orbit["seed"])
    output = args.collection / seed
    output.mkdir(parents=True, exist_ok=True)
    with (output / ".publish.lock").open("a+b") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        build = read(args.geometry / "build.json")
        still, request = [read(args.still / name) for name in ("receipt.json", "request.json")]
        require(
            build["seed"] == seed
            and build["identity"]["orbit_sha256"] == orbit["sha256"]
            and build["identity"]["recipe"]["source_fraction"] == 1
            and digest(args.geometry / "mesh.ply") == build["mesh_sha256"]
            and request["mesh_sha256"] == build["mesh_sha256"],
            "Still source differs",
        )
        require(
            still.get("complete") is True
            and still["identity_sha256"] == hashlib.sha256(encoded(request)).hexdigest(),
            "Still render is incomplete or has a different request",
        )
        sources = {"shell.png": (args.still / "render.png", still["artifacts"]["render.png"])}
        normal_path = args.normal / "normal.json" if args.normal else None
        if normal_path and normal_path.exists() and (normal := read(normal_path)).get("complete"):
            validate_normal(normal, alignment(orbit["count"]))
            require(
                normal["seed"] == seed
                and normal["source_samples_sha256"] == orbit["samples_sha256"]
                and normal["source_orbit_provenance"] == orbit["provenance"],
                "Normal source differs",
            )
            record = next(x for x in normal["artifacts"] if x["path"] == "master.png")
            sources["normal.png"] = (args.normal / "master.png", record)
        artifacts = {}
        for name, (source, record) in sources.items():
            checked_artifact(source, record)
            require(
                png_size(source) == ([3840, 3200] if name == "shell.png" else [3840, 2484]),
                "Expected native 4K still",
            )
            publish_copy(source, output / name, record)
            artifacts[name] = record
        previous = output / "stills.json"
        if previous.exists():
            earlier = read(previous)
            require(
                earlier["orbit_sha256"] == orbit["sha256"]
                and earlier["mesh_sha256"] == build["mesh_sha256"],
                "Published still source differs; use a new collection",
            )
            for name, record in earlier["artifacts"].items():
                checked_artifact(output / name, record)
                artifacts.setdefault(name, record)
        receipt = {
            "seed": seed,
            "complete": True,
            "orbit_sha256": orbit["sha256"],
            "mesh_sha256": build["mesh_sha256"],
            "artifacts": artifacts,
        }
        write(output / "stills.json", receipt)
        update_gallery(args.collection, stills=receipt)


def publish(args):
    normal = read(args.normal / "normal.json")
    seed = seed_name(normal["seed"])
    root, output = args.collection, args.collection / seed
    output.mkdir(parents=True, exist_ok=True)
    update_gallery(root, [seed])
    with (output / ".publish.lock").open("a+b") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        orbit = inspect_orbit(args.orbit)
        plan = alignment(orbit["count"])
        validate_normal(normal, plan)
        film, request, movie = [
            read(args.shell / name) for name in ("film.json", "request.json", "movie.json")
        ]
        validate_shell(film, request, movie, normal, orbit, plan)
        still, still_request = [
            read(args.still / name) for name in ("receipt.json", "request.json")
        ]
        require(
            still.get("complete") is True
            and still.get("identity_sha256") == hashlib.sha256(encoded(still_request)).hexdigest()
            and still_request["mesh_sha256"] == film["frames"][-1]["mesh_sha256"],
            "Still does not show the final shell",
        )
        normal_assets = {x["path"]: x for x in normal["artifacts"]}
        sources = {
            "normal-60fps.mp4": args.normal / "normal-hq.mp4",
            "normal.png": args.normal / "master.png",
            "shell.mp4": args.shell / "film.mp4",
            "shell.png": args.still / "render.png",
        }
        records = [
            normal_assets["normal-hq.mp4"],
            normal_assets["master.png"],
            movie,
            still["artifacts"]["render.png"],
        ]
        for source, record in zip(sources.values(), records, strict=True):
            checked_artifact(source, record)
        require(
            png_size(sources["normal.png"]) == [3840, 2484]
            and png_size(sources["shell.png"]) == [3840, 3200],
            "Expected native 4K stills",
        )
        identity = {
            "script_sha256": digest(Path(__file__)),
            "orbit_sha256": orbit["sha256"],
            "normal_sha256": digest(args.normal / "normal.json"),
            "shell_sha256": digest(args.shell / "film.json"),
            "still_sha256": digest(args.still / "receipt.json"),
            "ffmpeg_version": run([args.ffmpeg, "-version"]).splitlines()[0],
            "ffprobe_version": run([args.ffprobe, "-version"]).splitlines()[0],
        }
        receipt_path = output / "comparison.json"
        if receipt_path.exists():
            previous = read(receipt_path)
            require(previous["identity"] == identity, "Output already belongs to another request")
            if previous.get("complete"):
                for name, record in previous["artifacts"].items():
                    checked_artifact(output / name, record)
                update_gallery(root, completed=seed)
                return
        write(receipt_path, {"complete": False, "identity": identity, "seed": seed})
        count = len(plan["frames"])
        artifacts = {}
        artifacts["normal-60fps.mp4"] = verify_video(
            sources["normal-60fps.mp4"], args.ffmpeg, args.ffprobe, [3840, 2484], count * 2, 60
        )
        artifacts["shell.mp4"] = verify_video(
            sources["shell.mp4"], args.ffmpeg, args.ffprobe, [3840, 3200], count, 30
        )
        require(
            artifacts["normal-60fps.mp4"]["probe"].get("color_primaries") == "smpte432"
            and artifacts["shell.mp4"]["probe"].get("color_primaries") == "bt709",
            "Unexpected native movie primaries",
        )
        for name, source in sources.items():
            publish_copy(source, output / name, records[list(sources).index(name)])
            artifacts.setdefault(
                name, {"sha256": digest(output / name), "bytes": (output / name).stat().st_size}
            )
        jobs = [
            (
                "normal-paired.mp4",
                [output / "normal-60fps.mp4"],
                NORMAL_PAIR_FILTER,
                [3840, 2484],
            ),
            (
                "side-by-side.mp4",
                [output / "normal-paired.mp4", output / "shell.mp4"],
                "[0:v]scale=1920:1242:flags=lanczos,pad=1920:2160:0:459:color=0xe5e3dd,setsar=1[a];"
                "[1:v]scale=1920:1600:flags=lanczos,pad=1920:2160:0:280:color=0xe5e3dd,setsar=1[b];"
                "[a][b]hstack=inputs=2[out]",
                [3840, 2160],
            ),
        ]
        commands = []
        for name, inputs, graph, dimensions in jobs:
            partial, command = transcode(inputs, graph, output / name, args.ffmpeg, count)
            artifacts[name] = verify_video(
                partial, args.ffmpeg, args.ffprobe, dimensions, count, 30
            )
            partial.replace(output / name)
            commands.append(command)
        write(output / "alignment.json", plan)
        artifacts["alignment.json"] = {
            "sha256": digest(output / "alignment.json"),
            "bytes": (output / "alignment.json").stat().st_size,
        }
        write(
            receipt_path,
            {
                "schema_version": 1,
                "complete": True,
                "seed": seed,
                "identity": identity,
                "artifacts": artifacts,
                "commands": commands,
                "frames": count,
                "fps": 30,
                "duration_seconds": count / 30,
                "source_samples_sha256": orbit["samples_sha256"],
            },
        )
        update_gallery(root, completed=seed)


GALLERY = """<!doctype html><html lang="en"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>The Shell That Remembers · Complete trajectories</title><style>
*{box-sizing:border-box}body{margin:0;background:#edece6;color:#29372f;font:16px system-ui}
main{max-width:1800px;margin:auto;padding:3vw}h1{font:clamp(28px,4vw,56px) Georgia}
p{line-height:1.6;max-width:850px}select,button{font:inherit;padding:.7em;border:1px solid #879087;
background:#f9f8f2;color:inherit;border-radius:3px}a{color:inherit}nav{display:flex;gap:20px;flex-wrap:wrap}
.pair{display:grid;grid-template-columns:1fr 1fr;gap:16px}figure{margin:20px 0}
video,img{width:100%;height:52vw;max-height:75vh;object-fit:contain;background:#e5e3dd}
figcaption{font:22px Georgia;padding:14px 0}.controls{display:flex;align-items:center;gap:16px}
input{flex:1;min-width:60px}#notice{padding:30px 0}section[hidden]{display:none}
@media(max-width:650px){.pair{gap:6px}figcaption{font-size:16px}main{padding:20px 10px}}
</style><main><p>THREE BODIES · TWO EXPRESSIONS</p><h1>The Shell That Remembers</h1>
<p>The complete recorded trajectory, seen as accumulated light and a growing porcelain shell.
Both views show the same source history at every paired frame. Each film lasts 30.03 seconds.</p>
<label>Seed <select id="seed"></select></label><p id="notice" role="status">Loading collection…</p>
<section id="ready" hidden><video id="comparison" controls muted playsinline preload="metadata"></video>
<div class="pair"><figcaption>Accumulated light</figcaption><figcaption>The remembering shell</figcaption></div>
<div class="controls"><button id="play">Play</button>
<input id="seek" type="range" min="0" max="900" step="1" value="0" aria-label="Source history">
<output id="progress">0%</output></div><p>30 frames per second. Each pair uses the later of two
original light frames; the last pair includes the final recorded sample. No part of the source
history is removed. The two renderings use different visual coordinates.</p>
<nav id="downloads"></nav></section><section id="stills" hidden><h2>The completed forms</h2>
<p>Select an image to examine the native 4K still.</p><div class="pair">
<figure id="normalFigure"><figcaption>Accumulated light</figcaption><a id="normalStillLink"><img id="normalStill" alt="Complete accumulated light trajectory"></a></figure>
<figure><figcaption>The remembering shell</figcaption><a id="shellStillLink"><img id="shellStill" alt="Complete porcelain shell"></a></figure>
</div></section></main><script>
const $=id=>document.getElementById(id);let collection={},mapping=[],current='',loading=false;
const n=$('comparison');function stop(){n.pause();$('play').textContent='Play'}
async function choose(){stop();current=$('seed').value;const selected=current,item=collection.seeds[current];
$('ready').hidden=true;$('stills').hidden=!item.shell_still&&item.status!=='ready';
$('normalFigure').hidden=!item.normal_still&&item.status!=='ready';
for(const kind of ['normal','shell']){if(item[kind+'_still']||item.status==='ready'){
$(kind+'Still').src=current+'/'+kind+'.png';$(kind+'StillLink').href=current+'/'+kind+'.png'}}
$('notice').textContent=item.status==='ready'?'Loading verified media…':
(item.status==='stills'?'4K still ready · full-length films are rendering.':'Rendering · this seed is not ready yet.');
if(item.status!=='ready')return;try{const receipt=await fetch(item.receipt).then(r=>r.json());
if(!receipt.complete)throw Error('Incomplete result');const map=await fetch(selected+'/alignment.json').then(r=>r.json());
if(current!==selected)return;
mapping=map.frames;$('seek').max=mapping.length-1;n.src=current+'/side-by-side.mp4';$('seek').value=0;
for(const kind of ['normal','shell']){$(kind+'Still').src=current+'/'+kind+'.png';$(kind+'StillLink').href=current+'/'+kind+'.png'}
$('downloads').replaceChildren();for(const [file,label] of [['side-by-side.mp4','4K side-by-side film'],
['normal-60fps.mp4','Original 4K / 60 fps'],['shell.mp4','Shell 4K / 30 fps'],['normal.png','Light still'],
['shell.png','Shell still'],['alignment.json','Frame mapping']]){const a=document.createElement('a');
a.href=current+'/'+file;a.textContent=label;a.download='';$('downloads').append(a)}
$('ready').hidden=false;$('notice').textContent='Ready · 1,000,000 source samples · 901 matched frames';tick();
}catch(error){$('notice').textContent='Media could not be loaded: '+error.message}}
async function refresh(){if(loading)return;loading=true;try{const next=await fetch('collection.json',{cache:'no-store'}).then(r=>r.json());
const changed=JSON.stringify(next)!==JSON.stringify(collection),oldItem=JSON.stringify(collection.seeds?.[current]);
collection=next;if(changed){const previous=current;
$('seed').replaceChildren();for(const [seed,item] of Object.entries(next.seeds)){const option=document.createElement('option');
option.value=seed;option.textContent=seed+' · '+item.status;$('seed').append(option)}
if(next.seeds[previous])$('seed').value=previous;
if(!previous||oldItem!==JSON.stringify(next.seeds[previous]))await choose()}}
catch(error){$('notice').textContent=error.message}finally{loading=false}}
$('seed').onchange=choose;$('play').onclick=async()=>{if(!n.paused){stop();return}
if(n.ended)n.currentTime=0;
try{await n.play();$('play').textContent='Pause'}catch(error){stop();$('notice').textContent=error.message}};
$('seek').oninput=()=>{stop();n.currentTime=Number($('seek').value)/30;tick()};
n.onended=stop;n.onplay=()=>{$('play').textContent='Pause'};n.onpause=()=>{$('play').textContent='Play'};function tick(){if(mapping.length){const index=Math.min(mapping.length-1,Math.floor(n.currentTime*30+1e-5));
$('seek').value=index;$('progress').textContent=(mapping[index].source_fraction*100).toFixed(2)+'%'}}
function animate(){tick();requestAnimationFrame(animate)}refresh();setInterval(refresh,15000);animate();
</script></html>"""  # noqa: E501


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="action", required=True)
    plan = commands.add_parser("plan")
    plan.add_argument("--samples", type=int, default=1_000_000)
    plan.add_argument("--normal", type=Path)
    plan.add_argument("--output", type=Path, required=True)
    init = commands.add_parser("init")
    init.add_argument("--collection", type=Path, required=True)
    init.add_argument("--seeds", nargs="+", required=True)
    stills = commands.add_parser("publish-stills")
    for name in ("collection", "still", "geometry", "orbit"):
        stills.add_argument("--" + name, type=Path, required=True)
    stills.add_argument("--normal", type=Path)
    final = commands.add_parser("publish")
    for name in ("collection", "normal", "shell", "still", "orbit"):
        final.add_argument("--" + name, type=Path, required=True)
    final.add_argument("--ffmpeg", default="ffmpeg")
    final.add_argument("--ffprobe", default="ffprobe")
    args = parser.parse_args()
    if args.action == "plan":
        result = alignment(args.samples)
        if args.normal:
            validate_normal(read(args.normal), result)
        write(args.output / "alignment.json", result)
        write(
            args.output / "source-times.json",
            [frame["source_fraction"] for frame in result["frames"]],
        )
    elif args.action == "init":
        update_gallery(args.collection, args.seeds)
    elif args.action == "publish-stills":
        publish_stills(args)
    else:
        publish(args)


if __name__ == "__main__":
    main()
