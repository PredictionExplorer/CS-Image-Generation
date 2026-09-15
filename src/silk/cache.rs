//! Portable, validated little-endian orbit and cloth caches.
//!
//! Material and camera changes reuse the cloth cache. Floating-point samples
//! retain all source bits; JSON headers carry versioned provenance and settings.
use super::{ClothBake, FrameStats, Mesh, OrbitData, SilkResult, V3};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::fs::{self, File};
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::Path;

const ORBIT_MAGIC: &[u8; 8] = b"CSORBIT1";
const BAKE_MAGIC: &[u8; 8] = b"CSSILK01";
const MAX_HEADER: usize = 64 * 1024 * 1024;

#[derive(Serialize, Deserialize)]
struct OrbitHeader {
    seed: String,
    dt: f64,
    masses: [f64; 3],
    count: usize,
    provenance: serde_json::Value,
}

#[derive(Serialize, Deserialize)]
struct BakeHeader {
    vertices: usize,
    triangles: usize,
    frames: usize,
    fps: u32,
    stats: Vec<FrameStats>,
    recipe: serde_json::Value,
}

fn invalid(message: &str) -> std::io::Error {
    std::io::Error::new(std::io::ErrorKind::InvalidData, message)
}

fn atomic_write(
    path: &Path,
    write: impl FnOnce(&mut BufWriter<File>) -> SilkResult<()>,
) -> SilkResult<()> {
    if let Some(parent) = path.parent().filter(|p| !p.as_os_str().is_empty()) {
        fs::create_dir_all(parent)?;
    }
    let temporary = path.with_extension(format!("partial-{}", std::process::id()));
    let result = (|| {
        let mut writer = BufWriter::with_capacity(1 << 20, File::create(&temporary)?);
        write(&mut writer)?;
        writer.flush()?;
        writer.get_ref().sync_all()?;
        fs::rename(&temporary, path)?;
        Ok(())
    })();
    if result.is_err() {
        let _ = fs::remove_file(&temporary);
    }
    result
}

fn write_header(
    writer: &mut impl Write,
    magic: &[u8; 8],
    value: &impl Serialize,
) -> SilkResult<()> {
    let bytes = serde_json::to_vec(value)?;
    writer.write_all(magic)?;
    writer.write_all(&(bytes.len() as u64).to_le_bytes())?;
    writer.write_all(&bytes)?;
    Ok(())
}

fn read_header<T: for<'de> Deserialize<'de>>(
    reader: &mut impl Read,
    magic: &[u8; 8],
) -> SilkResult<(T, u64)> {
    let mut actual = [0; 8];
    reader.read_exact(&mut actual)?;
    if &actual != magic {
        return Err(invalid("Unrecognized cache type or version").into());
    }
    reader.read_exact(&mut actual)?;
    let length = u64::from_le_bytes(actual);
    if length > MAX_HEADER as u64 {
        return Err(invalid("Cache header exceeds size limit").into());
    }
    let mut bytes = vec![0; length as usize];
    reader.read_exact(&mut bytes)?;
    Ok((serde_json::from_slice(&bytes)?, length + 16))
}

fn write_v3(writer: &mut impl Write, v: V3) -> SilkResult<()> {
    if !v.is_finite() {
        return Err(invalid("Cannot cache non-finite geometry").into());
    }
    for component in [v.x, v.y, v.z] {
        writer.write_all(&component.to_le_bytes())?;
    }
    Ok(())
}

fn read_f64(reader: &mut impl Read) -> SilkResult<f64> {
    let mut bytes = [0; 8];
    reader.read_exact(&mut bytes)?;
    let value = f64::from_le_bytes(bytes);
    if !value.is_finite() {
        return Err(invalid("Cache contains non-finite number").into());
    }
    Ok(value)
}

fn read_v3(reader: &mut impl Read) -> SilkResult<V3> {
    Ok(V3::new(read_f64(reader)?, read_f64(reader)?, read_f64(reader)?))
}

/// Atomically store an original physical trajectory.
pub fn write_orbit(path: &Path, orbit: &OrbitData) -> SilkResult<()> {
    if orbit.samples.len() < 2
        || !orbit.dt.is_finite()
        || orbit.dt <= 0.0
        || orbit.masses.iter().any(|m| !m.is_finite() || *m <= 0.0)
    {
        return Err(
            invalid("Orbit requires at least two samples, positive timestep and masses").into()
        );
    }
    atomic_write(path, |writer| {
        write_header(
            writer,
            ORBIT_MAGIC,
            &OrbitHeader {
                seed: orbit.seed.clone(),
                dt: orbit.dt,
                masses: orbit.masses,
                count: orbit.samples.len(),
                provenance: orbit.provenance.clone(),
            },
        )?;
        for sample in &orbit.samples {
            for &point in sample {
                write_v3(writer, point)?;
            }
        }
        Ok(())
    })
}

/// Load a physical trajectory after validating dimensions and payload size.
pub fn read_orbit(path: &Path) -> SilkResult<OrbitData> {
    let file = File::open(path)?;
    let size = file.metadata()?.len();
    let mut reader = BufReader::with_capacity(1 << 20, file);
    let (header, offset): (OrbitHeader, u64) = read_header(&mut reader, ORBIT_MAGIC)?;
    let expected = (header.count as u64).checked_mul(72).and_then(|n| n.checked_add(offset));
    if expected != Some(size)
        || header.count < 2
        || header.dt <= 0.0
        || !header.dt.is_finite()
        || header.masses.iter().any(|m| !m.is_finite() || *m <= 0.0)
    {
        return Err(invalid("Invalid orbit dimensions or payload length").into());
    }
    let mut samples = Vec::with_capacity(header.count);
    for _ in 0..header.count {
        samples.push([read_v3(&mut reader)?, read_v3(&mut reader)?, read_v3(&mut reader)?]);
    }
    Ok(OrbitData {
        seed: header.seed,
        dt: header.dt,
        masses: header.masses,
        samples,
        provenance: header.provenance,
    })
}

/// Atomically store full-precision cloth geometry and its recipe.
pub fn write_bake(path: &Path, bake: &ClothBake) -> SilkResult<()> {
    let n = bake.mesh.positions.len();
    if n < 3
        || bake.mesh.uv.len() != n
        || bake.frames.is_empty()
        || bake.fps == 0
        || bake.stats.len() != bake.frames.len()
        || bake.frames.iter().any(|f| f.len() != n)
        || bake.stats.iter().any(|s| {
            !s.max_stretch.is_finite()
                || s.max_stretch < 0.0
                || !s.pin_error.is_finite()
                || s.pin_error < 0.0
        })
        || bake.mesh.triangles.iter().flatten().any(|&i| i as usize >= n)
    {
        return Err(invalid("Invalid cloth topology or frame dimensions").into());
    }
    atomic_write(path, |writer| {
        write_header(
            writer,
            BAKE_MAGIC,
            &BakeHeader {
                vertices: n,
                triangles: bake.mesh.triangles.len(),
                frames: bake.frames.len(),
                fps: bake.fps,
                stats: bake.stats.clone(),
                recipe: bake.recipe.clone(),
            },
        )?;
        for &point in &bake.mesh.positions {
            write_v3(writer, point)?;
        }
        for uv in &bake.mesh.uv {
            for &v in uv {
                if !v.is_finite() {
                    return Err(invalid("Non-finite material coordinates").into());
                }
                writer.write_all(&v.to_le_bytes())?;
            }
        }
        for triangle in &bake.mesh.triangles {
            for i in triangle {
                writer.write_all(&i.to_le_bytes())?;
            }
        }
        for frame in &bake.frames {
            for &point in frame {
                write_v3(writer, point)?;
            }
        }
        Ok(())
    })
}

/// Load a cloth bake without running any simulation.
pub fn read_bake(path: &Path) -> SilkResult<ClothBake> {
    let file = File::open(path)?;
    let size = file.metadata()?.len();
    let mut reader = BufReader::with_capacity(1 << 20, file);
    let (h, offset): (BakeHeader, u64) = read_header(&mut reader, BAKE_MAGIC)?;
    let expected = (h.vertices as u64)
        .checked_mul(40)
        .and_then(|n| (h.triangles as u64).checked_mul(12).and_then(|t| n.checked_add(t)))
        .and_then(|n| {
            (h.vertices as u64)
                .checked_mul(h.frames as u64)
                .and_then(|f| f.checked_mul(24))
                .and_then(|f| n.checked_add(f))
        })
        .and_then(|n| n.checked_add(offset));
    if expected != Some(size)
        || h.vertices < 3
        || h.frames == 0
        || h.fps == 0
        || h.stats.len() != h.frames
    {
        return Err(invalid("Invalid cloth dimensions or payload length").into());
    }
    let mut positions = Vec::with_capacity(h.vertices);
    for _ in 0..h.vertices {
        positions.push(read_v3(&mut reader)?);
    }
    let mut uv = Vec::with_capacity(h.vertices);
    for _ in 0..h.vertices {
        uv.push([read_f64(&mut reader)?, read_f64(&mut reader)?]);
    }
    let mut triangles = Vec::with_capacity(h.triangles);
    for _ in 0..h.triangles {
        let mut triangle = [0; 3];
        for i in &mut triangle {
            let mut bytes = [0; 4];
            reader.read_exact(&mut bytes)?;
            *i = u32::from_le_bytes(bytes);
            if *i as usize >= h.vertices {
                return Err(invalid("Triangle index out of range").into());
            }
        }
        triangles.push(triangle);
    }
    let mut frames = Vec::with_capacity(h.frames);
    for _ in 0..h.frames {
        let mut frame = Vec::with_capacity(h.vertices);
        for _ in 0..h.vertices {
            frame.push(read_v3(&mut reader)?);
        }
        frames.push(frame);
    }
    Ok(ClothBake {
        mesh: Mesh { positions, triangles, uv },
        frames,
        fps: h.fps,
        stats: h.stats,
        recipe: h.recipe,
    })
}

/// SHA-256 of a cache or output file for provenance and integrity checks.
pub fn file_hash(path: &Path) -> SilkResult<String> {
    let mut file = File::open(path)?;
    let mut digest = Sha256::new();
    let mut buffer = vec![0; 1 << 20];
    loop {
        let n = file.read(&mut buffer)?;
        if n == 0 {
            break;
        }
        digest.update(&buffer[..n]);
    }
    Ok(hex::encode(digest.finalize()))
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn orbit_roundtrip_preserves_bits_and_rejects_truncation() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("orbit.bin");
        let orbit = OrbitData {
            seed: "0x01".into(),
            dt: 0.001,
            masses: [1.0; 3],
            samples: vec![[V3::new(-0.0, 1.0 / 3.0, -12.0); 3]; 2],
            provenance: serde_json::json!({"version":1}),
        };
        write_orbit(&path, &orbit).unwrap();
        let read = read_orbit(&path).unwrap();
        assert_eq!(read.samples[0][0].x.to_bits(), orbit.samples[0][0].x.to_bits());
        assert_eq!(read.samples, orbit.samples);
        let file = File::options().write(true).open(&path).unwrap();
        file.set_len(20).unwrap();
        assert!(read_orbit(&path).is_err());
    }
    #[test]
    fn bake_roundtrip_preserves_topology_and_frames() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("cloth.bin");
        let positions = vec![V3::ZERO, V3::new(1.0, 0.0, 0.0), V3::new(0.0, 1.0, 0.0)];
        let bake = ClothBake {
            mesh: Mesh {
                positions: positions.clone(),
                triangles: vec![[0, 1, 2]],
                uv: vec![[0.0, 0.0]; 3],
            },
            frames: vec![positions],
            fps: 30,
            stats: vec![FrameStats::default()],
            recipe: serde_json::json!({"test":true}),
        };
        write_bake(&path, &bake).unwrap();
        let result = read_bake(&path).unwrap();
        assert_eq!(result.frames, bake.frames);
        assert_eq!(result.mesh.triangles, bake.mesh.triangles);
        let mut corrupt = fs::read(&path).unwrap();
        corrupt[0] = 0;
        fs::write(&path, corrupt).unwrap();
        assert!(read_bake(&path).is_err());
    }
}
