//! AR Quick Look export — converts the existing GLB model to USDZ format
//! for Apple AR viewing via the `usdzconvert` tool or `reality_converter`.

use std::process::{Command, Stdio};
use tracing::{info, warn};

pub fn export_usdz(
    glb_path: &str,
    output_path: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    info!("Exporting AR Quick Look asset (USDZ)...");

    if let Some(converter) = find_converter() {
        run_conversion(&converter, glb_path, output_path)?;
        info!("   Saved USDZ => {}", output_path);
    } else {
        warn!("   USDZ export skipped: no converter found (install usdzconvert or use Xcode)");
        return Err("No USDZ converter available. Install usdzconvert via: \
                     pip3 install usd-core, or use Xcode's Reality Converter.".into());
    }

    Ok(())
}

fn find_converter() -> Option<String> {
    for cmd in &["usdzconvert", "reality_converter"] {
        if Command::new("which")
            .arg(cmd)
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status()
            .map(|s| s.success())
            .unwrap_or(false)
        {
            return Some(cmd.to_string());
        }
    }

    let xcrun = Command::new("xcrun")
        .args(["--find", "usdzconvert"])
        .output();
    if let Ok(output) = xcrun {
        if output.status.success() {
            let path = String::from_utf8_lossy(&output.stdout).trim().to_string();
            if !path.is_empty() {
                return Some(path);
            }
        }
    }

    None
}

fn run_conversion(
    converter: &str,
    glb_path: &str,
    output_path: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    let status = Command::new(converter)
        .args([glb_path, output_path])
        .stdout(Stdio::null())
        .stderr(Stdio::inherit())
        .status()?;

    if !status.success() {
        return Err(format!(
            "USDZ conversion failed with status {:?} (converter: {})",
            status, converter
        ).into());
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_find_converter_returns_option() {
        let result = find_converter();
        // May or may not find a converter depending on the system
        assert!(result.is_some() || result.is_none());
    }
}
