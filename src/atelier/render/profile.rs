//! Bounded continuous emission curves, prepared once for each frame material.
use super::super::{EmissionProfile, SilkResult, UvFeather, V3};

#[derive(Clone, Copy)]
pub(super) struct ProfileSample {
    pub emission: V3,
    pub density: f64,
}

struct Segment {
    start: f64,
    inverse_width: f64,
    coefficients: [[f64; 4]; 4],
    minimum: [f64; 4],
    maximum: [f64; 4],
}

pub(super) struct CompiledProfile {
    origin: V3,
    axis: V3,
    inverse_extent: f64,
    first_position: f64,
    last_position: f64,
    first: [f64; 4],
    last: [f64; 4],
    segments: Vec<Segment>,
    uv_feather: Option<UvFeather>,
}

pub(super) fn validate_profile(profile: &EmissionProfile) -> SilkResult<()> {
    let axis_squared = profile.axis.length_squared();
    if !profile.origin.is_finite()
        || !profile.axis.is_finite()
        || !axis_squared.is_finite()
        || axis_squared < 1e-24
        || !profile.extent.is_finite()
        || profile.extent <= 0.0
        || !profile.extent.recip().is_finite()
        || profile.stops.len() < 2
    {
        return Err("invalid emission profile axis, extent, or stop count".into());
    }
    for (index, stop) in profile.stops.iter().enumerate() {
        if !stop.position.is_finite()
            || !(0.0..=1.0).contains(&stop.position)
            || !stop.emission.is_finite()
            || stop.emission.x < 0.0
            || stop.emission.y < 0.0
            || stop.emission.z < 0.0
            || !stop.density.is_finite()
            || !(0.0..=1.0).contains(&stop.density)
            || (index > 0 && stop.position <= profile.stops[index - 1].position)
        {
            return Err(
                "emission stops need ordered positions, nonnegative colors, and density in 0..1"
                    .into(),
            );
        }
    }
    if let Some(feather) = &profile.uv_feather
        && feather.u.iter().chain(&feather.v).any(|v| !v.is_finite() || !(0.0..=1.0).contains(v))
    {
        return Err("emission UV feather widths must be in 0..1".into());
    }
    Ok(())
}

impl CompiledProfile {
    pub fn new(profile: &EmissionProfile) -> SilkResult<Self> {
        validate_profile(profile)?;
        let values: Vec<[f64; 4]> = profile
            .stops
            .iter()
            .map(|stop| [stop.emission.x, stop.emission.y, stop.emission.z, stop.density])
            .collect();
        let widths: Vec<_> =
            profile.stops.windows(2).map(|s| s[1].position - s[0].position).collect();
        let secants: Vec<[f64; 4]> = values
            .windows(2)
            .zip(&widths)
            .map(|(v, width)| {
                std::array::from_fn(|channel| (v[1][channel] - v[0][channel]) / width)
            })
            .collect();
        let mut slopes = vec![[0.0; 4]; values.len()];
        // Weighted harmonic interior slopes preserve monotone intervals without
        // ringing or making every color stop a flat plateau. End slopes are zero
        // so clamping outside the world-height range stays continuously smooth.
        for index in 1..values.len() - 1 {
            for channel in 0..4 {
                let before = secants[index - 1][channel];
                let after = secants[index][channel];
                if before == 0.0
                    || after == 0.0
                    || before.is_sign_positive() != after.is_sign_positive()
                {
                    continue;
                }
                let w1 = 2.0 * widths[index] + widths[index - 1];
                let w2 = widths[index] + 2.0 * widths[index - 1];
                slopes[index][channel] = (w1 + w2) / (w1 / before + w2 / after);
            }
        }
        let mut segments = Vec::with_capacity(widths.len());
        for (index, &width) in widths.iter().enumerate() {
            let coefficients = std::array::from_fn(|channel| {
                let start = values[index][channel];
                let difference = values[index + 1][channel] - start;
                let first_slope = slopes[index][channel] * width;
                let last_slope = slopes[index + 1][channel] * width;
                [
                    start,
                    first_slope,
                    3.0 * difference - 2.0 * first_slope - last_slope,
                    -2.0 * difference + first_slope + last_slope,
                ]
            });
            if !width.recip().is_finite()
                || coefficients.iter().flatten().any(|value| !value.is_finite())
                || secants[index].iter().any(|value| !value.is_finite())
            {
                return Err("emission profile exceeds finite interpolation range".into());
            }
            segments.push(Segment {
                start: profile.stops[index].position,
                inverse_width: width.recip(),
                coefficients,
                minimum: std::array::from_fn(|c| values[index][c].min(values[index + 1][c])),
                maximum: std::array::from_fn(|c| values[index][c].max(values[index + 1][c])),
            });
        }
        Ok(Self {
            origin: profile.origin,
            axis: profile.axis.normalized(),
            inverse_extent: profile.extent.recip(),
            first_position: profile.stops[0].position,
            last_position: profile.stops[values.len() - 1].position,
            first: values[0],
            last: values[values.len() - 1],
            segments,
            uv_feather: profile.uv_feather.clone(),
        })
    }

    pub fn sample(&self, point: V3, uv: [f64; 2]) -> ProfileSample {
        let coordinate = (point - self.origin).dot(self.axis) * self.inverse_extent;
        let values = if coordinate <= self.first_position {
            self.first
        } else if coordinate >= self.last_position {
            self.last
        } else {
            let index = self.segments.partition_point(|s| s.start <= coordinate).saturating_sub(1);
            let segment = &self.segments[index];
            let t = ((coordinate - segment.start) * segment.inverse_width).clamp(0.0, 1.0);
            std::array::from_fn(|channel| {
                let [a, b, c, d] = segment.coefficients[channel];
                (((d * t + c) * t + b) * t + a)
                    .clamp(segment.minimum[channel], segment.maximum[channel])
            })
        };
        let envelope = self.uv_feather.as_ref().map_or(1.0, |feather| {
            edge(uv[0], feather.u[0])
                * edge(1.0 - uv[0], feather.u[1])
                * edge(uv[1], feather.v[0])
                * edge(1.0 - uv[1], feather.v[1])
        });
        ProfileSample {
            emission: V3::new(values[0], values[1], values[2]),
            density: values[3] * envelope,
        }
    }
}

fn edge(distance: f64, width: f64) -> f64 {
    if width == 0.0 {
        return 1.0;
    }
    let t = (distance / width).clamp(0.0, 1.0);
    (t * t * t * (10.0 + t * (-15.0 + 6.0 * t))).clamp(0.0, 1.0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::atelier::EmissionStop;

    fn profile() -> EmissionProfile {
        EmissionProfile {
            origin: V3::new(0.0, -2.0, 0.0),
            extent: 4.0,
            stops: vec![
                EmissionStop { position: 0.0, emission: V3::new(0.05, 2.0, 0.4), density: 0.0 },
                EmissionStop { position: 0.2, emission: V3::new(0.08, 1.7, 0.6), density: 0.8 },
                EmissionStop { position: 0.65, emission: V3::new(0.2, 0.6, 1.8), density: 1.0 },
                EmissionStop { position: 1.0, emission: V3::new(0.8, 0.05, 1.2), density: 0.0 },
            ],
            ..EmissionProfile::default()
        }
    }

    #[test]
    fn smooth_profiles_are_bounded_and_have_no_color_steps() {
        let source = profile();
        let curve = CompiledProfile::new(&source).unwrap();
        for i in 0..=2000 {
            let t = f64::from(i) / 2000.0;
            let value = curve.sample(V3::new(0.0, t * 4.0 - 2.0, 0.0), [0.5, 0.5]);
            let index = source
                .stops
                .partition_point(|s| s.position <= t)
                .saturating_sub(1)
                .min(source.stops.len() - 2);
            for channel in 0..3 {
                let a = source.stops[index].emission.axis(channel);
                let b = source.stops[index + 1].emission.axis(channel);
                assert!(
                    (a.min(b) - 1e-12..=a.max(b) + 1e-12).contains(&value.emission.axis(channel))
                );
            }
            assert!((0.0..=1.0).contains(&value.density));
        }
        for stop in &source.stops[1..source.stops.len() - 1] {
            let y = stop.position * 4.0 - 2.0;
            let step = 1e-5;
            let left = curve.sample(V3::new(0.0, y - step, 0.0), [0.5, 0.5]);
            let center = curve.sample(V3::new(0.0, y, 0.0), [0.5, 0.5]);
            let right = curve.sample(V3::new(0.0, y + step, 0.0), [0.5, 0.5]);
            assert!((center.emission - stop.emission).length() < 1e-12);
            assert!(
                ((center.emission - left.emission) / step
                    - (right.emission - center.emission) / step)
                    .length()
                    < 0.001
            );
            assert!(
                ((center.density - left.density) / step - (right.density - center.density) / step)
                    .abs()
                    < 0.001
            );
        }
    }

    #[test]
    fn world_height_is_shared_by_sheets_and_rays_and_uv_fade_is_smooth() {
        let mut source = profile();
        let ray = CompiledProfile::new(&source).unwrap();
        source.uv_feather = Some(UvFeather { u: [0.2, 0.15], v: [0.03, 0.4] });
        let sheet = CompiledProfile::new(&source).unwrap();
        let point = V3::new(1.3, 0.0, -0.4);
        let a = ray.sample(point, [12.5, 0.0]);
        let b = sheet.sample(point, [0.5, 0.5]);
        assert_eq!(a.emission, b.emission);
        assert_eq!(a.density, b.density);
        for uv in [[0.0, 0.5], [1.0, 0.5], [0.5, 0.0], [0.5, 1.0]] {
            assert_eq!(sheet.sample(point, uv).density, 0.0);
        }
        assert!(sheet.sample(point, [1e-5, 0.5]).density < 1e-10);
        assert_eq!(ray.sample(V3::new(0.0, -3.0, 0.0), [0.5, 0.5]).density, 0.0);
        assert_eq!(ray.sample(V3::new(0.0, 3.0, 0.0), [0.5, 0.5]).density, 0.0);
    }

    #[test]
    fn malformed_profiles_are_rejected_before_rendering() {
        let mut source = profile();
        source.stops[1].position = source.stops[0].position;
        assert!(CompiledProfile::new(&source).is_err());
        source = profile();
        source.stops[1].density = 1.1;
        assert!(CompiledProfile::new(&source).is_err());
        source = profile();
        source.axis = V3::ZERO;
        assert!(CompiledProfile::new(&source).is_err());
        source = profile();
        source.stops[1].emission.x = f64::INFINITY;
        assert!(CompiledProfile::new(&source).is_err());
        source = profile();
        source.uv_feather = Some(UvFeather { u: [0.2, f64::NAN], v: [0.0, 0.0] });
        assert!(CompiledProfile::new(&source).is_err());
    }
}
