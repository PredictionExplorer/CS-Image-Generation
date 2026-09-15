//! Conservative optical flux deposition, independent of the display treatment.
//!
//! Exact clipped-cell moments integrate a linear reconstruction basis. This
//! avoids pixel jumps in the point limit. Three normalized separable box passes
//! supply a compact, Gaussian-like optical footprint; at AA3 its RMS width is
//! close to the requested final-pixel footprint. Padding precedes deposition,
//! so light outside the receiving crop can still enter through that footprint.
use super::MappedTriangle;
use crate::atelier::SilkResult;
use rayon::prelude::*;
use smallvec::SmallVec;

const TILE: usize = 32;
const LINE_LOW: f64 = 0.000_025;
const LINE_HIGH: f64 = 0.000_1;

pub(super) struct AccumulatedBand {
    pub delta_irradiance: Vec<f64>,
    pub incident_flux: f64,
    pub received_flux: f64,
    pub escaped_flux: f64,
    pub baseline_removed_flux: f64,
    pub reference_escaped_flux: f64,
    pub conservation_residual: f64,
    pub degenerate_triangles: usize,
}

#[derive(Clone, Copy, Debug)]
struct Point {
    x: f64,
    y: f64,
}
impl Point {
    fn sub(self, other: Self) -> Self {
        Self { x: self.x - other.x, y: self.y - other.y }
    }
    fn add(self, other: Self) -> Self {
        Self { x: self.x + other.x, y: self.y + other.y }
    }
    fn scale(self, value: f64) -> Self {
        Self { x: self.x * value, y: self.y * value }
    }
    fn dot(self, other: Self) -> f64 {
        self.x * other.x + self.y * other.y
    }
    fn cross(self, other: Self) -> f64 {
        self.x * other.y - self.y * other.x
    }
}

#[derive(Clone, Copy)]
struct Bounds {
    x0: usize,
    y0: usize,
    x1: usize,
    y1: usize,
}
impl Bounds {
    fn contains(self, x: usize, y: usize) -> bool {
        x >= self.x0 && x < self.x1 && y >= self.y0 && y < self.y1
    }
    fn empty(self) -> bool {
        self.x0 == self.x1 || self.y0 == self.y1
    }
    fn remains_inside(self, crop: Self, blur_support: usize) -> bool {
        !self.empty()
            && self.x0 >= crop.x0 + blur_support
            && self.y0 >= crop.y0 + blur_support
            && self.x1 <= crop.x1.saturating_sub(blur_support)
            && self.y1 <= crop.y1.saturating_sub(blur_support)
    }
}

#[derive(Clone, Copy, PartialEq)]
enum Contribution {
    Reference,
    Refracted,
    Identity,
}

struct Plan {
    points: [Point; 3],
    center: Point,
    axis: Point,
    projections: [f64; 3],
    area: f64,
    flux: f64,
    point_weight: f64,
    line_weight: f64,
    bounds: Bounds,
    contribution: Contribution,
}

fn smooth(value: f64) -> f64 {
    let t = value.clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

impl Plan {
    fn new(
        vertices: [[f64; 2]; 3],
        flux: f64,
        contribution: Contribution,
        aa: usize,
        pad: usize,
        width: usize,
        height: usize,
    ) -> SilkResult<Self> {
        let mut points = vertices.map(|p| Point {
            x: p[0] * aa as f64 + pad as f64 - 0.5,
            y: p[1] * aa as f64 + pad as f64 - 0.5,
        });
        if points.iter().any(|p| !p.x.is_finite() || !p.y.is_finite()) {
            return Err("non-finite optical receiver coordinate".into());
        }
        let signed = points[1].sub(points[0]).cross(points[2].sub(points[0]));
        if !signed.is_finite() {
            return Err("optical triangle area exceeds finite range".into());
        }
        if signed < 0.0 {
            points.swap(1, 2);
        }
        let mut longest = Point { x: 0.0, y: 0.0 };
        let mut length = 0.0;
        for (a, b) in [(0, 1), (1, 2), (2, 0)] {
            let edge = points[b].sub(points[a]);
            let candidate = edge.x.hypot(edge.y);
            if candidate > length {
                length = candidate;
                longest = edge;
            }
        }
        if !length.is_finite() {
            return Err("optical edge exceeds finite range".into());
        }
        let axis =
            if length > 0.0 { longest.scale(length.recip()) } else { Point { x: 1.0, y: 0.0 } };
        let center = points[0]
            .scale(1.0 / 3.0)
            .add(points[1].scale(1.0 / 3.0))
            .add(points[2].scale(1.0 / 3.0));
        let mut projections = points.map(|p| p.sub(center).dot(axis));
        projections.sort_by(f64::total_cmp);
        let altitude = if length > 0.0 { signed.abs() / length } else { 0.0 };
        let point_weight = 1.0 - smooth((length - 1e-8) / (LINE_HIGH - 1e-8));
        let line_weight =
            (1.0 - point_weight) * smooth((LINE_HIGH - altitude) / (LINE_HIGH - LINE_LOW));
        // Projection onto the centroid line can move an endpoint just outside
        // the triangle AABB. Include that footprint before assigning tile bins.
        let support = [
            points[0],
            points[1],
            points[2],
            if line_weight > 0.0 { center.add(axis.scale(projections[0])) } else { center },
            if line_weight > 0.0 { center.add(axis.scale(projections[2])) } else { center },
        ];
        let xmin = support.iter().map(|p| p.x).fold(f64::INFINITY, f64::min);
        let xmax = support.iter().map(|p| p.x).fold(f64::NEG_INFINITY, f64::max);
        let ymin = support.iter().map(|p| p.y).fold(f64::INFINITY, f64::min);
        let ymax = support.iter().map(|p| p.y).fold(f64::NEG_INFINITY, f64::max);
        let bounds = Bounds {
            x0: xmin.floor().clamp(0.0, width as f64) as usize,
            y0: ymin.floor().clamp(0.0, height as f64) as usize,
            x1: (xmax.ceil() + 1.0).clamp(0.0, width as f64) as usize,
            y1: (ymax.ceil() + 1.0).clamp(0.0, height as f64) as usize,
        };
        Ok(Self {
            points,
            center,
            axis,
            projections,
            area: signed.abs() * 0.5,
            flux,
            point_weight,
            line_weight,
            bounds,
            contribution,
        })
    }
}

#[derive(Default, Clone, Copy)]
struct Sum {
    value: f64,
    error: f64,
}
impl Sum {
    fn add(&mut self, value: f64) {
        let corrected = value - self.error;
        let next = self.value + corrected;
        self.error = (next - self.value) - corrected;
        self.value = next;
    }
}

struct Deposit<'a> {
    pixels: &'a mut [f64],
    stride: usize,
    stripe_y: usize,
    bounds: Bounds,
    visible_x: &'a [f64],
    visible_y: &'a [f64],
    received: Sum,
    reference: Sum,
}
impl Deposit<'_> {
    fn add(&mut self, x: isize, y: isize, flux: f64, contribution: Contribution) {
        if x < 0 || y < 0 || !self.bounds.contains(x as usize, y as usize) {
            return;
        }
        let x = x as usize;
        let y = y as usize;
        let slot = (y - self.stripe_y) * self.stride + x;
        let visible = flux * self.visible_x[x] * self.visible_y[y];
        match contribution {
            Contribution::Reference => {
                self.pixels[slot] -= flux;
                self.reference.add(visible);
            }
            Contribution::Refracted => {
                self.pixels[slot] += flux;
                self.received.add(visible);
            }
            Contribution::Identity => {
                // A shared footprint has exactly zero redistribution. Adding
                // its many negative terms and then its positive terms would
                // leave roundoff in both empty and already illuminated pixels.
                self.reference.add(visible);
                self.received.add(visible);
            }
        }
    }
    fn point(&mut self, p: Point, flux: f64, contribution: Contribution) {
        let x = p.x.floor();
        let y = p.y.floor();
        let u = (p.x - x).clamp(0.0, 1.0);
        let v = (p.y - y).clamp(0.0, 1.0);
        self.add(x as isize, y as isize, flux * (1.0 - u) * (1.0 - v), contribution);
        self.add(x as isize + 1, y as isize, flux * u * (1.0 - v), contribution);
        self.add(x as isize, y as isize + 1, flux * (1.0 - u) * v, contribution);
        self.add(x as isize + 1, y as isize + 1, flux * u * v, contribution);
    }
}

fn clip(
    input: &[Point],
    output: &mut SmallVec<[Point; 8]>,
    axis: usize,
    bound: f64,
    greater: bool,
) {
    output.clear();
    if input.is_empty() {
        return;
    }
    let coordinate = |p: Point| if axis == 0 { p.x } else { p.y };
    let inside = |p: Point| if greater { coordinate(p) >= bound } else { coordinate(p) <= bound };
    let mut previous = input[input.len() - 1];
    let mut previous_inside = inside(previous);
    for &current in input {
        let current_inside = inside(current);
        if current_inside != previous_inside {
            let t = ((bound - coordinate(previous)) / (coordinate(current) - coordinate(previous)))
                .clamp(0.0, 1.0);
            let mut crossing = previous.add(current.sub(previous).scale(t));
            if axis == 0 {
                crossing.x = bound;
            } else {
                crossing.y = bound;
            }
            output.push(crossing);
        }
        if current_inside {
            output.push(current);
        }
        previous = current;
        previous_inside = current_inside;
    }
}

fn area_deposit(plan: &Plan, weight: f64, out: &mut Deposit<'_>) {
    if weight <= 0.0 || plan.area == 0.0 {
        return;
    }
    let x0 = plan.bounds.x0.saturating_sub(1).max(out.bounds.x0.saturating_sub(1));
    let y0 = plan.bounds.y0.saturating_sub(1).max(out.bounds.y0.saturating_sub(1));
    // Cell -1 still has a hat touching node0, so preserve it at padded edges.
    let x0 = if x0 == 0 { -1 } else { x0 as isize };
    let y0 = if y0 == 0 { -1 } else { y0 as isize };
    let x1 = plan.bounds.x1.min(out.bounds.x1) as isize;
    let y1 = plan.bounds.y1.min(out.bounds.y1) as isize;
    let mut polygon = SmallVec::<[Point; 8]>::new();
    let mut scratch = SmallVec::<[Point; 8]>::new();
    for y in y0..y1 {
        for x in x0..x1 {
            polygon.clear();
            polygon.extend(plan.points.map(|p| Point { x: p.x - x as f64, y: p.y - y as f64 }));
            for (axis, bound, greater) in
                [(0, 0.0, true), (0, 1.0, false), (1, 0.0, true), (1, 1.0, false)]
            {
                clip(&polygon, &mut scratch, axis, bound, greater);
                std::mem::swap(&mut polygon, &mut scratch);
            }
            if polygon.len() < 3 {
                continue;
            }
            let anchor = polygon[0];
            let mut moment = [0.0; 4];
            for index in 1..polygon.len() - 1 {
                let b = polygon[index];
                let c = polygon[index + 1];
                let area = b.sub(anchor).cross(c.sub(anchor)).abs() * 0.5;
                let sx = anchor.x + b.x + c.x;
                let sy = anchor.y + b.y + c.y;
                moment[0] += area;
                moment[1] += area * sx / 3.0;
                moment[2] += area * sy / 3.0;
                moment[3] += area * (sx * sy + anchor.x * anchor.y + b.x * b.y + c.x * c.y) / 12.0;
            }
            let [area, mx, my, mxy] = moment;
            if area <= 0.0 {
                continue;
            }
            let mut values = [area - mx - my + mxy, mx - mxy, my - mxy, mxy].map(|v| v.max(0.0));
            let total = values.iter().sum::<f64>();
            if total == 0.0 {
                continue;
            }
            let factor = plan.flux * weight * (area / plan.area) / total;
            for value in &mut values {
                *value *= factor;
            }
            for (value, (dx, dy)) in values.into_iter().zip([(0, 0), (1, 0), (0, 1), (1, 1)]) {
                out.add(x + dx, y + dy, value, plan.contribution);
            }
        }
    }
}

fn line_deposit(plan: &Plan, weight: f64, out: &mut Deposit<'_>) {
    if weight <= 0.0 {
        return;
    }
    let [first, middle, last] = plan.projections;
    if last <= first {
        out.point(plan.center, plan.flux * weight, plan.contribution);
        return;
    }
    for (lo, hi, rising) in [(first, middle, true), (middle, last, false)] {
        if hi <= lo {
            continue;
        }
        let mut start = lo;
        let mut end = hi;
        for (origin, direction, min, max) in [
            (plan.center.x, plan.axis.x, out.bounds.x0 as f64 - 1.0, out.bounds.x1 as f64),
            (plan.center.y, plan.axis.y, out.bounds.y0 as f64 - 1.0, out.bounds.y1 as f64),
        ] {
            if direction.abs() < 1e-15 {
                if origin < min || origin > max {
                    end = start;
                }
            } else {
                let a = (min - origin) / direction;
                let b = (max - origin) / direction;
                start = start.max(a.min(b));
                end = end.min(a.max(b));
            }
        }
        if end <= start {
            continue;
        }
        let mut breaks = vec![start, end];
        for (origin, direction) in [(plan.center.x, plan.axis.x), (plan.center.y, plan.axis.y)] {
            if direction.abs() < 1e-15 {
                continue;
            }
            let a = origin + direction * start;
            let b = origin + direction * end;
            for boundary in a.min(b).ceil() as isize..=a.max(b).floor() as isize {
                let q = (boundary as f64 - origin) / direction;
                if q > start && q < end {
                    breaks.push(q);
                }
            }
        }
        breaks.sort_by(f64::total_cmp);
        breaks.dedup();
        for pair in breaks.windows(2) {
            let center = f64::midpoint(pair[0], pair[1]);
            let half = (pair[1] - pair[0]) * 0.5;
            for sign in [-1.0, 1.0] {
                let q = center + sign * half / 3.0_f64.sqrt();
                let ramp = if rising { (q - lo) / (hi - lo) } else { (hi - q) / (hi - lo) };
                let flux = plan.flux * weight * half * (2.0 / (last - first)) * ramp;
                out.point(plan.center.add(plan.axis.scale(q)), flux, plan.contribution);
            }
        }
    }
}

#[derive(Clone, Copy)]
struct BoxKernel {
    interior: usize,
    edge: f64,
    norm: f64,
    support: usize,
}
impl BoxKernel {
    fn new(radius: f64) -> SilkResult<Self> {
        if !radius.is_finite() || radius <= 0.0 {
            return Err("optical footprint must be positive and finite".into());
        }
        if radius <= 0.5 {
            return Ok(Self { interior: 0, edge: 0.0, norm: 1.0, support: 0 });
        }
        let interior = (radius - 0.5).floor() as usize;
        let edge = radius - 0.5 - interior as f64;
        let support = interior
            .checked_add(usize::from(edge > 0.0))
            .ok_or("optical footprint support overflow")?;
        Ok(Self { interior, edge, norm: 2.0 * radius, support })
    }
}

fn box_line(input: &[f64], output: &mut [f64], kernel: BoxKernel) {
    let radius = kernel.interior;
    let mut sum = Sum::default();
    for &value in &input[..input.len().min(radius + 1)] {
        sum.add(value);
    }
    for index in 0..input.len() {
        if index > 0 {
            if index > radius {
                sum.add(-input[index - radius - 1]);
            }
            if index + radius < input.len() {
                sum.add(input[index + radius]);
            }
        }
        let mut value = sum.value;
        if index > radius {
            value += kernel.edge * input[index - radius - 1];
        }
        if index + radius + 1 < input.len() {
            value += kernel.edge * input[index + radius + 1];
        }
        output[index] = value / kernel.norm;
    }
}

fn visibility(size: usize, pad: usize, visible: usize, kernel: BoxKernel) -> Vec<f64> {
    let mut values: Vec<_> =
        (0..size).map(|i| if i >= pad && i < pad + visible { 1.0 } else { 0.0 }).collect();
    let mut scratch = vec![0.0; size];
    for _ in 0..3 {
        box_line(&values, &mut scratch, kernel);
        std::mem::swap(&mut values, &mut scratch);
    }
    values
}

fn blur(values: &mut Vec<f64>, width: usize, height: usize, kernel: BoxKernel) {
    let mut scratch = vec![0.0; values.len()];
    for _ in 0..3 {
        values
            .par_chunks(width)
            .zip(scratch.par_chunks_mut(width))
            .for_each(|(src, dst)| box_line(src, dst, kernel));
        std::mem::swap(values, &mut scratch);
    }
    // A transposed intermediate gives contiguous column walks and avoids false
    // sharing between worker writes. It is the same ordered one-dimensional sum.
    scratch.par_chunks_mut(height).enumerate().for_each(|(x, column)| {
        for (y, value) in column.iter_mut().enumerate() {
            *value = values[y * width + x];
        }
    });
    for _ in 0..3 {
        scratch
            .par_chunks(height)
            .zip(values.par_chunks_mut(height))
            .for_each(|(src, dst)| box_line(src, dst, kernel));
        std::mem::swap(values, &mut scratch);
    }
    values.par_chunks_mut(width).enumerate().for_each(|(y, row)| {
        for (x, value) in row.iter_mut().enumerate() {
            *value = scratch[x * height + y];
        }
    });
}

pub(super) fn accumulate(
    triangles: &[MappedTriangle],
    width: usize,
    height: usize,
    aa: usize,
    pixel_area: f64,
    footprint_pixels: f64,
) -> SilkResult<AccumulatedBand> {
    accumulate_impl::<true>(triangles, width, height, aa, pixel_area, footprint_pixels)
}

fn accumulate_impl<const DIRECT_IDENTITY_BUDGET: bool>(
    triangles: &[MappedTriangle],
    width: usize,
    height: usize,
    aa: usize,
    pixel_area: f64,
    footprint_pixels: f64,
) -> SilkResult<AccumulatedBand> {
    if width == 0 || height == 0 || aa == 0 || !pixel_area.is_finite() || pixel_area <= 0.0 {
        return Err("invalid optical receiving grid or pixel area".into());
    }
    let kernel = BoxKernel::new(footprint_pixels * aa as f64)?;
    let blur_support = kernel.support.checked_mul(3).ok_or("optical halo overflow")?;
    let pad = blur_support.checked_add(2).ok_or("optical halo overflow")?;
    let high_width = width.checked_mul(aa).ok_or("optical width overflow")?;
    let high_height = height.checked_mul(aa).ok_or("optical height overflow")?;
    let stride = pad
        .checked_mul(2)
        .and_then(|v| v.checked_add(high_width))
        .ok_or("optical padded width overflow")?;
    let rows = pad
        .checked_mul(2)
        .and_then(|v| v.checked_add(high_height))
        .ok_or("optical padded height overflow")?;
    let pixels = stride.checked_mul(rows).ok_or("optical receiving allocation overflow")?;
    if pixels > 4_000_000_000 || pixels.checked_mul(16).is_none_or(|n| n > isize::MAX as usize) {
        return Err("optical receiving buffers exceed the 64 GB safety bound".into());
    }
    let mut incident = Sum::default();
    let mut interior_identity = Sum::default();
    let crop = Bounds { x0: pad, y0: pad, x1: pad + high_width, y1: pad + high_height };
    let mut plans =
        Vec::with_capacity(triangles.len().checked_mul(2).ok_or("optical plan count overflow")?);
    let mut degenerate_triangles = 0;
    for triangle in triangles {
        if !triangle.flux.is_finite() || triangle.flux < 0.0 {
            return Err("optical flux must be nonnegative and finite".into());
        }
        incident.add(triangle.flux);
        let unchanged = triangle.source == triangle.receiver;
        let reference = if unchanged { Contribution::Identity } else { Contribution::Reference };
        // Keep unchanged triangles in both flux budgets, including crop loss,
        // while evaluating their common footprint only once.
        for (vertices, contribution) in
            [(triangle.source, reference), (triangle.receiver, Contribution::Refracted)]
                .into_iter()
                .take(if unchanged { 1 } else { 2 })
        {
            let plan = Plan::new(vertices, triangle.flux, contribution, aa, pad, stride, rows)?;
            if contribution != Contribution::Reference && plan.line_weight + plan.point_weight > 0.0
            {
                degenerate_triangles += 1;
            }
            if DIRECT_IDENTITY_BUDGET
                && contribution == Contribution::Identity
                && plan.bounds.remains_inside(crop, blur_support)
            {
                // The reconstruction bounds include the point/line blends;
                // three complete blur supports must also fit inside the crop.
                // This shared footprint redistributes nothing and retains all
                // its flux, so neither cell integration nor tile visits help.
                interior_identity.add(plan.flux);
                continue;
            }
            plans.push(plan);
        }
    }
    if !incident.value.is_finite() {
        return Err("total optical flux exceeds finite range".into());
    }
    let tiles_x = stride.div_ceil(TILE);
    let mut bins = vec![Vec::new(); tiles_x * rows.div_ceil(TILE)];
    for (index, plan) in plans.iter().enumerate() {
        if plan.bounds.empty() || plan.flux == 0.0 {
            continue;
        }
        for ty in plan.bounds.y0 / TILE..=(plan.bounds.y1 - 1) / TILE {
            for tx in plan.bounds.x0 / TILE..=(plan.bounds.x1 - 1) / TILE {
                bins[ty * tiles_x + tx].push(index);
            }
        }
    }
    let visible_x = visibility(stride, pad, high_width, kernel);
    let visible_y = visibility(rows, pad, high_height, kernel);
    let mut values = vec![0.0; pixels];
    let diagnostics: Vec<_> = values
        .par_chunks_mut(TILE * stride)
        .enumerate()
        .map(|(ty, stripe)| {
            let mut received = Sum::default();
            let mut reference = Sum::default();
            for tx in 0..tiles_x {
                let bounds = Bounds {
                    x0: tx * TILE,
                    y0: ty * TILE,
                    x1: ((tx + 1) * TILE).min(stride),
                    y1: ((ty + 1) * TILE).min(rows),
                };
                let mut out = Deposit {
                    pixels: stripe,
                    stride,
                    stripe_y: ty * TILE,
                    bounds,
                    visible_x: &visible_x,
                    visible_y: &visible_y,
                    received: Sum::default(),
                    reference: Sum::default(),
                };
                for &index in &bins[ty * tiles_x + tx] {
                    let plan = &plans[index];
                    if plan.point_weight > 0.0 {
                        out.point(plan.center, plan.flux * plan.point_weight, plan.contribution);
                    }
                    line_deposit(plan, plan.line_weight, &mut out);
                    area_deposit(
                        plan,
                        (1.0 - plan.point_weight - plan.line_weight).max(0.0),
                        &mut out,
                    );
                }
                received.add(out.received.value);
                reference.add(out.reference.value);
            }
            (received.value, reference.value)
        })
        .collect();
    let mut received = interior_identity;
    let mut reference = interior_identity;
    for (mapped, unbent) in diagnostics {
        received.add(mapped);
        reference.add(unbent);
    }
    drop(bins);
    drop(plans);
    blur(&mut values, stride, rows, kernel);
    let mut delta_irradiance = vec![0.0; width * height];
    delta_irradiance.par_chunks_mut(width).enumerate().for_each(|(y, row)| {
        for (x, pixel) in row.iter_mut().enumerate() {
            let mut flux = Sum::default();
            for dy in 0..aa {
                for dx in 0..aa {
                    flux.add(values[(pad + y * aa + dy) * stride + pad + x * aa + dx]);
                }
            }
            *pixel = flux.value / pixel_area;
        }
    });
    let mut integrated = Sum::default();
    for &value in &delta_irradiance {
        if !value.is_finite() {
            return Err("optical deposition produced a non-finite irradiance".into());
        }
        integrated.add(value * pixel_area);
    }
    let tolerance = incident.value.max(1.0) * 2e-8;
    let conservation_residual = integrated.value - (received.value - reference.value);
    if received.value > incident.value + tolerance
        || reference.value > incident.value + tolerance
        || conservation_residual.abs() > tolerance
    {
        return Err(format!("optical flux accounting exceeded tolerance: incident={}, received={}, reference={}, residual={conservation_residual}",incident.value,received.value,reference.value).into());
    }
    Ok(AccumulatedBand {
        delta_irradiance,
        incident_flux: incident.value,
        received_flux: received.value,
        escaped_flux: (incident.value - received.value).max(0.0),
        baseline_removed_flux: reference.value,
        reference_escaped_flux: (incident.value - reference.value).max(0.0),
        conservation_residual,
        degenerate_triangles,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    fn tri(receiver: [[f64; 2]; 3]) -> MappedTriangle {
        MappedTriangle { source: [[3.0, 3.0], [6.0, 3.0], [4.0, 6.0]], receiver, flux: 2.0 }
    }
    fn budgets(result: &AccumulatedBand) -> [f64; 6] {
        [
            result.incident_flux,
            result.received_flux,
            result.escaped_flux,
            result.baseline_removed_flux,
            result.reference_escaped_flux,
            result.conservation_residual,
        ]
    }
    fn identity_cases() -> Vec<MappedTriangle> {
        [
            [[12.25, 10.0], [46.25, 35.0], [12.25, 40.0]],
            [[12.25, 10.0], [12.75, 30.0], [12.25, 10.000_005]],
            [[10.0, 12.25], [50.0, 12.25], [22.0, 12.25]],
            [[31.25, 24.25]; 3],
            [[24.0, 26.0], [24.000_01, 26.0], [24.0, 26.000_01]],
        ]
        .into_iter()
        .enumerate()
        .map(|(i, vertices)| MappedTriangle {
            source: vertices,
            receiver: vertices,
            flux: (i as f64 + 0.25) / 3.0,
        })
        .collect()
    }
    #[test]
    fn identity_shortcut_requires_the_complete_reconstruction_and_blur_support() {
        for (aa, footprint) in [(1, 0.4), (1, 0.9), (3, 0.9)] {
            let blur_support = BoxKernel::new(footprint * aa as f64).unwrap().support * 3;
            let pad = blur_support + 2;
            let crop = Bounds { x0: pad, y0: pad, x1: pad + 48 * aa, y1: pad + 40 * aa };
            let left = (crop.x0 + blur_support) as f64;
            let top = (crop.y0 + blur_support) as f64;
            let right = (crop.x1 - blur_support - 1) as f64;
            let bottom = (crop.y1 - blur_support - 1) as f64;
            for (x, y, contained) in [
                (left, top, true),
                (right, bottom, true),
                (left - 0.25, top, false),
                (right + 0.25, bottom, false),
                (left, top - 0.25, false),
                (right, bottom + 0.25, false),
                (-100.0, top, false),
                (right + 1000.0, bottom, false),
            ] {
                let vertices =
                    [[(x - pad as f64 + 0.5) / aa as f64, (y - pad as f64 + 0.5) / aa as f64]; 3];
                let plan = Plan::new(
                    vertices,
                    1.0,
                    Contribution::Identity,
                    aa,
                    pad,
                    crop.x1 + pad,
                    crop.y1 + pad,
                )
                .unwrap();
                assert_eq!(plan.bounds.remains_inside(crop, blur_support), contained);
                let triangle = MappedTriangle { source: vertices, receiver: vertices, flux: 1.0 };
                let direct = accumulate(&[triangle], 48, 40, aa, 1.0, footprint).unwrap();
                let integrated =
                    accumulate_impl::<false>(&[triangle], 48, 40, aa, 1.0, footprint).unwrap();
                assert_eq!(direct.delta_irradiance, integrated.delta_irradiance);
                if contained {
                    assert_eq!(direct.received_flux, 1.0);
                    assert!((direct.received_flux - integrated.received_flux).abs() < 1e-12);
                } else {
                    // The old path must still determine partial visibility at
                    // every edge, including support clipped by the padded grid.
                    assert_eq!(budgets(&direct), budgets(&integrated));
                }
            }
        }
    }
    #[test]
    fn contained_identity_triangle_line_and_point_budgets_are_exact() {
        let triangles = identity_cases();
        let result = accumulate(&triangles, 72, 64, 3, 0.17, 0.9).unwrap();
        assert!(result.delta_irradiance.iter().all(|&v| v == 0.0));
        assert_eq!(result.received_flux, result.incident_flux);
        assert_eq!(result.baseline_removed_flux, result.incident_flux);
        assert_eq!(result.escaped_flux, 0.0);
        assert_eq!(result.reference_escaped_flux, 0.0);
        assert_eq!(result.conservation_residual, 0.0);
        assert_eq!(result.degenerate_triangles, 4);
    }
    #[test]
    fn identity_shortcut_matches_integrated_budgets_and_preserves_mixed_pixels() {
        let mut triangles = identity_cases();
        for vertices in [
            [[-0.2, 2.0], [2.0, 1.0], [0.5, 4.0]],
            [[71.9, 20.0]; 3],
            [[20.0, 63.9], [25.0, 63.9], [21.0, 63.9]],
            [[-20.0, 4.0]; 3],
        ] {
            triangles.push(MappedTriangle { source: vertices, receiver: vertices, flux: 1.3 });
        }
        triangles.insert(1, tri([[1.2, 1.1], [6.5, 2.7], [3.1, 7.3]]));
        triangles.push(MappedTriangle {
            source: [[18.0, 22.0], [51.0, 31.0], [22.0, 42.0]],
            receiver: [[22.0, 42.0], [61.0, 21.0], [18.0, 18.0]],
            flux: 3.7,
        });
        let integrated = accumulate_impl::<false>(&triangles, 72, 64, 3, 0.17, 0.9).unwrap();
        let run = |threads| {
            rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .unwrap()
                .install(|| accumulate(&triangles, 72, 64, 3, 0.17, 0.9).unwrap())
        };
        let direct = run(1);
        let parallel = run(4);
        assert_eq!(direct.delta_irradiance, integrated.delta_irradiance);
        assert_eq!(direct.degenerate_triangles, integrated.degenerate_triangles);
        for (a, b) in budgets(&direct).into_iter().zip(budgets(&integrated)) {
            assert!((a - b).abs() < direct.incident_flux * 1e-12, "budget {a} != {b}");
        }
        assert_eq!(direct.delta_irradiance, parallel.delta_irradiance);
        assert_eq!(budgets(&direct), budgets(&parallel));
        assert_eq!(direct.degenerate_triangles, parallel.degenerate_triangles);
    }
    #[test]
    fn identity_cancels_exactly_including_the_receiving_edge() {
        for vertices in [
            [[-0.2, 2.0], [2.0, 1.0], [0.5, 4.0]],
            [[-1.0, 2.0], [2.0, 2.0], [0.5, 2.0]],
            [[-0.2, 2.0]; 3],
        ] {
            let t = MappedTriangle { source: vertices, receiver: vertices, flux: 1.0 };
            let result = accumulate(&[t], 8, 8, 3, 1.0, 0.9).unwrap();
            assert!(result.delta_irradiance.iter().all(|&v| v == 0.0));
            assert_eq!(result.received_flux, result.baseline_removed_flux);
            assert_eq!(result.escaped_flux, result.reference_escaped_flux);
            assert_eq!(result.conservation_residual, 0.0);
            assert!(result.received_flux > 0.0 && result.escaped_flux > 0.0);
            assert!((result.received_flux + result.escaped_flux - 1.0).abs() < 1e-12);
        }
    }
    #[test]
    fn unchanged_flux_does_not_perturb_other_redistribution() {
        let changed = tri([[1.2, 1.1], [6.5, 2.7], [3.1, 7.3]]);
        let vertices = [[-0.2, 2.0], [2.0, 1.0], [0.5, 4.0]];
        let unchanged = MappedTriangle { source: vertices, receiver: vertices, flux: 1e6 };
        let baseline = accumulate(&[changed], 16, 12, 3, 1.0, 0.9).unwrap();
        let mixed = accumulate(&[unchanged, changed, unchanged], 16, 12, 3, 1.0, 0.9).unwrap();
        assert_eq!(mixed.delta_irradiance, baseline.delta_irradiance);
        assert_eq!(mixed.incident_flux, baseline.incident_flux + 2.0 * unchanged.flux);
        assert!(mixed.received_flux > baseline.received_flux);
        assert!(mixed.baseline_removed_flux > baseline.baseline_removed_flux);
    }
    #[test]
    fn footprint_allows_outside_light_to_enter_without_crop_renormalization() {
        let result = accumulate(&[tri([[-0.3, 4.0]; 3])], 8, 8, 3, 1.0, 0.9).unwrap();
        assert!(result.received_flux > 0.0 && result.received_flux < result.incident_flux);
        assert!((result.received_flux + result.escaped_flux - result.incident_flux).abs() < 1e-12);
        assert!(result.conservation_residual.abs() < 1e-10);
        let far = accumulate(&[tri([[-20.0, 4.0]; 3])], 8, 8, 3, 1.0, 0.9).unwrap();
        assert_eq!(far.received_flux, 0.0);
        assert_eq!(far.escaped_flux, far.incident_flux);
    }

    #[test]
    fn fully_contained_triangle_line_and_point_deliver_all_incident_flux() {
        for receiver in [
            [[12.25, 10.0], [46.25, 35.0], [12.25, 40.0]],
            [[12.25, 10.0], [12.75, 30.0], [12.25, 10.000_005]],
            [[10.0, 12.25], [50.0, 12.25], [22.0, 12.25]],
            [[31.25, 24.25]; 3],
        ] {
            let triangle = MappedTriangle {
                source: [[25.0, 25.0], [45.0, 25.0], [25.0, 45.0]],
                receiver,
                flux: 2.0,
            };
            let result = accumulate(&[triangle], 72, 64, 2, 1.0, 0.9).unwrap();
            assert!(
                (result.received_flux - result.incident_flux).abs() < 1e-10,
                "mapped flux {}",
                result.received_flux
            );
            assert!((result.baseline_removed_flux - result.incident_flux).abs() < 1e-10);
            assert!(result.escaped_flux < 1e-10);
            assert!(result.reference_escaped_flux < 1e-10);
        }
    }

    #[test]
    fn finite_footprint_is_normalized_and_has_the_requested_aa3_width() {
        let kernel = BoxKernel::new(0.9 * 3.0).unwrap();
        let mut values = vec![0.0; 129];
        values[64] = 1.0;
        let mut scratch = values.clone();
        for _ in 0..3 {
            box_line(&values, &mut scratch, kernel);
            std::mem::swap(&mut values, &mut scratch);
        }
        assert!((values.iter().sum::<f64>() - 1.0).abs() < 1e-12);
        let variance =
            values.iter().enumerate().map(|(i, &v)| v * (i as f64 - 64.0).powi(2)).sum::<f64>();
        assert!((variance.sqrt() / 3.0 - 0.9).abs() < 0.04);
        assert!(
            values
                .iter()
                .enumerate()
                .all(|(i, v)| i.abs_diff(64) <= 3 * kernel.support || v.abs() < 1e-14)
        );
    }
    #[test]
    fn collapsed_long_triangle_retains_extent_and_has_a_continuous_limit() {
        let line = [[1.0, 4.0], [7.0, 4.0], [2.0, 4.0]];
        let exact = accumulate(&[tri(line)], 10, 9, 2, 1.0, 0.9).unwrap();
        let near =
            accumulate(&[tri([[1.0, 4.0], [7.0, 4.0], [2.0, 4.000_000_01]])], 10, 9, 2, 1.0, 0.9)
                .unwrap();
        let error: f64 = exact
            .delta_irradiance
            .iter()
            .zip(&near.delta_irradiance)
            .map(|(a, b)| (a - b).abs())
            .sum();
        assert!(error < 1e-7, "line limit error {error}");
        let point = accumulate(&[tri([[10.0 / 3.0, 4.0]; 3])], 10, 9, 2, 1.0, 0.9).unwrap();
        let distinction: f64 = exact
            .delta_irradiance
            .iter()
            .zip(&point.delta_irradiance)
            .map(|(a, b)| (a - b).abs())
            .sum();
        assert!(distinction > 0.1, "long collapsed line was reduced to a point");
    }
    #[test]
    fn triangle_orientation_and_worker_count_do_not_change_flux() {
        let t = tri([[1.2, 1.1], [6.5, 2.7], [3.1, 7.3]]);
        let run = |threads| {
            rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .unwrap()
                .install(|| accumulate(&[t], 16, 12, 2, 0.1, 0.9).unwrap())
        };
        let a = run(1);
        let b = run(4);
        assert_eq!(a.delta_irradiance, b.delta_irradiance);
        assert_eq!(a.received_flux, b.received_flux);
        let mut reversed = t;
        reversed.receiver.swap(1, 2);
        let c = accumulate(&[reversed], 16, 12, 2, 0.1, 0.9).unwrap();
        assert_eq!(a.delta_irradiance, c.delta_irradiance);
    }
    #[test]
    fn refining_a_source_grid_converges_at_a_fold() {
        let make = |n: usize| {
            let mut triangles = Vec::new();
            for y in 0..n {
                for x in 0..n {
                    let corners = [(x, y), (x + 1, y), (x + 1, y + 1), (x, y + 1)].map(|(x, y)| {
                        let u = 2.0 * x as f64 / n as f64 - 1.0;
                        let v = 2.0 * y as f64 / n as f64 - 1.0;
                        (
                            [6.0 + 3.0 * u, 6.0 + 3.0 * v],
                            [6.0 + 3.0 * (u * u * u - 0.4 * u), 6.0 + 3.0 * v + 0.2 * u * u],
                        )
                    });
                    for indices in [[0, 1, 2], [0, 2, 3]] {
                        triangles.push(MappedTriangle {
                            source: indices.map(|i| corners[i].0),
                            receiver: indices.map(|i| corners[i].1),
                            flux: 2.0 / (n * n) as f64,
                        });
                    }
                }
            }
            triangles
        };
        let a = accumulate(&make(8), 12, 12, 2, 1.0, 0.9).unwrap();
        let b = accumulate(&make(16), 12, 12, 2, 1.0, 0.9).unwrap();
        let c = accumulate(&make(32), 12, 12, 2, 1.0, 0.9).unwrap();
        let error =
            |v: &[f64]| v.iter().zip(&c.delta_irradiance).map(|(a, b)| (a - b).abs()).sum::<f64>();
        assert!(error(&b.delta_irradiance) < error(&a.delta_irradiance));
        assert!(c.conservation_residual.abs() < 1e-9);
    }
}
