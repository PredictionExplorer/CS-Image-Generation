//! Small explicit double-precision vector operations for stable geometry.
use serde::{Deserialize, Serialize};
use std::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign};

/// Cartesian vector stored without platform-dependent padding in cache files.
#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
#[must_use]
pub struct V3 {
    /// Horizontal component.
    pub x: f64,
    /// Vertical component.
    pub y: f64,
    /// Depth component.
    pub z: f64,
}
impl V3 {
    /// Zero vector.
    pub const ZERO: Self = Self::new(0.0, 0.0, 0.0);
    /// Construct a vector.
    pub const fn new(x: f64, y: f64, z: f64) -> Self {
        Self { x, y, z }
    }
    /// Ordered scalar product.
    pub fn dot(self, rhs: Self) -> f64 {
        (self.x * rhs.x + self.y * rhs.y) + self.z * rhs.z
    }
    /// Oriented cross product.
    pub fn cross(self, b: Self) -> Self {
        Self::new(
            self.y * b.z - self.z * b.y,
            self.z * b.x - self.x * b.z,
            self.x * b.y - self.y * b.x,
        )
    }
    /// Squared Euclidean length.
    pub fn length_squared(self) -> f64 {
        self.dot(self)
    }
    /// Euclidean length.
    pub fn length(self) -> f64 {
        self.length_squared().sqrt()
    }
    /// Unit direction, or zero for a degenerate vector.
    pub fn normalized(self) -> Self {
        let n = self.length();
        if n > 1e-14 { self / n } else { Self::ZERO }
    }
    /// Linear interpolation.
    pub fn lerp(self, rhs: Self, t: f64) -> Self {
        self * (1.0 - t) + rhs * t
    }
    /// Whether every component is finite.
    pub fn is_finite(self) -> bool {
        self.x.is_finite() && self.y.is_finite() && self.z.is_finite()
    }
    /// Componentwise product.
    pub fn hadamard(self, rhs: Self) -> Self {
        Self::new(self.x * rhs.x, self.y * rhs.y, self.z * rhs.z)
    }
    /// Componentwise minimum.
    pub fn min(self, rhs: Self) -> Self {
        Self::new(self.x.min(rhs.x), self.y.min(rhs.y), self.z.min(rhs.z))
    }
    /// Componentwise maximum.
    pub fn max(self, rhs: Self) -> Self {
        Self::new(self.x.max(rhs.x), self.y.max(rhs.y), self.z.max(rhs.z))
    }
    /// Component by axis index.
    pub fn axis(self, axis: usize) -> f64 {
        match axis {
            0 => self.x,
            1 => self.y,
            _ => self.z,
        }
    }
}
impl Add for V3 {
    type Output = Self;
    fn add(self, b: Self) -> Self {
        Self::new(self.x + b.x, self.y + b.y, self.z + b.z)
    }
}
impl Sub for V3 {
    type Output = Self;
    fn sub(self, b: Self) -> Self {
        Self::new(self.x - b.x, self.y - b.y, self.z - b.z)
    }
}
impl Mul<f64> for V3 {
    type Output = Self;
    fn mul(self, b: f64) -> Self {
        Self::new(self.x * b, self.y * b, self.z * b)
    }
}
impl Mul<V3> for f64 {
    type Output = V3;
    fn mul(self, b: V3) -> V3 {
        b * self
    }
}
impl Div<f64> for V3 {
    type Output = Self;
    fn div(self, b: f64) -> Self {
        Self::new(self.x / b, self.y / b, self.z / b)
    }
}
impl Neg for V3 {
    type Output = Self;
    fn neg(self) -> Self {
        self * (-1.0)
    }
}
impl AddAssign for V3 {
    fn add_assign(&mut self, b: Self) {
        *self = *self + b;
    }
}
impl SubAssign for V3 {
    fn sub_assign(&mut self, b: Self) {
        *self = *self - b;
    }
}
impl MulAssign<f64> for V3 {
    fn mul_assign(&mut self, b: f64) {
        *self = *self * b;
    }
}
impl DivAssign<f64> for V3 {
    fn div_assign(&mut self, b: f64) {
        *self = *self / b;
    }
}
