//! Pinhole camera intrinsics with Brown–Conrady (radial + tangential) distortion — the model
//! event-camera calibrations ship (e.g. EV-IMO `calib.txt` = `fx fy cx cy k1 k2 p1 p2`). Used
//! by [`EventStream::undistort`](crate::EventStream::undistort) to rectify event coordinates.

/// Number of fixed-point iterations for the distortion inverse. The grid is small and the LUT
/// is built once, so we can afford to iterate well past convergence.
const UNDISTORT_ITERS: usize = 20;

/// Pinhole intrinsics `(fx, fy, cx, cy)` plus distortion `(k1, k2, p1, p2, k3)`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Camera {
    pub fx: f64,
    pub fy: f64,
    pub cx: f64,
    pub cy: f64,
    pub k1: f64,
    pub k2: f64,
    pub p1: f64,
    pub p2: f64,
    pub k3: f64,
}

impl Camera {
    /// A distortion-free pinhole camera (all distortion coefficients zero).
    pub fn new(fx: f64, fy: f64, cx: f64, cy: f64) -> Self {
        Self {
            fx,
            fy,
            cx,
            cy,
            k1: 0.0,
            k2: 0.0,
            p1: 0.0,
            p2: 0.0,
            k3: 0.0,
        }
    }

    /// A camera with the full radial (`k1, k2, k3`) and tangential (`p1, p2`) distortion model.
    #[allow(clippy::too_many_arguments)]
    pub fn with_distortion(
        fx: f64,
        fy: f64,
        cx: f64,
        cy: f64,
        k1: f64,
        k2: f64,
        p1: f64,
        p2: f64,
        k3: f64,
    ) -> Self {
        Self {
            fx,
            fy,
            cx,
            cy,
            k1,
            k2,
            p1,
            p2,
            k3,
        }
    }

    /// Maps a **distorted** pixel `(u, v)` to its **undistorted** pixel location in the same
    /// camera (same intrinsics), inverting the distortion model iteratively (OpenCV's
    /// `undistortPoints` scheme).
    pub fn undistort_point(&self, u: f64, v: f64) -> (f64, f64) {
        let (x0, y0) = ((u - self.cx) / self.fx, (v - self.cy) / self.fy);
        let (mut x, mut y) = (x0, y0);
        for _ in 0..UNDISTORT_ITERS {
            let r2 = x * x + y * y;
            let radial = 1.0 + ((self.k3 * r2 + self.k2) * r2 + self.k1) * r2;
            let dx = 2.0 * self.p1 * x * y + self.p2 * (r2 + 2.0 * x * x);
            let dy = self.p1 * (r2 + 2.0 * y * y) + 2.0 * self.p2 * x * y;
            x = (x0 - dx) / radial;
            y = (y0 - dy) / radial;
        }
        (x * self.fx + self.cx, y * self.fy + self.cy)
    }

    /// Forward model: maps **ideal normalized** coordinates to a **distorted** pixel. The
    /// inverse of [`Self::undistort_point`]'s normalized stage; used to validate the inverse.
    pub fn distort_point(&self, x: f64, y: f64) -> (f64, f64) {
        let r2 = x * x + y * y;
        let radial = 1.0 + ((self.k3 * r2 + self.k2) * r2 + self.k1) * r2;
        let dx = 2.0 * self.p1 * x * y + self.p2 * (r2 + 2.0 * x * x);
        let dy = self.p1 * (r2 + 2.0 * y * y) + 2.0 * self.p2 * x * y;
        (
            (x * radial + dx) * self.fx + self.cx,
            (y * radial + dy) * self.fy + self.cy,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::Camera;

    #[test]
    fn undistort_without_distortion_is_identity() {
        let camera = Camera::new(256.9, 256.9, 131.6, 188.6);
        for &(u, v) in &[(0.0, 0.0), (100.0, 50.0), (345.0, 259.0)] {
            let (nu, nv) = camera.undistort_point(u, v);
            assert!((nu - u).abs() < 1e-9 && (nv - v).abs() < 1e-9);
        }
    }

    #[test]
    fn undistort_inverts_the_forward_distortion() {
        // EV-IMO `box/seq_10` calib values.
        let camera = Camera::with_distortion(
            256.919, 256.862, 131.651, 188.576, -0.366475, 0.130235, 0.000262, 0.000191, 0.0,
        );
        // For several ideal normalized points: distort to a pixel, then undistort back.
        for &(xn, yn) in &[(0.0, 0.0), (0.2, -0.3), (-0.4, 0.1), (0.5, 0.5)] {
            let (u, v) = camera.distort_point(xn, yn);
            let (xu, yu) = camera.undistort_point(u, v);
            let (ideal_u, ideal_v) = (xn * camera.fx + camera.cx, yn * camera.fy + camera.cy);
            assert!(
                (xu - ideal_u).abs() < 1e-3 && (yu - ideal_v).abs() < 1e-3,
                "round-trip drift at ({xn}, {yn}): got ({xu}, {yu}) vs ({ideal_u}, {ideal_v})"
            );
        }
    }
}
