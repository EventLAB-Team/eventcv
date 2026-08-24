//! Spatial (geometric) transforms on the event stream. Coordinates are remapped and rounded;
//! out-of-bounds events drop and the sensor size is recomputed per op.

use crate::camera::Camera;
use crate::EventStream;

impl EventStream {
    /// Keeps events inside the `w`×`h` window at `(x0, y0)` and shifts them to a new origin.
    /// The result is a `w`×`h` stream.
    pub fn crop(&self, x0: i64, y0: i64, w: usize, h: usize) -> EventStream {
        self.remap(w, h, |x, y, t, p| Some((x - x0, y - y0, t, p)))
    }

    /// Mirrors horizontally (`x → width-1-x`). Sensor size unchanged.
    pub fn flip_x(&self) -> EventStream {
        let (width, height) = self.sensor_size();
        let max_x = width.saturating_sub(1) as u16;
        self.map_columns(width, height, |out| {
            for x in &mut out.xs {
                *x = max_x - *x;
            }
        })
    }

    /// Mirrors vertically (`y → height-1-y`). Sensor size unchanged.
    pub fn flip_y(&self) -> EventStream {
        let (width, height) = self.sensor_size();
        let max_y = height.saturating_sub(1) as u16;
        self.map_columns(width, height, |out| {
            for y in &mut out.ys {
                *y = max_y - *y;
            }
        })
    }

    /// Rotates by `k * 90°` clockwise. `k` is taken mod 4; quarter turns swap the sensor dims.
    ///
    /// A quarter turn swaps the two coordinate columns before rewriting one of them, so it copies
    /// one column rather than four.
    pub fn rotate90(&self, k: i32) -> EventStream {
        let (width, height) = self.sensor_size();
        let (max_x, max_y) = (
            width.saturating_sub(1) as u16,
            height.saturating_sub(1) as u16,
        );
        match k.rem_euclid(4) {
            0 => self.map_columns(width, height, |_| {}),
            1 => self.map_columns(height, width, |out| {
                std::mem::swap(&mut out.xs, &mut out.ys);
                for x in &mut out.xs {
                    *x = max_y - *x;
                }
            }),
            2 => self.map_columns(width, height, |out| {
                for x in &mut out.xs {
                    *x = max_x - *x;
                }
                for y in &mut out.ys {
                    *y = max_y - *y;
                }
            }),
            _ => self.map_columns(height, width, |out| {
                std::mem::swap(&mut out.xs, &mut out.ys);
                for y in &mut out.ys {
                    *y = max_x - *y;
                }
            }),
        }
    }

    /// Reflects across the main diagonal (`(x, y) → (y, x)`); swaps the sensor dims.
    pub fn transpose(&self) -> EventStream {
        let (width, height) = self.sensor_size();
        self.map_columns(height, width, |out| {
            std::mem::swap(&mut out.xs, &mut out.ys)
        })
    }

    /// Translates by `(dx, dy)`; events shifted off the sensor are dropped. Sensor unchanged.
    pub fn translate(&self, dx: i64, dy: i64) -> EventStream {
        let (width, height) = self.sensor_size();
        self.remap(width, height, |x, y, t, p| Some((x + dx, y + dy, t, p)))
    }

    /// Resizes the sensor grid to `w`×`h`, rebinning each coordinate proportionally (floored —
    /// the destination bin, no interpolation). Every event maps into `[0, w)×[0, h)`, so the
    /// count is conserved; on downscale several events may share a pixel (lossless).
    pub fn resize(&self, w: usize, h: usize) -> EventStream {
        let (width, height) = self.sensor_size();
        let sx = if width > 0 {
            w as f64 / width as f64
        } else {
            0.0
        };
        let sy = if height > 0 {
            h as f64 / height as f64
        } else {
            0.0
        };
        self.map_columns(w, h, |out| {
            for x in &mut out.xs {
                *x = (f64::from(*x) * sx).floor() as u16;
            }
            for y in &mut out.ys {
                *y = (f64::from(*y) * sy).floor() as u16;
            }
        })
    }

    /// Scales the sensor by `(sx, sy)`, rounding the new dimensions. See [`Self::resize`].
    pub fn scale(&self, sx: f64, sy: f64) -> EventStream {
        let (width, height) = self.sensor_size();
        let w = (width as f64 * sx).round().max(0.0) as usize;
        let h = (height as f64 * sy).round().max(0.0) as usize;
        self.resize(w, h)
    }

    /// Applies a 2×3 affine matrix `[[a,b,c],[d,e,f]]` (`x' = a·x+b·y+c`, rounded). Sensor
    /// size unchanged; events warped off the sensor are dropped.
    pub fn warp_affine(&self, m: [[f64; 3]; 2]) -> EventStream {
        let (width, height) = self.sensor_size();
        self.remap(width, height, |x, y, t, p| {
            let (xf, yf) = (x as f64, y as f64);
            let nx = m[0][0] * xf + m[0][1] * yf + m[0][2];
            let ny = m[1][0] * xf + m[1][1] * yf + m[1][2];
            Some((nx.round() as i64, ny.round() as i64, t, p))
        })
    }

    /// Applies a 3×3 perspective (homography) matrix, dividing by the homogeneous coordinate.
    /// Events whose denominator is zero, or that warp off the sensor, are dropped.
    pub fn warp_perspective(&self, m: [[f64; 3]; 3]) -> EventStream {
        let (width, height) = self.sensor_size();
        self.remap(width, height, |x, y, t, p| {
            let (xf, yf) = (x as f64, y as f64);
            let w = m[2][0] * xf + m[2][1] * yf + m[2][2];
            if w == 0.0 {
                return None;
            }
            let nx = (m[0][0] * xf + m[0][1] * yf + m[0][2]) / w;
            let ny = (m[1][0] * xf + m[1][1] * yf + m[1][2]) / w;
            Some((nx.round() as i64, ny.round() as i64, t, p))
        })
    }

    /// Rectifies events with a [`Camera`]'s intrinsics + distortion, mapping each event from its
    /// distorted pixel to the undistorted location on the same grid. Builds a per-pixel lookup
    /// once (the sensor grid is small), then remaps every event through it; events landing off
    /// the sensor after rectification are dropped. Sensor size unchanged.
    pub fn undistort(&self, camera: &Camera) -> EventStream {
        let (width, height) = self.sensor_size();
        if width == 0 || height == 0 {
            return self.clone();
        }
        let lut: Vec<(i64, i64)> = (0..width * height)
            .map(|i| {
                let (u, v) = camera.undistort_point((i % width) as f64, (i / width) as f64);
                (u.round() as i64, v.round() as i64)
            })
            .collect();
        self.remap(width, height, |x, y, t, p| {
            let (nx, ny) = lut[y as usize * width + x as usize];
            Some((nx, ny, t, p))
        })
    }

    /// Keeps only events where the `mask_w`×`mask_h` row-major boolean grid is `true`. Events
    /// outside the mask are dropped. Sensor size unchanged.
    pub fn mask(&self, mask: &[bool], mask_w: usize, mask_h: usize) -> EventStream {
        let (width, height) = self.sensor_size();
        self.remap(width, height, |x, y, t, p| {
            let (ux, uy) = (x as usize, y as usize);
            let keep = ux < mask_w && uy < mask_h && mask[uy * mask_w + ux];
            keep.then_some((x, y, t, p))
        })
    }
}

#[cfg(test)]
mod tests {
    use crate::{EventStream, EventStreamBuilder};

    /// A 4×3 stream with one event per column on a diagonal-ish path.
    fn sample() -> EventStream {
        let mut builder = EventStreamBuilder::new(4, 3, 0.001);
        builder.push(0, 0, 10, true);
        builder.push(1, 1, 20, false);
        builder.push(2, 2, 30, true);
        builder.push(3, 0, 40, false);
        builder.build()
    }

    fn coords(stream: &EventStream) -> Vec<(u16, u16)> {
        stream
            .xs()
            .iter()
            .copied()
            .zip(stream.ys().iter().copied())
            .collect()
    }

    #[test]
    fn crop_subsets_and_shifts_to_new_origin() {
        let cropped = sample().crop(1, 1, 2, 2);
        assert_eq!(cropped.sensor_size(), (2, 2));
        assert_eq!(coords(&cropped), vec![(0, 0), (1, 1)]); // events (1,1) and (2,2)
        assert_eq!(cropped.ts(), &[20, 30]);
    }

    #[test]
    fn flips_are_their_own_inverse() {
        let s = sample();
        assert_eq!(coords(&s.flip_x().flip_x()), coords(&s));
        assert_eq!(coords(&s.flip_y().flip_y()), coords(&s));
        assert_eq!(s.flip_x().sensor_size(), (4, 3));
        assert_eq!(coords(&s.flip_x()), vec![(3, 0), (2, 1), (1, 2), (0, 0)]);
    }

    #[test]
    fn rotate90_swaps_dims_and_round_trips() {
        let s = sample();
        assert_eq!(s.rotate90(1).sensor_size(), (3, 4)); // W×H -> H×W
        assert_eq!(s.rotate90(2).sensor_size(), (4, 3));
        // Four quarter turns and a (1 then 3) pair both return to the original.
        assert_eq!(coords(&s.rotate90(4)), coords(&s));
        assert_eq!(coords(&s.rotate90(1).rotate90(3)), coords(&s));
        assert_eq!(coords(&s.rotate90(-1)), coords(&s.rotate90(3)));
    }

    #[test]
    fn transpose_swaps_axes_and_dims() {
        let t = sample().transpose();
        assert_eq!(t.sensor_size(), (3, 4));
        assert_eq!(coords(&t), vec![(0, 0), (1, 1), (2, 2), (0, 3)]);
    }

    #[test]
    fn translate_shifts_and_drops_out_of_bounds() {
        // x + 2 = [2, 3, 4, 5]; only x = 2, 3 stay on the 4-wide sensor (the rest fall off).
        let shifted = sample().translate(2, 0);
        assert_eq!(shifted.xs(), &[2, 3]);
        assert_eq!(shifted.ys(), &[0, 1]);
        // Translating the survivors back restores their original coordinates.
        assert_eq!(coords(&shifted.translate(-2, 0)), vec![(0, 0), (1, 1)]);
    }

    #[test]
    fn resize_rebins_and_conserves_count() {
        let down = sample().resize(2, 2); // 4x3 -> 2x2; floor keeps every event
        assert_eq!(down.sensor_size(), (2, 2));
        assert_eq!(down.len(), 4);
        assert!(down.xs().iter().all(|&x| x < 2) && down.ys().iter().all(|&y| y < 2));
        assert_eq!(sample().scale(2.0, 2.0).sensor_size(), (8, 6));
    }

    #[test]
    fn warp_affine_identity_and_translation() {
        let s = sample();
        let identity = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];
        assert_eq!(coords(&s.warp_affine(identity)), coords(&s));
        let shift = [[1.0, 0.0, 1.0], [0.0, 1.0, 0.0]];
        assert_eq!(coords(&s.warp_affine(shift)), coords(&s.translate(1, 0)));
    }

    #[test]
    fn warp_perspective_identity_round_trips() {
        let identity = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
        let s = sample();
        assert_eq!(coords(&s.warp_perspective(identity)), coords(&s));
    }

    #[test]
    fn mask_keeps_only_selected_pixels() {
        let s = sample(); // 4x3
        let mut mask = vec![false; 4 * 3];
        mask[4 + 1] = true; // keep only pixel (1,1): row 1 (×4) + col 1
        let masked = s.mask(&mask, 4, 3);
        assert_eq!(coords(&masked), vec![(1, 1)]);
    }

    #[test]
    fn undistort_without_distortion_keeps_events_in_place() {
        use crate::camera::Camera;
        let s = sample(); // 4×3
        let camera = Camera::new(100.0, 100.0, 2.0, 1.5); // no distortion -> identity map
        assert_eq!(coords(&s.undistort(&camera)), coords(&s));
        assert_eq!(s.undistort(&camera).sensor_size(), (4, 3));
    }

    #[test]
    fn undistort_remaps_under_distortion() {
        use crate::camera::Camera;
        let s = sample();
        // Barrel distortion pulls events toward the principal point; the result stays on-grid
        // and conserves count here (no event leaves the 4×3 sensor).
        let camera = Camera::with_distortion(50.0, 50.0, 2.0, 1.5, -0.2, 0.0, 0.0, 0.0, 0.0);
        let out = s.undistort(&camera);
        assert_eq!(out.sensor_size(), (4, 3));
        assert!(out.len() <= s.len());
        assert!(out.xs().iter().all(|&x| x < 4) && out.ys().iter().all(|&y| y < 3));
    }

    /// The count-preserving transforms bypass `remap` and edit the columns directly, so the one
    /// thing that has to stay true is that they still agree with `remap` event for event.
    #[test]
    fn bijective_transforms_agree_with_the_general_path() {
        let mut builder = EventStreamBuilder::new(13, 7, 0.001);
        for index in 0..60_u16 {
            builder.push(index % 13, index % 7, i64::from(index) * 3, index % 3 == 0);
        }
        let s = builder.build();
        let (w, h) = s.sensor_size();
        let (max_x, max_y) = (w as i64 - 1, h as i64 - 1);
        let (sx, sy) = (5.0 / w as f64, 4.0 / h as f64);

        let cases: [(EventStream, EventStream); 6] = [
            (
                s.flip_x(),
                s.remap(w, h, |x, y, t, p| Some((max_x - x, y, t, p))),
            ),
            (
                s.flip_y(),
                s.remap(w, h, |x, y, t, p| Some((x, max_y - y, t, p))),
            ),
            (
                s.transpose(),
                s.remap(h, w, |x, y, t, p| Some((y, x, t, p))),
            ),
            (
                s.rotate90(1),
                s.remap(h, w, |x, y, t, p| Some((max_y - y, x, t, p))),
            ),
            (
                s.rotate90(3),
                s.remap(h, w, |x, y, t, p| Some((y, max_x - x, t, p))),
            ),
            (
                s.resize(5, 4),
                s.remap(5, 4, |x, y, t, p| {
                    Some((
                        (x as f64 * sx).floor() as i64,
                        (y as f64 * sy).floor() as i64,
                        t,
                        p,
                    ))
                }),
            ),
        ];

        for (fast, general) in cases {
            assert_eq!(fast.sensor_size(), general.sensor_size());
            assert_eq!(coords(&fast), coords(&general));
            assert_eq!(fast.ts(), general.ts());
            assert_eq!(fast.ps(), general.ps());
        }
    }

    /// Every transform that bypasses `remap` claims to map onto the grid it declares. If one ever
    /// did not, it would silently produce out-of-range coordinates instead of dropping the event.
    #[test]
    fn bijective_transforms_keep_every_event_in_bounds() {
        let s = sample();
        for out in [
            s.flip_x(),
            s.flip_y(),
            s.transpose(),
            s.rotate90(1),
            s.rotate90(2),
            s.rotate90(3),
            s.resize(2, 2),
            s.resize(9, 7),
            s.time_shift(-5),
            s.invert_polarity(),
        ] {
            let (width, height) = out.sensor_size();
            assert_eq!(out.len(), s.len());
            assert!(out.xs().iter().all(|&x| (x as usize) < width));
            assert!(out.ys().iter().all(|&y| (y as usize) < height));
        }
    }

    #[test]
    fn transforms_handle_the_empty_stream() {
        let empty = EventStreamBuilder::new(4, 3, 0.001).build();
        assert!(empty.flip_x().is_empty());
        assert!(empty.rotate90(1).is_empty());
        assert_eq!(empty.crop(0, 0, 2, 2).sensor_size(), (2, 2));
        assert!(empty.resize(2, 2).is_empty());
    }
}
