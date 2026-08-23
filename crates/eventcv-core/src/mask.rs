//! Region-of-interest masks — the shapes behind [`EventStream::mask`](crate::EventStream::mask).
//!
//! Each function rasterises one shape into a row-major `width·height` grid where `true` means
//! **keep** the pixel, the layout every masking path in the crate takes ([`EventStream::mask`],
//! [`EventStream::drop_masked_pixels`](crate::EventStream::drop_masked_pixels) inverted, and the
//! live capture's per-event filter). Combine shapes by OR-ing/AND-ing the grids elementwise.
//!
//! Coordinates are continuous, not pixel indices: a pixel is kept when its **centre**
//! (`x + 0.5`, `y + 0.5`) falls inside the shape, so `rect(w, h, 0.0, 0.0, 64.0, 64.0)` keeps
//! exactly pixels `0..64`. Geometry off the sensor is clamped rather than rejected, and degenerate
//! input (an empty sensor, a non-positive radius, a polygon with fewer than three points) gives a
//! mask that keeps nothing.

use std::cmp::Ordering;

/// Keeps the axis-aligned rectangle of size `w`×`h` at `(x0, y0)`.
pub fn rect(width: usize, height: usize, x0: f64, y0: f64, w: f64, h: f64) -> Vec<bool> {
    let mut mask = vec![false; width * height];
    for y in span(y0, y0 + h, height) {
        for x in span(x0, x0 + w, width) {
            mask[y * width + x] = true;
        }
    }
    mask
}

/// Keeps the axis-aligned ellipse centred on `(cx, cy)` with semi-axes `rx`, `ry` — pass the same
/// value for both to get a circle.
pub fn ellipse(width: usize, height: usize, cx: f64, cy: f64, rx: f64, ry: f64) -> Vec<bool> {
    let mut mask = vec![false; width * height];
    // Negated so a NaN radius (which compares false either way) also keeps nothing.
    if !(rx > 0.0 && ry > 0.0) {
        return mask;
    }
    for y in span(cy - ry, cy + ry, height) {
        let dy = (y as f64 + 0.5 - cy) / ry;
        for x in span(cx - rx, cx + rx, width) {
            let dx = (x as f64 + 0.5 - cx) / rx;
            if dx * dx + dy * dy <= 1.0 {
                mask[y * width + x] = true;
            }
        }
    }
    mask
}

/// Keeps the interior of the closed polygon through `points`, by the even-odd rule (so a
/// self-intersecting outline leaves holes). The last point joins back to the first automatically,
/// which is what makes this the freehand tool: hand it the cursor track.
pub fn polygon(width: usize, height: usize, points: &[(f64, f64)]) -> Vec<bool> {
    let mut mask = vec![false; width * height];
    if points.len() < 3 {
        return mask;
    }
    let mut crossings: Vec<f64> = Vec::new();
    for y in 0..height {
        let scanline = y as f64 + 0.5;
        crossings.clear();
        for (index, &(x0, y0)) in points.iter().enumerate() {
            let (x1, y1) = points[(index + 1) % points.len()];
            // Half-open in y, so a vertex on the scanline counts once and horizontal edges (which
            // would divide by zero) never match.
            if (y0 <= scanline) != (y1 <= scanline) {
                crossings.push(x0 + (scanline - y0) / (y1 - y0) * (x1 - x0));
            }
        }
        crossings.sort_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));
        for pair in crossings.as_chunks::<2>().0 {
            for x in span(pair[0], pair[1], width) {
                mask[y * width + x] = true;
            }
        }
    }
    mask
}

/// The pixel indices whose centres lie in `[lo, hi)`, clamped to `0..limit`. Saturating casts turn
/// infinities and NaN into an empty or fully clamped range rather than a panic.
fn span(lo: f64, hi: f64, limit: usize) -> std::ops::Range<usize> {
    let start = ((lo - 0.5).ceil().max(0.0) as usize).min(limit);
    let end = ((hi - 0.5).ceil().max(0.0) as usize).min(limit);
    start.min(end)..end
}

#[cfg(test)]
mod tests {
    use super::{ellipse, polygon, rect, span};

    fn kept(mask: &[bool]) -> usize {
        mask.iter().filter(|&&keep| keep).count()
    }

    #[test]
    fn rect_keeps_exactly_its_pixels_and_clamps_off_sensor() {
        let mask = rect(8, 8, 2.0, 3.0, 3.0, 2.0);
        assert_eq!(kept(&mask), 6);
        for y in 0..8 {
            for x in 0..8 {
                let inside = (2..5).contains(&x) && (3..5).contains(&y);
                assert_eq!(mask[y * 8 + x], inside, "pixel ({x}, {y})");
            }
        }
        // A rectangle hanging off the sensor keeps the overlap; one entirely outside keeps nothing.
        assert_eq!(kept(&rect(8, 8, -4.0, -4.0, 6.0, 6.0)), 4);
        assert_eq!(kept(&rect(8, 8, 20.0, 0.0, 4.0, 4.0)), 0);
        assert_eq!(kept(&rect(8, 8, 0.0, 0.0, 0.0, 4.0)), 0); // zero width
    }

    #[test]
    fn ellipse_keeps_the_centre_rejects_the_corners_and_has_the_right_area() {
        let (width, height, radius) = (64, 64, 20.0);
        let mask = ellipse(width, height, 32.0, 32.0, radius, radius);
        assert!(mask[32 * width + 32]);
        assert!(!mask[0]);
        assert!(!mask[(height - 1) * width + width - 1]);
        // Pixel-centre sampling puts the area within a few percent of πr².
        let area = std::f64::consts::PI * radius * radius;
        assert!((kept(&mask) as f64 - area).abs() < 0.03 * area, "{}", kept(&mask));
        // Squashing one axis halves it, and a non-positive radius keeps nothing.
        assert!(kept(&ellipse(width, height, 32.0, 32.0, radius, radius / 2.0)) < kept(&mask));
        assert_eq!(kept(&ellipse(width, height, 32.0, 32.0, 0.0, radius)), 0);
    }

    #[test]
    fn polygon_fills_a_rectangle_exactly_like_rect() {
        let corners = [(2.0, 3.0), (5.0, 3.0), (5.0, 5.0), (2.0, 5.0)];
        assert_eq!(polygon(8, 8, &corners), rect(8, 8, 2.0, 3.0, 3.0, 2.0));
    }

    #[test]
    fn polygon_fills_a_triangle_by_the_even_odd_rule() {
        // A right triangle with legs on x = 0 and y = 0 and the hypotenuse x + y = 5: a pixel
        // centre is inside when x + y < 4, so the rows hold 4, 3, 2, and 1 pixels.
        let mask = polygon(8, 8, &[(0.0, 0.0), (5.0, 0.0), (0.0, 5.0)]);
        assert_eq!(kept(&mask), 4 + 3 + 2 + 1);
        assert!(mask[0]);
        assert!(!mask[3 * 8 + 3]);
        // Fewer than three points cannot enclose anything.
        assert_eq!(kept(&polygon(8, 8, &[(0.0, 0.0), (4.0, 4.0)])), 0);
    }

    #[test]
    fn degenerate_sensors_and_coordinates_produce_an_empty_mask() {
        assert!(rect(0, 0, 0.0, 0.0, 4.0, 4.0).is_empty());
        assert!(ellipse(0, 8, 0.0, 0.0, 2.0, 2.0).is_empty());
        assert_eq!(kept(&rect(8, 8, f64::NAN, 0.0, 4.0, 4.0)), 0);
        assert_eq!(span(f64::INFINITY, f64::INFINITY, 8).len(), 0);
    }
}
