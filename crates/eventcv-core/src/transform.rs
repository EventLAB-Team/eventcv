//! Event-domain geometry and stream algebra — OpenCV-style transforms that operate on the
//! sparse [`EventStream`] itself (not a dense frame), preserving per-event timestamps. Every
//! op returns a **new** stream so calls chain. Coordinates are remapped and rounded — there is
//! no pixel interpolation — and out-of-bounds events are dropped (downscale therefore lets
//! several events share a pixel, which is lossless: representations handle the collisions).
//!
//! Frame-domain resize lives separately in [`crate::image`]; the two are complementary.

mod algebra;
mod spatial;
mod temporal;

use crate::{EventStream, EventStreamBuilder};

impl EventStream {
    /// Builds a new stream over a `(width, height)` grid by mapping each event through `f`.
    /// `f` returns `None` to drop an event, or new `(x, y, t, p)` as `i64` coordinates;
    /// events landing outside the grid (including negative coordinates) are dropped before the
    /// `u16` cast. The single construction path shared by the geometric transforms.
    ///
    /// `f` is `FnMut` so the random augmentations can carry their seeded RNG in the closure; events
    /// are visited once each, in order, so a stateful `f` sees a well-defined sequence.
    pub(crate) fn remap(
        &self,
        width: usize,
        height: usize,
        mut f: impl FnMut(i64, i64, i64, bool) -> Option<(i64, i64, i64, bool)>,
    ) -> EventStream {
        let mut builder =
            EventStreamBuilder::with_capacity(width, height, self.timestamp_scale_ms(), self.len());
        let (xs, ys, ts, ps) = (self.xs(), self.ys(), self.ts(), self.ps());
        for index in 0..self.len() {
            let Some((x, y, t, p)) = f(
                i64::from(xs[index]),
                i64::from(ys[index]),
                ts[index],
                ps[index],
            ) else {
                continue;
            };
            if (0..width as i64).contains(&x) && (0..height as i64).contains(&y) {
                builder.push(x as u16, y as u16, t, p);
            }
        }
        builder.build()
    }
}
