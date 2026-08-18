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
            // One bounds test per event, not two. It has to happen here rather than in the
            // builder because the coordinates are still `i64` — a negative one would wrap into
            // range on the cast — and testing here also lets an off-sensor event be rejected
            // before the second coordinate is looked at, which is most of them for a `crop`.
            if (0..width as i64).contains(&x) && (0..height as i64).contains(&y) {
                builder.push_in_bounds(x as u16, y as u16, t, p);
            }
        }
        builder.build()
    }

    /// Builds a new stream over a `(width, height)` grid by editing the columns in place, for the
    /// transforms that map every event onto the new grid and so cannot drop one.
    ///
    /// [`remap`](Self::remap) exists to *select*: it pays a closure, an `Option`, a bounds check
    /// and four `Vec::push`es per event so that an event can land off the sensor and be dropped. A
    /// flip, a rotation, a transpose, a rebin or a timestamp shift never does — each is onto the
    /// grid it declares — so all of that machinery is dead weight, and what is left is one pass
    /// over one column. `edit` gets the whole stream rather than a coordinate mapper so that it can
    /// touch only the columns it changes: the others are shared, not copied, and a transpose is a
    /// swap of two handles.
    ///
    /// A zero-width or zero-height grid can hold no events at all; `remap` expressed that by
    /// dropping every event at the bounds check, and this expresses it directly.
    pub(crate) fn map_columns(
        &self,
        width: usize,
        height: usize,
        edit: impl FnOnce(&mut EventStream),
    ) -> EventStream {
        if width == 0 || height == 0 {
            return EventStreamBuilder::new(width, height, self.timestamp_scale_ms()).build();
        }
        let mut out = self.clone();
        out.width = width;
        out.height = height;
        edit(&mut out);
        out
    }
}
