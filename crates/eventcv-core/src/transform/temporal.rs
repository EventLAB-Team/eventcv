//! Temporal transforms — operate on timestamps (and event selection in time); coordinates and
//! sensor size are unchanged.

use crate::{EventStream, EventStreamBuilder};

impl EventStream {
    /// Keeps events whose timestamp lies in the half-open window `[t0, t1)`.
    ///
    /// In a time-ordered stream — what every reader and the simulator produce — the window is a
    /// contiguous range, so this is two binary searches and a slice rather than a predicate per
    /// event; a window covering the whole stream copies nothing at all. The scan that establishes
    /// the ordering is one pass over the timestamps and vectorises, and an unordered stream falls
    /// back to the general path rather than being silently mis-windowed.
    pub fn time_window(&self, t0: i64, t1: i64) -> EventStream {
        let (width, height) = self.sensor_size();
        let ts = self.ts();
        if !ts.is_sorted() {
            return self.remap(width, height, |x, y, t, p| {
                (t >= t0 && t < t1).then_some((x, y, t, p))
            });
        }
        let (lo, hi) = (
            ts.partition_point(|&t| t < t0),
            ts.partition_point(|&t| t < t1),
        );
        if (lo, hi) == (0, self.len()) {
            return self.clone();
        }
        let mut builder =
            EventStreamBuilder::with_capacity(width, height, self.timestamp_scale_ms(), hi - lo);
        builder.extend_from_columns(
            &self.xs()[lo..hi],
            &self.ys()[lo..hi],
            &ts[lo..hi],
            &self.ps()[lo..hi],
        );
        builder.build()
    }

    /// Shifts every timestamp by `dt` (same units as the stored timestamps).
    pub fn time_shift(&self, dt: i64) -> EventStream {
        let (width, height) = self.sensor_size();
        self.map_columns(width, height, |out| {
            for t in &mut out.ts {
                *t += dt;
            }
        })
    }

    /// Scales every timestamp by `factor` (rounded), e.g. to change playback speed.
    pub fn time_scale(&self, factor: f64) -> EventStream {
        let (width, height) = self.sensor_size();
        self.map_columns(width, height, |out| {
            for t in &mut out.ts {
                *t = (*t as f64 * factor).round() as i64;
            }
        })
    }

    /// Shifts timestamps so the earliest event starts at zero. A no-op on an empty stream.
    pub fn normalize_time(&self) -> EventStream {
        match self.ts().iter().min() {
            Some(&t_min) => self.time_shift(-t_min),
            None => self.clone(),
        }
    }

    /// Keeps every `k`-th event by index (`k = 1` is the identity); `k = 0` is treated as 1.
    pub fn decimate(&self, k: usize) -> EventStream {
        let k = k.max(1);
        let (width, height) = self.sensor_size();
        let mut builder = EventStreamBuilder::with_capacity(
            width,
            height,
            self.timestamp_scale_ms(),
            self.len() / k + 1,
        );
        let (xs, ys, ts, ps) = (self.xs(), self.ys(), self.ts(), self.ps());
        for index in (0..self.len()).step_by(k) {
            builder.push(xs[index], ys[index], ts[index], ps[index]);
        }
        builder.build()
    }
}

#[cfg(test)]
mod tests {
    use crate::{EventStream, EventStreamBuilder};

    fn sample() -> EventStream {
        let mut builder = EventStreamBuilder::new(4, 3, 0.001);
        for i in 0..6u16 {
            builder.push(i % 4, i % 3, 100 + i64::from(i) * 10, i % 2 == 0);
        }
        builder.build()
    }

    #[test]
    fn time_window_is_half_open() {
        let windowed = sample().time_window(110, 140); // keeps t = 110, 120, 130
        assert_eq!(windowed.ts(), &[110, 120, 130]);
    }

    /// The sorted fast path is a different code path from the general one, so both have to agree
    /// — including on a stream whose timestamps are out of order, where only the general one is
    /// correct.
    #[test]
    fn time_window_agrees_on_unordered_streams() {
        let mut builder = EventStreamBuilder::new(4, 3, 0.001);
        for (index, t) in [130_i64, 100, 150, 110, 140, 120].into_iter().enumerate() {
            let index = index as u16;
            builder.push(index % 4, index % 3, t, index.is_multiple_of(2));
        }
        let unordered = builder.build();

        let windowed = unordered.time_window(110, 140);
        assert_eq!(windowed.ts(), &[130, 110, 120]); // input order kept, window applied per event

        // Sorting first must select the same events, only ordered.
        let sorted = unordered.sort_by_time().time_window(110, 140);
        assert_eq!(sorted.ts(), &[110, 120, 130]);

        // A window covering everything is the stream itself, on either path.
        assert_eq!(unordered.time_window(0, 1_000).ts(), unordered.ts());
        assert_eq!(sample().time_window(0, 1_000).ts(), sample().ts());
    }

    #[test]
    fn time_shift_inverts_and_preserves_count() {
        let s = sample();
        let back = s.time_shift(1000).time_shift(-1000);
        assert_eq!(back.ts(), s.ts());
        assert_eq!(back.len(), s.len());
    }

    #[test]
    fn normalize_time_starts_at_zero() {
        let n = sample().normalize_time();
        assert_eq!(n.ts()[0], 0);
        assert_eq!(n.ts(), &[0, 10, 20, 30, 40, 50]);
    }

    #[test]
    fn time_scale_rounds() {
        let scaled = sample().time_scale(0.5);
        assert_eq!(scaled.ts(), &[50, 55, 60, 65, 70, 75]);
    }

    #[test]
    fn decimate_keeps_every_kth_event() {
        let d = sample().decimate(2);
        assert_eq!(d.len(), 3);
        assert_eq!(d.ts(), &[100, 120, 140]);
        assert_eq!(sample().decimate(0).len(), 6); // k=0 treated as identity
    }

    #[test]
    fn temporal_ops_handle_empty() {
        let empty = EventStreamBuilder::new(4, 3, 0.001).build();
        assert!(empty.normalize_time().is_empty());
        assert!(empty.time_shift(5).is_empty());
        assert!(empty.decimate(3).is_empty());
    }
}
