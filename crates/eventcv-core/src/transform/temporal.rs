//! Temporal transforms — operate on timestamps (and event selection in time); coordinates and
//! sensor size are unchanged.

use crate::{EventStream, EventStreamBuilder};

impl EventStream {
    /// Keeps events whose timestamp lies in the half-open window `[t0, t1)`.
    pub fn time_window(&self, t0: i64, t1: i64) -> EventStream {
        let (width, height) = self.sensor_size();
        self.remap(width, height, |x, y, t, p| {
            (t >= t0 && t < t1).then_some((x, y, t, p))
        })
    }

    /// Shifts every timestamp by `dt` (same units as the stored timestamps).
    pub fn time_shift(&self, dt: i64) -> EventStream {
        let (width, height) = self.sensor_size();
        self.remap(width, height, |x, y, t, p| Some((x, y, t + dt, p)))
    }

    /// Scales every timestamp by `factor` (rounded), e.g. to change playback speed.
    pub fn time_scale(&self, factor: f64) -> EventStream {
        let (width, height) = self.sensor_size();
        self.remap(width, height, |x, y, t, p| {
            Some((x, y, (t as f64 * factor).round() as i64, p))
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
