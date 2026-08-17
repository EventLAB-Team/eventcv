//! Temporal analytics — how event activity varies over the length of a recording.
//!
//! The spatial counterpart already exists elsewhere: `io::SliceSource::pixel_counts` totals events
//! per pixel across a whole file, which is the heatmap. This module is the other axis, and answers
//! the questions that come up when a recording behaves oddly — where did the sensor saturate, where
//! is the scene actually still, is one polarity dominating.
//!
//! Counts are returned rather than plotted: the library depends on nothing but `ndarray` here, and a
//! caller who wants a figure already has matplotlib.

use crate::EventStream;

/// Activity over time, binned into fixed-width intervals.
///
/// `starts` holds each bin's left edge in the stream's own timestamp units, so it lines up with
/// `EventStream::ts()` without conversion. The three count arrays are the same length as `starts`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct EventRate {
    /// Left edge of each bin, in stream timestamp units (µs).
    pub starts: Vec<i64>,
    /// Events of either polarity in each bin.
    pub counts: Vec<u64>,
    /// Positive-polarity events in each bin.
    pub positive: Vec<u64>,
    /// Negative-polarity events in each bin.
    pub negative: Vec<u64>,
    /// Bin width in stream timestamp units (µs).
    pub bin_us: i64,
}

impl EventRate {
    /// Events per second in each bin — `counts` divided by the bin width.
    ///
    /// Separate from `counts` because the bins are uniform: dividing is only meaningful if every bin
    /// covers the same span, which is exactly what this binning guarantees and what a caller
    /// aggregating by some other rule could not assume.
    pub fn per_second(&self) -> Vec<f64> {
        let seconds = self.bin_us as f64 / 1_000_000.0;
        self.counts.iter().map(|&n| n as f64 / seconds).collect()
    }

    /// Number of bins.
    pub fn len(&self) -> usize {
        self.starts.len()
    }

    /// True when the stream was empty and there is nothing to plot.
    pub fn is_empty(&self) -> bool {
        self.starts.is_empty()
    }
}

impl EventStream {
    /// Bins the stream into fixed-width intervals of `bin_us` and counts events in each.
    ///
    /// Bins span the stream's own extent — from the earliest to the latest timestamp — so an empty
    /// stream produces no bins and a stream that does not divide evenly gets a final short bin that
    /// is still counted. A `bin_us` below 1 is clamped, since a zero-width bin has no rate.
    ///
    /// Does not require sorted input: bins are indexed arithmetically from the minimum timestamp
    /// rather than by walking in order, so this is safe to call before `sort_by_time`.
    pub fn event_rate(&self, bin_us: i64) -> EventRate {
        let bin_us = bin_us.max(1);
        let ts = self.ts();
        let (Some(&t_min), Some(&t_max)) = (ts.iter().min(), ts.iter().max()) else {
            return EventRate {
                starts: Vec::new(),
                counts: Vec::new(),
                positive: Vec::new(),
                negative: Vec::new(),
                bin_us,
            };
        };

        // +1 because the span is inclusive of the last event: a recording from t=0 to t=100 with
        // 100µs bins needs two, not one.
        let bins = ((t_max - t_min) / bin_us + 1) as usize;
        let mut counts = vec![0u64; bins];
        let mut positive = vec![0u64; bins];
        let mut negative = vec![0u64; bins];

        for (&t, &p) in ts.iter().zip(self.ps()) {
            let bin = ((t - t_min) / bin_us) as usize;
            counts[bin] += 1;
            if p {
                positive[bin] += 1;
            } else {
                negative[bin] += 1;
            }
        }

        EventRate {
            starts: (0..bins).map(|i| t_min + i as i64 * bin_us).collect(),
            counts,
            positive,
            negative,
            bin_us,
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::{EventStream, EventStreamBuilder};

    fn sample() -> EventStream {
        // Events at t = 0, 10, 20, … 90; polarity alternates.
        let mut builder = EventStreamBuilder::new(4, 4, 0.001);
        for i in 0..10u16 {
            builder.push(i % 4, i % 4, i64::from(i) * 10, i % 2 == 0);
        }
        builder.build()
    }

    #[test]
    fn bins_cover_the_whole_span() {
        let rate = sample().event_rate(50);
        assert_eq!(rate.starts, vec![0, 50]);
        assert_eq!(rate.counts, vec![5, 5]);
        assert_eq!(rate.counts.iter().sum::<u64>(), 10);
    }

    #[test]
    fn polarities_sum_to_the_total() {
        let rate = sample().event_rate(30);
        for i in 0..rate.len() {
            assert_eq!(rate.positive[i] + rate.negative[i], rate.counts[i]);
        }
        assert_eq!(rate.positive.iter().sum::<u64>(), 5);
        assert_eq!(rate.negative.iter().sum::<u64>(), 5);
    }

    #[test]
    fn a_trailing_partial_bin_is_still_counted() {
        // Span is 0..=90 with 40µs bins: 0-39, 40-79, 80-90 (short).
        let rate = sample().event_rate(40);
        assert_eq!(rate.len(), 3);
        assert_eq!(rate.counts, vec![4, 4, 2]);
    }

    #[test]
    fn per_second_scales_by_bin_width() {
        let rate = sample().event_rate(50); // 50µs bins, 5 events each
        assert_eq!(rate.per_second(), vec![100_000.0, 100_000.0]);
    }

    #[test]
    fn unsorted_input_bins_identically() {
        let sorted = sample();
        let shuffled = sorted.time_scale(-1.0).time_scale(-1.0); // same values, rebuilt
        assert_eq!(sorted.event_rate(30), shuffled.event_rate(30));
    }

    #[test]
    fn zero_bin_width_is_clamped() {
        let rate = sample().event_rate(0);
        assert_eq!(rate.bin_us, 1);
        assert_eq!(rate.counts.iter().sum::<u64>(), 10);
    }

    #[test]
    fn empty_stream_has_no_bins() {
        let empty = EventStreamBuilder::new(4, 4, 0.001).build();
        let rate = empty.event_rate(100);
        assert!(rate.is_empty());
        assert_eq!(rate.len(), 0);
        assert!(rate.per_second().is_empty());
    }
}
