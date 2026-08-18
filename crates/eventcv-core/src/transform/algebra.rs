//! Polarity and stream-algebra transforms — filtering, reordering and combining streams.

use crate::{EventStream, EventStreamBuilder};

impl EventStream {
    /// Keeps only events of the given polarity.
    pub fn filter_polarity(&self, polarity: bool) -> EventStream {
        let (width, height) = self.sensor_size();
        self.remap(width, height, |x, y, t, p| {
            (p == polarity).then_some((x, y, t, p))
        })
    }

    /// Flips every event's polarity. Sensor and timestamps unchanged.
    pub fn invert_polarity(&self) -> EventStream {
        let (width, height) = self.sensor_size();
        self.map_columns(width, height, |out| {
            for p in &mut out.ps {
                *p = !*p;
            }
        })
    }

    /// Returns a copy reordered by ascending timestamp (stable for equal timestamps).
    pub fn sort_by_time(&self) -> EventStream {
        let (width, height) = self.sensor_size();
        let ts = self.ts();
        let mut order: Vec<usize> = (0..self.len()).collect();
        order.sort_by_key(|&index| ts[index]); // stable
        let (xs, ys, ps) = (self.xs(), self.ys(), self.ps());
        let mut builder =
            EventStreamBuilder::with_capacity(width, height, self.timestamp_scale_ms(), self.len());
        for index in order {
            builder.push(xs[index], ys[index], ts[index], ps[index]);
        }
        builder.build()
    }

    /// Concatenates several streams into one (in argument order, not time-sorted). The sensor
    /// size is the element-wise maximum of the inputs; the timestamp scale comes from `self`.
    pub fn concat(&self, others: &[&EventStream]) -> EventStream {
        let (mut width, mut height) = self.sensor_size();
        for other in others {
            let (w, h) = other.sensor_size();
            width = width.max(w);
            height = height.max(h);
        }
        let total = self.len() + others.iter().map(|s| s.len()).sum::<usize>();
        let mut builder =
            EventStreamBuilder::with_capacity(width, height, self.timestamp_scale_ms(), total);
        for stream in std::iter::once(self).chain(others.iter().copied()) {
            builder.extend_from_stream(stream);
        }
        builder.build()
    }
}

#[cfg(test)]
mod tests {
    use crate::{EventStream, EventStreamBuilder};

    fn sample() -> EventStream {
        let mut builder = EventStreamBuilder::new(4, 3, 0.001);
        builder.push(0, 0, 30, true);
        builder.push(1, 1, 10, false);
        builder.push(2, 2, 20, true);
        builder.build()
    }

    #[test]
    fn filter_polarity_splits_and_conserves_count() {
        let s = sample();
        let on = s.filter_polarity(true);
        let off = s.filter_polarity(false);
        assert_eq!(on.len() + off.len(), s.len());
        assert!(on.ps().iter().all(|&p| p));
        assert!(off.ps().iter().all(|&p| !p));
    }

    #[test]
    fn invert_polarity_is_its_own_inverse() {
        let s = sample();
        assert_eq!(s.invert_polarity().invert_polarity().ps(), s.ps());
        assert_eq!(s.invert_polarity().ps(), &[false, true, false]);
    }

    #[test]
    fn sort_by_time_orders_ascending() {
        let sorted = sample().sort_by_time();
        assert_eq!(sorted.ts(), &[10, 20, 30]);
        assert_eq!(sorted.xs(), &[1, 2, 0]); // rows follow their timestamps
    }

    #[test]
    fn concat_appends_and_takes_max_sensor() {
        let a = sample();
        let mut b_builder = EventStreamBuilder::new(8, 6, 0.001);
        b_builder.push(7, 5, 50, true);
        let b = b_builder.build();

        let combined = a.concat(&[&b]);
        assert_eq!(combined.len(), 4);
        assert_eq!(combined.sensor_size(), (8, 6)); // element-wise max
        assert_eq!(combined.ts(), &[30, 10, 20, 50]); // argument order, not sorted
    }

    #[test]
    fn algebra_ops_handle_empty() {
        let empty = EventStreamBuilder::new(4, 3, 0.001).build();
        assert!(empty.filter_polarity(true).is_empty());
        assert!(empty.sort_by_time().is_empty());
        assert_eq!(empty.concat(&[&sample()]).len(), 3);
    }
}
