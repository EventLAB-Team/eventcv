use super::{
    age_ms, event_index, frame_len, reference_time, validate_positive, EventFrame, EventFrameData,
    Representation, RepresentationError, RepresentationKind,
};
use crate::EventStream;

/// Averaged time surface (HATS-style): where [`super::TimeSurface`] keeps only each pixel's
/// *latest* exponential response, this averages `exp(-age / τ)` over **all** events that hit
/// the pixel, so recurring activity reads brighter than a single stale hit. Two channels
/// (positive / negative), `f32`, each pixel the mean response of its events (0 where none).
#[derive(Clone, Copy, Debug)]
pub struct AveragedTimeSurface {
    tau_ms: f64,
}

impl AveragedTimeSurface {
    pub fn new(tau_ms: f64) -> Self {
        Self { tau_ms }
    }
}

impl Default for AveragedTimeSurface {
    fn default() -> Self {
        Self::new(30.0)
    }
}

impl Representation for AveragedTimeSurface {
    type Output = EventFrame;

    fn generate(&self, stream: &EventStream) -> Result<EventFrame, RepresentationError> {
        validate_positive(self.tau_ms, "tau_ms")?;
        let (width, height, length) = frame_len(stream, 2)?;
        let plane_len = width * height;
        let mut sums = vec![0.0_f64; length];
        let mut counts = vec![0_u64; length];

        if let Some(reference) = reference_time(stream) {
            for event in stream.iter() {
                let index =
                    event_index(event, width, height)? + if event.polarity { 0 } else { plane_len };
                let response = (-age_ms(stream, reference, event.timestamp) / self.tau_ms).exp();
                sums[index] += response;
                counts[index] += 1;
            }
        }

        let values = sums
            .into_iter()
            .zip(counts)
            .map(|(sum, count)| {
                if count == 0 {
                    0.0
                } else {
                    (sum / count as f64) as f32
                }
            })
            .collect();

        Ok(frame(values, width, height))
    }

    /// The kernel accumulates the same `exp(-age/tau)` responses in Q16.16 and counts them
    /// alongside, then divides here. Fixed point again, for the same reason as the voxel grid: the
    /// mean of a pixel's responses should not depend on which invocation got there first.
    fn generate_on(
        &self,
        stream: &EventStream,
        device: crate::accel::Device,
    ) -> Result<EventFrame, RepresentationError> {
        if device == crate::accel::Device::Cpu {
            return self.generate(stream);
        }
        validate_positive(self.tau_ms, "tau_ms")?;
        let (width, height, length) = frame_len(stream, 2)?;
        // Two planes of sums followed by two of counts — one dispatch instead of two passes over
        // the events.
        let cells = super::on_gpu(
            stream,
            &crate::accel::GpuDispatch {
                entry: "averaged_time_surface",
                cells: length * 2,
                initial: 0,
                bins: 2,
                span_ms: self.tau_ms as f32,
                fixed_one: crate::accel::FIXED_ONE,
                window_ms: None,
                needs_ages: true,
            },
        )?;
        let (sums, counts) = cells.split_at(length);
        let values = sums
            .iter()
            .zip(counts)
            .map(|(sum, count)| match count {
                0 => 0.0,
                count => *sum as f32 / crate::accel::FIXED_ONE / *count as f32,
            })
            .collect();
        Ok(frame(values, width, height))
    }
}

fn frame(values: Vec<f32>, width: usize, height: usize) -> EventFrame {
    EventFrame {
        data: EventFrameData::F32(values),
        channels: 2,
        width,
        height,
        kind: RepresentationKind::AveragedTimeSurface,
        channel_names: vec!["positive".to_owned(), "negative".to_owned()],
    }
}

#[cfg(test)]
mod tests {
    use ndarray::array;

    use super::{AveragedTimeSurface, Representation};
    use crate::{representation::EventFrameData, EventStream};

    #[test]
    fn averages_all_events_at_a_pixel() {
        // Two positive events at (0,0): the newest (t=30_000) gives 1.0, the older
        // (t=20_000, age 10 ms, τ=10 ms) gives e^-1; the pixel is their mean.
        let stream = EventStream::from_array2(
            array![[0, 0, 30_000, 1], [0, 0, 20_000, 1], [1, 0, 10_000, 0]],
            2,
            1,
            0.001,
        );

        let frame = AveragedTimeSurface::new(10.0).generate(&stream).unwrap();
        let EventFrameData::F32(values) = frame.data() else {
            panic!("averaged time surfaces must use float32 data");
        };

        let expected_mean = (1.0 + (-1.0_f32).exp()) / 2.0;
        assert!((values[0] - expected_mean).abs() < 1e-6);
        assert_eq!(values[1], 0.0); // no positive events at (1,0)
                                    // negative channel: the single event at (1,0) has age 20 ms → e^-2
        assert!((values[3] - (-2.0_f32).exp()).abs() < 1e-6);
    }

    #[test]
    fn rejects_invalid_tau() {
        let stream = EventStream::from_array2(array![[0, 0, 10, 1]], 1, 1, 0.001);

        let error = AveragedTimeSurface::new(0.0).generate(&stream).unwrap_err();

        assert_eq!(error.to_string(), "tau_ms must be finite and positive");
    }
}
