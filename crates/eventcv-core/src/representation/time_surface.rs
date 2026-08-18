use super::{
    age_ms, event_index, frame_len, reference_time, validate_positive, EventFrame, EventFrameData,
    Representation, RepresentationError, RepresentationKind,
};
use crate::EventStream;

#[derive(Clone, Copy, Debug)]
pub struct TimeSurface {
    tau_ms: f64,
}

impl TimeSurface {
    pub fn new(tau_ms: f64) -> Self {
        Self { tau_ms }
    }
}

impl Default for TimeSurface {
    fn default() -> Self {
        Self::new(30.0)
    }
}

impl Representation for TimeSurface {
    type Output = EventFrame;

    fn generate(&self, stream: &EventStream) -> Result<EventFrame, RepresentationError> {
        validate_positive(self.tau_ms, "tau_ms")?;
        let (width, height, length) = frame_len(stream, 2)?;
        let plane_len = width * height;
        // Two zeroed planes rather than a `Vec<Option<u64>>`: `vec![0; n]` is one `alloc_zeroed`,
        // so a cell no event touches is never written or even faulted in, where the `Option` was
        // 16 bytes a cell and had to be filled end to end before the first event was read — about
        // a millisecond at 640x480, however few events the window held. `seen` rather than a
        // sentinel timestamp because 0 is a timestamp a stream can hold.
        //
        // `touched` then drives the exponential over just those cells. A window cannot touch more
        // cells than it has events, so `stream.len()` says before the loop whether that is worth
        // doing: a dense window would reach most cells anyway, and scattering through a
        // sensor-sized list costs more than one sequential pass. See `AveragedTimeSurface` for the
        // measurement behind the threshold.
        let track_touched = stream.len().saturating_mul(4) <= length;
        let mut latest = vec![0_u64; length];
        let mut seen = vec![false; length];
        let mut touched = if track_touched {
            Vec::with_capacity(stream.len())
        } else {
            Vec::new()
        };

        for event in stream.iter() {
            let index =
                event_index(event, width, height)? + if event.polarity { 0 } else { plane_len };
            if !seen[index] {
                seen[index] = true;
                if track_touched {
                    touched.push(index);
                }
                latest[index] = event.timestamp;
            } else if event.timestamp > latest[index] {
                latest[index] = event.timestamp;
            }
        }

        let mut values = vec![0_f32; length];
        if let Some(reference) = reference_time(stream) {
            if track_touched {
                for index in touched {
                    values[index] =
                        (-age_ms(stream, reference, latest[index]) / self.tau_ms).exp() as f32;
                }
            } else {
                for (index, value) in values.iter_mut().enumerate() {
                    if seen[index] {
                        *value =
                            (-age_ms(stream, reference, latest[index]) / self.tau_ms).exp() as f32;
                    }
                }
            }
        }

        Ok(EventFrame {
            data: EventFrameData::F32(values),
            channels: 2,
            width,
            height,
            kind: RepresentationKind::TimeSurface,
            channel_names: vec!["positive".to_owned(), "negative".to_owned()],
        })
    }

    /// The kernel keeps the *smallest age* per pixel and polarity with an integer `atomicMin`,
    /// which is the same quantity the CPU's "latest timestamp" is, and maps it through the same
    /// `exp` on readback. Order cannot matter to a minimum, so this is exact up to the one `exp`.
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
        let ages = super::on_gpu(
            stream,
            &crate::accel::GpuDispatch {
                entry: "time_surface",
                cells: length,
                initial: i32::MAX,
                bins: 2,
                span_ms: self.tau_ms as f32,
                fixed_one: 1.0,
                window_ms: None,
                needs_ages: true,
            },
        )?;
        let scale = stream.timestamp_scale_ms();
        let values = ages
            .iter()
            .map(|age| match *age {
                i32::MAX => 0.0, // no event ever landed on this pixel
                age => (-(f64::from(age) * scale) / self.tau_ms).exp() as f32,
            })
            .collect();
        Ok(EventFrame {
            data: EventFrameData::F32(values),
            channels: 2,
            width,
            height,
            kind: RepresentationKind::TimeSurface,
            channel_names: vec!["positive".to_owned(), "negative".to_owned()],
        })
    }
}

#[cfg(test)]
mod tests {
    use ndarray::array;

    use super::{Representation, TimeSurface};
    use crate::{representation::EventFrameData, EventStream};

    #[test]
    fn uses_latest_event_per_pixel_and_polarity() {
        let stream = EventStream::from_array2(
            array![[0, 0, 30_000, 1], [1, 0, 20_000, 0], [0, 0, 10_000, 1]],
            2,
            1,
            0.001,
        );

        let frame = TimeSurface::new(10.0).generate(&stream).unwrap();
        let EventFrameData::F32(values) = frame.data() else {
            panic!("time surfaces must use float32 data");
        };

        assert_eq!(values[0], 1.0);
        assert_eq!(values[1], 0.0);
        assert!((values[3] - (-1.0_f32).exp()).abs() < 1e-6);
    }

    /// `0` is a timestamp a stream can hold, and it is also what the zeroed scratch plane holds
    /// for a cell no event reached. The two must not be confused.
    #[test]
    fn a_timestamp_of_zero_is_an_event_not_an_empty_pixel() {
        let stream = EventStream::from_array2(array![[0, 0, 0, 1]], 2, 1, 0.001);

        let frame = TimeSurface::new(10.0).generate(&stream).unwrap();
        let EventFrameData::F32(values) = frame.data() else {
            panic!("time surfaces must use float32 data");
        };

        assert_eq!(values[0], 1.0); // the event, at age 0
        assert_eq!(values[1], 0.0); // no event ever landed here
        assert!(values[2..].iter().all(|&value| value == 0.0)); // nor on the negative plane
    }
}
