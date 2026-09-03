use super::{
    age_ms, event_index, frame_len, reference_time, validate_positive, EventFrame, EventFrameData,
    Representation, RepresentationError, RepresentationKind,
};
use crate::EventStream;

/// Averaged time surface: where [`super::TimeSurface`] keeps only each pixel's *latest*
/// exponential response, this averages `exp(-age / τ)` over **all** events that hit the pixel, so
/// recurring activity reads brighter than a single stale hit. Two channels (positive / negative),
/// `f32`, each pixel the mean response of its events (0 where none).
///
/// Related in spirit to HATS (Sironi et al., CVPR 2018) but deliberately not named after it: HATS
/// partitions the sensor into cells, keeps a local memory per cell and polarity, and builds a
/// local neighbourhood surface for every event before averaging into per-cell histograms. This is
/// a per-pixel mean in a single pass, with a different output shape.
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
        // The accumulators stay `f64`/`u64` — the output is bit-identical either way. What changes
        // is how the means are written out at the end.
        //
        // A short window touches a handful of cells out of hundreds of thousands, and the old
        // `sums.zip(counts).map().collect()` walked every one of them and allocated a third
        // sensor-sized buffer to hold the result — about 0.7 ms at 640x480 before a single event
        // was read. Listing the cells an event reached and dividing only those removes it.
        //
        // A *dense* window is the opposite case: it reaches most cells, the list becomes a
        // sensor-sized allocation of its own, and scattering through it at the end is a random
        // walk over the plane that costs more than the single sequential pass it replaced. A
        // window cannot touch more cells than it has events, so `stream.len()` decides which case
        // this is before the loop starts — the dense path then never builds a list at all.
        // Mispredicting can only cost the optimisation, never correctness: a stream whose events
        // pile onto a few pixels is called dense and simply walks the plane, as it always did.
        let track_touched = stream.len().saturating_mul(4) <= length;
        let mut sums = vec![0.0_f64; length];
        let mut counts = vec![0_u64; length];
        let mut touched = if track_touched {
            Vec::with_capacity(stream.len())
        } else {
            Vec::new()
        };

        if let Some(reference) = reference_time(stream) {
            // Split rather than testing `track_touched` per event: the dense loop is then exactly
            // the accumulation this has always done, with nothing added to a path that runs once
            // per event over a whole recording.
            // The response depends on the timestamp alone, and a window holds far fewer
            // distinct timestamps than events: a sensor emits many events at one tick --- an
            // EVT3 vector word alone carries up to twelve --- so on a real recording this is
            // tens of events per timestamp. Remembering the last one turns `exp`, the most
            // expensive thing in the loop, from once per event into once per tick. Exact by
            // construction: the same input returns the same value, so nothing is approximated.
            let mut last: Option<(u64, f64)> = None;
            let mut cell = |event: crate::Event| -> Result<(usize, f64), RepresentationError> {
                let index =
                    event_index(event, width, height)? + if event.polarity { 0 } else { plane_len };
                let response = match last {
                    Some((timestamp, response)) if timestamp == event.timestamp => response,
                    _ => {
                        let response =
                            (-age_ms(stream, reference, event.timestamp) / self.tau_ms).exp();
                        last = Some((event.timestamp, response));
                        response
                    }
                };
                Ok((index, response))
            };
            if track_touched {
                for event in stream.iter() {
                    let (index, response) = cell(event)?;
                    if counts[index] == 0 {
                        touched.push(index);
                    }
                    sums[index] += response;
                    counts[index] += 1;
                }
            } else {
                for event in stream.iter() {
                    let (index, response) = cell(event)?;
                    sums[index] += response;
                    counts[index] += 1;
                }
            }
        }

        let values = if track_touched {
            let mut values = vec![0.0_f32; length];
            for index in touched {
                values[index] = (sums[index] / counts[index] as f64) as f32;
            }
            values
        } else {
            // Writing every cell anyway, so build the buffer straight from the accumulators
            // rather than zeroing one first and then testing each cell.
            sums.into_iter()
                .zip(counts)
                .map(|(sum, count)| {
                    if count == 0 {
                        0.0
                    } else {
                        (sum / count as f64) as f32
                    }
                })
                .collect()
        };

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
