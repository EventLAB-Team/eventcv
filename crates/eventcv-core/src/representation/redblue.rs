use super::{
    countmask::percentile_linear, frame_len, polarity::polarity_counts_on, EventFrame,
    EventFrameData, Representation, RepresentationError, RepresentationKind,
};
use crate::{accel::Device, EventStream};

/// White-background RGB counts with independent polarity scales and positive ties.
#[derive(Clone, Copy, Debug)]
pub struct RedBlue {
    pct: f64,
}

impl RedBlue {
    pub fn new(pct: f64) -> Self {
        Self { pct }
    }
}

impl Default for RedBlue {
    fn default() -> Self {
        Self::new(99.0)
    }
}

// Repeated float32 additions of 1 stop changing at 2^24, as in the NumPy reference.
fn reference_count(count: u64) -> f32 {
    count.min(1 << 24) as f32
}

fn normalize(counts: &[u64], pct: f64) -> Vec<f32> {
    let mut nonzero: Vec<f64> = counts
        .iter()
        .filter(|&&count| count > 0)
        .map(|&count| f64::from(reference_count(count)))
        .collect();
    let threshold = if nonzero.is_empty() {
        1.0
    } else {
        percentile_linear(&mut nonzero, pct) as f32
    };
    counts
        .iter()
        .map(|&count| reference_count(count).min(threshold) / threshold)
        .collect()
}

impl Representation for RedBlue {
    type Output = EventFrame;

    fn generate(&self, stream: &EventStream) -> Result<EventFrame, RepresentationError> {
        self.generate_on(stream, Device::Cpu)
    }

    fn generate_on(
        &self,
        stream: &EventStream,
        device: Device,
    ) -> Result<EventFrame, RepresentationError> {
        if !self.pct.is_finite() || !(0.0..=100.0).contains(&self.pct) {
            return Err(RepresentationError::InvalidParameter("pct"));
        }
        let (width, height, length) = frame_len(stream, 3)?;
        let plane = width * height;
        let (_, _, counts) = polarity_counts_on(stream, device)?;
        let pos = normalize(&counts[..plane], self.pct);
        let neg = normalize(&counts[plane..], self.pct);
        let mut data = vec![255; length];
        for i in 0..plane {
            let positive = pos[i] >= neg[i];
            let intensity = if positive { pos[i] } else { neg[i] };
            let faded = ((1.0 - intensity).clamp(0.0, 1.0) * 255.0) as u8;
            data[plane + i] = faded;
            data[if positive { 2 * plane + i } else { i }] = faded;
        }
        Ok(EventFrame {
            data: EventFrameData::U8(data),
            channels: 3,
            width,
            height,
            kind: RepresentationKind::RedBlue,
            channel_names: vec!["red".into(), "green".into(), "blue".into()],
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::viz::{render_frame, Colormap};
    use ndarray::{array, Array2};

    #[test]
    fn preserves_rgb_when_rendered() {
        let stream = EventStream::from_array2(array![[0, 0, 0, 1], [1, 0, 0, 0]], 3, 1, 0.001);
        let frame = RedBlue::default().generate(&stream).unwrap();
        for normalize in [false, true] {
            assert_eq!(
                render_frame(&frame, Colormap::Turbo, normalize).pixels,
                vec![255, 0, 0, 0, 0, 255, 255, 255, 255]
            );
        }
        assert_eq!(RepresentationKind::from_tag("redblue"), Some(frame.kind()));
        assert_eq!(frame.kind().as_str(), "redblue");
    }

    #[test]
    fn matches_float32_count_saturation() {
        assert_eq!(reference_count((1 << 24) + 1), 16_777_216.0);
        assert_eq!(reference_count(u64::MAX), 16_777_216.0);
        assert_eq!(normalize(&[1 << 24, u64::MAX], 99.0), vec![1.0, 1.0]);
    }

    #[test]
    fn rejects_invalid_coordinates_and_dimensions() {
        let outside = EventStream::from_array2(array![[2, 0, 0, 1]], 2, 1, 0.001);
        assert!(matches!(
            RedBlue::default().generate(&outside),
            Err(RepresentationError::EventOutOfBounds { .. })
        ));
        let oversized = EventStream::from_array2(Array2::zeros((0, 4)), usize::MAX, 2, 0.001);
        assert_eq!(
            RedBlue::default().generate(&oversized).unwrap_err(),
            RepresentationError::SizeOverflow
        );
    }
}
