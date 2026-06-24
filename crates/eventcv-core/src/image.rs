use std::{error::Error, fmt};

use crate::representation::{EventFrame, EventFrameData};

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum PoolingMethod {
    #[default]
    Average,
    Sum,
}

impl EventFrame {
    /// Returns a resized frame, pooling shrinking axes and interpolating growing axes.
    pub fn resize(
        &self,
        width: usize,
        height: usize,
        pooling: PoolingMethod,
    ) -> Result<Self, ResizeError> {
        if width == 0 || height == 0 {
            return Err(ResizeError::InvalidDimensions);
        }
        width
            .checked_mul(height)
            .and_then(|plane_len| plane_len.checked_mul(self.channels))
            .ok_or(ResizeError::SizeOverflow)?;
        if width == self.width && height == self.height && pooling == PoolingMethod::Average {
            return Ok(self.clone());
        }

        let data = match (&self.data, pooling) {
            (EventFrameData::U8(data), PoolingMethod::Average) => EventFrameData::U8(
                resize_average(data, self.width, self.height, width, height, self.channels),
            ),
            (EventFrameData::U16(data), PoolingMethod::Average) => EventFrameData::U16(
                resize_average(data, self.width, self.height, width, height, self.channels),
            ),
            (EventFrameData::U64(data), PoolingMethod::Average) => EventFrameData::U64(
                resize_average(data, self.width, self.height, width, height, self.channels),
            ),
            (EventFrameData::F32(data), PoolingMethod::Average) => EventFrameData::F32(
                resize_average(data, self.width, self.height, width, height, self.channels),
            ),
            (EventFrameData::U8(data), PoolingMethod::Sum) => EventFrameData::U64(resize_sum(
                data,
                self.width,
                self.height,
                width,
                height,
                self.channels,
            )?),
            (EventFrameData::U16(data), PoolingMethod::Sum) => EventFrameData::U64(resize_sum(
                data,
                self.width,
                self.height,
                width,
                height,
                self.channels,
            )?),
            (EventFrameData::U64(data), PoolingMethod::Sum) => EventFrameData::U64(resize_sum(
                data,
                self.width,
                self.height,
                width,
                height,
                self.channels,
            )?),
            (EventFrameData::F32(data), PoolingMethod::Sum) => EventFrameData::F32(
                resize_float_sum(data, self.width, self.height, width, height, self.channels),
            ),
        };

        Ok(Self {
            data,
            channels: self.channels,
            width,
            height,
            kind: self.kind,
            channel_names: self.channel_names.clone(),
        })
    }
}

trait ResizeValue: Copy {
    fn to_f64(self) -> f64;
    fn from_f64(value: f64) -> Self;
}

trait IntegerResizeValue: ResizeValue {
    fn to_u64(self) -> u64;
}

macro_rules! impl_resize_value {
    ($type:ty) => {
        impl ResizeValue for $type {
            fn to_f64(self) -> f64 {
                self as f64
            }

            fn from_f64(value: f64) -> Self {
                value.round() as Self
            }
        }

        impl IntegerResizeValue for $type {
            fn to_u64(self) -> u64 {
                self as u64
            }
        }
    };
}

impl_resize_value!(u8);
impl_resize_value!(u16);
impl_resize_value!(u64);

impl ResizeValue for f32 {
    fn to_f64(self) -> f64 {
        f64::from(self)
    }

    fn from_f64(value: f64) -> Self {
        value as f32
    }
}

fn resize_average<T: ResizeValue>(
    data: &[T],
    source_width: usize,
    source_height: usize,
    width: usize,
    height: usize,
    channels: usize,
) -> Vec<T> {
    let x_samples = average_axis_samples(source_width, width);
    let y_samples = average_axis_samples(source_height, height);
    resize_weighted(
        data,
        source_width,
        source_height,
        width,
        height,
        channels,
        &x_samples,
        &y_samples,
    )
}

fn resize_sum<T: IntegerResizeValue>(
    data: &[T],
    source_width: usize,
    source_height: usize,
    width: usize,
    height: usize,
    channels: usize,
) -> Result<Vec<u64>, ResizeError> {
    let summed_width = width.min(source_width);
    let summed_height = height.min(source_height);
    let summed = sum_bins(
        data,
        source_width,
        source_height,
        summed_width,
        summed_height,
        channels,
    )?;

    if summed_width == width && summed_height == height {
        return Ok(summed);
    }

    Ok(resize_average(
        &summed,
        summed_width,
        summed_height,
        width,
        height,
        channels,
    ))
}

fn resize_float_sum(
    data: &[f32],
    source_width: usize,
    source_height: usize,
    width: usize,
    height: usize,
    channels: usize,
) -> Vec<f32> {
    let summed_width = width.min(source_width);
    let summed_height = height.min(source_height);
    let summed = sum_float_bins(
        data,
        source_width,
        source_height,
        summed_width,
        summed_height,
        channels,
    );

    if summed_width == width && summed_height == height {
        return summed;
    }

    resize_average(
        &summed,
        summed_width,
        summed_height,
        width,
        height,
        channels,
    )
}

fn sum_bins<T: IntegerResizeValue>(
    data: &[T],
    source_width: usize,
    source_height: usize,
    width: usize,
    height: usize,
    channels: usize,
) -> Result<Vec<u64>, ResizeError> {
    let x_ranges = axis_ranges(source_width, width);
    let y_ranges = axis_ranges(source_height, height);
    let mut resized = Vec::with_capacity(channels * width * height);
    let source_plane_len = source_width * source_height;

    for channel in 0..channels {
        let channel_offset = channel * source_plane_len;
        for &(y_start, y_end) in &y_ranges {
            for &(x_start, x_end) in &x_ranges {
                let mut sum = 0_u64;
                for source_y in y_start..y_end {
                    for source_x in x_start..x_end {
                        sum = sum
                            .checked_add(
                                data[channel_offset + source_y * source_width + source_x].to_u64(),
                            )
                            .ok_or(ResizeError::SumOverflow)?;
                    }
                }
                resized.push(sum);
            }
        }
    }

    Ok(resized)
}

fn sum_float_bins(
    data: &[f32],
    source_width: usize,
    source_height: usize,
    width: usize,
    height: usize,
    channels: usize,
) -> Vec<f32> {
    let x_ranges = axis_ranges(source_width, width);
    let y_ranges = axis_ranges(source_height, height);
    let mut resized = Vec::with_capacity(channels * width * height);
    let source_plane_len = source_width * source_height;

    for channel in 0..channels {
        let channel_offset = channel * source_plane_len;
        for &(y_start, y_end) in &y_ranges {
            for &(x_start, x_end) in &x_ranges {
                let mut sum = 0.0;
                for source_y in y_start..y_end {
                    for source_x in x_start..x_end {
                        sum += data[channel_offset + source_y * source_width + source_x];
                    }
                }
                resized.push(sum);
            }
        }
    }

    resized
}

#[allow(clippy::too_many_arguments)]
fn resize_weighted<T: ResizeValue>(
    data: &[T],
    source_width: usize,
    source_height: usize,
    width: usize,
    height: usize,
    channels: usize,
    x_samples: &[Vec<(usize, f64)>],
    y_samples: &[Vec<(usize, f64)>],
) -> Vec<T> {
    let mut resized = Vec::with_capacity(channels * width * height);
    let source_plane_len = source_width * source_height;

    for channel in 0..channels {
        let channel_offset = channel * source_plane_len;
        for y_sample in y_samples {
            for x_sample in x_samples {
                let mut value = 0.0;
                for &(source_y, y_weight) in y_sample {
                    for &(source_x, x_weight) in x_sample {
                        value += data[channel_offset + source_y * source_width + source_x].to_f64()
                            * x_weight
                            * y_weight;
                    }
                }
                resized.push(T::from_f64(value));
            }
        }
    }

    resized
}

fn average_axis_samples(source_length: usize, length: usize) -> Vec<Vec<(usize, f64)>> {
    if length < source_length {
        return axis_ranges(source_length, length)
            .into_iter()
            .map(|(start, end)| {
                let weight = 1.0 / (end - start) as f64;
                (start..end).map(|index| (index, weight)).collect()
            })
            .collect();
    }

    if length == source_length {
        return (0..length).map(|index| vec![(index, 1.0)]).collect();
    }

    (0..length)
        .map(|index| {
            let position = ((index as f64 + 0.5) * source_length as f64 / length as f64 - 0.5)
                .clamp(0.0, (source_length - 1) as f64);
            let lower = position.floor() as usize;
            let upper = (lower + 1).min(source_length - 1);
            if lower == upper {
                vec![(lower, 1.0)]
            } else {
                let upper_weight = position - lower as f64;
                vec![(lower, 1.0 - upper_weight), (upper, upper_weight)]
            }
        })
        .collect()
}

fn axis_ranges(source_length: usize, length: usize) -> Vec<(usize, usize)> {
    (0..length)
        .map(|index| {
            (
                proportional_boundary(index, source_length, length),
                proportional_boundary(index + 1, source_length, length),
            )
        })
        .collect()
}

fn proportional_boundary(index: usize, source_length: usize, length: usize) -> usize {
    let numerator = index as u128 * source_length as u128;
    numerator.div_ceil(length as u128) as usize
}

#[derive(Debug, PartialEq, Eq)]
pub enum ResizeError {
    InvalidDimensions,
    SizeOverflow,
    SumOverflow,
}

impl fmt::Display for ResizeError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidDimensions => formatter.write_str("resize dimensions must be positive"),
            Self::SizeOverflow => formatter.write_str("resize dimensions are too large"),
            Self::SumOverflow => formatter.write_str("pooled value exceeds uint64 capacity"),
        }
    }
}

impl Error for ResizeError {}

#[cfg(test)]
mod tests {
    use super::{PoolingMethod, ResizeError};
    use crate::representation::{EventFrame, EventFrameData, RepresentationKind};

    #[test]
    fn average_pools_proportional_bins_and_preserves_metadata() {
        let frame = EventFrame {
            data: EventFrameData::U8(vec![1, 2, 3, 4, 5, 5, 4, 3, 2, 1]),
            channels: 2,
            width: 5,
            height: 1,
            kind: RepresentationKind::Polarity,
            channel_names: vec!["positive".to_owned(), "negative".to_owned()],
        };

        let resized = frame.resize(2, 1, PoolingMethod::Average).unwrap();

        assert_eq!(resized.shape(), (2, 1, 2));
        assert_eq!(resized.channel_names(), ["positive", "negative"]);
        assert_eq!(resized.kind(), RepresentationKind::Polarity);
        assert_eq!(resized.data(), &EventFrameData::U8(vec![2, 5, 4, 2]));
        assert_eq!(frame.shape(), (2, 1, 5));
    }

    #[test]
    fn sum_pooling_widens_and_preserves_channel_totals() {
        let frame = EventFrame {
            data: EventFrameData::U16(vec![1, 2, 3, 4, 5, 5, 4, 3, 2, 1]),
            channels: 2,
            width: 5,
            height: 1,
            kind: RepresentationKind::Polarity,
            channel_names: vec!["positive".to_owned(), "negative".to_owned()],
        };

        let resized = frame.resize(2, 1, PoolingMethod::Sum).unwrap();

        assert_eq!(resized.data(), &EventFrameData::U64(vec![6, 9, 12, 3]));
        let EventFrameData::U64(values) = resized.data() else {
            panic!("sum pooling must return uint64 data");
        };
        assert_eq!(values.iter().sum::<u64>(), 30);
    }

    #[test]
    fn bilinear_enlargement_is_center_aligned_and_preserves_dtype() {
        let frame = EventFrame {
            data: EventFrameData::U8(vec![0, 10, 10, 0]),
            channels: 2,
            width: 2,
            height: 1,
            kind: RepresentationKind::Polarity,
            channel_names: vec!["positive".to_owned(), "negative".to_owned()],
        };

        let resized = frame.resize(4, 1, PoolingMethod::Average).unwrap();

        assert_eq!(
            resized.data(),
            &EventFrameData::U8(vec![0, 3, 8, 10, 10, 8, 3, 0])
        );
    }

    #[test]
    fn supports_mixed_pooling_and_interpolation() {
        let frame = EventFrame {
            data: EventFrameData::U16(
                [vec![7_u16; 8], vec![9_u16; 8]]
                    .into_iter()
                    .flatten()
                    .collect(),
            ),
            channels: 2,
            width: 2,
            height: 4,
            kind: RepresentationKind::Polarity,
            channel_names: vec!["positive".to_owned(), "negative".to_owned()],
        };

        let resized = frame.resize(4, 2, PoolingMethod::Average).unwrap();
        let summed = frame.resize(4, 2, PoolingMethod::Sum).unwrap();

        assert_eq!(resized.shape(), (2, 2, 4));
        assert_eq!(
            resized.data(),
            &EventFrameData::U16(
                [vec![7_u16; 8], vec![9_u16; 8]]
                    .into_iter()
                    .flatten()
                    .collect()
            )
        );
        assert_eq!(
            summed.data(),
            &EventFrameData::U64(
                [vec![14_u64; 8], vec![18_u64; 8]]
                    .into_iter()
                    .flatten()
                    .collect()
            )
        );
    }

    #[test]
    fn identity_resize_preserves_values() {
        let frame = EventFrame {
            data: EventFrameData::U16(vec![1, 2, 3, 4]),
            channels: 2,
            width: 2,
            height: 1,
            kind: RepresentationKind::Polarity,
            channel_names: vec!["positive".to_owned(), "negative".to_owned()],
        };

        let resized = frame.resize(2, 1, PoolingMethod::Average).unwrap();

        assert_eq!(resized.data(), frame.data());
    }

    #[test]
    fn resizes_float_frames_without_integer_conversion() {
        let frame = EventFrame {
            data: EventFrameData::F32(vec![0.25, 0.75, -1.0, 2.0]),
            channels: 1,
            width: 2,
            height: 2,
            kind: RepresentationKind::Voxel,
            channel_names: vec!["bin_0".to_owned()],
        };

        let average = frame.resize(1, 1, PoolingMethod::Average).unwrap();
        let sum = frame.resize(1, 1, PoolingMethod::Sum).unwrap();

        assert_eq!(average.data(), &EventFrameData::F32(vec![0.5]));
        assert_eq!(sum.data(), &EventFrameData::F32(vec![2.0]));
    }

    #[test]
    fn rejects_invalid_resize_dimensions_and_sum_overflow() {
        let frame = EventFrame {
            data: EventFrameData::U64(vec![u64::MAX, 1, 0, 0]),
            channels: 2,
            width: 2,
            height: 1,
            kind: RepresentationKind::Polarity,
            channel_names: vec!["positive".to_owned(), "negative".to_owned()],
        };

        assert_eq!(
            frame.resize(0, 1, PoolingMethod::Average).unwrap_err(),
            ResizeError::InvalidDimensions
        );
        assert_eq!(
            frame
                .resize(usize::MAX, 2, PoolingMethod::Average)
                .unwrap_err(),
            ResizeError::SizeOverflow
        );
        assert_eq!(
            frame.resize(1, 1, PoolingMethod::Sum).unwrap_err(),
            ResizeError::SumOverflow
        );
    }
}
