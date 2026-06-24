use std::ops::Range;
use std::path::Path;

use hdf5::types::{IntSize, TypeDescriptor};
use hdf5::{Dataset, File, H5Type};

use super::{IoError, LoadOptions};
use crate::{EventStream, EventStreamBuilder};

/// Events are read in blocks of this many so capping with `max_events` and reading
/// huge files both stay bounded in memory.
const BLOCK: usize = 1_000_000;

/// Reads event datasets `x`, `y`, `t`, `p` from an HDF5 file (looked up under an
/// `events/` group, then at the root). Timestamps are converted to microseconds via
/// `options.time_unit` — Brisbane-Event-VPR stores nanoseconds, so pass `time_unit="ns"`.
pub fn read_hdf5(path: impl AsRef<Path>, options: &LoadOptions) -> Result<EventStream, IoError> {
    let path = path.as_ref();
    let (width, height) = options.sensor_size.ok_or_else(|| {
        IoError::Format("HDF5 files require sensor_size = (width, height)".to_owned())
    })?;
    if !path.exists() {
        return Err(IoError::Io(std::io::Error::new(
            std::io::ErrorKind::NotFound,
            path.display().to_string(),
        )));
    }

    let file = File::open(path).map_err(map_hdf5_error)?;
    let [x, y, t, p] = open_event_datasets(&file)?;

    let total = x.shape().first().copied().unwrap_or(0);
    for (axis, dataset) in [("y", &y), ("t", &t), ("p", &p)] {
        if dataset.shape().first().copied().unwrap_or(0) != total {
            return Err(IoError::Format(format!(
                "event column '{axis}' length does not match 'x'"
            )));
        }
    }
    let target = options.max_events.map_or(total, |max| max.min(total));

    let mut builder = EventStreamBuilder::with_capacity(width, height, 0.001, target);
    let mut start = 0;
    while start < target {
        let end = (start + BLOCK).min(target);
        let xs = read_integers(&x, start..end)?;
        let ys = read_integers(&y, start..end)?;
        let ts = read_integers(&t, start..end)?;
        let ps = read_polarities(&p, start..end)?;
        for index in 0..(end - start) {
            builder.push(
                xs[index] as u16,
                ys[index] as u16,
                options.time_unit.microseconds_from_int(ts[index]),
                ps[index],
            );
        }
        start = end;
    }
    Ok(builder.build())
}

fn open_event_datasets(file: &File) -> Result<[Dataset; 4], IoError> {
    for prefix in ["events/", ""] {
        let names = ["x", "y", "t", "p"].map(|axis| format!("{prefix}{axis}"));
        if let (Ok(x), Ok(y), Ok(t), Ok(p)) = (
            file.dataset(&names[0]),
            file.dataset(&names[1]),
            file.dataset(&names[2]),
            file.dataset(&names[3]),
        ) {
            return Ok([x, y, t, p]);
        }
    }
    Err(IoError::Format(
        "could not find event datasets x/y/t/p (looked under 'events/' and the root)".to_owned(),
    ))
}

/// Reads an integer (or unsigned) column as `i64`, dispatching on its on-disk width.
fn read_integers(dataset: &Dataset, range: Range<usize>) -> Result<Vec<i64>, IoError> {
    let descriptor = dataset
        .dtype()
        .and_then(|dtype| dtype.to_descriptor())
        .map_err(map_hdf5_error)?;
    match descriptor {
        TypeDescriptor::Unsigned(IntSize::U1) => widen::<u8>(dataset, range),
        TypeDescriptor::Unsigned(IntSize::U2) => widen::<u16>(dataset, range),
        TypeDescriptor::Unsigned(IntSize::U4) => widen::<u32>(dataset, range),
        TypeDescriptor::Unsigned(IntSize::U8) => widen::<u64>(dataset, range),
        TypeDescriptor::Integer(IntSize::U1) => widen::<i8>(dataset, range),
        TypeDescriptor::Integer(IntSize::U2) => widen::<i16>(dataset, range),
        TypeDescriptor::Integer(IntSize::U4) => widen::<i32>(dataset, range),
        TypeDescriptor::Integer(IntSize::U8) => widen::<i64>(dataset, range),
        other => Err(IoError::Format(format!(
            "unsupported HDF5 integer column type: {other:?}"
        ))),
    }
}

/// Reads a polarity column as `bool`, accepting HDF5 boolean, enum, or 1-byte int.
fn read_polarities(dataset: &Dataset, range: Range<usize>) -> Result<Vec<bool>, IoError> {
    let descriptor = dataset
        .dtype()
        .and_then(|dtype| dtype.to_descriptor())
        .map_err(map_hdf5_error)?;
    match descriptor {
        TypeDescriptor::Boolean => read_block::<bool>(dataset, range),
        TypeDescriptor::Enum(_) | TypeDescriptor::Integer(IntSize::U1) => {
            Ok(read_block::<i8>(dataset, range)?
                .into_iter()
                .map(|value| value != 0)
                .collect())
        }
        TypeDescriptor::Unsigned(IntSize::U1) => Ok(read_block::<u8>(dataset, range)?
            .into_iter()
            .map(|value| value != 0)
            .collect()),
        other => Err(IoError::Format(format!(
            "unsupported HDF5 polarity column type: {other:?}"
        ))),
    }
}

fn widen<T: H5Type + Clone + IntoI64>(
    dataset: &Dataset,
    range: Range<usize>,
) -> Result<Vec<i64>, IoError> {
    Ok(read_block::<T>(dataset, range)?
        .into_iter()
        .map(IntoI64::into_i64)
        .collect())
}

fn read_block<T: H5Type + Clone>(
    dataset: &Dataset,
    range: Range<usize>,
) -> Result<Vec<T>, IoError> {
    dataset
        .read_slice_1d::<T, _>(range)
        .map(|array| array.to_vec())
        .map_err(map_hdf5_error)
}

fn map_hdf5_error(error: hdf5::Error) -> IoError {
    IoError::Format(format!("hdf5: {error}"))
}

trait IntoI64 {
    fn into_i64(self) -> i64;
}

macro_rules! impl_into_i64 {
    ($($type:ty),*) => {
        $(impl IntoI64 for $type {
            fn into_i64(self) -> i64 {
                self as i64
            }
        })*
    };
}

impl_into_i64!(u8, u16, u32, u64, i8, i16, i32, i64);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::io::TimeUnit;

    fn options(width: usize, height: usize, time_unit: TimeUnit) -> LoadOptions {
        LoadOptions {
            sensor_size: Some((width, height)),
            time_unit,
            ..LoadOptions::default()
        }
    }

    #[test]
    fn reads_grouped_event_datasets_and_drops_out_of_bounds() {
        let dir = std::env::temp_dir().join(format!("eventcv-h5-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("events.h5");
        {
            let file = File::create(&path).unwrap();
            let group = file.create_group("events").unwrap();
            group
                .new_dataset_builder()
                .with_data(&[1u16, 3, 0, 4][..])
                .create("x")
                .unwrap();
            group
                .new_dataset_builder()
                .with_data(&[2u16, 0, 1, 0][..])
                .create("y")
                .unwrap();
            group
                .new_dataset_builder()
                .with_data(&[1000u64, 2000, 3000, 4000][..])
                .create("t")
                .unwrap();
            group
                .new_dataset_builder()
                .with_data(&[true, false, true, false][..])
                .create("p")
                .unwrap();
        }

        let stream = read_hdf5(&path, &options(4, 4, TimeUnit::Microseconds)).unwrap();

        assert_eq!(stream.len(), 3); // (4, 0) dropped: x == width
        assert_eq!(stream.xs(), &[1, 3, 0]);
        assert_eq!(stream.ys(), &[2, 0, 1]);
        assert_eq!(stream.ts(), &[1000, 2000, 3000]);
        assert_eq!(stream.ps(), &[true, false, true]);
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn nanosecond_timestamps_convert_to_microseconds() {
        let dir = std::env::temp_dir().join(format!("eventcv-h5ns-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("ns.h5");
        {
            let file = File::create(&path).unwrap();
            file.new_dataset_builder()
                .with_data(&[0u16, 1][..])
                .create("x")
                .unwrap();
            file.new_dataset_builder()
                .with_data(&[0u16, 1][..])
                .create("y")
                .unwrap();
            file.new_dataset_builder()
                .with_data(&[1_000_000u64, 2_500_000][..])
                .create("t")
                .unwrap();
            file.new_dataset_builder()
                .with_data(&[true, false][..])
                .create("p")
                .unwrap();
        }

        let stream = read_hdf5(&path, &options(8, 8, TimeUnit::Nanoseconds)).unwrap();

        assert_eq!(stream.ts(), &[1000, 2500]); // ns -> us
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn requires_sensor_size() {
        let error = read_hdf5("missing.h5", &LoadOptions::default()).unwrap_err();
        assert!(matches!(error, IoError::Format(_)));
    }
}
