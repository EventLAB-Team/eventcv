use std::path::Path;

use npyz::npz::NpzArchive;

use super::IoError;
use crate::{EventStream, EventStreamBuilder};

const EVENT_DATA_KEY: &str = "event_data";
const N_IMAGENET_WIDTH: usize = 640;
const N_IMAGENET_HEIGHT: usize = 480;

#[derive(npyz::Deserialize)]
struct NImageNetEvent {
    x: u16,
    y: u16,
    t: u16,
    p: bool,
}

/// Reads an N-ImageNet `.npz` archive (an `event_data` structured array). `sensor`
/// overrides the default 640x480 grid; out-of-bounds coordinates are an error.
pub fn read_npz(
    path: impl AsRef<Path>,
    sensor: Option<(usize, usize)>,
) -> Result<EventStream, IoError> {
    let (width, height) = sensor.unwrap_or((N_IMAGENET_WIDTH, N_IMAGENET_HEIGHT));
    let mut archive = NpzArchive::open(path).map_err(|error| {
        if error.kind() == std::io::ErrorKind::InvalidData {
            IoError::Format(error.to_string())
        } else {
            IoError::Io(error)
        }
    })?;
    let npy = archive
        .by_name(EVENT_DATA_KEY)
        .map_err(|error| IoError::Format(error.to_string()))?
        .ok_or_else(|| IoError::Format("missing event_data array".to_owned()))?;

    let [event_count] = npy.shape() else {
        return Err(IoError::Format(
            "event_data must be one-dimensional".to_owned(),
        ));
    };
    let event_count = usize::try_from(*event_count)
        .map_err(|_| IoError::Format("event_data is too large".to_owned()))?;
    let records = npy
        .data::<NImageNetEvent>()
        .map_err(|error| IoError::Format(error.to_string()))?;
    let mut builder = EventStreamBuilder::with_capacity(width, height, 0.001, event_count);

    for record in records {
        let record = record.map_err(|error| IoError::Format(error.to_string()))?;
        if !builder.push(record.x, record.y, i64::from(record.t), record.p) {
            return Err(IoError::Format(format!(
                "event coordinate ({}, {}) exceeds sensor size {}x{}",
                record.x, record.y, width, height
            )));
        }
    }

    Ok(builder.build())
}
