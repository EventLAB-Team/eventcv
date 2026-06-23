use std::{error::Error, fmt, io, path::Path};

use ndarray::{Array2, ArrayView2};
use npyz::npz::NpzArchive;

pub mod image;
pub mod representation;

const COLUMN_COUNT: usize = 4;
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

#[derive(Debug)]
pub enum LoadError {
    Io(io::Error),
    InvalidFormat(String),
}

impl fmt::Display for LoadError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io(error) => error.fmt(formatter),
            Self::InvalidFormat(message) => formatter.write_str(message),
        }
    }
}

impl Error for LoadError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Io(error) => Some(error),
            Self::InvalidFormat(_) => None,
        }
    }
}

#[derive(Debug)]
pub struct EventStream {
    events: Array2<u64>,
    width: usize,
    height: usize,
    timestamp_scale_ms: f64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Event {
    pub x: usize,
    pub y: usize,
    pub timestamp: u64,
    pub polarity: bool,
}

impl EventStream {
    pub fn len(&self) -> usize {
        self.events.nrows()
    }

    pub fn is_empty(&self) -> bool {
        self.events.nrows() == 0
    }

    pub fn as_array(&self) -> ArrayView2<'_, u64> {
        self.events.view()
    }

    pub fn sensor_size(&self) -> (usize, usize) {
        (self.width, self.height)
    }

    pub fn timestamp_scale_ms(&self) -> f64 {
        self.timestamp_scale_ms
    }

    pub fn iter(&self) -> impl Iterator<Item = Event> + '_ {
        self.events.rows().into_iter().map(|row| Event {
            x: row[0] as usize,
            y: row[1] as usize,
            timestamp: row[2],
            polarity: row[3] != 0,
        })
    }
}

pub fn load(path: impl AsRef<Path>) -> Result<EventStream, LoadError> {
    let mut archive = NpzArchive::open(path).map_err(|error| {
        if error.kind() == io::ErrorKind::InvalidData {
            LoadError::InvalidFormat(error.to_string())
        } else {
            LoadError::Io(error)
        }
    })?;
    let npy = archive
        .by_name(EVENT_DATA_KEY)
        .map_err(|error| LoadError::InvalidFormat(error.to_string()))?
        .ok_or_else(|| LoadError::InvalidFormat("missing event_data array".to_owned()))?;

    let [event_count] = npy.shape() else {
        return Err(LoadError::InvalidFormat(
            "event_data must be one-dimensional".to_owned(),
        ));
    };
    let event_count = usize::try_from(*event_count)
        .map_err(|_| LoadError::InvalidFormat("event_data is too large".to_owned()))?;
    let capacity = event_count
        .checked_mul(COLUMN_COUNT)
        .ok_or_else(|| LoadError::InvalidFormat("event_data is too large".to_owned()))?;
    let records = npy
        .data::<NImageNetEvent>()
        .map_err(|error| LoadError::InvalidFormat(error.to_string()))?;
    let mut values = Vec::with_capacity(capacity);

    for record in records {
        let record = record.map_err(|error| LoadError::InvalidFormat(error.to_string()))?;
        if usize::from(record.x) >= N_IMAGENET_WIDTH
            || usize::from(record.y) >= N_IMAGENET_HEIGHT
        {
            return Err(LoadError::InvalidFormat(format!(
                "event coordinate ({}, {}) exceeds sensor size {}x{}",
                record.x, record.y, N_IMAGENET_WIDTH, N_IMAGENET_HEIGHT
            )));
        }
        values.extend([
            u64::from(record.x),
            u64::from(record.y),
            u64::from(record.t),
            u64::from(record.p),
        ]);
    }

    let events = Array2::from_shape_vec((event_count, COLUMN_COUNT), values)
        .map_err(|error| LoadError::InvalidFormat(error.to_string()))?;
    Ok(EventStream {
        events,
        width: N_IMAGENET_WIDTH,
        height: N_IMAGENET_HEIGHT,
        timestamp_scale_ms: 0.001,
    })
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use super::load;

    #[test]
    fn loads_n_imagenet_events() {
        let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../data/test/example.npz");
        let stream = load(path).unwrap();
        let events = stream.as_array();

        assert!(!stream.is_empty());
        assert_eq!(stream.sensor_size(), (640, 480));
        assert_eq!(events.dim(), (stream.len(), 4));
        assert!(events.column(0).iter().all(|&x| x < 640));
        assert!(events.column(1).iter().all(|&y| y < 480));
        assert!(events.column(3).iter().all(|&polarity| polarity <= 1));
    }
}
