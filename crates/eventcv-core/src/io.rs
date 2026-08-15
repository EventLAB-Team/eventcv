use std::{error::Error, fmt, io, path::Path};

use crate::{EventStream, EventStreamBuilder};

mod aedat;
mod bag;
mod e2vid;
#[cfg(feature = "hdf5")]
mod h5;
mod npz;
mod prophesee;
mod prophesee_raw;
mod text;

pub use aedat::read_aedat;
pub use bag::{open_bag_slice, read_bag, write_bag, BagSliceSource};
pub use e2vid::{write_e2vid, E2vidWriter};
#[cfg(feature = "hdf5")]
pub use h5::{
    open_hdf5_slice, read_hdf5, read_hdf5_frame, write_hdf5_frame, write_hdf5_stream,
    Hdf5EventSink, Hdf5FrameSink, Hdf5SliceSource,
};
pub use npz::{read_npz, read_npz_frame, write_npz_frame, write_npz_stream};
pub use prophesee::read_dat;
pub use prophesee_raw::{open_raw_slice, read_raw};
pub use text::{
    load_rows, open_text_slice, read_text, write_text_stream, ColumnOrder, RawRow, TextOptions,
    TextReader, TimeUnit,
};

use crate::representation::{EventFrame, EventFrameData};
use crate::viz::{render_frame, Colormap};

/// A single event as produced by a reader, before it is placed on the sensor grid.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RawEvent {
    pub x: u16,
    pub y: u16,
    pub t: i64,
    pub p: bool,
}

/// A bounded-memory source of events. Readers parse on demand so multi-gigabyte
/// files never need to be resident; the consumer decides what to accumulate.
pub trait EventSource {
    fn sensor_size(&self) -> (usize, usize);
    fn timestamp_scale_ms(&self) -> f64;
    /// Returns the next event, or `None` at end of stream.
    fn next_event(&mut self) -> Result<Option<RawEvent>, IoError>;
}

/// Drains a source into an in-memory [`EventStream`], dropping out-of-bounds events.
pub fn read_all(source: impl EventSource) -> Result<EventStream, IoError> {
    read_capped(source, None)
}

/// Like [`read_all`] but stops after `max` kept events (for previewing huge files).
pub fn read_capped(
    mut source: impl EventSource,
    max: Option<usize>,
) -> Result<EventStream, IoError> {
    let (width, height) = source.sensor_size();
    let mut builder = EventStreamBuilder::new(width, height, source.timestamp_scale_ms());
    while let Some(event) = source.next_event()? {
        builder.push(event.x, event.y, event.t, event.p);
        if max.is_some_and(|max| builder.len() >= max) {
            break;
        }
    }
    Ok(builder.build())
}

/// Options for the unified [`load`] entry point. Most fields apply to a single
/// format; readers ignore the ones they do not need.
#[derive(Clone, Debug, Default)]
pub struct LoadOptions {
    /// `(width, height)`. `None` infers it from the data (coordinate range), or from the
    /// message for rosbag. An explicit value overrides and, for HDF5, skips the scan.
    pub sensor_size: Option<(usize, usize)>,
    /// Timestamp unit. `None` infers it (HDF5/text): a fractional text value means
    /// seconds, otherwise the unit is chosen from the recording span (see
    /// `TimeUnit::infer_from_span`). Ignored for rosbag (always ROS `sec`+`nsec`).
    pub time_unit: Option<TimeUnit>,
    /// Text column order.
    pub order: ColumnOrder,
    /// Rosbag topic to read (defaults to `/davis/left/events`).
    pub topic: Option<String>,
    /// Cap on the number of events to read.
    pub max_events: Option<usize>,
    /// Absolute timestamp (µs, the file's own time base): events before it are skipped.
    /// `max_events` then caps the events *after* the offset. `None`/`<= 0` reads from the start.
    pub offset: Option<i64>,
    /// Explicit x/y/t/p column names, overriding auto-detection when a file's layout can't
    /// be guessed. For HDF5 each value is a dataset path (a compound field is `dataset/field`);
    /// for text/CSV a header column name or 0-based index. `None` auto-detects. Readers that
    /// don't need it (rosbag, aedat, dat) ignore it.
    pub keys: Option<EventKeys>,
}

/// User-supplied names for the four event columns, the escape hatch when auto-detection
/// can't identify them. Interpreted per format (HDF5 dataset paths, text header
/// names/indices); see [`LoadOptions::keys`].
#[derive(Clone, Debug)]
pub struct EventKeys {
    pub x: String,
    pub y: String,
    pub t: String,
    pub p: String,
}

/// Column index of each event field in the `[x, y, t, p]` ordering the readers share.
pub(crate) const X: usize = 0;
pub(crate) const Y: usize = 1;
pub(crate) const T: usize = 2;
pub(crate) const P: usize = 3;

/// Case-insensitive name synonyms for each event field, used by the HDF5 and text readers
/// to identify x/y/t/p under varied headings (dataset names, CSV headers). Order within a
/// list is the tie-break preference — earlier is more canonical.
pub(crate) const ROLE_KEYS: [&[&str]; 4] = [
    &[
        "x",
        "xs",
        "x_coordinate",
        "x_coordinates",
        "u",
        "col",
        "cols",
        "column",
        "columns",
    ],
    &[
        "y",
        "ys",
        "y_coordinate",
        "y_coordinates",
        "v",
        "row",
        "rows",
    ],
    &[
        "t",
        "ts",
        "time",
        "times",
        "timestamp",
        "timestamps",
        "time_stamp",
    ],
    &[
        "p",
        "ps",
        "pol",
        "pols",
        "polarity",
        "polarities",
        "polarity_bit",
        "polarity_bits",
        "sign",
    ],
];

/// The event field (`X`/`Y`/`T`/`P`) a column or dataset base name denotes, matched
/// case-insensitively against [`ROLE_KEYS`]; `None` if it matches none. The synonym lists
/// are disjoint, so at most one role matches.
pub(crate) fn role_of(name: &str) -> Option<usize> {
    let lower = name.to_ascii_lowercase();
    ROLE_KEYS
        .iter()
        .position(|keys| keys.contains(&lower.as_str()))
}

/// Rank of `name` within `role`'s synonym list (lower = more canonical), used to choose
/// between several names in one group that map to the same role; `None` if it isn't one.
#[cfg_attr(not(feature = "hdf5"), allow(dead_code))]
pub(crate) fn role_rank(role: usize, name: &str) -> Option<usize> {
    let lower = name.to_ascii_lowercase();
    ROLE_KEYS[role].iter().position(|key| *key == lower)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Format {
    Npz,
    Text,
    Hdf5,
    Rosbag,
    Aedat,
    Aedat4,
    PropheseeDat,
    PropheseeRaw,
    Png,
    /// E2VID's `t x y p` text interchange (see [`e2vid`]) — write-only.
    E2vid,
}

fn detect_format(path: &Path) -> Result<Format, IoError> {
    let extension = path
        .extension()
        .and_then(|extension| extension.to_str())
        .map(str::to_ascii_lowercase);
    match extension.as_deref() {
        Some("npz") => Ok(Format::Npz),
        Some("txt") | Some("csv") => Ok(Format::Text),
        Some("h5") | Some("hdf5") => Ok(Format::Hdf5),
        Some("bag") => Ok(Format::Rosbag),
        Some("aedat") => Ok(Format::Aedat),
        Some("aedat4") => Ok(Format::Aedat4),
        Some("dat") => Ok(Format::PropheseeDat),
        Some("raw") => Ok(Format::PropheseeRaw),
        Some("png") => Ok(Format::Png),
        Some("zip") => Ok(Format::E2vid),
        Some(other) => Err(IoError::Unsupported(format!(
            "unrecognised file extension: .{other}"
        ))),
        None => Err(IoError::Unsupported(
            "file has no extension to detect its format".to_owned(),
        )),
    }
}

/// Whether `path`'s format can be written incrementally with [`Hdf5EventSink`] — currently only
/// HDF5. The live recorder uses this to stream a camera to disk window-by-window when it can, and
/// fall back to buffering the whole recording in memory (then [`save_stream`]) when it can't.
pub fn supports_event_append(path: impl AsRef<Path>) -> bool {
    matches!(detect_format(path.as_ref()), Ok(Format::Hdf5))
}

/// Whether `path` (with an optional explicit `format`) names an E2VID export — the target
/// [`E2vidWriter`] streams into. Lets a caller that owns the read loop, like the Python
/// `save(reader, …)`, pick the incremental path without duplicating extension rules.
pub fn is_e2vid_target(path: impl AsRef<Path>, format: Option<&str>) -> bool {
    let options = SaveOptions {
        format: format.map(str::to_owned),
        ..SaveOptions::default()
    };
    matches!(requested_format(path.as_ref(), &options), Ok(Format::E2vid))
}

/// Loads events from any supported file, detected by extension — the OpenCV-style
/// single entry point. Supported today: `.npz`, `.txt`/`.csv`, `.bag`, `.h5`/`.hdf5`,
/// `.aedat` (AEDAT 2.0), and `.dat` (Prophesee CD).
pub fn load(path: impl AsRef<Path>, options: LoadOptions) -> Result<EventStream, IoError> {
    let path = path.as_ref();
    let Some(cutoff) = options.offset.filter(|&offset| offset > 0) else {
        return load_format(path, &options);
    };
    // The offset is an absolute timestamp, so the whole recording must be read before
    // skipping; the cap then applies to the events at/after the offset.
    let mut read_options = options.clone();
    read_options.max_events = None;
    load_format(path, &read_options).map(|stream| skip_before(stream, cutoff, options.max_events))
}

/// Drops events before the absolute timestamp `cutoff` (µs), then keeps at most `max` of the rest.
fn skip_before(stream: EventStream, cutoff: i64, max: Option<usize>) -> EventStream {
    let (ts, (xs, ys, ps)) = (stream.ts(), (stream.xs(), stream.ys(), stream.ps()));
    if ts.is_empty() {
        return stream;
    }
    let (width, height) = stream.sensor_size();
    let mut builder = EventStreamBuilder::new(width, height, stream.timestamp_scale_ms());
    for index in (0..ts.len()).filter(|&index| ts[index] >= cutoff) {
        builder.push(xs[index], ys[index], ts[index], ps[index]);
        if max.is_some_and(|max| builder.len() >= max) {
            break;
        }
    }
    builder.build()
}

fn load_format(path: &Path, options: &LoadOptions) -> Result<EventStream, IoError> {
    match detect_format(path)? {
        Format::Npz => npz::read_npz(path, options.sensor_size),
        Format::Text => text::load_text(path, options),
        Format::Rosbag => bag::read_bag(path, options),
        Format::Hdf5 => {
            #[cfg(feature = "hdf5")]
            {
                h5::read_hdf5(path, options)
            }
            #[cfg(not(feature = "hdf5"))]
            {
                Err(IoError::Unsupported(
                    "HDF5 support is not built in; rebuild with --features hdf5".to_owned(),
                ))
            }
        }
        Format::Aedat => aedat::read_aedat(path, options),
        Format::Aedat4 => Err(IoError::Unsupported(
            "AEDAT4 (.aedat4, iniVation DV FlatBuffer/LZ4) reading is not implemented yet"
                .to_owned(),
        )),
        Format::PropheseeDat => prophesee::read_dat(path, options),
        Format::PropheseeRaw => prophesee_raw::read_raw(path, options),
        Format::Png => Err(IoError::Unsupported(
            "PNG is a frame export format, not an event stream; use save_frame".to_owned(),
        )),
        Format::E2vid => Err(IoError::Unsupported(
            "E2VID's .zip is an export format for that reconstruction pipeline, not one eventcv \
             reads back; save an npz/h5/bag alongside it to keep the recording"
                .to_owned(),
        )),
    }
}

/// Random-access view over a file's events: fetch an arbitrary time or count range
/// without materialising the whole stream. This backs the lazy [`open`] handle — the
/// OpenCV `VideoCapture` to [`load`]'s `imread`. Each call returns a new [`EventStream`].
pub trait SliceSource: Send {
    fn sensor_size(&self) -> (usize, usize);
    fn timestamp_scale_ms(&self) -> f64;
    /// Total events in the file.
    fn n_events(&self) -> usize;
    /// `(t_min, t_max)` in microseconds across the whole file; `(0, 0)` when empty.
    fn time_span(&self) -> (i64, i64);
    /// Events whose index lies in `[i0, i1)` (clamped to the file).
    fn slice_index(&self, i0: usize, i1: usize) -> Result<EventStream, IoError>;
    /// Events whose timestamp (µs) lies in the half-open window `[t0, t1)`.
    fn slice_time(&self, t0: i64, t1: i64) -> Result<EventStream, IoError>;

    /// Per-pixel event counts over the whole file (row-major `width·height`), out-of-bounds
    /// events dropped — what the reader's hot-pixel pre-scan needs. The default tallies through
    /// `slice_index` in bounded chunks; a source that can read coordinates alone (HDF5) overrides
    /// this to skip the `t`/`p` columns and stream construction, which is most of the work.
    fn pixel_counts(&self) -> Result<Vec<u64>, IoError> {
        const CHUNK: usize = 8_000_000;
        let (width, height) = self.sensor_size();
        let mut counts = vec![0u64; width * height];
        let total = self.n_events();
        let mut start = 0;
        while start < total {
            let end = (start + CHUNK).min(total);
            self.slice_index(start, end)?.add_pixel_counts(&mut counts);
            start = end;
        }
        Ok(counts)
    }
}

/// A [`SliceSource`] backed by an already-loaded stream — the universal fallback for
/// formats without native random access (npz/txt/bag today). Slicing is entirely
/// in-RAM, so `open()` returns a working handle for every supported format from day one.
pub struct MemorySliceSource {
    stream: EventStream,
}

impl MemorySliceSource {
    pub fn new(stream: EventStream) -> Self {
        Self { stream }
    }

    fn rebuild(&self, indices: impl Iterator<Item = usize>) -> EventStream {
        let (width, height) = self.stream.sensor_size();
        let mut builder = EventStreamBuilder::new(width, height, self.stream.timestamp_scale_ms());
        let (xs, ys, ts, ps) = (
            self.stream.xs(),
            self.stream.ys(),
            self.stream.ts(),
            self.stream.ps(),
        );
        for index in indices {
            builder.push(xs[index], ys[index], ts[index], ps[index]);
        }
        builder.build()
    }
}

impl SliceSource for MemorySliceSource {
    fn sensor_size(&self) -> (usize, usize) {
        self.stream.sensor_size()
    }

    fn timestamp_scale_ms(&self) -> f64 {
        self.stream.timestamp_scale_ms()
    }

    fn n_events(&self) -> usize {
        self.stream.len()
    }

    fn time_span(&self) -> (i64, i64) {
        let ts = self.stream.ts();
        match (ts.iter().min(), ts.iter().max()) {
            (Some(&min), Some(&max)) => (min, max),
            _ => (0, 0),
        }
    }

    fn slice_index(&self, i0: usize, i1: usize) -> Result<EventStream, IoError> {
        let i0 = i0.min(self.stream.len());
        let i1 = i1.clamp(i0, self.stream.len());
        Ok(self.rebuild(i0..i1))
    }

    fn slice_time(&self, t0: i64, t1: i64) -> Result<EventStream, IoError> {
        let ts = self.stream.ts();
        Ok(self.rebuild((0..ts.len()).filter(|&index| ts[index] >= t0 && ts[index] < t1)))
    }

    fn pixel_counts(&self) -> Result<Vec<u64>, IoError> {
        // Already resident — tally straight from the stream, no chunked rebuilds.
        let (width, height) = self.stream.sensor_size();
        let mut counts = vec![0u64; width * height];
        self.stream.add_pixel_counts(&mut counts);
        Ok(counts)
    }
}

/// A boxed [`SliceSource`] — the handle [`open`] returns.
pub type Reader = Box<dyn SliceSource>;

/// Opens a file for lazy slicing, detected by extension (the `VideoCapture` analogue to
/// [`load`]'s `imread`). HDF5 is sliced in place by binary-searching its timestamp
/// dataset; every other format is loaded once and sliced in memory. Same `LoadOptions`
/// as [`load`] (`max_events` is ignored — slicing supersedes it).
pub fn open(path: impl AsRef<Path>, options: LoadOptions) -> Result<Reader, IoError> {
    let path = path.as_ref();
    match detect_format(path)? {
        Format::Hdf5 => {
            #[cfg(feature = "hdf5")]
            {
                Ok(Box::new(h5::open_hdf5_slice(path, &options)?))
            }
            #[cfg(not(feature = "hdf5"))]
            {
                Err(IoError::Unsupported(
                    "HDF5 support is not built in; rebuild with --features hdf5".to_owned(),
                ))
            }
        }
        Format::Text => Ok(Box::new(text::open_text_slice(path, &options)?)),
        Format::Rosbag => Ok(Box::new(bag::open_bag_slice(path, &options)?)),
        Format::PropheseeRaw => Ok(Box::new(prophesee_raw::open_raw_slice(path, &options)?)),
        _ => Ok(Box::new(MemorySliceSource::new(load(path, options)?))),
    }
}

/// Options for the [`save_stream`] / [`save_frame`] writers — the symmetric mirror of
/// [`LoadOptions`]. Most formats ignore every field; readers and writers agree on the rest.
#[derive(Clone, Debug, Default)]
pub struct SaveOptions {
    /// Rosbag topic to write the `dvs_msgs/EventArray` messages on (defaults to
    /// `/davis/left/events`, matching the reader).
    pub topic: Option<String>,
    /// PNG frame export only: the colour map (default [`Colormap::Viridis`]).
    pub colormap: Colormap,
    /// PNG frame export only: auto-contrast the field to its data range. `None` = `true`.
    pub normalize: Option<bool>,
    /// Overrides the format the extension implies. The one case that needs it is writing E2VID's
    /// layout to a `.txt` (which otherwise means eventcv's own text format); `.zip` already
    /// implies it. `None` follows the extension.
    pub format: Option<String>,
}

/// The format `path` and `options` together ask for: an explicit `format` name, else the
/// extension.
fn requested_format(path: &Path, options: &SaveOptions) -> Result<Format, IoError> {
    match options.format.as_deref() {
        None => detect_format(path),
        Some("e2vid") => Ok(Format::E2vid),
        Some("npz") => Ok(Format::Npz),
        Some("txt") | Some("csv") | Some("text") => Ok(Format::Text),
        Some("h5") | Some("hdf5") => Ok(Format::Hdf5),
        Some("bag") | Some("rosbag") => Ok(Format::Rosbag),
        Some("png") => Ok(Format::Png),
        Some(other) => Err(IoError::Unsupported(format!(
            "unknown format: {other} (expected npz, txt, h5, bag, png, or e2vid)"
        ))),
    }
}

/// Persists an [`EventStream`] to `path`, the format chosen by extension — the symmetric
/// counterpart of [`load`]. npz/HDF5/rosbag round-trip exactly (metadata stored); txt
/// stores `t x y p` and recovers sensor size / time unit on load via inference or options.
/// `.zip` (or `format: "e2vid"`) writes E2VID's interchange text instead, which eventcv does not
/// read back.
pub fn save_stream(
    path: impl AsRef<Path>,
    stream: &EventStream,
    options: &SaveOptions,
) -> Result<(), IoError> {
    let path = path.as_ref();
    match requested_format(path, options)? {
        Format::Npz => npz::write_npz_stream(path, stream),
        Format::Text => text::write_text_stream(path, stream),
        Format::E2vid => e2vid::write_e2vid(path, stream),
        Format::Rosbag => bag::write_bag(path, stream, options.topic.as_deref()),
        Format::Hdf5 => {
            #[cfg(feature = "hdf5")]
            {
                h5::write_hdf5_stream(path, stream)
            }
            #[cfg(not(feature = "hdf5"))]
            {
                Err(IoError::Unsupported(
                    "HDF5 support is not built in; rebuild with --features hdf5".to_owned(),
                ))
            }
        }
        other => Err(IoError::Unsupported(format!(
            "saving an event stream as {other:?} is not supported"
        ))),
    }
}

/// Persists an [`EventFrame`] (a computed representation) to `path`, preserving its shape,
/// dtype, `kind`, and `channel_names`. Supported: npz (default build) and HDF5.
pub fn save_frame(
    path: impl AsRef<Path>,
    frame: &EventFrame,
    options: &SaveOptions,
) -> Result<(), IoError> {
    let path = path.as_ref();
    match detect_format(path)? {
        Format::Npz => npz::write_npz_frame(path, frame),
        Format::Hdf5 => {
            #[cfg(feature = "hdf5")]
            {
                h5::write_hdf5_frame(path, frame)
            }
            #[cfg(not(feature = "hdf5"))]
            {
                Err(IoError::Unsupported(
                    "HDF5 support is not built in; rebuild with --features hdf5".to_owned(),
                ))
            }
        }
        Format::Png => write_png_frame(path, frame, options),
        other => Err(IoError::Unsupported(format!(
            "saving an event frame as {other:?} is not supported"
        ))),
    }
}

/// Renders a frame through [`render_frame`] (colormapped 2-D view) and encodes it as an
/// 8-bit RGB PNG. Unlike npz/HDF5 this is a *view*, not a round-trippable dump.
fn write_png_frame(path: &Path, frame: &EventFrame, options: &SaveOptions) -> Result<(), IoError> {
    let image = render_frame(frame, options.colormap, options.normalize.unwrap_or(true));
    let file = std::fs::File::create(path)?;
    let writer = std::io::BufWriter::new(file);
    let mut encoder = png::Encoder::new(writer, image.width as u32, image.height as u32);
    encoder.set_color(png::ColorType::Rgb);
    encoder.set_depth(png::BitDepth::Eight);
    encoder
        .write_header()
        .and_then(|mut writer| writer.write_image_data(&image.pixels))
        .map_err(png_encoding_error)
}

fn png_encoding_error(error: png::EncodingError) -> IoError {
    match error {
        png::EncodingError::IoError(error) => IoError::Io(error),
        other => IoError::Format(other.to_string()),
    }
}

/// Writes an ROI mask (row-major `width·height`, `true` = keep) as an 8-bit greyscale `.png`:
/// white where events are kept, black where they are dropped. Lets an ROI drawn or computed once
/// be reused across sessions, and inspected in any image viewer. Read it back with [`read_mask`].
pub fn write_mask(
    path: impl AsRef<Path>,
    mask: &[bool],
    width: usize,
    height: usize,
) -> Result<(), IoError> {
    let path = path.as_ref();
    if !matches!(detect_format(path)?, Format::Png) {
        return Err(IoError::Unsupported("masks are saved as .png".to_owned()));
    }
    if width == 0 || height == 0 {
        return Err(IoError::InvalidSensorSize);
    }
    if mask.len() != width * height {
        return Err(IoError::Format(format!(
            "mask has {} pixels, expected {} for a {width}x{height} grid",
            mask.len(),
            width * height
        )));
    }
    let pixels: Vec<u8> = mask.iter().map(|&keep| if keep { 255 } else { 0 }).collect();
    let writer = std::io::BufWriter::new(std::fs::File::create(path)?);
    let mut encoder = png::Encoder::new(writer, width as u32, height as u32);
    encoder.set_color(png::ColorType::Grayscale);
    encoder.set_depth(png::BitDepth::Eight);
    encoder
        .write_header()
        .and_then(|mut writer| writer.write_image_data(&pixels))
        .map_err(png_encoding_error)
}

/// Reads an ROI mask from a `.png`, returning it as `(mask, width, height)` (row-major, `true` =
/// keep). Any 8-bit-normalisable PNG works — greyscale, palette, or colour, with or without alpha —
/// so a mask binarised in another tool loads as-is: a pixel is **kept where it is non-black and not
/// fully transparent**. The counterpart of [`write_mask`].
pub fn read_mask(path: impl AsRef<Path>) -> Result<(Vec<bool>, usize, usize), IoError> {
    let png = decode_png(path.as_ref(), "masks are loaded from .png")?;
    let mut mask = Vec::with_capacity(png.width * png.height);
    png.for_each_pixel(|colour, opacity| {
        // Any non-black, non-transparent pixel is "keep" — the intensities themselves don't matter
        // for a mask, only whether the pixel was painted.
        mask.push(colour.iter().any(|&value| value != 0) && opacity.first() != Some(&0));
    });
    Ok((mask, png.width, png.height))
}

/// Reads a PNG as a single-channel 8-bit intensity frame.
///
/// Colour images are collapsed to luma with the Rec. 601 weights the rest of the library uses;
/// an alpha channel is ignored rather than composited, since there is no background to composite
/// against and a half-transparent pixel still had a measured brightness.
pub fn read_png_frame(path: impl AsRef<Path>) -> Result<EventFrame, IoError> {
    let png = decode_png(path.as_ref(), "frames are loaded from .png")?;
    let mut samples = Vec::with_capacity(png.width * png.height);
    png.for_each_pixel(|colour, _| samples.push(luma(colour)));
    EventFrame::intensity(EventFrameData::U8(samples), png.width, png.height)
        .map_err(|error| IoError::Format(error.to_string()))
}

/// Rec. 601 luma. A single sample is already grey and passes through untouched, which keeps a
/// greyscale PNG bit-exact rather than round-tripping it through the weights.
fn luma(colour: &[u8]) -> u8 {
    match colour {
        [grey] => *grey,
        [r, g, b, ..] => {
            (0.299 * f32::from(*r) + 0.587 * f32::from(*g) + 0.114 * f32::from(*b)).round() as u8
        }
        _ => 0,
    }
}

/// An 8-bit PNG decoded into memory, with enough shape to walk its pixels.
struct DecodedPng {
    buffer: Vec<u8>,
    width: usize,
    height: usize,
    line_size: usize,
    samples: usize,
    alpha: bool,
}

impl DecodedPng {
    /// Calls `visit(colour, opacity)` per pixel in row-major order, where `colour` excludes any
    /// alpha sample and `opacity` is the alpha (empty when the image has none).
    fn for_each_pixel(&self, mut visit: impl FnMut(&[u8], &[u8])) {
        for row in self.buffer.chunks_exact(self.line_size).take(self.height) {
            for pixel in row[..self.width * self.samples].chunks_exact(self.samples) {
                let (colour, opacity) = pixel.split_at(self.samples - usize::from(self.alpha));
                visit(colour, opacity);
            }
        }
    }
}

/// Shared 8-bit PNG decode behind [`read_mask`] and [`read_png_frame`]. `unsupported` is the
/// message used when the path is not a PNG at all, so each caller can say what it wanted.
fn decode_png(path: &Path, unsupported: &str) -> Result<DecodedPng, IoError> {
    if !matches!(detect_format(path)?, Format::Png) {
        return Err(IoError::Unsupported(unsupported.to_owned()));
    }
    let mut decoder = png::Decoder::new(std::io::BufReader::new(std::fs::File::open(path)?));
    // Expands palette and sub-byte images and strips 16-bit samples, so everything below is 8-bit.
    decoder.set_transformations(png::Transformations::normalize_to_color8());
    let mut reader = decoder.read_info().map_err(png_decoding_error)?;
    let mut buffer = vec![0; reader.output_buffer_size()];
    let info = reader.next_frame(&mut buffer).map_err(png_decoding_error)?;
    Ok(DecodedPng {
        width: info.width as usize,
        height: info.height as usize,
        line_size: info.line_size,
        samples: info.color_type.samples(),
        alpha: matches!(
            info.color_type,
            png::ColorType::GrayscaleAlpha | png::ColorType::Rgba
        ),
        buffer,
    })
}

fn png_decoding_error(error: png::DecodingError) -> IoError {
    match error {
        png::DecodingError::IoError(error) => IoError::Io(error),
        other => IoError::Format(other.to_string()),
    }
}

/// Reads an [`EventFrame`] previously written by [`save_frame`], reconstructing its dtype,
/// `kind`, and `channel_names`. Supported: npz (default build) and HDF5.
pub fn load_frame(path: impl AsRef<Path>) -> Result<EventFrame, IoError> {
    let path = path.as_ref();
    match detect_format(path)? {
        Format::Npz => npz::read_npz_frame(path),
        // A PNG has no stored kind or channel names, so it comes back as a greyscale intensity
        // frame — the one representation that is not derived from events.
        Format::Png => read_png_frame(path),
        Format::Hdf5 => {
            #[cfg(feature = "hdf5")]
            {
                h5::read_hdf5_frame(path)
            }
            #[cfg(not(feature = "hdf5"))]
            {
                Err(IoError::Unsupported(
                    "HDF5 support is not built in; rebuild with --features hdf5".to_owned(),
                ))
            }
        }
        other => Err(IoError::Unsupported(format!(
            "loading an event frame from {other:?} is not supported"
        ))),
    }
}

#[derive(Debug)]
pub enum IoError {
    Io(io::Error),
    Parse { line: usize, message: String },
    Format(String),
    InvalidSensorSize,
    Unsupported(String),
}

impl fmt::Display for IoError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io(error) => error.fmt(formatter),
            Self::Parse { line, message } => write!(formatter, "line {line}: {message}"),
            Self::Format(message) => formatter.write_str(message),
            Self::InvalidSensorSize => {
                formatter.write_str("sensor width and height must be positive")
            }
            Self::Unsupported(message) => formatter.write_str(message),
        }
    }
}

impl Error for IoError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Io(error) => Some(error),
            _ => None,
        }
    }
}

impl From<io::Error> for IoError {
    fn from(error: io::Error) -> Self {
        Self::Io(error)
    }
}

#[cfg(test)]
mod tests {
    use super::{
        load, open, read_mask, write_mask, IoError, LoadOptions, MemorySliceSource, SliceSource,
    };
    use crate::EventStreamBuilder;

    #[test]
    fn unknown_extension_is_unsupported() {
        let error = load("recording.mp4", LoadOptions::default()).unwrap_err();
        assert!(matches!(error, IoError::Unsupported(_)));
    }

    #[test]
    fn hdf5_extension_dispatches_per_feature() {
        // `.h5` is always recognised; with the feature it reaches the reader (a missing
        // file is then an IO error), without it the dispatch reports missing support.
        let error = load("recording.h5", LoadOptions::default()).unwrap_err();
        #[cfg(feature = "hdf5")]
        assert!(matches!(error, IoError::Io(_)));
        #[cfg(not(feature = "hdf5"))]
        match error {
            IoError::Unsupported(message) => assert!(message.contains("HDF5")),
            other => panic!("expected unsupported error, got {other:?}"),
        }
    }

    #[test]
    fn text_missing_file_is_reported() {
        // Text no longer requires sensor_size (it infers); a missing file is an IO error.
        let error = load("events.txt", LoadOptions::default()).unwrap_err();
        assert!(matches!(error, IoError::Io(_)));
    }

    #[test]
    fn aedat_dispatches_to_the_reader() {
        // `.aedat` reaches the AEDAT 2.0 reader, so a missing file is an IO error.
        let error = load("recording.aedat", LoadOptions::default()).unwrap_err();
        assert!(matches!(error, IoError::Io(_)));
    }

    #[test]
    fn prophesee_dat_dispatches_to_the_reader() {
        let error = load("recording.dat", LoadOptions::default()).unwrap_err();
        assert!(matches!(error, IoError::Io(_)));
    }

    #[test]
    fn aedat4_is_unsupported_and_raw_dispatches() {
        match load("recording.aedat4", LoadOptions::default()) {
            Err(IoError::Unsupported(_)) => {}
            other => panic!("expected unsupported for recording.aedat4, got {other:?}"),
        }
        assert!(matches!(
            load("recording.raw", LoadOptions::default()),
            Err(IoError::Io(_))
        ));
    }

    fn sample_source() -> MemorySliceSource {
        let mut builder = EventStreamBuilder::new(4, 4, 0.001);
        builder.push(0, 0, 0, true);
        builder.push(1, 1, 10, false);
        builder.push(2, 2, 20, true);
        builder.push(0, 1, 30, false);
        MemorySliceSource::new(builder.build())
    }

    #[test]
    fn memory_source_reports_span_and_count() {
        let source = sample_source();
        assert_eq!(source.n_events(), 4);
        assert_eq!(source.time_span(), (0, 30));
    }

    #[test]
    fn memory_source_slices_by_time_and_index() {
        let source = sample_source();

        // Half-open [10, 30) keeps t = 10 and 20.
        assert_eq!(source.slice_time(10, 30).unwrap().ts(), &[10, 20]);
        assert_eq!(source.slice_index(1, 3).unwrap().ts(), &[10, 20]);
        assert_eq!(source.slice_index(2, 100).unwrap().len(), 2); // hi clamped to len
        assert!(source.slice_time(100, 200).unwrap().is_empty());
    }

    #[test]
    fn open_rejects_unknown_extension() {
        // `Reader` is a trait object (not `Debug`), so match rather than `unwrap_err`.
        match open("recording.mp4", LoadOptions::default()) {
            Err(IoError::Unsupported(_)) => {}
            Err(other) => panic!("expected unsupported error, got {other:?}"),
            Ok(_) => panic!("expected an error for an unknown extension"),
        }
    }

    #[test]
    fn mask_png_round_trips_and_validates() {
        let path = std::env::temp_dir().join(format!("eventcv_mask_{}.png", std::process::id()));
        let mask = crate::mask::ellipse(16, 12, 8.0, 6.0, 5.0, 4.0);

        write_mask(&path, &mask, 16, 12).unwrap();
        let (loaded, width, height) = read_mask(&path).unwrap();
        assert_eq!((width, height), (16, 12));
        assert_eq!(loaded, mask);
        std::fs::remove_file(&path).ok();

        // A mask that doesn't match the grid it claims, and a non-PNG target, both report why.
        assert!(matches!(
            write_mask(&path, &mask, 16, 11),
            Err(IoError::Format(_))
        ));
        assert!(matches!(
            write_mask("mask.npz", &mask, 16, 12),
            Err(IoError::Unsupported(_))
        ));
    }
}
