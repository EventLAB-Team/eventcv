//! `cameras` and `record`: USB event cameras without Python, e.g. on a headless capture box.
//!
//! The commands are always listed, so `--help` reads the same for every build; a build without the
//! `camera` feature says so when one is run.

use clap::Args;

#[cfg(not(feature = "camera"))]
use crate::Failure;

/// Source-side limits applied as the camera opens, shared by `record` and `tui --camera`.
#[derive(Args, Clone, Debug, Default)]
pub(crate) struct CameraOptions {
    /// cap on the events per second the sensor emits, enforced on-chip (Prophesee EVK4/EVK3-HD)
    #[arg(long, value_name = "EV/S")]
    pub max_event_rate: Option<f64>,
    /// only pixels inside this rectangle produce events
    #[arg(long, num_args = 4, value_names = ["X0", "Y0", "W", "H"])]
    pub roi: Option<Vec<i64>>,
    /// retune the sensor's biases as it runs to hold the event rate steady across changing light
    #[arg(long)]
    pub adaptive_bias: bool,
}

#[derive(Args, Debug)]
pub(crate) struct RecordArgs {
    /// destination; the extension picks the format (.h5 streams straight to disk)
    pub output: String,
    /// stop after this long (default: until Ctrl+C)
    #[arg(long, value_name = "S")]
    pub seconds: Option<f64>,
    /// camera serial, from `eventcv cameras` (default: the first one found)
    #[arg(long)]
    pub serial: Option<String>,
    /// HDF5 gzip level 0..9 (default: uncompressed, so the disk keeps up with the sensor)
    #[arg(long, value_name = "LEVEL")]
    pub compression: Option<u8>,
    #[command(flatten)]
    pub camera: CameraOptions,
    /// suppress the summary line
    #[arg(short, long)]
    pub quiet: bool,
}

#[cfg(not(feature = "camera"))]
pub(crate) const NO_CAMERA: &str =
    "eventcv was built without camera support; rebuild with --features camera";

#[cfg(not(feature = "camera"))]
pub(crate) fn cameras() -> Result<(), Failure> {
    Err(NO_CAMERA.into())
}

#[cfg(not(feature = "camera"))]
pub(crate) fn record(_: &RecordArgs) -> Result<(), Failure> {
    Err(NO_CAMERA.into())
}

#[cfg(feature = "camera")]
pub(crate) use live::{cameras, open, record};

#[cfg(feature = "camera")]
mod live {
    use std::time::{Duration, Instant};

    use eventcv_core::bias::BiasOverrides;
    use eventcv_core::device::{list_cameras, Capture, Limits, RoiPlacement, Window};
    use eventcv_core::io::{self, Compression, SaveOptions};
    use eventcv_core::representation::DEFAULT_SPAN_MS;

    use super::{CameraOptions, RecordArgs};
    use crate::commands::format_count;
    use crate::{emit, Failure};

    const FLUSH_INTERVAL: Duration = Duration::from_secs(1);

    impl CameraOptions {
        /// Checked before the device is opened, so a bad value is rejected the same way with or
        /// without hardware attached (and with the same words as `ecv.stream`).
        fn limits(&self) -> Result<Limits, Failure> {
            let max_event_rate = self
                .max_event_rate
                .map(|rate| {
                    if !rate.is_finite() || rate < 1.0 {
                        return Err("max_event_rate must be at least 1 event per second");
                    }
                    Ok(rate as u64)
                })
                .transpose()?;
            let roi = match self.roi.as_deref() {
                None => None,
                Some(&[x0, y0, width, height]) => {
                    if x0 < 0 || y0 < 0 {
                        return Err(
                            "roi must be (x0, y0, width, height) with a non-negative origin".into(),
                        );
                    }
                    if width < 1 || height < 1 {
                        return Err("roi width and height must be at least 1".into());
                    }
                    Some((x0 as usize, y0 as usize, width as usize, height as usize))
                }
                Some(_) => unreachable!("clap takes exactly four values"),
            };
            Ok(Limits {
                max_event_rate,
                roi,
            })
        }
    }

    /// Opens a camera the way `ecv.stream()` does, warning when a ROI can only be applied after
    /// the events have crossed the cable.
    pub(crate) fn open(serial: Option<&str>, options: &CameraOptions) -> Result<Capture, Failure> {
        let limits = options.limits()?;
        let bias = options.adaptive_bias.then(BiasOverrides::default);
        let capture = Capture::open(serial, Window::Duration(DEFAULT_SPAN_MS), limits, bias)?;
        if let Some(((x0, y0, width, height), RoiPlacement::Host)) = capture.roi() {
            eprintln!(
                "warning: roi=({x0}, {y0}, {width}, {height}) is filtered on the host: the {} has \
                 no on-chip region masks, so those events are still sent over USB and decoded \
                 before being dropped",
                capture.name()
            );
        }
        Ok(capture)
    }

    pub(crate) fn cameras() -> Result<(), Failure> {
        let listing = list_cameras();
        if let Some(warning) = &listing.warning {
            eprintln!("warning: {warning}");
        }
        if listing.cameras.is_empty() {
            emit("no event cameras found");
            return Ok(());
        }
        let mut rows = vec![["name", "kind", "serial", "speed", "usb"].map(str::to_owned)];
        for camera in &listing.cameras {
            rows.push([
                camera.name.clone(),
                camera.kind.clone(),
                camera.serial.clone().unwrap_or_else(|| "?".to_owned()),
                camera.speed.clone(),
                match (camera.bus_number, camera.address) {
                    (Some(bus), Some(address)) => format!("{bus}:{address}"),
                    _ => "-".to_owned(),
                },
            ]);
        }
        let widths: Vec<usize> = (0..5)
            .map(|column| rows.iter().map(|row| row[column].len()).max().unwrap_or(0))
            .collect();
        for row in rows {
            let cells: Vec<String> = row
                .iter()
                .zip(&widths)
                .map(|(cell, width)| format!("{cell:<width$}"))
                .collect();
            emit(cells.join("  ").trim_end());
        }
        Ok(())
    }

    pub(crate) fn record(args: &RecordArgs) -> Result<(), Failure> {
        // All of this before the device opens, so a typo costs nothing and leaves no empty file.
        if args.compression.is_some_and(|level| level > 9) {
            return Err("compression must be a gzip level in 0..=9".into());
        }
        if !io::supports_event_append(&args.output) {
            return Err(format!(
                "{} is not an event file; every event format can be appended window-by-window \
                 (.h5, .npz, .txt, .csv, .bag, .aedat, .aedat4, .dat, .raw), but .png is a frame \
                 export",
                args.output
            )
            .into());
        }
        let duration = args
            .seconds
            .map(|seconds| {
                Duration::try_from_secs_f64(seconds)
                    .ok()
                    .filter(|duration| !duration.is_zero())
                    .ok_or("seconds must be positive")
            })
            .transpose()?;

        let mut capture = open(args.serial.as_deref(), &args.camera)?;
        let compression = match args.compression {
            None | Some(0) => Compression::None,
            Some(level) => Compression::Gzip(level),
        };
        let mut sink = io::open_sink(
            &args.output,
            &SaveOptions {
                compression,
                ..SaveOptions::default()
            },
        )?;

        // Ctrl+C ends the recording and keeps what was captured, rather than discarding it.
        crate::graceful();
        let start = Instant::now();
        let mut last_flush = start;
        let mut overflows = 0;
        let mut fill = || -> Result<(), Failure> {
            while !crate::interrupted() && duration.is_none_or(|limit| start.elapsed() < limit) {
                if let Some(window) = capture.poll()? {
                    overflows += usize::from(window.first_after_overflow);
                    sink.append(&window.stream)?;
                    if last_flush.elapsed() >= FLUSH_INTERVAL {
                        sink.flush()?;
                        last_flush = Instant::now();
                    }
                }
            }
            // The window still filling belongs in the file too, or it would stop at the last whole
            // window boundary.
            let tail = capture.finish();
            for window in tail
                .into_iter()
                .chain(std::iter::from_fn(|| capture.take_pending()))
            {
                sink.append(&window.stream)?;
            }
            Ok(())
        };
        // Closed on both exits: a write error must not drop the last second of the recording.
        let result = fill();
        let n_events = sink.n_events();
        let closed = sink.finish();
        result?;
        closed?;

        if !args.quiet {
            let note = if overflows > 0 {
                format!(", {overflows} overflows")
            } else {
                String::new()
            };
            emit(format!(
                "recorded {} events in {:.1} s -> {}{note}",
                format_count(n_events as u64),
                start.elapsed().as_secs_f64(),
                args.output
            ));
        }
        Ok(())
    }
}
