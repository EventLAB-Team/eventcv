//! `info`, `convert`, `render` and `simulate` — ports of the Python CLI's commands of the same
//! names, over the same core calls the Python bindings make.

use std::io::IsTerminal;
use std::path::Path;
use std::time::{Duration, Instant};

use clap::Args;
use eventcv_core::io::{self, ColumnOrder, Compression, LoadOptions, SaveOptions, TimeUnit};
use eventcv_core::representation::ReprSpec;
use eventcv_core::simulate::{
    simulate_video_on, ProgressReporter, SimulatorConfig, Upsample, MAX_UPSAMPLE,
};
use eventcv_core::video::{encoder_for, AnimationEncoder, Fps};
use eventcv_core::viz::{
    frame_extent, Colormap, RawSurface, RenderView, Scale, CLIM_CALIBRATION_FRAMES,
};

use crate::{emit, Failure};

/// Events read per pass when streaming a recording to another format (as the bindings do).
const EXPORT_CHUNK: usize = 4_000_000;

/// How often a simulation's output file is flushed, so a long run is readable while it goes.
const FLUSH_INTERVAL: Duration = Duration::from_secs(1);

/// How to *read* the input, shared by every command that opens one. Text formats carry no unit or
/// column order in the file, and a rosbag has no single obvious connection.
#[derive(Args, Clone, Debug, Default)]
pub(crate) struct Source {
    /// timestamp unit of the source, when it cannot be inferred (text formats)
    #[arg(long, value_parser = ["s", "ms", "us", "ns"])]
    pub time_unit: Option<String>,
    /// column order of a text source, e.g. txyp or xytp
    #[arg(long, value_name = "COLS")]
    pub order: Option<String>,
    /// rosbag connection to read
    #[arg(long, value_name = "TOPIC")]
    pub topic: Option<String>,
}

impl Source {
    pub(crate) fn options(&self) -> Result<LoadOptions, Failure> {
        Ok(LoadOptions {
            time_unit: self
                .time_unit
                .as_deref()
                .map(str::parse::<TimeUnit>)
                .transpose()?,
            order: match self.order.as_deref() {
                None | Some("txyp") => ColumnOrder::Txyp,
                Some("xytp") => ColumnOrder::Xytp,
                Some(other) => return Err(format!("unsupported order: {other}").into()),
            },
            topic: self.topic.clone(),
            ..LoadOptions::default()
        })
    }
}

#[derive(Args, Debug)]
pub(crate) struct InfoArgs {
    /// event file to inspect
    pub path: String,
    #[command(flatten)]
    pub source: Source,
    /// also report the peak rate, measured over bins of this width
    #[arg(long, value_name = "MS")]
    pub rate_bin_ms: Option<f64>,
}

#[derive(Args, Debug)]
pub(crate) struct ConvertArgs {
    /// source recording
    pub input: String,
    /// destination; the extension picks the format
    pub output: String,
    #[command(flatten)]
    pub source: Source,
    /// suppress the summary line
    #[arg(short, long)]
    pub quiet: bool,
}

#[derive(Args, Debug)]
pub(crate) struct RenderArgs {
    /// source recording
    pub input: String,
    /// destination; the extension picks the format
    pub output: String,
    #[command(flatten)]
    pub source: Source,
    /// time per frame (default: 30)
    #[arg(long, value_name = "MS", default_value_t = 30.0)]
    pub dt_ms: f64,
    /// playback frame rate (default: 30)
    #[arg(long, default_value_t = 30.0)]
    pub fps: f64,
    /// what to draw: "raw" for the event stream itself (default), or a representation name
    /// (tencode, count, tsurf, voxel, …)
    #[arg(long, default_value = "raw")]
    pub repr: String,
    /// raw view only: fade time for an event's trail (default: one frame's worth)
    #[arg(long, value_name = "MS")]
    pub decay_ms: Option<f64>,
    /// colormap for scalar representations
    #[arg(long, default_value = "viridis")]
    pub colormap: String,
    /// fix the value mapped to the top of the colormap; omit to calibrate from the first frames,
    /// or pass 0 for per-frame auto-contrast
    #[arg(long, value_name = "MAX", allow_negative_numbers = true)]
    pub clim: Option<f64>,
    /// stop after N frames
    #[arg(long, value_name = "N")]
    pub max_frames: Option<usize>,
    /// suppress the summary line
    #[arg(short, long)]
    pub quiet: bool,
}

#[derive(Args, Debug)]
pub(crate) struct SimulateArgs {
    /// source video
    pub input: String,
    /// destination event file; the extension picks the format
    pub output: String,
    /// log-contrast threshold for both polarities (default: 0.2)
    #[arg(long, value_name = "C", default_value_t = 0.2)]
    pub threshold: f32,
    /// per-pixel threshold spread; 0 for an ideal sensor (default: 0.03)
    #[arg(long, value_name = "S", default_value_t = 0.03)]
    pub sigma_thres: f32,
    /// photoreceptor bandwidth for a white pixel; 0 disables (default: 200)
    #[arg(long, value_name = "HZ", default_value_t = 200.0)]
    pub cutoff_hz: f32,
    /// spontaneous ON events per pixel per second (default: 1)
    #[arg(long, value_name = "HZ", default_value_t = 1.0)]
    pub leak_rate_hz: f32,
    /// noise floor in dark pixels (default: 10)
    #[arg(long, value_name = "HZ", default_value_t = 10.0)]
    pub shot_noise_rate_hz: f32,
    /// dead time after each event (default: 100)
    #[arg(long, value_name = "US", default_value_t = 100)]
    pub refractory_us: i64,
    /// frame subdivision: "adaptive" (default), "off", or an integer factor
    #[arg(long, value_name = "MODE")]
    pub upsample: Option<String>,
    /// decode the video at this size instead of its native one
    #[arg(long, num_args = 2, value_names = ["W", "H"])]
    pub scale: Option<Vec<usize>>,
    /// stop after N frames
    #[arg(long, value_name = "N")]
    pub max_frames: Option<usize>,
    /// cap the sub-steps any frame pair is split into (default: 64). Each costs a full pass over
    /// every pixel and the busiest pixel sets the factor, so lowering this bounds the worst case
    /// at some cost in timestamp accuracy
    #[arg(long, value_name = "N")]
    pub max_upsample: Option<usize>,
    /// HDF5 gzip level 0..9; 0 stores uncompressed (default: 1)
    #[arg(long, value_name = "LEVEL")]
    pub compression: Option<i64>,
    /// seed for mismatch and noise
    #[arg(long, default_value_t = 0)]
    pub seed: u64,
    /// suppress the summary line
    #[arg(short, long)]
    pub quiet: bool,
}

/// Report what is in a recording without decoding all of it.
pub(crate) fn info(args: &InfoArgs) -> Result<(), Failure> {
    // No windowing: this opens the index only, so it stays fast on a multi-GB file.
    let reader = io::open(&args.path, args.source.options()?)?;
    let (width, height) = reader.sensor_size();
    let n = reader.n_events();
    let (lo, hi) = reader.time_span();
    let seconds = (hi - lo) as f64 / 1000.0 / 1000.0;

    let mut rows = vec![
        ("file", args.path.clone()),
        ("sensor", format!("{width} x {height}")),
        ("events", format_count(n as u64)),
        ("duration", format!("{seconds:.3} s")),
        // Mean rate is the number people actually compare against a camera's spec sheet.
        (
            "mean rate",
            if seconds > 0.0 {
                format!("{} ev/s", format_count((n as f64 / seconds) as u64))
            } else {
                "n/a ev/s".to_owned()
            },
        ),
    ];
    if let Some(bin_ms) = args.rate_bin_ms.filter(|&bin| bin != 0.0) {
        let bin_us = duration_us("bin", bin_ms)?;
        let rate = reader
            .slice_time(lo, hi.saturating_add(1))?
            .event_rate(bin_us);
        if !rate.is_empty() {
            let peak = rate.per_second().into_iter().fold(0.0, f64::max);
            rows.push(("peak rate", format!("{} ev/s", format_count(peak as u64))));
            rows.push(("bins", format!("{} x {bin_ms:?} ms", rate.len())));
        }
    }

    let key_width = rows.iter().map(|(key, _)| key.len()).max().unwrap_or(0);
    for (key, value) in rows {
        emit(format!("{key:>key_width$}  {value}"));
    }
    Ok(())
}

/// Re-encode a recording into another format.
pub(crate) fn convert(args: &ConvertArgs) -> Result<(), Failure> {
    let options = args.source.options()?;
    let reader = io::open(&args.input, options.clone())?;
    let n_events = reader.n_events();

    // E2VID is the one target written window by window, so a multi-GB recording never has to fit
    // in memory; the rest are loaded whole — the same split as the Python CLI.
    let streamed = io::is_e2vid_target(&args.output, None);
    if streamed {
        let mut writer = io::E2vidWriter::create(&args.output)?;
        let mut start = 0;
        while start < n_events {
            let end = (start + EXPORT_CHUNK).min(n_events);
            writer.append(&reader.slice_index(start, end)?)?;
            start = end;
        }
        writer.finish()?;
    } else {
        drop(reader);
        let stream = io::load(&args.input, options)?;
        io::save_stream(&args.output, &stream, &SaveOptions::default())?;
    }

    if !args.quiet {
        let how = if streamed { "streamed" } else { "loaded" };
        emit(format!(
            "wrote {} ({} events, {how})",
            args.output,
            format_count(n_events as u64)
        ));
    }
    Ok(())
}

/// Render a recording to an animation (.gif / .apng / .mp4).
pub(crate) fn render(args: &RenderArgs) -> Result<(), Failure> {
    let dt_us = duration_us("dt", args.dt_ms)?;
    let reader = io::open(&args.input, args.source.options()?)?;
    // "raw" is not a representation — it draws the events themselves. A representation's own time
    // span follows the frame, as `open(dt_ms=…).with_repr(…)` makes it in Python.
    let spec = (args.repr != "raw")
        .then(|| ReprSpec::named_with_span(&args.repr, None, Some(dt_us as f64 / 1000.0)))
        .transpose()?;

    let (lo, hi) = reader.time_span();
    let total = if reader.n_events() == 0 {
        0
    } else {
        ((hi - lo) / dt_us + 1) as usize
    };
    let frames = args.max_frames.map_or(total, |limit| limit.min(total));
    if frames == 0 {
        return Err("nothing to render: the reader has no slices".into());
    }
    let window = |index: usize| {
        let t0 = lo + index as i64 * dt_us;
        reader.slice_time(t0, t0 + dt_us)
    };

    let mut view = match spec {
        None => {
            let (width, height) = reader.sensor_size();
            // One window's worth of trail is what makes motion legible without smearing it.
            let decay = args.decay_ms.unwrap_or((dt_us as f64 / 1000.0).max(1.0));
            RenderView::Raw(RawSurface::new(width, height, decay))
        }
        Some(spec) => {
            let scale = Scale::for_sequence(
                args.clim,
                (0..frames.min(CLIM_CALIBRATION_FRAMES))
                    .map(|index| Ok::<_, Failure>(frame_extent(&spec.generate(&window(index)?)?))),
            )?;
            RenderView::Repr {
                spec,
                colormap: parse_colormap(&args.colormap)?,
                scale,
            }
        }
    };

    let mut encoder: Option<Box<dyn AnimationEncoder + Send>> = None;
    for index in 0..frames {
        let image = view.render(&window(index)?)?;
        // Opened on the first frame so the encoder learns the dimensions from real output.
        let encoder = match encoder.as_mut() {
            Some(encoder) => encoder,
            None => encoder.insert(encoder_for(
                Path::new(&args.output),
                frames as u32,
                image.width,
                image.height,
                Fps::new(args.fps),
            )?),
        };
        encoder.write_frame(&image)?;
    }
    if let Some(encoder) = encoder {
        encoder.finish()?;
    }

    if !args.quiet {
        emit(format!(
            "wrote {} ({frames} frames at {} fps)",
            args.output,
            python_g(args.fps)
        ));
    }
    Ok(())
}

/// Turn a video into the events a DVS would have produced watching it.
pub(crate) fn simulate(args: &SimulateArgs) -> Result<(), Failure> {
    let upsample = match args.upsample.as_deref() {
        None | Some("adaptive") => Upsample::Adaptive {
            max_events_per_pixel: 1.0,
        },
        Some("off") | Some("none") => Upsample::Off,
        Some(other) => other.parse().map(Upsample::Fixed).map_err(|_| {
            format!("upsample must be \"adaptive\", \"off\", or an integer factor, not {other:?}")
        })?,
    };
    let compression = match args.compression {
        None => Compression::default(),
        Some(0) => Compression::None,
        Some(level @ 1..=9) => Compression::Gzip(level as u8),
        Some(_) => {
            return Err(
                "compression must be None (the default, gzip 1), False or 0 to store \
                        uncompressed, or a gzip level in 1..=9"
                    .into(),
            )
        }
    };
    let config = SimulatorConfig {
        pos_thres: args.threshold,
        neg_thres: args.threshold,
        sigma_thres: args.sigma_thres,
        refractory_us: args.refractory_us,
        cutoff_hz: args.cutoff_hz,
        leak_rate_hz: args.leak_rate_hz,
        shot_noise_rate_hz: args.shot_noise_rate_hz,
        seed: args.seed,
        upsample,
        max_upsample: args.max_upsample.unwrap_or(MAX_UPSAMPLE),
    };

    // Opened first, so a target that cannot hold events fails before the whole clip is simulated.
    // Every event format appends, so the run streams to disk and never holds the output in memory.
    let mut sink = io::open_sink(
        &args.output,
        &SaveOptions {
            compression,
            ..SaveOptions::default()
        },
    )?;
    let mut last_flush = Instant::now();
    // Progress goes to stderr, so it stays out of a redirected stdout.
    let mut reporter = ProgressReporter::new(!args.quiet && std::io::stderr().is_terminal());
    let mut frames = 0;
    let result = simulate_video_on(
        Path::new(&args.input),
        config,
        eventcv_core::accel::Device::Cpu,
        None,
        args.scale.as_deref().map(|size| (size[0], size[1])),
        args.max_frames,
        |stream| {
            if stream.is_empty() {
                return Ok(());
            }
            let io_error = |error: io::IoError| std::io::Error::other(error.to_string());
            sink.append(&stream).map_err(io_error)?;
            if last_flush.elapsed() >= FLUSH_INTERVAL {
                sink.flush().map_err(io_error)?;
                last_flush = Instant::now();
            }
            Ok(())
        },
        |progress| {
            frames = progress.frames;
            reporter.report(progress);
            Ok(())
        },
    );
    reporter.finish();
    result?;
    let events = sink.n_events();
    sink.finish()?;

    if !args.quiet {
        emit(format!(
            "wrote {} ({} events from {frames} frames)",
            args.output,
            format_count(events as u64)
        ));
    }
    Ok(())
}

/// A window length in whole microseconds, rejected the way the Python bindings reject it.
pub(crate) fn duration_us(name: &str, ms: f64) -> Result<i64, Failure> {
    if ms.is_nan() {
        return Err(format!("{name} must be a number").into());
    }
    let us = ms * 1000.0;
    if us <= 0.0 {
        return Err(format!("{name} must be positive").into());
    }
    let rounded = us.round() as i64;
    if rounded < 1 {
        return Err(format!(
            "{name} of {us} us is below eventcv's one-microsecond timestamp resolution"
        )
        .into());
    }
    Ok(rounded)
}

pub(crate) fn parse_colormap(name: &str) -> Result<Colormap, Failure> {
    Colormap::from_name(name).ok_or_else(|| format!("unsupported colormap: {name}").into())
}

/// `1234567` → `1,234,567`, Python's `f"{n:,}"`.
pub(crate) fn format_count(value: u64) -> String {
    let digits = value.to_string();
    let mut out = String::with_capacity(digits.len() + digits.len() / 3);
    for (index, digit) in digits.chars().enumerate() {
        if index > 0 && (digits.len() - index).is_multiple_of(3) {
            out.push(',');
        }
        out.push(digit);
    }
    out
}

/// Python's `f"{x:g}"`: six significant digits, trailing zeros dropped, and exponent notation
/// outside `1e-4 <= |x| < 1e6` — so the summary line reads `30 fps`, not `30.0 fps`.
fn python_g(value: f64) -> String {
    if !value.is_finite() {
        return if value.is_nan() {
            "nan".to_owned()
        } else if value > 0.0 {
            "inf".to_owned()
        } else {
            "-inf".to_owned()
        };
    }
    if value == 0.0 {
        return if value.is_sign_negative() { "-0" } else { "0" }.to_owned();
    }
    let trim = |digits: String| {
        if digits.contains('.') {
            digits
                .trim_end_matches('0')
                .trim_end_matches('.')
                .to_owned()
        } else {
            digits
        }
    };
    // The exponent after rounding to six digits, which is what decides the notation.
    let scientific = format!("{value:.5e}");
    let (mantissa, exponent) = scientific
        .split_once('e')
        .expect("{:e} always has an exponent");
    let exponent: i32 = exponent.parse().expect("{:e} exponent is an integer");
    if (-4..6).contains(&exponent) {
        trim(format!("{value:.*}", (5 - exponent) as usize))
    } else {
        let sign = if exponent < 0 { '-' } else { '+' };
        format!("{}e{sign}{:02}", trim(mantissa.to_owned()), exponent.abs())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn counts_are_grouped_in_thousands() {
        assert_eq!(format_count(0), "0");
        assert_eq!(format_count(999), "999");
        assert_eq!(format_count(1000), "1,000");
        assert_eq!(format_count(1_234_567), "1,234,567");
    }

    #[test]
    fn python_g_matches_python() {
        // Expected values are Python's own `f"{x:g}"`.
        for (value, expected) in [
            (30.0, "30"),
            (29.97, "29.97"),
            (23.976023976, "23.976"),
            (0.5, "0.5"),
            (1e-5, "1e-05"),
            (0.0001, "0.0001"),
            (999999.0, "999999"),
            (999999.5, "1e+06"),
            (1234567.0, "1.23457e+06"),
            (-2.5, "-2.5"),
            (0.0, "0"),
        ] {
            assert_eq!(python_g(value), expected, "{value}");
        }
    }

    #[test]
    fn durations_are_validated_like_the_bindings() {
        assert_eq!(duration_us("dt", 30.0).unwrap(), 30_000);
        assert_eq!(
            duration_us("dt", 0.0).unwrap_err().message,
            "dt must be positive"
        );
        assert!(duration_us("dt", 0.0001)
            .unwrap_err()
            .message
            .contains("one-microsecond"));
        assert_eq!(
            duration_us("bin", f64::NAN).unwrap_err().message,
            "bin must be a number"
        );
    }

    #[test]
    fn source_options_reject_unknown_orders() {
        let source = Source {
            order: Some("ptxy".to_owned()),
            ..Source::default()
        };
        assert_eq!(
            source.options().unwrap_err().message,
            "unsupported order: ptxy"
        );
    }
}
