//! Animated export — a sequence of rendered frames written as APNG, GIF or MP4.
//!
//! The single-frame path is `io::write_png_frame`; this is its moving counterpart, and takes the
//! same [`Rgb8Image`] that path already produces so nothing is repacked on the way out.
//!
//! # Why these three formats
//!
//! **APNG** costs nothing: the `png` crate is already a dependency and supports animation, so an
//! animated PNG is written by the same encoder as a still one, losslessly and at full colour.
//!
//! **GIF** is the format that pastes into an issue or a README. It is limited to 256 colours per
//! frame, so a palette is built per frame with `color_quant` — for event visualisations, which are
//! mostly a colormap ramp over a dark ground, the loss is hard to see.
//!
//! **MP4** is handed to a system `ffmpeg` over a pipe rather than encoded in-process. Linking an
//! H.264 encoder is not free the way the other two are: x264 is GPL, which EventCV's Apache-2.0
//! cannot absorb, and openh264 carries patent obligations that are only cleanly discharged by
//! shipping Cisco's prebuilt binary. Piping raw frames to a tool the user already has keeps the
//! licence position clean, adds no dependency, and adds nothing to the wheel — at the cost of
//! requiring ffmpeg on `PATH`, which [`FfmpegEncoder::new`] reports clearly when it is missing.

use std::io::{BufWriter, Write};
use std::path::Path;
use std::process::{Child, Command, Stdio};

use crate::viz::Rgb8Image;

/// Frames per second for an exported animation.
///
/// Kept as a struct rather than a bare `f64` because the three encoders express timing differently —
/// APNG wants a rational delay, GIF wants hundredths of a second, ffmpeg wants a rate — and the
/// conversions are easy to get subtly wrong at the boundaries.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Fps(f64);

impl Fps {
    /// Clamps to a sane playable range: below 0.1 fps a GIF delay overflows its 16-bit field, and
    /// above 1000 fps the per-frame delay rounds to zero and players fall back to their own default.
    pub fn new(fps: f64) -> Self {
        Self(if fps.is_finite() {
            fps.clamp(0.1, 1000.0)
        } else {
            30.0
        })
    }

    pub fn get(self) -> f64 {
        self.0
    }

    /// Frame delay in hundredths of a second (GIF's unit), at least 1 so players don't
    /// substitute their own default for a zero delay.
    fn centiseconds(self) -> u16 {
        ((100.0 / self.0).round() as u64).clamp(1, u16::MAX as u64) as u16
    }
}

impl Default for Fps {
    fn default() -> Self {
        Self(30.0)
    }
}

/// Accepts rendered frames one at a time and finishes a file. Implemented per container so the
/// caller can stream a long recording without holding every frame in memory.
///
/// `Send` because finishing an MP4 blocks on ffmpeg exiting, and the Python bindings release the
/// GIL around that — a wait that can take seconds must not hold up every other thread.
pub trait AnimationEncoder: Send {
    /// Writes one frame. Frames must all share the dimensions of the first.
    fn write_frame(&mut self, image: &Rgb8Image) -> std::io::Result<()>;
    /// Finalises the file. Not folded into `Drop` because it can fail and the caller must see that.
    fn finish(self: Box<Self>) -> std::io::Result<()>;
}

/// The container an animation is written into, chosen from the output path's extension.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AnimationFormat {
    Apng,
    Gif,
    Mp4,
}

impl AnimationFormat {
    /// Picks a format from a file extension, case-insensitively. `.png` means APNG here: a caller
    /// asking for an animation with a `.png` path wants a moving one.
    pub fn from_path(path: &Path) -> Option<Self> {
        let ext = path.extension()?.to_str()?.to_ascii_lowercase();
        Some(match ext.as_str() {
            "apng" | "png" => Self::Apng,
            "gif" => Self::Gif,
            "mp4" | "m4v" | "mov" => Self::Mp4,
            _ => return None,
        })
    }
}

/// Opens an encoder for `path`, choosing the container from its extension.
///
/// `frames` is the total number to be written; APNG needs it up front because the count goes in a
/// header chunk before any frame data, and it cannot be backfilled on a streaming write.
pub fn encoder_for(
    path: &Path,
    frames: u32,
    width: usize,
    height: usize,
    fps: Fps,
) -> std::io::Result<Box<dyn AnimationEncoder + Send>> {
    let format = AnimationFormat::from_path(path).ok_or_else(|| {
        std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            format!(
                "cannot infer an animation format from {}: expected .gif, .mp4 or .png/.apng",
                path.display()
            ),
        )
    })?;
    Ok(match format {
        AnimationFormat::Apng => Box::new(ApngEncoder::new(path, frames, width, height, fps)?)
            as Box<dyn AnimationEncoder + Send>,
        AnimationFormat::Gif => Box::new(GifEncoder::new(path, width, height, fps)?),
        AnimationFormat::Mp4 => Box::new(FfmpegEncoder::new(path, width, height, fps)?),
    })
}

/// Animated PNG, via the same `png` encoder that writes single frames.
pub struct ApngEncoder {
    writer: png::Writer<BufWriter<std::fs::File>>,
}

impl ApngEncoder {
    pub fn new(
        path: &Path,
        frames: u32,
        width: usize,
        height: usize,
        fps: Fps,
    ) -> std::io::Result<Self> {
        let file = BufWriter::new(std::fs::File::create(path)?);
        let mut encoder = png::Encoder::new(file, width as u32, height as u32);
        encoder.set_color(png::ColorType::Rgb);
        encoder.set_depth(png::BitDepth::Eight);
        // `set_animated` rejects zero frames, and a zero-frame animation is not a thing we can write.
        encoder
            .set_animated(frames.max(1), 0)
            .map_err(to_io_error)?;
        // Delay is a rational: numerator/1000 seconds, which keeps non-integer frame rates exact.
        encoder
            .set_frame_delay((1000.0 / fps.get()).round() as u16, 1000)
            .map_err(to_io_error)?;
        let writer = encoder.write_header().map_err(to_io_error)?;
        Ok(Self { writer })
    }
}

impl AnimationEncoder for ApngEncoder {
    fn write_frame(&mut self, image: &Rgb8Image) -> std::io::Result<()> {
        self.writer
            .write_image_data(&image.pixels)
            .map_err(to_io_error)
    }

    fn finish(self: Box<Self>) -> std::io::Result<()> {
        self.writer.finish().map_err(to_io_error)
    }
}

/// Animated GIF, with a per-frame 256-colour palette.
pub struct GifEncoder {
    encoder: gif::Encoder<BufWriter<std::fs::File>>,
    delay: u16,
}

impl GifEncoder {
    pub fn new(path: &Path, width: usize, height: usize, fps: Fps) -> std::io::Result<Self> {
        let file = BufWriter::new(std::fs::File::create(path)?);
        let mut encoder =
            gif::Encoder::new(file, width as u16, height as u16, &[]).map_err(to_io_error)?;
        encoder
            .set_repeat(gif::Repeat::Infinite)
            .map_err(to_io_error)?;
        Ok(Self {
            encoder,
            delay: fps.centiseconds(),
        })
    }
}

impl AnimationEncoder for GifEncoder {
    fn write_frame(&mut self, image: &Rgb8Image) -> std::io::Result<()> {
        // `from_rgb` quantises to a 256-colour palette internally (color_quant's NeuQuant).
        let mut frame =
            gif::Frame::from_rgb(image.width as u16, image.height as u16, &image.pixels);
        frame.delay = self.delay;
        self.encoder.write_frame(&frame).map_err(to_io_error)
    }

    fn finish(self: Box<Self>) -> std::io::Result<()> {
        // Dropping the encoder writes the trailer; there is no fallible explicit finish.
        drop(self.encoder);
        Ok(())
    }
}

/// H.264 in MP4, by piping raw RGB frames to a system `ffmpeg`.
pub struct FfmpegEncoder {
    child: Child,
}

impl FfmpegEncoder {
    pub fn new(path: &Path, width: usize, height: usize, fps: Fps) -> std::io::Result<Self> {
        let child = Command::new("ffmpeg")
            .args(["-hide_banner", "-loglevel", "error", "-y"])
            .args(["-f", "rawvideo", "-pix_fmt", "rgb24"])
            .args(["-s", &format!("{width}x{height}")])
            .args(["-r", &format!("{}", fps.get())])
            .args(["-i", "-"])
            // yuv420p rather than ffmpeg's default for RGB input: it is what QuickTime, PowerPoint
            // and most browsers will actually play. Dimensions must be even for 4:2:0 chroma, so
            // pad rather than fail on an odd-sized sensor.
            .args(["-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2"])
            .args(["-c:v", "libx264", "-pix_fmt", "yuv420p"])
            .arg(path)
            .stdin(Stdio::piped())
            .stdout(Stdio::null())
            .stderr(Stdio::inherit())
            .spawn()
            .map_err(|error| {
                missing_ffmpeg(
                    error,
                    "writing .mp4 needs ffmpeg on PATH",
                    "Write .gif or .apng instead to avoid the dependency.",
                )
            })?;
        Ok(Self { child })
    }
}

/// Rewrites a spawn failure into an actionable message when `ffmpeg` simply is not installed.
///
/// Shared by the encoder and the decoder because "command not found" is the overwhelmingly common
/// failure for both, and the bare `NotFound` the OS returns names neither the tool nor the fix.
/// `alternative` is the way out specific to the caller — there is one for writing, none for reading.
fn missing_ffmpeg(error: std::io::Error, what: &str, alternative: &str) -> std::io::Error {
    if error.kind() != std::io::ErrorKind::NotFound {
        return error;
    }
    std::io::Error::new(
        std::io::ErrorKind::NotFound,
        format!(
            "{what} (macOS: `brew install ffmpeg`, Debian/Ubuntu: `apt install ffmpeg`, \
             conda: `conda install -c conda-forge ffmpeg`). {alternative}"
        )
        .trim_end()
        .to_owned(),
    )
}

/// Waits for a finished ffmpeg and turns a non-zero exit into an error.
///
/// Its stderr is inherited rather than captured, so the diagnostics have already reached the user's
/// terminal by the time this runs — hence pointing at them rather than repeating them.
fn wait_for_ffmpeg(child: &mut Child) -> std::io::Result<()> {
    let status = child.wait()?;
    if status.success() {
        Ok(())
    } else {
        Err(std::io::Error::other(format!(
            "ffmpeg exited with {status} — its error output is above"
        )))
    }
}

impl AnimationEncoder for FfmpegEncoder {
    fn write_frame(&mut self, image: &Rgb8Image) -> std::io::Result<()> {
        let stdin = self.child.stdin.as_mut().ok_or_else(|| {
            std::io::Error::new(std::io::ErrorKind::BrokenPipe, "ffmpeg stdin was closed")
        })?;
        stdin.write_all(&image.pixels)
    }

    fn finish(mut self: Box<Self>) -> std::io::Result<()> {
        // Closing stdin is what tells ffmpeg the stream ended; without it, wait() deadlocks.
        drop(self.child.stdin.take());
        wait_for_ffmpeg(&mut self.child)
    }
}

/// What `ffprobe` reports about a video before any of it is decoded.
///
/// Raw `rgb24` on a pipe carries no framing at all — just a byte stream — so the decoder cannot know
/// where one frame ends without being told the dimensions first. That is the only reason this exists.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct VideoInfo {
    pub width: usize,
    pub height: usize,
    pub fps: f64,
}

impl VideoInfo {
    /// Probes `path` with `ffprobe`.
    pub fn probe(path: &Path) -> std::io::Result<Self> {
        let output = Command::new("ffprobe")
            .args(["-v", "error", "-select_streams", "v:0"])
            .args(["-show_entries", "stream=width,height,r_frame_rate"])
            .args(["-of", "csv=p=0"])
            .arg(path)
            .output()
            .map_err(|error| missing_ffmpeg(error, "reading video needs ffmpeg on PATH", ""))?;
        if !output.status.success() {
            return Err(std::io::Error::other(format!(
                "ffprobe could not read {}: {}",
                path.display(),
                String::from_utf8_lossy(&output.stderr).trim()
            )));
        }
        Self::parse(&String::from_utf8_lossy(&output.stdout), path)
    }

    /// Parses ffprobe's `width,height,num/den` CSV. Split out so the parsing is testable without
    /// having ffprobe installed.
    fn parse(text: &str, path: &Path) -> std::io::Result<Self> {
        let malformed = || {
            std::io::Error::other(format!(
                "could not read the video stream of {} — is it a video file?",
                path.display()
            ))
        };
        let line = text
            .lines()
            .find(|line| !line.trim().is_empty())
            .ok_or_else(malformed)?;
        let mut fields = line.trim().split(',');
        let width: usize = fields
            .next()
            .ok_or_else(malformed)?
            .trim()
            .parse()
            .map_err(|_| malformed())?;
        let height: usize = fields
            .next()
            .ok_or_else(malformed)?
            .trim()
            .parse()
            .map_err(|_| malformed())?;
        // The frame rate is a rational like `30000/1001`, not a decimal.
        let rate = fields.next().ok_or_else(malformed)?.trim();
        let (num, den) = rate.split_once('/').unwrap_or((rate, "1"));
        let num: f64 = num.parse().map_err(|_| malformed())?;
        let den: f64 = den.parse().unwrap_or(1.0);
        let fps = if den > 0.0 && num > 0.0 {
            num / den
        } else {
            30.0
        };
        if width == 0 || height == 0 {
            return Err(malformed());
        }
        Ok(Self { width, height, fps })
    }
}

/// Decodes a video into [`Rgb8Image`] frames by pulling raw `rgb24` from a system `ffmpeg`.
///
/// The mirror of [`FfmpegEncoder`], and deliberately a *pulling* iterator rather than a callback:
/// the simulator consumes frames in pairs and needs to hold one back, which a push API makes awkward.
pub struct FfmpegDecoder {
    child: Child,
    info: VideoInfo,
    frame_bytes: usize,
    buffer: Vec<u8>,
    finished: bool,
}

impl FfmpegDecoder {
    /// Opens `path` for decoding. `scale` optionally resizes on the way out, which is far cheaper
    /// than decoding at full resolution and downsampling afterwards.
    pub fn open(path: &Path, scale: Option<(usize, usize)>) -> std::io::Result<Self> {
        let probed = VideoInfo::probe(path)?;
        let info = match scale {
            Some((width, height)) if width > 0 && height > 0 => VideoInfo {
                width,
                height,
                fps: probed.fps,
            },
            _ => probed,
        };
        let mut command = Command::new("ffmpeg");
        command
            .args(["-hide_banner", "-loglevel", "error"])
            .arg("-i")
            .arg(path);
        if scale.is_some() {
            command.args(["-vf", &format!("scale={}:{}", info.width, info.height)]);
        }
        let child = command
            .args(["-f", "rawvideo", "-pix_fmt", "rgb24", "-"])
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit())
            .spawn()
            .map_err(|error| missing_ffmpeg(error, "reading video needs ffmpeg on PATH", ""))?;
        Ok(Self {
            child,
            info,
            frame_bytes: info.width * info.height * 3,
            buffer: vec![0; info.width * info.height * 3],
            finished: false,
        })
    }

    pub fn info(&self) -> VideoInfo {
        self.info
    }

    /// Pulls the next frame, or `None` at the end of the stream.
    ///
    /// A partial read at the end is treated as end-of-stream rather than an error: ffmpeg closing
    /// the pipe mid-frame is how a truncated source presents, and there is nothing useful to do with
    /// half a frame.
    pub fn next_frame(&mut self) -> std::io::Result<Option<Rgb8Image>> {
        if self.finished {
            return Ok(None);
        }
        let stdout = self.child.stdout.as_mut().ok_or_else(|| {
            std::io::Error::new(std::io::ErrorKind::BrokenPipe, "ffmpeg stdout was closed")
        })?;
        match read_exact_or_eof(stdout, &mut self.buffer[..self.frame_bytes])? {
            true => Ok(Some(Rgb8Image {
                width: self.info.width,
                height: self.info.height,
                pixels: self.buffer[..self.frame_bytes].to_vec(),
            })),
            false => {
                self.finished = true;
                wait_for_ffmpeg(&mut self.child)?;
                Ok(None)
            }
        }
    }
}

impl Drop for FfmpegDecoder {
    fn drop(&mut self) {
        // A caller that stops early (`max_frames`) leaves ffmpeg writing into a pipe nobody reads.
        // Killing it is the only way to avoid a process stuck on a full pipe buffer.
        if !self.finished {
            let _ = self.child.kill();
            let _ = self.child.wait();
        }
    }
}

/// Fills `buffer` completely, returning `false` if the stream ended before any byte was read.
/// A short read after at least one byte is a truncated frame and reports end-of-stream too.
fn read_exact_or_eof(reader: &mut impl std::io::Read, buffer: &mut [u8]) -> std::io::Result<bool> {
    let mut filled = 0;
    while filled < buffer.len() {
        match reader.read(&mut buffer[filled..]) {
            Ok(0) => return Ok(false),
            Ok(n) => filled += n,
            Err(error) if error.kind() == std::io::ErrorKind::Interrupted => {}
            Err(error) => return Err(error),
        }
    }
    Ok(true)
}

fn to_io_error<E: std::fmt::Display>(error: E) -> std::io::Error {
    std::io::Error::other(error.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::viz::Rgb8Image;

    fn frame(width: usize, height: usize, shade: u8) -> Rgb8Image {
        Rgb8Image {
            width,
            height,
            pixels: vec![shade; width * height * 3],
        }
    }

    fn temp_path(name: &str) -> std::path::PathBuf {
        let mut path = std::env::temp_dir();
        path.push(format!(
            "eventcv-video-test-{}-{}",
            std::process::id(),
            name
        ));
        path
    }

    #[test]
    fn format_is_read_from_the_extension() {
        let cases = [
            ("a.gif", Some(AnimationFormat::Gif)),
            ("a.GIF", Some(AnimationFormat::Gif)),
            ("a.png", Some(AnimationFormat::Apng)),
            ("a.apng", Some(AnimationFormat::Apng)),
            ("a.mp4", Some(AnimationFormat::Mp4)),
            ("a.mov", Some(AnimationFormat::Mp4)),
            ("a.txt", None),
            ("a", None),
        ];
        for (name, expected) in cases {
            assert_eq!(
                AnimationFormat::from_path(Path::new(name)),
                expected,
                "{name}"
            );
        }
    }

    #[test]
    fn fps_clamps_and_converts() {
        assert_eq!(Fps::new(f64::NAN).get(), 30.0);
        assert_eq!(Fps::new(0.0).get(), 0.1);
        assert_eq!(Fps::new(1e9).get(), 1000.0);
        assert_eq!(Fps::new(100.0).centiseconds(), 1); // never zero
        assert_eq!(Fps::new(50.0).centiseconds(), 2);
        assert_eq!(Fps::new(10.0).centiseconds(), 10);
    }

    #[test]
    fn apng_writes_a_multi_frame_file() {
        let path = temp_path("apng.png");
        let mut encoder: Box<dyn AnimationEncoder> =
            Box::new(ApngEncoder::new(&path, 3, 4, 4, Fps::new(10.0)).unwrap());
        for shade in [0u8, 128, 255] {
            encoder.write_frame(&frame(4, 4, shade)).unwrap();
        }
        encoder.finish().unwrap();

        let bytes = std::fs::read(&path).unwrap();
        assert_eq!(&bytes[..8], b"\x89PNG\r\n\x1a\n");
        // acTL is the APNG animation-control chunk — its presence is what makes this animated
        // rather than three frames silently collapsed into one still image.
        assert!(bytes.windows(4).any(|w| w == b"acTL"));
        assert!(bytes.windows(4).any(|w| w == b"fcTL"));
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn gif_writes_a_multi_frame_file() {
        let path = temp_path("gif.gif");
        let mut encoder: Box<dyn AnimationEncoder> =
            Box::new(GifEncoder::new(&path, 4, 4, Fps::new(10.0)).unwrap());
        for shade in [0u8, 128, 255] {
            encoder.write_frame(&frame(4, 4, shade)).unwrap();
        }
        encoder.finish().unwrap();

        let bytes = std::fs::read(&path).unwrap();
        assert_eq!(&bytes[..6], b"GIF89a");
        assert_eq!(bytes.last(), Some(&0x3B)); // trailer, so the file is complete
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn video_info_parses_ffprobe_csv() {
        let path = Path::new("clip.mp4");
        // Integer rate.
        let info = VideoInfo::parse("64,48,30/1\n", path).unwrap();
        assert_eq!((info.width, info.height), (64, 48));
        assert!((info.fps - 30.0).abs() < 1e-9);
        // NTSC rational — the case a naive `parse::<f64>()` gets wrong.
        let ntsc = VideoInfo::parse("1920,1080,30000/1001", path).unwrap();
        assert!((ntsc.fps - 29.97).abs() < 0.01);
        // Bare rate with no denominator.
        assert!((VideoInfo::parse("8,8,25", path).unwrap().fps - 25.0).abs() < 1e-9);
    }

    #[test]
    fn video_info_rejects_nonsense() {
        let path = Path::new("notes.txt");
        for text in ["", "\n", "not,a,video", "0,0,30/1"] {
            assert!(VideoInfo::parse(text, path).is_err(), "{text:?}");
        }
    }

    #[test]
    fn decoder_reads_back_every_frame_it_was_given() {
        // Round trip through the encoder so this needs no fixture on disk: write a known number of
        // frames, decode them back, and check the count and dimensions survive.
        if Command::new("ffmpeg").arg("-version").output().is_err() {
            return; // covered by the encoder's own skip; nothing to assert without ffmpeg
        }
        let path = temp_path("roundtrip.mp4");
        let mut encoder: Box<dyn AnimationEncoder + Send> =
            Box::new(FfmpegEncoder::new(&path, 32, 24, Fps::new(10.0)).unwrap());
        for shade in [0u8, 60, 120, 180, 240] {
            encoder.write_frame(&frame(32, 24, shade)).unwrap();
        }
        encoder.finish().unwrap();

        let mut decoder = FfmpegDecoder::open(&path, None).unwrap();
        assert_eq!((decoder.info().width, decoder.info().height), (32, 24));
        let mut decoded = 0;
        while let Some(image) = decoder.next_frame().unwrap() {
            assert_eq!(image.pixels.len(), 32 * 24 * 3);
            decoded += 1;
        }
        assert_eq!(decoded, 5);
        // Exhausted decoders keep returning None rather than erroring.
        assert!(decoder.next_frame().unwrap().is_none());
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn decoder_can_scale_on_the_way_out() {
        if Command::new("ffmpeg").arg("-version").output().is_err() {
            return;
        }
        let path = temp_path("scaled.mp4");
        let mut encoder: Box<dyn AnimationEncoder + Send> =
            Box::new(FfmpegEncoder::new(&path, 64, 64, Fps::new(10.0)).unwrap());
        encoder.write_frame(&frame(64, 64, 128)).unwrap();
        encoder.finish().unwrap();

        let mut decoder = FfmpegDecoder::open(&path, Some((16, 16))).unwrap();
        let image = decoder.next_frame().unwrap().expect("one frame");
        assert_eq!((image.width, image.height), (16, 16));
        assert_eq!(image.pixels.len(), 16 * 16 * 3);
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn read_exact_or_eof_reports_a_clean_end() {
        let mut full = [0u8; 4];
        assert!(read_exact_or_eof(&mut &b"abcd"[..], &mut full).unwrap());
        assert_eq!(&full, b"abcd");
        // Nothing at all is a clean end...
        assert!(!read_exact_or_eof(&mut &b""[..], &mut full).unwrap());
        // ...and so is a truncated frame, which is what a cut-off source looks like.
        assert!(!read_exact_or_eof(&mut &b"ab"[..], &mut full).unwrap());
    }

    #[test]
    fn unknown_extension_is_rejected_with_a_useful_message() {
        let error = encoder_for(Path::new("out.avi"), 1, 4, 4, Fps::default())
            .err()
            .expect("an unknown extension must not open an encoder");
        assert_eq!(error.kind(), std::io::ErrorKind::InvalidInput);
        assert!(error.to_string().contains(".gif"));
    }

    #[test]
    fn missing_ffmpeg_names_the_fix() {
        // Only meaningful where ffmpeg is genuinely absent; where it exists this asserts nothing,
        // which is the right trade for a test that must pass on any machine.
        if Command::new("ffmpeg").arg("-version").output().is_err() {
            let error = FfmpegEncoder::new(&temp_path("x.mp4"), 4, 4, Fps::default())
                .err()
                .expect("spawning ffmpeg must fail when it is not installed");
            assert_eq!(error.kind(), std::io::ErrorKind::NotFound);
            assert!(error.to_string().contains("ffmpeg"));
            assert!(error.to_string().contains(".gif"));
        }
    }
}
