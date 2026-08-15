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
                if error.kind() == std::io::ErrorKind::NotFound {
                    std::io::Error::new(
                        std::io::ErrorKind::NotFound,
                        "writing .mp4 needs ffmpeg on PATH (macOS: `brew install ffmpeg`, \
                         Debian/Ubuntu: `apt install ffmpeg`, conda: `conda install -c conda-forge \
                         ffmpeg`). Write .gif or .apng instead to avoid the dependency.",
                    )
                } else {
                    error
                }
            })?;
        Ok(Self { child })
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
        let status = self.child.wait()?;
        if status.success() {
            Ok(())
        } else {
            Err(std::io::Error::other(format!(
                "ffmpeg exited with {status} — its error output is above"
            )))
        }
    }
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
