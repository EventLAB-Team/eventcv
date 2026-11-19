//! The event image in real pixels, for terminals with a graphics protocol (Kitty, Sixel or iTerm2):
//! the half-block view's two pixels a cell become the cell's full font size.

use std::cell::Cell;
use std::io::Cursor;
use std::num::NonZeroU16;
use std::sync::mpsc::{self, Receiver, SyncSender};

use eventcv_core::viz::Rgb8Image;
use icy_sixel::{sixel_encode, EncodeOptions};
use image::{ImageFormat, RgbImage};
use ratatui::buffer::{Buffer, CellDiffOption};
use ratatui::layout::{Rect, Size};
use ratatui::widgets::Widget;
use ratatui_image::picker::{Capability, Picker, ProtocolType};
use ratatui_image::protocol::kitty::Kitty;
use ratatui_image::protocol::Protocol;
use ratatui_image::{Image, Resize};

use super::draw::{downsample, fit};

/// Frames take these Kitty ids in turn, so each transmission replaces an old image instead of
/// piling up a new one at every refresh. The id is the placeholders' colour, so a fresh one makes
/// ratatui rewrite every cell; with a single id it rewrites only the first, and Kitty re-places the
/// new image on that row alone. Three, as up to two frames can finish between draws, so the frame
/// shown never shares an id with the one on screen.
const KITTY_IDS: [u32; 3] = [0x00ec_0001, 0x00ec_0002, 0x00ec_0003];

/// Encodes frames on a worker thread: a full-screen Sixel frame takes ~100 ms, which on the UI
/// thread would stall the keys and the camera drain. While the worker is busy, newer frames
/// replace the waiting one, so the view shows the latest it can keep up with.
pub(super) struct Pixels {
    jobs: SyncSender<(Rgb8Image, Rect)>,
    done: Receiver<(Encoded, Rect)>,
    shown: Option<(Encoded, Rect)>,
    /// Where the view last drew the image, which the next frame is encoded to fill.
    area: Cell<Rect>,
    /// The area the last frame sent was encoded for.
    sent: Rect,
    /// A frame is waiting to be sent: a new one was rendered, or the worker was busy.
    pub dirty: bool,
}

impl Pixels {
    pub fn new(picker: Picker) -> Self {
        let (jobs, inbox) = mpsc::sync_channel::<(Rgb8Image, Rect)>(1);
        let (outbox, done) = mpsc::channel();
        // Ends once the viewer drops `jobs`.
        std::thread::spawn(move || {
            for ((image, area), kitty_id) in inbox.into_iter().zip(KITTY_IDS.into_iter().cycle()) {
                if let Some(encoded) = encode(&picker, &image, area, kitty_id) {
                    if outbox.send(encoded).is_err() {
                        break;
                    }
                }
            }
        });
        Self {
            jobs,
            done,
            shown: None,
            area: Cell::new(Rect::ZERO),
            sent: Rect::ZERO,
            dirty: false,
        }
    }

    /// Picks up finished frames, and hands the worker `image` if it is new (or the view moved)
    /// and the worker is free.
    pub fn sync(&mut self, image: Option<&Rgb8Image>) {
        while let Ok(encoded) = self.done.try_recv() {
            self.shown = Some(encoded);
        }
        let area = self.area.get();
        let Some(image) = image.filter(|_| !area.is_empty() && (self.dirty || area != self.sent))
        else {
            return;
        };
        if self.jobs.try_send((image.clone(), area)).is_ok() {
            self.dirty = false;
            self.sent = area;
        }
    }
}

impl Widget for &Pixels {
    fn render(self, area: Rect, buf: &mut Buffer) {
        self.area.set(area);
        // A frame made for an older, larger area waits for its replacement rather than spill.
        let shown = self.shown.as_ref();
        match shown.filter(|(_, rect)| area.union(*rect) == area) {
            Some((Encoded::Widget(protocol), rect)) => Image::new(protocol).render(*rect, buf),
            // As the crate draws its own: the sequence in the first cell, the rest left alone.
            Some((Encoded::Inline(sequence), rect)) => {
                for (index, position) in rect.positions().enumerate() {
                    let cell = &mut buf[position];
                    match index {
                        0 => cell
                            .set_symbol(sequence)
                            .set_diff_option(CellDiffOption::ForcedWidth(NonZeroU16::MIN)),
                        _ => cell.set_diff_option(CellDiffOption::Skip),
                    };
                }
            }
            None => {}
        }
    }
}

enum Encoded {
    /// Drawn by ratatui-image: Kitty, which replaces the image in place by its id, and anything
    /// inside tmux, whose passthrough the crate wraps.
    Widget(Protocol),
    /// Our own Sixel or iTerm2 sequence. The crate's erases the area before each image, which the
    /// terminal shows as a blank frame between every two.
    Inline(String),
}

/// `image` scaled to fill `area` at the terminal's font size by the half-block view's
/// brightest-pixel rule, centred, and encoded for the terminal's protocol.
fn encode(
    picker: &Picker,
    image: &Rgb8Image,
    area: Rect,
    kitty_id: u32,
) -> Option<(Encoded, Rect)> {
    let font = picker.font_size();
    let (font_width, font_height) = (usize::from(font.width), usize::from(font.height));
    let (width, height) = fit(
        (image.width, image.height),
        (
            usize::from(area.width) * font_width,
            usize::from(area.height) * font_height,
        ),
    );
    if width == 0 || height == 0 {
        return None;
    }
    let pixels = downsample(image, width, height);
    let cells = Size::new(
        width.div_ceil(font_width) as u16,
        height.div_ceil(font_height) as u16,
    );
    let tmux = picker.tmux_detected();
    let encoded = match picker.protocol_type() {
        ProtocolType::Sixel if !tmux => {
            let rgba: Vec<u8> = pixels
                .iter()
                .flat_map(|&[r, g, b]| [r, g, b, 255])
                .collect();
            // No dithering: it would scatter speckle around every lone event.
            let options = EncodeOptions {
                diffusion: 0.0,
                ..Default::default()
            };
            Encoded::Inline(sixel_encode(&rgba, width, height, &options).ok()?)
        }
        protocol => {
            let image = RgbImage::from_raw(width as u32, height as u32, pixels.into_flattened())?;
            match protocol {
                ProtocolType::Iterm2 if !tmux => {
                    let mut png = Vec::new();
                    image
                        .write_to(&mut Cursor::new(&mut png), ImageFormat::Png)
                        .ok()?;
                    let mut sequence = format!(
                        "\x1b]1337;File=inline=1;size={};width={width}px;height={height}px;\
                         doNotMoveCursor=1:",
                        png.len()
                    );
                    base64_simd::STANDARD.encode_append(&png, &mut sequence);
                    sequence.push('\x07');
                    Encoded::Inline(sequence)
                }
                ProtocolType::Kitty => Encoded::Widget(Protocol::Kitty(
                    Kitty::new(
                        image.into(),
                        cells,
                        kitty_id,
                        tmux,
                        picker
                            .capabilities()
                            .contains(&Capability::KittyCompression),
                    )
                    .ok()?,
                )),
                _ => Encoded::Widget(
                    picker
                        .new_protocol(image.into(), cells, Resize::Fit(None))
                        .ok()?,
                ),
            }
        }
    };
    let centred = Rect::new(
        area.x + (area.width - cells.width) / 2,
        area.y + (area.height - cells.height) / 2,
        cells.width,
        cells.height,
    );
    Some((encoded, centred))
}
