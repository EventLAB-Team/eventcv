//! The event image in real pixels, for terminals with a graphics protocol (Kitty, Sixel or iTerm2):
//! the half-block view's two pixels a cell become the cell's full font size.

use std::cell::Cell;
use std::fmt::Write as _;
use std::io::{Cursor, Write as _};
use std::num::NonZeroU16;
use std::sync::mpsc::{self, Receiver, SyncSender};

use eventcv_core::viz::Rgb8Image;
use flate2::write::ZlibEncoder;
use flate2::Compression;
use icy_sixel::{sixel_encode, EncodeOptions};
use image::{ImageFormat, RgbImage};
use ratatui::buffer::{Buffer, CellDiffOption};
use ratatui::layout::{Rect, Size};
use ratatui::style::Color;
use ratatui::widgets::Widget;
use ratatui_image::picker::cap_parser::Parser;
use ratatui_image::picker::{Capability, Picker, ProtocolType};
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
            // Unicode placeholders, as the crate draws them: the transmission rides in the first
            // cell, every cell is coloured with the image id, and each row's first cell names its
            // row (the terminal infers the columns after it).
            Some((Encoded::Kitty { sequence, id }, rect)) => {
                let [_, red, green, blue] = id.to_be_bytes();
                let mut symbol = String::new();
                for (row, y) in (rect.top()..rect.bottom())
                    .enumerate()
                    .take(DIACRITICS.len())
                {
                    for x in rect.left()..rect.right() {
                        symbol.clear();
                        if (x, y) == (rect.x, rect.y) {
                            symbol.push_str(sequence);
                        }
                        symbol.push(PLACEHOLDER);
                        if x == rect.x {
                            // Row, column 0, and the id's top byte (0 for every `KITTY_IDS`).
                            symbol.extend([diacritic(row), diacritic(0), diacritic(0)]);
                        }
                        buf[(x, y)]
                            .set_symbol(&symbol)
                            .set_fg(Color::Rgb(red, green, blue))
                            .set_diff_option(CellDiffOption::ForcedWidth(NonZeroU16::MIN));
                    }
                }
            }
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
    /// Drawn by ratatui-image: Sixel and iTerm2 inside tmux, whose passthrough the crate wraps.
    Widget(Protocol),
    /// Our own Sixel or iTerm2 sequence. The crate's erases the area before each image, which the
    /// terminal shows as a blank frame between every two.
    Inline(String),
    /// Our own Kitty transmission, shown through placeholder cells coloured with image `id`. The
    /// crate's places the image at its pixel size, so filling the view meant upscaling it here.
    Kitty { sequence: String, id: u32 },
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
    let cells = Size::new(
        width.div_ceil(font_width) as u16,
        height.div_ceil(font_height) as u16,
    );
    let tmux = picker.tmux_detected();
    let encoded = match picker.protocol_type() {
        ProtocolType::Kitty => {
            let compress = picker
                .capabilities()
                .contains(&Capability::KittyCompression);
            kitty(image, (width, height), cells, kitty_id, compress, tmux)
        }
        ProtocolType::Sixel if !tmux => {
            let pixels = downsample(image, width, height);
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
            let pixels = downsample(image, width, height).into_flattened();
            let image = RgbImage::from_raw(width as u32, height as u32, pixels)?;
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

/// `image` as Kitty image `id`, with a virtual placement `cells` in size for Unicode placeholders.
/// `c`/`r` have the terminal fit it to those cells, so it goes at no more than its own resolution
/// and the terminal scales it up: filling a Retina view itself took ~5× the sensor's pixels, each
/// one deflated, base64'd and inflated again every frame. `display` is the size the view shows it
/// at; when that is smaller, it is shrunk here by the brightest-pixel rule as for the other
/// protocols. RGB (`f=24`), as the view has no alpha to send.
fn kitty(
    image: &Rgb8Image,
    display: (usize, usize),
    cells: Size,
    id: u32,
    compress: bool,
    tmux: bool,
) -> Encoded {
    let shrunk;
    let (pixels, (width, height)) = if display.0 < image.width && display.1 < image.height {
        shrunk = downsample(image, display.0, display.1).into_flattened();
        (&shrunk[..], display)
    } else {
        (&image.pixels[..], (image.width, image.height))
    };
    let deflated;
    let (payload, encoding) = if compress {
        // Fast: the frames are mostly black, so it costs little size and runs every frame.
        let mut encoder = ZlibEncoder::new(Vec::new(), Compression::fast());
        encoder.write_all(pixels).expect("deflating into a Vec");
        deflated = encoder.finish().expect("deflating into a Vec");
        (&deflated[..], "o=z,")
    } else {
        (pixels, "")
    };
    let (start, escape, end) = Parser::tmux_start_escape_end(tmux);
    // The protocol's limit: 4096 base64 characters a chunk.
    let chunks = payload.chunks(3072);
    let count = chunks.len();
    let mut sequence = String::with_capacity(payload.len() / 3 * 4 + 32 * count + 128);
    for (index, chunk) in chunks.enumerate() {
        sequence.push_str(start);
        let _ = write!(sequence, "{escape}_Gq=2,");
        if index == 0 {
            let _ = write!(
                sequence,
                "i={id},a=T,U=1,f=24,{encoding}t=d,s={width},v={height},c={},r={},",
                cells.width, cells.height
            );
        }
        let _ = write!(sequence, "m={};", u8::from(index + 1 < count));
        base64_simd::STANDARD.encode_append(chunk, &mut sequence);
        let _ = write!(sequence, "{escape}\\{end}");
    }
    Encoded::Kitty { sequence, id }
}

/// The character Kitty reserves for image placeholders.
const PLACEHOLDER: char = '\u{10EEEE}';

fn diacritic(index: usize) -> char {
    char::from_u32(DIACRITICS[index]).expect("a combining character")
}

/// Kitty's row/column diacritics, `rowcolumn-diacritics.txt` from its graphics protocol: the n-th
/// marks a placeholder cell's row (or column) n. Copied from ratatui-image (MIT), which keeps its
/// own private.
const DIACRITICS: [u32; 297] = [
    0x0305, 0x030D, 0x030E, 0x0310, 0x0312, 0x033D, 0x033E, 0x033F, 0x0346, 0x034A, 0x034B, 0x034C,
    0x0350, 0x0351, 0x0352, 0x0357, 0x035B, 0x0363, 0x0364, 0x0365, 0x0366, 0x0367, 0x0368, 0x0369,
    0x036A, 0x036B, 0x036C, 0x036D, 0x036E, 0x036F, 0x0483, 0x0484, 0x0485, 0x0486, 0x0487, 0x0592,
    0x0593, 0x0594, 0x0595, 0x0597, 0x0598, 0x0599, 0x059C, 0x059D, 0x059E, 0x059F, 0x05A0, 0x05A1,
    0x05A8, 0x05A9, 0x05AB, 0x05AC, 0x05AF, 0x05C4, 0x0610, 0x0611, 0x0612, 0x0613, 0x0614, 0x0615,
    0x0616, 0x0617, 0x0657, 0x0658, 0x0659, 0x065A, 0x065B, 0x065D, 0x065E, 0x06D6, 0x06D7, 0x06D8,
    0x06D9, 0x06DA, 0x06DB, 0x06DC, 0x06DF, 0x06E0, 0x06E1, 0x06E2, 0x06E4, 0x06E7, 0x06E8, 0x06EB,
    0x06EC, 0x0730, 0x0732, 0x0733, 0x0735, 0x0736, 0x073A, 0x073D, 0x073F, 0x0740, 0x0741, 0x0743,
    0x0745, 0x0747, 0x0749, 0x074A, 0x07EB, 0x07EC, 0x07ED, 0x07EE, 0x07EF, 0x07F0, 0x07F1, 0x07F3,
    0x0816, 0x0817, 0x0818, 0x0819, 0x081B, 0x081C, 0x081D, 0x081E, 0x081F, 0x0820, 0x0821, 0x0822,
    0x0823, 0x0825, 0x0826, 0x0827, 0x0829, 0x082A, 0x082B, 0x082C, 0x082D, 0x0951, 0x0953, 0x0954,
    0x0F82, 0x0F83, 0x0F86, 0x0F87, 0x135D, 0x135E, 0x135F, 0x17DD, 0x193A, 0x1A17, 0x1A75, 0x1A76,
    0x1A77, 0x1A78, 0x1A79, 0x1A7A, 0x1A7B, 0x1A7C, 0x1B6B, 0x1B6D, 0x1B6E, 0x1B6F, 0x1B70, 0x1B71,
    0x1B72, 0x1B73, 0x1CD0, 0x1CD1, 0x1CD2, 0x1CDA, 0x1CDB, 0x1CE0, 0x1DC0, 0x1DC1, 0x1DC3, 0x1DC4,
    0x1DC5, 0x1DC6, 0x1DC7, 0x1DC8, 0x1DC9, 0x1DCB, 0x1DCC, 0x1DD1, 0x1DD2, 0x1DD3, 0x1DD4, 0x1DD5,
    0x1DD6, 0x1DD7, 0x1DD8, 0x1DD9, 0x1DDA, 0x1DDB, 0x1DDC, 0x1DDD, 0x1DDE, 0x1DDF, 0x1DE0, 0x1DE1,
    0x1DE2, 0x1DE3, 0x1DE4, 0x1DE5, 0x1DE6, 0x1DFE, 0x20D0, 0x20D1, 0x20D4, 0x20D5, 0x20D6, 0x20D7,
    0x20DB, 0x20DC, 0x20E1, 0x20E7, 0x20E9, 0x20F0, 0x2CEF, 0x2CF0, 0x2CF1, 0x2DE0, 0x2DE1, 0x2DE2,
    0x2DE3, 0x2DE4, 0x2DE5, 0x2DE6, 0x2DE7, 0x2DE8, 0x2DE9, 0x2DEA, 0x2DEB, 0x2DEC, 0x2DED, 0x2DEE,
    0x2DEF, 0x2DF0, 0x2DF1, 0x2DF2, 0x2DF3, 0x2DF4, 0x2DF5, 0x2DF6, 0x2DF7, 0x2DF8, 0x2DF9, 0x2DFA,
    0x2DFB, 0x2DFC, 0x2DFD, 0x2DFE, 0x2DFF, 0xA66F, 0xA67C, 0xA67D, 0xA6F0, 0xA6F1, 0xA8E0, 0xA8E1,
    0xA8E2, 0xA8E3, 0xA8E4, 0xA8E5, 0xA8E6, 0xA8E7, 0xA8E8, 0xA8E9, 0xA8EA, 0xA8EB, 0xA8EC, 0xA8ED,
    0xA8EE, 0xA8EF, 0xA8F0, 0xA8F1, 0xAAB0, 0xAAB2, 0xAAB3, 0xAAB7, 0xAAB8, 0xAABE, 0xAABF, 0xAAC1,
    0xFE20, 0xFE21, 0xFE22, 0xFE23, 0xFE24, 0xFE25, 0xFE26, 0x10A0F, 0x10A38, 0x1D185, 0x1D186,
    0x1D187, 0x1D188, 0x1D189, 0x1D1AA, 0x1D1AB, 0x1D1AC, 0x1D1AD, 0x1D242, 0x1D243, 0x1D244,
];

#[cfg(test)]
mod tests {
    use std::io::Read;

    use super::*;

    fn sensor() -> Rgb8Image {
        let mut pixels = vec![0; 1280 * 720 * 3];
        pixels[..3].copy_from_slice(&[0xff, 0x49, 0x6c]);
        Rgb8Image {
            width: 1280,
            height: 720,
            pixels,
        }
    }

    /// The control keys of the first chunk, and the whole payload decoded back from base64.
    fn parse(sequence: &str) -> (String, Vec<u8>) {
        let mut keys = String::new();
        let mut payload = Vec::new();
        for chunk in sequence.split("\x1b_G").skip(1) {
            let (control, data) = chunk.split_once(';').unwrap();
            let data = data.strip_suffix("\x1b\\").unwrap();
            assert!(data.len() <= 4096, "chunk of {} characters", data.len());
            if keys.is_empty() {
                keys = control.to_owned();
            }
            payload.extend(base64_simd::STANDARD.decode_to_vec(data).unwrap());
        }
        (keys, payload)
    }

    #[test]
    fn kitty_sends_the_sensor_and_lets_the_terminal_scale_it_up() {
        let image = sensor();
        let Encoded::Kitty { sequence, id } =
            kitty(&image, (2972, 1672), Size::new(166, 44), 7, false, false)
        else {
            panic!("not a Kitty image");
        };
        let (keys, payload) = parse(&sequence);
        assert_eq!(id, 7);
        assert!(
            keys.contains("i=7,a=T,U=1,f=24,t=d,s=1280,v=720,c=166,r=44,"),
            "{keys}"
        );
        assert_eq!(payload, image.pixels);
    }

    #[test]
    fn kitty_shrinks_to_a_smaller_view_and_deflates() {
        let Encoded::Kitty { sequence, .. } =
            kitty(&sensor(), (160, 90), Size::new(20, 6), 7, true, false)
        else {
            panic!("not a Kitty image");
        };
        let (keys, payload) = parse(&sequence);
        assert!(keys.contains("f=24,o=z,t=d,s=160,v=90,c=20,r=6,"), "{keys}");
        let mut pixels = Vec::new();
        flate2::read::ZlibDecoder::new(&payload[..])
            .read_to_end(&mut pixels)
            .unwrap();
        assert_eq!(pixels.len(), 160 * 90 * 3);
        // The lone event survives the shrink.
        assert_eq!(pixels[..3], [0xff, 0x49, 0x6c]);
    }

    #[test]
    fn kitty_wraps_every_chunk_for_tmux() {
        let Encoded::Kitty { sequence, .. } =
            kitty(&sensor(), (2972, 1672), Size::new(166, 44), 7, false, true)
        else {
            panic!("not a Kitty image");
        };
        let chunks = sequence.matches("\x1bPtmux;\x1b\x1b_G").count();
        assert!(chunks > 1);
        assert_eq!(sequence.matches("\x1b\x1b\\\x1b\\").count(), chunks);
    }

    #[test]
    fn placeholders_cover_the_image_and_carry_its_id() {
        let (sequence, id) = ("<image>".to_owned(), 0x00ec_0002);
        let shown = Pixels {
            shown: Some((Encoded::Kitty { sequence, id }, Rect::new(2, 1, 3, 2))),
            ..Pixels::new(Picker::halfblocks())
        };
        let mut buf = Buffer::empty(Rect::new(0, 0, 6, 4));
        (&shown).render(buf.area, &mut buf);
        let first = [PLACEHOLDER, diacritic(0), diacritic(0), diacritic(0)];
        let second = [PLACEHOLDER, diacritic(1), diacritic(0), diacritic(0)];
        assert_eq!(
            buf[(2, 1)].symbol(),
            format!("<image>{}", String::from_iter(first))
        );
        assert_eq!(buf[(3, 1)].symbol(), PLACEHOLDER.to_string());
        assert_eq!(buf[(2, 2)].symbol(), String::from_iter(second));
        assert_eq!(buf[(4, 2)].fg, Color::Rgb(0xec, 0x00, 0x02));
        assert_eq!(buf[(1, 1)].symbol(), " ");
    }
}
