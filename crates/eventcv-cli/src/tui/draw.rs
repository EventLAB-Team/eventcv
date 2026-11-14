//! The dashboard's widgets: the event image in half blocks, a braille rate graph, and the bars.

use eventcv_core::viz::Rgb8Image;
use ratatui::buffer::Buffer;
use ratatui::layout::Rect;
use ratatui::style::{Color, Style};
use ratatui::text::Span;
use ratatui::widgets::Widget;

use super::theme::Theme;

/// An [`Rgb8Image`] scaled to fit its area (aspect kept, centred). Each cell is an upper half block
/// `▀` whose foreground is one pixel and background the pixel below it, so a cell carries two
/// square-ish pixels.
pub(super) struct HalfBlock<'a> {
    pub image: &'a Rgb8Image,
    pub theme: &'a Theme,
}

impl Widget for HalfBlock<'_> {
    fn render(self, area: Rect, buf: &mut Buffer) {
        let (width, height) = fit(
            (self.image.width, self.image.height),
            (usize::from(area.width), usize::from(area.height) * 2),
        );
        if width == 0 || height == 0 {
            return;
        }
        let pixels = downsample(self.image, width, height);
        let rows = height.div_ceil(2);
        let left = area.x + (area.width - width as u16) / 2;
        let top = area.y + (area.height - rows as u16) / 2;
        let color = |rgb| self.theme.color(rgb);
        for row in 0..rows {
            for column in 0..width {
                let upper = pixels[2 * row * width + column];
                // An odd pixel height leaves the last row's lower half off the image.
                let lower = pixels
                    .get((2 * row + 1) * width + column)
                    .copied()
                    .unwrap_or_default();
                buf[(left + column as u16, top + row as u16)]
                    .set_char('▀')
                    .set_fg(color(upper))
                    .set_bg(color(lower));
            }
        }
    }
}

/// The largest `(width, height)` with `image`'s aspect that fits in `bounds`, in pixels.
fn fit(image: (usize, usize), bounds: (usize, usize)) -> (usize, usize) {
    let ((width, height), (max_width, max_height)) = (image, bounds);
    if width == 0 || height == 0 || max_width == 0 || max_height == 0 {
        return (0, 0);
    }
    let scale = (max_width as f64 / width as f64).min(max_height as f64 / height as f64);
    (
        ((width as f64 * scale) as usize).clamp(1, max_width),
        ((height as f64 * scale) as usize).clamp(1, max_height),
    )
}

/// Resizes to `width × height` by keeping the **brightest** pixel of each source box. An average
/// would wash sparse events out: shrinking a 1280×720 sensor into a terminal folds ~64 pixels into
/// each one, and a lone event would come out at 1/64 of its brightness. Upscaling repeats pixels.
fn downsample(image: &Rgb8Image, width: usize, height: usize) -> Vec<[u8; 3]> {
    let span = |index: usize, out: usize, source: usize| {
        let start = index * source / out;
        start..((index + 1) * source / out).max(start + 1)
    };
    let mut pixels = Vec::with_capacity(width * height);
    for y in 0..height {
        let rows = span(y, height, image.height);
        for x in 0..width {
            let columns = span(x, width, image.width);
            let mut brightest = [0u8; 3];
            for source_y in rows.clone() {
                for source_x in columns.clone() {
                    let offset = 3 * (source_y * image.width + source_x);
                    let rgb = [
                        image.pixels[offset],
                        image.pixels[offset + 1],
                        image.pixels[offset + 2],
                    ];
                    let sum = |rgb: [u8; 3]| rgb.iter().map(|&c| u16::from(c)).sum::<u16>();
                    if sum(rgb) > sum(brightest) {
                        brightest = rgb;
                    }
                }
            }
            pixels.push(brightest);
        }
    }
    pixels
}

/// A filled braille area graph with the newest sample at the right edge, as btop draws them: two
/// samples per cell across, four dots per cell up, shaded from OFF blue at the floor to ON red at
/// the top so a burst reads as heat.
pub(super) struct RateGraph<'a> {
    pub samples: &'a [u64],
    pub theme: &'a Theme,
}

/// Braille dot bits for the left and right halves of a cell, bottom dot first.
const LEFT_DOTS: [u32; 4] = [0x40, 0x04, 0x02, 0x01];
const RIGHT_DOTS: [u32; 4] = [0x80, 0x20, 0x10, 0x08];

impl Widget for RateGraph<'_> {
    fn render(self, area: Rect, buf: &mut Buffer) {
        let (columns, rows) = (usize::from(area.width), usize::from(area.height));
        if columns == 0 || rows == 0 {
            return;
        }
        let shown = &self.samples[self.samples.len().saturating_sub(2 * columns)..];
        let peak = shown.iter().copied().max().unwrap_or(0).max(1) as f64;
        // Dots lit under each sample. Anything above zero shows at least one, so a quiet stretch
        // is told apart from a dead one.
        let levels: Vec<usize> = shown
            .iter()
            .map(|&sample| match sample {
                0 => 0,
                _ => ((sample as f64 / peak) * (4 * rows) as f64)
                    .round()
                    .max(1.0) as usize,
            })
            .collect();
        let offset = 2 * columns - levels.len();
        for row in 0..rows {
            let style = Style::new().fg(self.theme.heat((row as f64 + 0.5) / rows as f64));
            for column in 0..columns {
                let mut bits = 0;
                for (side, dots) in [LEFT_DOTS, RIGHT_DOTS].iter().enumerate() {
                    let level = (2 * column + side)
                        .checked_sub(offset)
                        .map_or(0, |index| levels[index]);
                    let lit = level.saturating_sub(4 * row).min(4);
                    bits |= dots[..lit].iter().fold(0, |acc, dot| acc | dot);
                }
                if bits != 0 {
                    let symbol = char::from_u32(0x2800 + bits).expect("braille block");
                    buf[(area.x + column as u16, area.bottom() - 1 - row as u16)]
                        .set_char(symbol)
                        .set_style(style);
                }
            }
        }
    }
}

/// `███░░░░` — `share` of `width` cells filled in `color`, the rest a faint track.
pub(super) fn meter(share: f64, width: usize, color: Color, theme: &Theme) -> [Span<'static>; 2] {
    let filled = ((share.clamp(0.0, 1.0) * width as f64).round() as usize).min(width);
    [
        Span::styled("█".repeat(filled), Style::new().fg(color)),
        Span::styled("░".repeat(width - filled), Style::new().fg(theme.border)),
    ]
}

/// `━━━━●─────` — a playhead `share` of the way along `width` cells.
pub(super) fn scrub(share: f64, width: usize, theme: &Theme) -> [Span<'static>; 3] {
    let width = width.max(1);
    let head = (share.clamp(0.0, 1.0) * (width - 1) as f64).round() as usize;
    [
        Span::styled("━".repeat(head), Style::new().fg(theme.on)),
        Span::styled("●", theme.value()),
        Span::styled("─".repeat(width - 1 - head), Style::new().fg(theme.border)),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    fn image(width: usize, height: usize) -> Rgb8Image {
        Rgb8Image {
            width,
            height,
            pixels: vec![0; width * height * 3],
        }
    }

    #[test]
    fn fit_keeps_the_aspect() {
        assert_eq!(fit((1280, 720), (160, 90)), (160, 90));
        assert_eq!(fit((1280, 720), (200, 90)), (160, 90));
        assert_eq!(fit((346, 260), (80, 48)), (63, 48));
        assert_eq!(fit((10, 10), (0, 5)), (0, 0));
    }

    #[test]
    fn a_lone_event_survives_downsampling() {
        let mut image = image(1280, 720);
        let offset = 3 * (300 * 1280 + 500);
        image.pixels[offset..offset + 3].copy_from_slice(&[0xff, 0x49, 0x6c]);
        let small = downsample(&image, 160, 90);
        assert_eq!(small.iter().filter(|&&rgb| rgb != [0; 3]).count(), 1);
        assert_eq!(small[(300 / 8) * 160 + 500 / 8], [0xff, 0x49, 0x6c]);
    }

    #[test]
    fn upscaling_repeats_pixels() {
        let mut image = image(2, 1);
        image.pixels[3..6].copy_from_slice(&[9, 9, 9]);
        assert_eq!(
            downsample(&image, 4, 2),
            [[0; 3], [0; 3], [9; 3], [9; 3]].repeat(2)
        );
    }

    #[test]
    fn odd_heights_leave_the_last_lower_half_black() {
        let mut image = image(1, 3);
        image.pixels.fill(200);
        let mut buf = Buffer::empty(Rect::new(0, 0, 1, 2));
        HalfBlock {
            image: &image,
            theme: &Theme::new(true),
        }
        .render(buf.area, &mut buf);
        assert_eq!(buf[(0, 0)].bg, Color::Rgb(200, 200, 200));
        assert_eq!(buf[(0, 1)].fg, Color::Rgb(200, 200, 200));
        assert_eq!(buf[(0, 1)].bg, Color::Rgb(0, 0, 0));
    }

    fn graph(samples: &[u64], width: u16, height: u16) -> Buffer {
        let mut buf = Buffer::empty(Rect::new(0, 0, width, height));
        RateGraph {
            samples,
            theme: &Theme::new(true),
        }
        .render(buf.area, &mut buf);
        buf
    }

    #[test]
    fn the_peak_fills_its_column_and_the_newest_sits_right() {
        let buf = graph(&[0, 0, 0, 8], 2, 2);
        // The peak is the right half of the last cell, full height: dots 4, 5, 6 and 8.
        assert_eq!(buf[(1, 0)].symbol(), "⢸");
        assert_eq!(buf[(1, 1)].symbol(), "⢸");
        assert_eq!(buf[(0, 1)].symbol(), " ");
    }

    #[test]
    fn a_small_sample_still_shows() {
        let buf = graph(&[1, 1000], 1, 2);
        // The left half's one bottom dot, beside a full right half.
        assert_eq!(buf[(0, 1)].symbol(), "⣸");
    }

    #[test]
    fn bars_fill_their_share() {
        let theme = Theme::new(true);
        let [on, track] = meter(0.25, 8, theme.on, &theme);
        assert_eq!(
            (on.content.as_ref(), track.content.as_ref()),
            ("██", "░░░░░░")
        );
        let [done, head, rest] = scrub(0.5, 5, &theme);
        let bar = format!("{}{}{}", done.content, head.content, rest.content);
        assert_eq!(bar, "━━●──");
    }
}
