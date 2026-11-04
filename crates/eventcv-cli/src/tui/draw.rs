//! Drawing an image into terminal cells: each cell is an upper half block `▀` whose foreground is
//! one pixel and background the pixel below it, so a cell carries two square-ish pixels.

use eventcv_core::viz::Rgb8Image;
use ratatui::buffer::Buffer;
use ratatui::layout::Rect;
use ratatui::style::Color;
use ratatui::widgets::Widget;

/// An [`Rgb8Image`] scaled to fit its area (aspect kept, centred), drawn in half blocks.
pub(super) struct HalfBlock<'a> {
    pub image: &'a Rgb8Image,
    /// 24-bit colour; otherwise the nearest xterm-256 colour, for terminals without it.
    pub truecolor: bool,
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
        let color = |rgb: [u8; 3]| {
            if self.truecolor {
                Color::Rgb(rgb[0], rgb[1], rgb[2])
            } else {
                Color::Indexed(xterm256(rgb))
            }
        };
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

/// The nearest colour in the xterm-256 6×6×6 cube.
fn xterm256([r, g, b]: [u8; 3]) -> u8 {
    // The cube's levels are 0, 95, 135, 175, 215, 255; these are the midpoints between them.
    let level = |value: u8| match value {
        0..48 => 0,
        48..115 => 1,
        _ => (value - 35) / 40,
    };
    16 + 36 * level(r) + 6 * level(g) + level(b)
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
            truecolor: true,
        }
        .render(buf.area, &mut buf);
        assert_eq!(buf[(0, 0)].bg, Color::Rgb(200, 200, 200));
        assert_eq!(buf[(0, 1)].fg, Color::Rgb(200, 200, 200));
        assert_eq!(buf[(0, 1)].bg, Color::Rgb(0, 0, 0));
    }

    #[test]
    fn xterm256_hits_the_cube_corners() {
        assert_eq!(xterm256([0, 0, 0]), 16);
        assert_eq!(xterm256([255, 255, 255]), 231);
        assert_eq!(xterm256([255, 0, 0]), 196);
        assert_eq!(xterm256([0x27, 0xc2, 0xff]), 16 + 6 * 3 + 5);
    }
}
