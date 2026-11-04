//! The EventCV palette: the logo's red and blue, which are also the raw view's ON and OFF colours,
//! on quiet slate chrome so the events stay the brightest thing on screen.

use ratatui::style::{Color, Modifier, Style};

const ON: [u8; 3] = [0xff, 0x49, 0x6c];
const OFF: [u8; 3] = [0x27, 0xc2, 0xff];
const TEXT: [u8; 3] = [0xe5, 0xe9, 0xf0];
const MUTED: [u8; 3] = [0x7b, 0x83, 0x94];
const BORDER: [u8; 3] = [0x3b, 0x42, 0x52];
const GOOD: [u8; 3] = [0x8b, 0xd5, 0xa0];
const WARN: [u8; 3] = [0xf5, 0xc2, 0x6b];

/// Colours resolved once for the terminal: 24-bit where it is supported, the nearest xterm-256
/// colour otherwise (macOS Terminal.app, most tmux setups).
pub(super) struct Theme {
    truecolor: bool,
    pub on: Color,
    pub off: Color,
    pub text: Color,
    pub muted: Color,
    pub border: Color,
    pub good: Color,
    pub warn: Color,
}

impl Theme {
    /// 24-bit colour when the terminal says it has it (and always on Windows, whose terminals do).
    pub fn detect() -> Self {
        let colorterm = std::env::var("COLORTERM");
        Self::new(matches!(colorterm.as_deref(), Ok("truecolor") | Ok("24bit")) || cfg!(windows))
    }

    pub fn new(truecolor: bool) -> Self {
        let color = |rgb| paint(rgb, truecolor);
        Self {
            truecolor,
            on: color(ON),
            off: color(OFF),
            text: color(TEXT),
            muted: color(MUTED),
            border: color(BORDER),
            good: color(GOOD),
            warn: color(WARN),
        }
    }

    pub fn color(&self, rgb: [u8; 3]) -> Color {
        paint(rgb, self.truecolor)
    }

    /// A blend from OFF blue (`t = 0`) to ON red (`t = 1`), for graphs that heat up as they rise.
    pub fn heat(&self, t: f64) -> Color {
        let t = t.clamp(0.0, 1.0);
        let mix = |a: u8, b: u8| (f64::from(a) + (f64::from(b) - f64::from(a)) * t).round() as u8;
        self.color([mix(OFF[0], ON[0]), mix(OFF[1], ON[1]), mix(OFF[2], ON[2])])
    }

    pub fn label(&self) -> Style {
        Style::new().fg(self.muted)
    }

    pub fn value(&self) -> Style {
        Style::new().fg(self.text).add_modifier(Modifier::BOLD)
    }

    pub fn key(&self) -> Style {
        Style::new().fg(self.off).add_modifier(Modifier::BOLD)
    }

    /// A count that should be zero: quiet green while it is, amber once it is not.
    #[cfg(feature = "camera")]
    pub fn alarm(&self, count: usize) -> Style {
        Style::new().fg(if count == 0 { self.good } else { self.warn })
    }
}

fn paint([r, g, b]: [u8; 3], truecolor: bool) -> Color {
    if truecolor {
        Color::Rgb(r, g, b)
    } else {
        Color::Indexed(xterm256([r, g, b]))
    }
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

    #[test]
    fn xterm256_hits_the_cube_corners() {
        assert_eq!(xterm256([0, 0, 0]), 16);
        assert_eq!(xterm256([255, 255, 255]), 231);
        assert_eq!(xterm256([255, 0, 0]), 196);
        assert_eq!(xterm256(OFF), 16 + 6 * 3 + 5);
    }

    #[test]
    fn heat_runs_from_off_to_on() {
        let theme = Theme::new(true);
        assert_eq!(theme.heat(0.0), theme.off);
        assert_eq!(theme.heat(1.0), theme.on);
        assert_eq!(theme.heat(7.0), theme.on);
    }
}
