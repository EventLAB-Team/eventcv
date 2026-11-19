//! The filter box: core's streaming filters, each switched on and tuned from the keyboard.

use eventcv_core::filter::{
    BackgroundProcessor, HotPixelProcessor, Pipeline, Processor, RefractoryProcessor,
};
use ratatui::crossterm::event::KeyCode;

#[derive(Clone, Copy)]
enum Kind {
    Background,
    Refractory,
    HotPixels,
}

pub(super) struct Filter {
    kind: Kind,
    pub enabled: bool,
    /// The time constant in µs, or the hot-pixel threshold in standard deviations.
    value: f64,
}

impl Filter {
    pub fn name(&self) -> &'static str {
        match self.kind {
            Kind::Background => "background",
            Kind::Refractory => "refractory",
            Kind::HotPixels => "hot pixels",
        }
    }

    pub fn value(&self) -> String {
        match self.kind {
            Kind::HotPixels => format!("{}σ", self.value),
            _ if self.value >= 1_000.0 => format!("{} ms", self.value / 1_000.0),
            _ => format!("{} µs", self.value),
        }
    }

    fn build(&self) -> Box<dyn Processor> {
        match self.kind {
            Kind::Background => Box::new(BackgroundProcessor::new(self.value as i64)),
            Kind::Refractory => Box::new(RefractoryProcessor::new(self.value as i64)),
            Kind::HotPixels => Box::new(HotPixelProcessor::new(self.value)),
        }
    }

    /// Times move along 1-2-5 steps from 10 µs to 1 s; the threshold by half a σ.
    fn adjust(&mut self, up: bool) {
        self.value = match self.kind {
            Kind::HotPixels => (self.value + if up { 0.5 } else { -0.5 }).clamp(0.5, 10.0),
            _ => {
                let steps = (1..=6)
                    .flat_map(|power| [1.0, 2.0, 5.0].map(|step| step * 10f64.powi(power)))
                    .filter(|&step| step <= 1e6);
                let next = if up {
                    steps.clone().find(|&step| step > self.value)
                } else {
                    steps.rev().find(|&step| step < self.value)
                };
                next.unwrap_or(self.value)
            }
        };
    }
}

pub(super) struct Filters {
    pub list: [Filter; 3],
    /// The highlighted filter while the box has the keys.
    pub focus: Option<usize>,
}

impl Default for Filters {
    /// Off, at the GUI viewer's defaults, and `open(hot_pixel_filter=True)`'s threshold.
    fn default() -> Self {
        let filter = |kind, value| Filter {
            kind,
            enabled: false,
            value,
        };
        Self {
            list: [
                filter(Kind::Background, 25_000.0),
                filter(Kind::Refractory, 1_000.0),
                filter(Kind::HotPixels, 3.0),
            ],
            focus: None,
        }
    }
}

impl Filters {
    pub fn any(&self) -> bool {
        self.list.iter().any(|filter| filter.enabled)
    }

    pub fn pipeline(&self) -> Pipeline {
        Pipeline::new(
            self.list
                .iter()
                .filter(|filter| filter.enabled)
                .map(Filter::build)
                .collect(),
        )
    }

    /// Handles `code` if it is the box's: `f` to take or give back the keys, and while it has
    /// them ↑↓ to pick, space to switch, ←→ to tune and Esc to leave. `Some(true)` means the
    /// pipeline changed; `None` leaves the key to the viewer.
    pub fn key(&mut self, code: KeyCode) -> Option<bool> {
        let Some(index) = self.focus else {
            if code == KeyCode::Char('f') {
                self.focus = Some(0);
                return Some(false);
            }
            return None;
        };
        let last = self.list.len() - 1;
        let filter = &mut self.list[index];
        match code {
            KeyCode::Char('f') | KeyCode::Esc => self.focus = None,
            KeyCode::Up => self.focus = Some(index.saturating_sub(1)),
            KeyCode::Down => self.focus = Some((index + 1).min(last)),
            KeyCode::Char(' ') => {
                filter.enabled = !filter.enabled;
                return Some(true);
            }
            KeyCode::Left | KeyCode::Right => {
                filter.adjust(code == KeyCode::Right);
                return Some(filter.enabled);
            }
            _ => return None,
        }
        Some(false)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn times_step_along_one_two_five() {
        let mut filters = Filters::default();
        let background = &mut filters.list[0];
        background.adjust(true);
        assert_eq!(background.value(), "50 ms");
        background.adjust(false);
        background.adjust(false);
        assert_eq!(background.value(), "10 ms");
        let refractory = &mut filters.list[1];
        for _ in 0..10 {
            refractory.adjust(false);
        }
        assert_eq!(refractory.value(), "10 µs");
    }

    #[test]
    fn the_box_takes_keys_only_while_focused() {
        let mut filters = Filters::default();
        assert_eq!(filters.key(KeyCode::Char(' ')), None);
        assert_eq!(filters.key(KeyCode::Char('f')), Some(false));
        assert_eq!(filters.key(KeyCode::Down), Some(false));
        assert_eq!(filters.key(KeyCode::Char(' ')), Some(true));
        assert!(filters.list[1].enabled && filters.any());
        // Tuning a filter that is off changes nothing that runs.
        assert_eq!(filters.key(KeyCode::Up), Some(false));
        assert_eq!(filters.key(KeyCode::Right), Some(false));
        assert_eq!(filters.key(KeyCode::Char('r')), None);
        assert_eq!(filters.key(KeyCode::Esc), Some(false));
        assert_eq!(filters.key(KeyCode::Esc), None);
    }
}
