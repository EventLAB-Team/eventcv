//! The live camera's bias panel, opened with `b` in place of the camera box.
//!
//! ```text
//! ╭─ biases ───────────────────────────────────╮
//! │adaptive  Tracking · 2 slow steps · a off   │
//! │1.23 Mev/s in 346.0 kev/s–1.73 Mev/s        │
//! │▸ ON threshold       ━━━━━━━●──────    1564 │
//! │  OFF threshold      ━━●───────────     583 │
//! │  Refractory period  ━━━━━━●───────     993 │
//! │  ── 16 advanced · m to show ──             │
//! │ Contrast an ON event needs. Higher: fewer  │
//! │ ON events.                                 │
//! │ onbn · 0..2040 · default 1564              │
//! ╰────────────────────────────────────────────╯
//! ```
//!
//! Only the biases worth tuning for a scene are listed until `m` shows the advanced ones. `a` turns
//! adaptive biasing on and off, and the top lines show what it is doing; the biases it drives then
//! move on their own, dimmed.

use eventcv_core::bias::{BiasOverrides, BiasState};
use eventcv_core::device::{BiasInfo, BiasPreset, Capture};
use ratatui::crossterm::event::KeyCode;
use ratatui::layout::{Constraint, Layout, Rect};
use ratatui::style::Style;
use ratatui::text::{Line, Span};
use ratatui::widgets::{Padding, Paragraph, Wrap};
use ratatui::Frame;

use super::draw::scrub;
use super::theme::Theme;
use super::ui::{boxed, format_rate};

/// Width of the side column while the panel is open: wide enough for a readable label and a gauge.
pub(super) const WIDTH: u16 = 48;
/// Characters a label gets before it is cut.
const LABEL: usize = 18;
/// Characters the value column gets.
const VALUE: usize = 7;
/// Lines under the list for what the selected bias does.
const FOOTER: u16 = 3;
/// Lines above the list for the adaptive-bias controller.
const STATUS: u16 = 2;
/// A coarse step (page up/down) crosses a bias's range in this many presses.
const COARSE_STEPS: i64 = 20;

/// What a key asks of the camera.
#[derive(Debug, PartialEq, Eq)]
pub(super) enum Action {
    Set(String, i64),
    ResetAll,
    Save,
    Load,
    ToggleAdaptive,
}

pub(super) struct BiasPanel {
    /// Primary biases first, as [`Capture::biases`] orders them.
    pub rows: Vec<BiasInfo>,
    pub selected: usize,
    /// Whether the advanced biases are listed.
    pub advanced: bool,
    /// A value being typed in for the selected bias.
    pub editing: Option<String>,
    /// The last write's outcome, shown under the list.
    pub message: Option<String>,
    /// The adaptive-bias controller, when it is running.
    pub adaptive: Option<BiasState>,
}

impl BiasPanel {
    pub fn open(capture: &Capture) -> Self {
        let mut panel = Self {
            rows: Vec::new(),
            selected: 0,
            advanced: false,
            editing: None,
            message: None,
            adaptive: None,
        };
        panel.refresh(capture);
        panel
    }

    /// Re-reads the biases, which adaptive biasing may have moved since the last frame.
    pub fn refresh(&mut self, capture: &Capture) {
        match capture.biases() {
            Ok(rows) => self.rows = rows,
            Err(message) => self.message = Some(message),
        }
        self.adaptive = capture.bias_state();
        self.selected = self.selected.min(self.shown().saturating_sub(1));
    }

    fn primaries(&self) -> usize {
        self.rows.iter().filter(|bias| bias.primary).count()
    }

    /// How many rows are listed: the primary ones, or all of them when the advanced view is open
    /// or eventcv knows none of this camera's biases.
    fn shown(&self) -> usize {
        match self.primaries() {
            0 => self.rows.len(),
            _ if self.advanced => self.rows.len(),
            primaries => primaries,
        }
    }

    fn select(&mut self, delta: isize) {
        let last = self.shown().saturating_sub(1);
        self.selected = self.selected.saturating_add_signed(delta).min(last);
    }

    /// The selected bias, if it can be changed by hand.
    fn adjustable(&self) -> Option<&BiasInfo> {
        self.rows.get(self.selected).filter(|bias| !bias.adaptive)
    }

    /// The selected bias moved by `delta`, clamped to its range — `None` if it cannot move.
    fn nudged(&self, delta: i64) -> Option<Action> {
        let bias = self.adjustable()?;
        let value = (bias.value + delta).clamp(bias.minimum, bias.maximum);
        (value != bias.value).then(|| Action::Set(bias.name.clone(), value))
    }

    /// Turns a key into an action, or updates the panel itself (selection, typing). `None` with
    /// `true` means the key was used up here; `false` leaves it to the viewer.
    pub fn command(&mut self, code: KeyCode) -> (bool, Option<Action>) {
        if let Some(text) = &mut self.editing {
            let action = match code {
                KeyCode::Char(digit) if digit.is_ascii_digit() => {
                    text.push(digit);
                    None
                }
                KeyCode::Char('-') if text.is_empty() => {
                    text.push('-');
                    None
                }
                KeyCode::Backspace => {
                    text.pop();
                    None
                }
                KeyCode::Esc => {
                    self.editing = None;
                    None
                }
                KeyCode::Enter => {
                    let text = self.editing.take().unwrap_or_default();
                    match (text.parse::<i64>(), self.adjustable()) {
                        (Ok(value), Some(bias)) => Some(Action::Set(bias.name.clone(), value)),
                        _ => {
                            self.message = Some(format!("`{text}` is not a whole number"));
                            None
                        }
                    }
                }
                _ => None,
            };
            // Everything goes to the field while it is open, so `q` or `b` cannot slip through.
            return (true, action);
        }
        let coarse = self
            .rows
            .get(self.selected)
            .map_or(1, |bias| ((bias.maximum - bias.minimum) / COARSE_STEPS).max(1));
        // No key here means anything else on this screen: seeking and speed only exist for
        // recordings, and the panel only for a camera. None needs shift; `esc` only cancels typing.
        let action = match code {
            KeyCode::Up => {
                self.select(-1);
                None
            }
            KeyCode::Down => {
                self.select(1);
                None
            }
            KeyCode::Left => self.nudged(-1),
            KeyCode::Right => self.nudged(1),
            KeyCode::PageDown => self.nudged(-coarse),
            KeyCode::PageUp => self.nudged(coarse),
            KeyCode::Enter => {
                if self.adjustable().is_some() {
                    self.editing = Some(String::new());
                    self.message = None;
                }
                None
            }
            KeyCode::Char('m') => {
                self.advanced = !self.advanced;
                self.select(0);
                None
            }
            KeyCode::Char('d') => self
                .adjustable()
                .filter(|bias| bias.value != bias.default)
                .map(|bias| Action::Set(bias.name.clone(), bias.default)),
            KeyCode::Char('c') => Some(Action::ResetAll),
            KeyCode::Char('s') => Some(Action::Save),
            KeyCode::Char('l') => Some(Action::Load),
            KeyCode::Char('a') => Some(Action::ToggleAdaptive),
            _ => return (false, None),
        };
        (true, action)
    }

    /// Handles a key against the open camera. `false` leaves it to the viewer.
    pub fn key(&mut self, capture: &mut Capture, code: KeyCode) -> bool {
        let (used, action) = self.command(code);
        let Some(action) = action else {
            return used;
        };
        let path = std::path::PathBuf::from(format!("biases-{}.json", capture.serial()));
        // A note on success, the refusal on failure.
        let outcome = match action {
            Action::Set(name, value) => capture.set_biases(&[(name, value)]).map(|()| None),
            Action::ResetAll => capture.reset_biases().map(|()| None),
            Action::Save => capture
                .biases()
                .and_then(|biases| BiasPreset::write(&path, capture.name(), &biases))
                .map(|()| Some(format!("saved {}", path.display()))),
            Action::Load => BiasPreset::read(&path)
                .and_then(|preset| capture.apply_preset(&preset))
                .map(|()| Some(format!("loaded {}", path.display()))),
            Action::ToggleAdaptive => {
                let on = capture.bias_state().is_none();
                capture
                    .set_adaptive_bias(on.then(BiasOverrides::default))
                    .map(|()| on.then(|| "adaptive biasing on: measuring the scene first".into()))
            }
        };
        self.message = outcome.unwrap_or_else(Some);
        self.refresh(capture);
        true
    }

    pub fn draw(&self, frame: &mut Frame, area: Rect, theme: &Theme) {
        let block = boxed(theme, " biases ").padding(Padding::horizontal(1));
        let inner = block.inner(area);
        frame.render_widget(block, area);
        let [status, list, footer] = Layout::vertical([
            Constraint::Length(STATUS),
            Constraint::Min(1),
            Constraint::Length(FOOTER),
        ])
        .areas(inner);
        frame.render_widget(Paragraph::new(self.status(theme)), status);

        let gauge = usize::from(list.width)
            .saturating_sub(2 + LABEL + 1 + 1 + VALUE)
            .max(1);
        let primaries = self.primaries();
        let mut lines: Vec<(Option<usize>, Line)> = Vec::new();
        for (index, bias) in self.rows.iter().enumerate().take(self.shown()) {
            if index == primaries && primaries > 0 {
                lines.push((None, self.divider(theme)));
            }
            lines.push((Some(index), self.row(index, bias, gauge, theme)));
        }
        if self.shown() == primaries && primaries < self.rows.len() {
            lines.push((None, self.divider(theme)));
        }
        // Scrolled to keep the selection on screen.
        let height = usize::from(list.height).max(1);
        let at = lines
            .iter()
            .position(|(index, _)| *index == Some(self.selected))
            .unwrap_or(0);
        let first = at.saturating_sub(height - 1);
        let visible: Vec<Line> = lines
            .into_iter()
            .skip(first)
            .take(height)
            .map(|(_, line)| line)
            .collect();
        frame.render_widget(Paragraph::new(visible), list);
        frame.render_widget(
            Paragraph::new(self.footer(theme)).wrap(Wrap { trim: true }),
            footer,
        );
    }

    fn row(&self, index: usize, bias: &BiasInfo, gauge: usize, theme: &Theme) -> Line<'static> {
        let selected = index == self.selected;
        let label: String = bias
            .label
            .unwrap_or(bias.name.as_str())
            .chars()
            .take(LABEL)
            .collect();
        let style = match (bias.adaptive, selected) {
            (true, _) => theme.label(),
            (false, true) => theme.key(),
            (false, false) => theme.value(),
        };
        let span = (bias.maximum - bias.minimum).max(1) as f64;
        let share = (bias.value - bias.minimum) as f64 / span;
        let mut spans = vec![
            Span::styled(if selected { "▸ " } else { "  " }, theme.key()),
            Span::styled(format!("{label:<LABEL$} "), style),
        ];
        spans.extend(scrub(share, gauge, theme));
        let value = match (&self.editing, selected) {
            (Some(text), true) => Span::styled(format!("{:>VALUE$}", format!("{text}▏")), theme.key()),
            _ => Span::styled(format!("{:>VALUE$}", bias.value), style),
        };
        spans.extend([Span::raw(" "), value]);
        Line::from(spans)
    }

    /// What adaptive biasing is doing: its state and slow steps, then the rate against its band.
    fn status(&self, theme: &Theme) -> Vec<Line<'static>> {
        let Some(state) = &self.adaptive else {
            return vec![Line::from(vec![
                Span::styled("adaptive  ", theme.label()),
                Span::styled("off", theme.value()),
                Span::styled(" · a to turn on", theme.label()),
            ])];
        };
        let control = match state.calibrating {
            true => "measuring the scene".to_owned(),
            false => format!("{:?} · {} slow steps", state.authority, state.slow_steps),
        };
        let (low, high) = state.target_rate;
        vec![
            Line::from(vec![
                Span::styled("adaptive  ", theme.label()),
                Span::styled(control, Style::new().fg(theme.good)),
                Span::styled(" · a off", theme.label()),
            ]),
            Line::from(vec![
                Span::styled(format_rate(state.event_rate), theme.value()),
                Span::styled(
                    format!(" in {}–{}", format_rate(low), format_rate(high)),
                    theme.label(),
                ),
            ]),
        ]
    }

    fn divider(&self, theme: &Theme) -> Line<'static> {
        let text = match self.advanced {
            true => "  ── advanced · m to hide ──".to_owned(),
            false => format!(
                "  ── {} advanced · m to show ──",
                self.rows.len() - self.primaries()
            ),
        };
        Line::styled(text, theme.label())
    }

    fn footer(&self, theme: &Theme) -> Vec<Line<'static>> {
        let Some(bias) = self.rows.get(self.selected) else {
            return vec![Line::styled("this camera exposes no biases", theme.label())];
        };
        let range = format!("{}..{}", bias.minimum, bias.maximum);
        if self.editing.is_some() {
            return vec![Line::styled(
                format!("type a value in {range} · enter sets · esc cancels"),
                theme.key(),
            )];
        }
        let mut lines = Vec::new();
        if let Some(message) = &self.message {
            lines.push(Line::styled(message.clone(), Style::new().fg(theme.warn)));
        } else if bias.adaptive {
            lines.push(Line::styled("driven by adaptive_bias", Style::new().fg(theme.warn)));
        } else if let Some(description) = bias.description {
            lines.push(Line::styled(description, theme.value()));
        }
        lines.push(Line::styled(
            format!("{} · {range} · default {}", bias.name, bias.default),
            theme.label(),
        ));
        lines
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn bias(name: &str, value: i64, primary: bool, adaptive: bool) -> BiasInfo {
        BiasInfo {
            name: name.to_owned(),
            value,
            minimum: -100,
            maximum: 100,
            default: 0,
            adaptive,
            label: primary.then_some("ON threshold"),
            description: primary.then_some("Contrast an ON event needs."),
            primary,
        }
    }

    fn panel() -> BiasPanel {
        BiasPanel {
            rows: vec![
                bias("diff_on", 8, true, false),
                bias("refr", 0, true, true),
                bias("reqpuy", 5, false, false),
            ],
            selected: 0,
            advanced: false,
            editing: None,
            message: None,
            adaptive: None,
        }
    }

    fn press(panel: &mut BiasPanel, code: KeyCode) -> Option<Action> {
        panel.command(code).1
    }

    fn set(value: i64) -> Option<Action> {
        Some(Action::Set("diff_on".to_owned(), value))
    }

    #[test]
    fn steps_are_fine_or_coarse() {
        let mut panel = panel();
        assert_eq!(press(&mut panel, KeyCode::Right), set(9));
        assert_eq!(press(&mut panel, KeyCode::Left), set(7));
        // A coarse step is a twentieth of the range: 200 / 20.
        assert_eq!(press(&mut panel, KeyCode::PageUp), set(18));
        assert_eq!(press(&mut panel, KeyCode::PageDown), set(-2));
        // Clamped to the range.
        panel.rows[0].value = 99;
        assert_eq!(press(&mut panel, KeyCode::PageUp), set(100));
        assert_eq!(press(&mut panel, KeyCode::Char('d')), set(0));
        assert_eq!(press(&mut panel, KeyCode::Char('c')), Some(Action::ResetAll));
        assert_eq!(press(&mut panel, KeyCode::Char('a')), Some(Action::ToggleAdaptive));
    }

    #[test]
    fn the_viewers_own_keys_are_left_alone() {
        let mut panel = panel();
        for code in [
            KeyCode::Char('-'),
            KeyCode::Char('+'),
            KeyCode::Char('['),
            KeyCode::Char('r'),
            KeyCode::Char('b'),
            KeyCode::Char('q'),
            KeyCode::Char(' '),
            KeyCode::Esc,
        ] {
            assert_eq!(panel.command(code), (false, None), "{code:?}");
        }
    }

    #[test]
    fn a_typed_value_is_set_on_enter() {
        let mut panel = panel();
        assert_eq!(press(&mut panel, KeyCode::Enter), None);
        for key in ['-', '4', 'q', '2'] {
            // `q` would quit the viewer; while typing it is just ignored.
            assert!(panel.command(KeyCode::Char(key)).0);
        }
        assert_eq!(panel.editing.as_deref(), Some("-42"));
        assert_eq!(press(&mut panel, KeyCode::Enter), set(-42));
        assert_eq!(panel.editing, None);

        press(&mut panel, KeyCode::Enter);
        press(&mut panel, KeyCode::Char('7'));
        assert_eq!(press(&mut panel, KeyCode::Esc), None);
        assert_eq!(panel.editing, None, "esc cancels");
    }

    #[test]
    fn advanced_biases_stay_hidden_until_asked_for() {
        let mut panel = panel();
        panel.select(5);
        assert_eq!(panel.selected, 1, "only the two primary biases are listed");
        // An adaptively driven bias cannot be nudged, defaulted or typed into.
        assert_eq!(press(&mut panel, KeyCode::Right), None);
        press(&mut panel, KeyCode::Enter);
        assert_eq!(panel.editing, None);
        press(&mut panel, KeyCode::Char('m'));
        panel.select(5);
        assert_eq!(panel.selected, 2);
        // Hiding them again pulls the selection back into view.
        press(&mut panel, KeyCode::Char('m'));
        assert_eq!(panel.selected, 1);
    }

    fn screen(panel: &BiasPanel) -> String {
        let backend = ratatui::backend::TestBackend::new(WIDTH, 12);
        let mut terminal = ratatui::Terminal::new(backend).unwrap();
        let theme = Theme::new(false);
        terminal
            .draw(|frame| panel.draw(frame, frame.area(), &theme))
            .unwrap();
        terminal
            .backend()
            .buffer()
            .content()
            .iter()
            .map(|cell| cell.symbol())
            .collect()
    }

    #[test]
    fn draws_labels_the_description_and_the_advanced_divider() {
        let mut panel = panel();
        let text = screen(&panel);
        for needle in ["biases", "▸ ON threshold", "1 advanced · m to show", "Contrast an ON"] {
            assert!(text.contains(needle), "{needle:?} not on screen");
        }
        assert!(!text.contains("reqpuy"));
        assert!(text.contains("adaptive  off · a to turn on"));
        panel.advanced = true;
        panel.adaptive = Some(BiasState {
            event_rate: 1.23e6,
            values: eventcv_core::bias::BiasValues {
                refractory: 0,
                photoreceptor: 0,
                follower: 0,
                on_threshold: 0,
                off_threshold: 0,
            },
            target_rate: (3.46e5, 1.73e6),
            calibrating: false,
            authority: eventcv_core::bias::Authority::Tracking,
            slow_steps: 2,
        });
        let text = screen(&panel);
        assert!(text.contains("reqpuy"));
        assert!(text.contains("Tracking · 2 slow steps"), "{text}");
        assert!(text.contains("1.23 Mev/s in 346.0 kev/s–1.73 Mev/s"));
    }
}
