//! The dashboard layout, after btop: rounded boxes titled in their borders, the EventCV badge and
//! source in the frame, the event view beside the stream and playback (or camera) boxes, and a
//! braille rate graph along the bottom.
//!
//! ```text
//! ╭─ ◉ EventCV ────────────────── sunset2.hdf5 · 1280×720 · ▶ ×1 ─╮
//! │╭─ raw ─────────────────────── r next ─╮╭─ stream ────────────╮│
//! ││                                      ││ events       1,813  ││
//! ││          (the events)                ││ ON  ██░░░░░░░    9% ││
//! ││                                      │╰─────────────────────╯│
//! ││                                      │╭─ playback ──────────╮│
//! │╰──────────────────────────────────────╯╰─────────────────────╯│
//! │╭─ event rate ──────────────── now 60.4 kev/s · peak 2.71 Mev/s ╮│
//! ││ ⣀⣠⣤⣴⣶⣾⣿⣷⣶⣤⣀⣀⣠⣤⣶⣿⣿⣶⣤⣀⡀ ⢀⣀⣤⣴⣶⣤⣀⣀⣠⣤⣶⣾⣿⣷⣶⣤⣀                    ││
//! │╰──────────────────────────────────────────────────────────────╯│
//! ╰─ space play · ←→ seek · +- speed · [] window · r view · q quit ─╯
//! ```

use ratatui::layout::{Alignment, Constraint, Layout, Rect};
use ratatui::style::{Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, BorderType, Padding, Paragraph, Wrap};
use ratatui::Frame;

use super::draw::{meter, scrub, HalfBlock, RateGraph};
use super::theme::Theme;
use super::{App, Feed};
use crate::commands::format_count;

/// Width of the stream / playback column.
const SIDE: u16 = 30;
/// Rows the rate graph takes, borders included, when the terminal is tall enough to spare them.
const GRAPH: u16 = 7;

impl App {
    pub(super) fn draw(&self, frame: &mut Frame) {
        let frame_block = boxed(&self.theme, badge(&self.theme))
            .title(self.source_line().right_aligned())
            .title_bottom(self.keys());
        let body = frame_block.inner(frame.area());
        frame.render_widget(frame_block, frame.area());

        let graph = if body.height >= 3 * GRAPH { GRAPH } else { 0 };
        let [top, bottom] =
            Layout::vertical([Constraint::Min(4), Constraint::Length(graph)]).areas(body);
        let [view, side] =
            Layout::horizontal([Constraint::Min(16), Constraint::Length(self.side_width())]).areas(top);
        let [stream, detail] =
            Layout::vertical([Constraint::Length(6), Constraint::Min(0)]).areas(side);

        self.draw_view(frame, view);
        self.draw_stream(frame, stream);
        match self.bias_panel() {
            #[cfg(feature = "camera")]
            Some(panel) => panel.draw(frame, detail, &self.theme),
            _ => self.draw_detail(frame, detail),
        }
        if graph > 0 {
            self.draw_graph(frame, bottom);
        }
    }

    /// `sunset2.hdf5 · 1280×720 · ▶ ×1`, or the camera and `● LIVE`.
    fn source_line(&self) -> Line<'static> {
        let theme = &self.theme;
        let (width, height) = self.sensor;
        let state = match &self.feed {
            Feed::File(feed) if feed.playback.playing => Span::styled(
                format!("▶ ×{}", format_speed(feed.playback.speed)),
                Style::new().fg(theme.good),
            ),
            Feed::File(_) => Span::styled("‖ paused", Style::new().fg(theme.warn)),
            #[cfg(feature = "camera")]
            Feed::Live(_) if self.paused => Span::styled("‖ paused", Style::new().fg(theme.warn)),
            #[cfg(feature = "camera")]
            Feed::Live(_) => Span::styled(
                "● LIVE",
                Style::new().fg(theme.on).add_modifier(Modifier::BOLD),
            ),
        };
        let dot = Span::styled(" · ", theme.label());
        Line::from(vec![
            Span::raw(" "),
            Span::styled(self.name.clone(), theme.value()),
            dot.clone(),
            Span::styled(format!("{width}×{height}"), theme.label()),
            dot,
            state,
            Span::raw(" "),
        ])
    }

    /// The key hints for this source, set into the bottom border.
    fn keys(&self) -> Line<'static> {
        let file = matches!(self.feed, Feed::File(_));
        let quit = ("q", if self.back { "back" } else { "quit" });
        if self.bias_panel().is_some() {
            return key_hints(
                &self.theme,
                &[
                    ("↑↓", "pick"),
                    ("←→", "±1"),
                    ("PgDn PgUp", "coarse"),
                    ("enter", "type"),
                    ("m", "more"),
                    ("a", "adaptive"),
                    ("d", "default"),
                    ("c", "clear all"),
                    ("s", "save"),
                    ("l", "load"),
                    ("b", "close"),
                    quit,
                ],
            );
        }
        let mut keys = vec![("space", if file { "play" } else { "pause" })];
        if file {
            keys.extend([("←→", "seek"), ("+-", "speed")]);
        } else {
            keys.push(("b", "biases"));
        }
        // Opened from the home screen, `q` goes back to it rather than quitting.
        keys.extend([
            ("[]", "window"),
            ("r", "view"),
            quit,
        ]);
        key_hints(&self.theme, &keys)
    }

    /// The side column widens while the bias panel is open, to fit its labels and gauges.
    fn side_width(&self) -> u16 {
        match self.bias_panel() {
            #[cfg(feature = "camera")]
            Some(_) => super::biases::WIDTH,
            _ => SIDE,
        }
    }

    #[cfg(feature = "camera")]
    fn bias_panel(&self) -> Option<&super::biases::BiasPanel> {
        self.biases.as_ref()
    }

    /// Without camera support there is never a bias panel.
    #[cfg(not(feature = "camera"))]
    fn bias_panel(&self) -> Option<std::convert::Infallible> {
        None
    }

    fn draw_view(&self, frame: &mut Frame, area: Rect) {
        let theme = &self.theme;
        let block = boxed(
            &self.theme,
            Line::from(vec![
                Span::raw(" "),
                Span::styled(self.view_name(), theme.value()),
                Span::raw(" "),
            ]),
        )
        .title(
            Line::from(vec![
                Span::raw(" "),
                Span::styled("r", theme.key()),
                Span::styled(" next ", theme.label()),
            ])
            .right_aligned(),
        );
        let inner = block.inner(area);
        frame.render_widget(block, area);
        match (&self.image, &self.problem) {
            (Some(image), _) => frame.render_widget(HalfBlock { image, theme }, inner),
            (None, Some(problem)) => frame.render_widget(
                Paragraph::new(problem.as_str())
                    .style(Style::new().fg(theme.on))
                    .alignment(Alignment::Center)
                    .wrap(Wrap { trim: true }),
                inner,
            ),
            (None, None) => {}
        }
    }

    fn draw_stream(&self, frame: &mut Frame, area: Rect) {
        let theme = &self.theme;
        let block = boxed(&self.theme, " stream ").padding(Padding::horizontal(1));
        let inner = block.inner(area);
        frame.render_widget(block, area);

        let stats = self.stats;
        let negative = stats.total - stats.positive;
        let share = |count: usize| match stats.total {
            0 => 0.0,
            total => count as f64 / total as f64,
        };
        // Label, bar, and a right-aligned percentage, all inside the box.
        let bar_width = usize::from(inner.width).saturating_sub(10);
        let polarity = |label: &'static str, count: usize, color| {
            let mut spans = vec![Span::styled(label, Style::new().fg(color))];
            spans.extend(meter(share(count), bar_width, color, theme));
            spans.push(Span::styled(
                format!("{:>5.0}%", 100.0 * share(count)),
                theme.value(),
            ));
            Line::from(spans)
        };
        let rate_label = if matches!(self.feed, Feed::File(_)) {
            "rate"
        } else {
            "decoded"
        };
        let lines = vec![
            row(
                theme,
                "events",
                format_count(stats.total as u64),
                inner.width,
            ),
            row(theme, rate_label, format_rate(stats.rate), inner.width),
            polarity("ON  ", stats.positive, theme.on),
            polarity("OFF ", negative, theme.off),
        ];
        frame.render_widget(Paragraph::new(lines), inner);
    }

    /// Playback for a recording; the driver's health for a camera.
    fn draw_detail(&self, frame: &mut Frame, area: Rect) {
        let theme = &self.theme;
        let title = match self.feed {
            Feed::File(_) => " playback ",
            #[cfg(feature = "camera")]
            Feed::Live(_) => " camera ",
        };
        let block = boxed(&self.theme, title).padding(Padding::horizontal(1));
        let inner = block.inner(area);
        frame.render_widget(block, area);

        let width = inner.width;
        let window = format!("{} ms · {}", self.dt_ms(), self.view_name());
        let lines = match &self.feed {
            Feed::File(feed) => {
                let playback = &feed.playback;
                let share = match playback.duration_us {
                    0 => 0.0,
                    duration => playback.cursor_us as f64 / duration as f64,
                };
                let position = format!(
                    "{} / {}",
                    format_time(playback.cursor_us),
                    format_time(playback.duration_us)
                );
                let speed = format!("×{}", format_speed(playback.speed));
                vec![
                    Line::styled(position, theme.value()),
                    Line::from(scrub(share, usize::from(width), theme).to_vec()),
                    row(theme, "speed", speed, width),
                    row(theme, "window", window, width),
                    row(
                        theme,
                        "events",
                        format_count(feed.reader.n_events() as u64),
                        width,
                    ),
                ]
            }
            #[cfg(feature = "camera")]
            Feed::Live(feed) => {
                let capture = &feed.capture;
                let count = |label, value: usize, unit: &str| {
                    let text = format!("{}{unit}", format_count(value as u64));
                    padded(theme, label, Span::styled(text, theme.alarm(value)), width)
                };
                let mut lines = vec![
                    count("backlog", capture.backlog(), " buf"),
                    count("overflows", feed.overflows, ""),
                    count("dropped", capture.n_budget_dropped(), " buf"),
                    row(theme, "window", window, width),
                ];
                if let Some(bias) = capture.bias_state() {
                    lines.push(Line::default());
                    lines.push(row(theme, "bias rate", format_rate(bias.event_rate), width));
                    let (low, high) = bias.target_rate;
                    let target = format!("{}–{}", format_rate(low), format_rate(high));
                    lines.push(row(theme, "target", target, width));
                    let control = if bias.calibrating {
                        "calibrating".to_owned()
                    } else {
                        format!("{:?} · {} slow", bias.authority, bias.slow_steps)
                    };
                    lines.push(row(theme, "control", control, width));
                }
                lines
            }
        };
        frame.render_widget(Paragraph::new(lines), inner);
    }

    fn draw_graph(&self, frame: &mut Frame, area: Rect) {
        let theme = &self.theme;
        let samples: Vec<u64> = self.history.iter().copied().collect();
        let peak = samples.iter().copied().max().unwrap_or(0);
        let title = if matches!(self.feed, Feed::File(_)) {
            " event rate "
        } else {
            " decoded rate "
        };
        let block = boxed(&self.theme, title).title(
            Line::from(vec![
                Span::styled(" now ", theme.label()),
                Span::styled(format_rate(self.stats.rate), theme.value()),
                Span::styled(" · peak ", theme.label()),
                Span::styled(format_rate(peak as f64), Style::new().fg(theme.on)),
                Span::raw(" "),
            ])
            .right_aligned(),
        );
        let inner = block.inner(area);
        frame.render_widget(block, area);
        frame.render_widget(
            RateGraph {
                samples: &samples,
                theme,
            },
            inner,
        );
    }
}

/// A rounded box in the chrome colour, titled in its top border.
pub(super) fn boxed<'a>(theme: &Theme, title: impl Into<Line<'a>>) -> Block<'a> {
    Block::bordered()
        .border_type(BorderType::Rounded)
        .border_style(Style::new().fg(theme.border))
        .title(title)
}

/// `◉ EventCV`: the ON dot, "Event" in the logo's red and "CV" in its blue.
pub(super) fn badge(theme: &Theme) -> Line<'static> {
    Line::from(vec![
        Span::raw(" "),
        Span::styled("◉ ", Style::new().fg(theme.on)),
        Span::styled(
            "Event",
            Style::new().fg(theme.on).add_modifier(Modifier::BOLD),
        ),
        Span::styled(
            "CV",
            Style::new().fg(theme.off).add_modifier(Modifier::BOLD),
        ),
        Span::raw(" "),
    ])
}

/// `space play · q quit`: each key in blue, its action muted.
pub(super) fn key_hints(theme: &Theme, keys: &[(&'static str, &'static str)]) -> Line<'static> {
    let mut spans = vec![Span::raw(" ")];
    for (index, &(key, action)) in keys.iter().enumerate() {
        if index > 0 {
            spans.push(Span::styled(" · ", Style::new().fg(theme.border)));
        }
        spans.push(Span::styled(key, theme.key()));
        spans.push(Span::styled(format!(" {action}"), theme.label()));
    }
    spans.push(Span::raw(" "));
    Line::from(spans)
}

/// `label ……… value`: a muted label and a bold value pushed to the right edge of `width`.
fn row(theme: &Theme, label: &'static str, value: String, width: u16) -> Line<'static> {
    padded(theme, label, Span::styled(value, theme.value()), width)
}

fn padded(theme: &Theme, label: &'static str, value: Span<'static>, width: u16) -> Line<'static> {
    let gap = usize::from(width).saturating_sub(label.chars().count() + value.width());
    Line::from(vec![
        Span::styled(label, theme.label()),
        Span::raw(" ".repeat(gap)),
        value,
    ])
}

/// `1.234 s`, or `m:ss.ss` past a minute — as the GUI player shows it.
fn format_time(microseconds: i64) -> String {
    let seconds = microseconds.max(0) as f64 / 1_000_000.0;
    if seconds >= 60.0 {
        format!("{}:{:05.2}", (seconds / 60.0) as u64, seconds % 60.0)
    } else {
        format!("{seconds:.3} s")
    }
}

pub(super) fn format_rate(rate: f64) -> String {
    if rate >= 1_000_000.0 {
        format!("{:.2} Mev/s", rate / 1_000_000.0)
    } else if rate >= 1_000.0 {
        format!("{:.1} kev/s", rate / 1_000.0)
    } else {
        format!("{rate:.0} ev/s")
    }
}

fn format_speed(speed: f64) -> String {
    if speed >= 1.0 {
        format!("{speed}")
    } else {
        format!("1/{}", (1.0 / speed).round())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rates_and_times_read_like_the_gui() {
        assert_eq!(format_rate(512.0), "512 ev/s");
        assert_eq!(format_rate(2_500.0), "2.5 kev/s");
        assert_eq!(format_rate(3_210_000.0), "3.21 Mev/s");
        assert_eq!(format_time(1_234_000), "1.234 s");
        assert_eq!(format_time(83_500_000), "1:23.50");
        assert_eq!(format_speed(2.0), "2");
        assert_eq!(format_speed(0.25), "1/4");
    }

    #[test]
    fn rows_push_the_value_to_the_edge() {
        let theme = Theme::new(true);
        let line = row(&theme, "rate", "60 ev/s".to_owned(), 20);
        assert_eq!(line.width(), 20);
        assert_eq!(line.to_string(), "rate         60 ev/s");
    }
}
