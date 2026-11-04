//! `eventcv tui`: a recording or a live camera, watched in the terminal.
//!
//! The quick look that does not need a GPU window — over SSH, on a headless capture box, or just
//! to check a camera is producing what it should. The events are drawn in half blocks next to the
//! diagnostics that matter when something is off: rates and polarity balance, and for a camera the
//! driver backlog, overflows, decode drops and the adaptive-bias state.

mod draw;
mod source;

use std::collections::VecDeque;
use std::ffi::OsString;
use std::io::IsTerminal;
use std::time::{Duration, Instant};

use clap::{ArgGroup, Args, Parser};
use eventcv_core::representation::ReprSpec;
use eventcv_core::viz::{Colormap, RawSurface, RenderView, Rgb8Image, Scale};
use eventcv_core::EventStream;
use ratatui::crossterm::event::{self, Event, KeyCode, KeyEventKind, KeyModifiers};
use ratatui::layout::{Constraint, Layout, Rect};
use ratatui::style::{Style, Stylize};
use ratatui::text::Line;
use ratatui::widgets::{Block, Paragraph, Sparkline};
use ratatui::Frame;

use crate::camera::CameraOptions;
use crate::commands::{duration_us, format_count, parse_colormap, Source};
use crate::Failure;
use draw::HalfBlock;
use source::{FileFeed, Step};

/// Window lengths `[` and `]` stay within, as in the GUI player.
const DT_RANGE_US: (i64, i64) = (100, 1_000_000);
/// Rate samples kept for the sparkline — more than any terminal is wide.
const HISTORY: usize = 512;

#[derive(Args, Debug)]
#[command(group(ArgGroup::new("input").required(true).args(["path", "camera"])))]
pub(crate) struct TuiArgs {
    /// recording to play
    pub path: Option<String>,
    /// watch a live USB camera instead (optionally the one with this serial)
    #[arg(long, value_name = "SERIAL", num_args = 0..=1)]
    pub camera: Option<Option<String>>,
    #[command(flatten)]
    pub source: Source,
    #[command(flatten)]
    pub camera_options: CameraOptions,
    /// time shown per frame (default: 30)
    #[arg(long, value_name = "MS", default_value_t = 30.0)]
    pub dt_ms: f64,
    /// what to draw: "raw" for the event stream itself (default), or a representation name; `r`
    /// cycles through them
    #[arg(long, default_value = "raw")]
    pub repr: String,
    /// raw view only: fade time for an event's trail (default: one frame's worth)
    #[arg(long, value_name = "MS")]
    pub decay_ms: Option<f64>,
    /// colormap for scalar representations
    #[arg(long, default_value = "viridis")]
    pub colormap: String,
    /// playback rate multiplier (default: 1)
    #[arg(long, default_value_t = 1.0)]
    pub speed: f64,
    /// screen refreshes per second (default: 30)
    #[arg(long, value_name = "HZ", default_value_t = 30.0)]
    pub refresh_hz: f64,
}

/// `eventcv tui …` as its own command line, for the Python package's `eventcv` to hand over to.
#[derive(Parser)]
#[command(
    name = "eventcv tui",
    about = "watch a recording or a live camera in the terminal",
    infer_long_args = true
)]
struct Standalone {
    #[command(flatten)]
    args: TuiArgs,
}

/// Runs `eventcv tui` from `argv` (program name first) and returns the exit status.
pub fn main<I, T>(argv: I) -> i32
where
    I: IntoIterator<Item = T>,
    T: Into<OsString> + Clone,
{
    let args = match Standalone::try_parse_from(argv) {
        Ok(standalone) => standalone.args,
        Err(error) => {
            let _ = error.print();
            return error.exit_code();
        }
    };
    let path = args.path.clone();
    match run(args) {
        Ok(()) => 0,
        Err(failure) => {
            crate::report(failure, path.as_deref());
            1
        }
    }
}

pub(crate) fn run(args: TuiArgs) -> Result<(), Failure> {
    // Everything that can be wrong with the arguments or the source is found before the terminal
    // is taken over, so the message lands in a normal shell even when output is piped.
    let mut app = App::new(&args)?;
    if !std::io::stdout().is_terminal() {
        return Err("eventcv tui needs an interactive terminal".into());
    }
    if !(args.refresh_hz.is_finite() && args.refresh_hz > 0.0) {
        return Err("refresh_hz must be finite and positive".into());
    }
    let tick = Duration::from_secs_f64(1.0 / args.refresh_hz.min(120.0));

    crate::graceful();
    let mut terminal = ratatui::init();
    // Restores the terminal on every way out, early returns included; `init` covers panics.
    struct Restore;
    impl Drop for Restore {
        fn drop(&mut self) {
            ratatui::restore();
        }
    }
    let _restore = Restore;

    let mut last = Instant::now();
    loop {
        let now = Instant::now();
        app.step(now - last)?;
        last = now;
        terminal.draw(|frame| app.draw(frame))?;
        let deadline = now + tick;
        while let Some(remaining) = deadline.checked_duration_since(Instant::now()) {
            if !event::poll(remaining)? {
                break;
            }
            // Windows reports releases too; acting on those would fire every key twice.
            if let Event::Key(key) = event::read()? {
                if key.kind == KeyEventKind::Press && !app.key(key.code, key.modifiers) {
                    return Ok(());
                }
            }
        }
        if crate::interrupted() {
            return Ok(());
        }
    }
}

// One per viewer, so the size gap between a camera and a file reader is not worth a box.
#[allow(clippy::large_enum_variant)]
enum Feed {
    File(FileFeed),
    #[cfg(feature = "camera")]
    Live(source::LiveFeed),
}

/// Counts over the window on screen.
#[derive(Clone, Copy, Default)]
struct Stats {
    total: usize,
    positive: usize,
    rate: f64,
}

impl Stats {
    fn of(window: &EventStream, dt_us: i64) -> Self {
        let total = window.len();
        Self {
            total,
            positive: window.ps().iter().filter(|&&on| on).count(),
            rate: total as f64 / (dt_us as f64 / 1e6),
        }
    }
}

struct App {
    feed: Feed,
    name: String,
    sensor: (usize, usize),
    dt_us: i64,
    decay_ms: Option<f64>,
    colormap: Colormap,
    /// 0 is the raw view; `n` is `ReprSpec::NAMES[n - 1]`.
    view_index: usize,
    view: RenderView,
    /// A live view frozen on screen; a recording pauses through its own playback.
    #[cfg(feature = "camera")]
    paused: bool,
    image: Option<Rgb8Image>,
    /// Why the current view could not be drawn, shown in place of the image.
    problem: Option<String>,
    stats: Stats,
    history: VecDeque<u64>,
    truecolor: bool,
}

impl App {
    fn new(args: &TuiArgs) -> Result<Self, Failure> {
        let dt_us = duration_us("dt", args.dt_ms)?;
        if !(args.speed.is_finite() && args.speed > 0.0) {
            return Err("speed must be finite and positive".into());
        }
        let view_index = match args.repr.as_str() {
            "raw" => 0,
            name => {
                // Checked through the spec itself, so an unknown name reads as everywhere else.
                ReprSpec::named(name, None)?;
                1 + ReprSpec::NAMES
                    .iter()
                    .position(|&known| known == name)
                    .unwrap_or(0)
            }
        };
        let colormap = parse_colormap(&args.colormap)?;
        let (feed, name, sensor) = match (&args.path, &args.camera) {
            (Some(path), _) => {
                let reader = eventcv_core::io::open(path, args.source.options()?)?;
                let sensor = reader.sensor_size();
                let name = std::path::Path::new(path)
                    .file_name()
                    .map_or(path.clone(), |name| name.to_string_lossy().into_owned());
                (
                    Feed::File(FileFeed::new(reader, args.speed, dt_us)),
                    name,
                    sensor,
                )
            }
            (None, Some(serial)) => Self::open_camera(serial.as_deref(), &args.camera_options)?,
            (None, None) => unreachable!("clap requires a path or --camera"),
        };
        let mut app = Self {
            feed,
            name,
            sensor,
            dt_us,
            decay_ms: args.decay_ms,
            colormap,
            view_index,
            view: RenderView::Raw(RawSurface::new(0, 0, 1.0)),
            #[cfg(feature = "camera")]
            paused: false,
            image: None,
            problem: None,
            stats: Stats::default(),
            history: VecDeque::with_capacity(HISTORY),
            truecolor: matches!(
                std::env::var("COLORTERM").as_deref(),
                Ok("truecolor") | Ok("24bit")
            ) || cfg!(windows),
        };
        app.rebuild_view()?;
        Ok(app)
    }

    #[cfg(feature = "camera")]
    fn open_camera(
        serial: Option<&str>,
        options: &CameraOptions,
    ) -> Result<(Feed, String, (usize, usize)), Failure> {
        let capture = crate::camera::open(serial, options)?;
        let name = format!("{} {}", capture.name(), capture.serial());
        let sensor = (capture.width(), capture.height());
        Ok((Feed::Live(source::LiveFeed::new(capture)), name, sensor))
    }

    #[cfg(not(feature = "camera"))]
    fn open_camera(
        _: Option<&str>,
        _: &CameraOptions,
    ) -> Result<(Feed, String, (usize, usize)), Failure> {
        Err(crate::camera::NO_CAMERA.into())
    }

    fn dt_ms(&self) -> f64 {
        self.dt_us as f64 / 1000.0
    }

    /// Starts the view over — after a change of view or window, or a seek — so a raw view's
    /// trails and the window on screen are rebuilt from the recording rather than carried over.
    fn rebuild_view(&mut self) -> Result<(), Failure> {
        let (width, height) = self.sensor;
        self.view = match self.view_index {
            0 => RenderView::Raw(RawSurface::new(
                width,
                height,
                self.decay_ms.unwrap_or(self.dt_ms()),
            )),
            index => RenderView::Repr {
                // Its time span follows the window, so it covers exactly the events on screen.
                spec: ReprSpec::named_with_span(
                    ReprSpec::NAMES[index - 1],
                    None,
                    Some(self.dt_ms()),
                )?,
                colormap: self.colormap,
                scale: Scale::Auto,
            },
        };
        match &mut self.feed {
            Feed::File(feed) => feed.reset(),
            #[cfg(feature = "camera")]
            Feed::Live(feed) => feed.reset(),
        }
        Ok(())
    }

    fn view_name(&self) -> &'static str {
        match self.view_index {
            0 => "raw",
            index => ReprSpec::NAMES[index - 1],
        }
    }

    fn step(&mut self, elapsed: Duration) -> Result<(), Failure> {
        // A raw view needs the trail behind the window as well, or a seek starts black.
        let lookback = match self.view {
            RenderView::Raw(_) => (self.decay_ms.unwrap_or(self.dt_ms()) * 6_500.0).round() as i64,
            RenderView::Repr { .. } => 0,
        };
        let step = match &mut self.feed {
            Feed::File(feed) => {
                feed.playback.advance(elapsed);
                feed.step(self.dt_us, lookback)?
            }
            #[cfg(feature = "camera")]
            Feed::Live(feed) => feed.step(self.dt_us, self.paused)?,
        };
        let Some(Step { fresh, window }) = step else {
            return Ok(());
        };
        self.stats = Stats::of(&window, self.dt_us);
        if self.history.len() == HISTORY {
            self.history.pop_front();
        }
        self.history.push_back(self.stats.rate as u64);
        // The raw surface keeps its own history, so it is fed only what is new; a representation
        // is computed over the whole window.
        let events = match self.view {
            RenderView::Raw(_) => &fresh,
            RenderView::Repr { .. } => &window,
        };
        match self.view.render(events) {
            Ok(image) => {
                self.image = Some(image);
                self.problem = None;
            }
            Err(error) => {
                self.image = None;
                self.problem = Some(format!("{}: {error}", self.view_name()));
            }
        }
        Ok(())
    }

    /// Handles one key; `false` means quit.
    #[cfg_attr(not(feature = "camera"), allow(irrefutable_let_patterns))]
    fn key(&mut self, code: KeyCode, modifiers: KeyModifiers) -> bool {
        let mut rebuild = false;
        match code {
            KeyCode::Char('q') | KeyCode::Esc => return false,
            KeyCode::Char('c') if modifiers.contains(KeyModifiers::CONTROL) => return false,
            KeyCode::Char(' ') => match &mut self.feed {
                Feed::File(feed) => {
                    let playback = &mut feed.playback;
                    // Play from the start again once the end has been reached.
                    if !playback.playing && playback.cursor_us == playback.duration_us {
                        playback.cursor_us = self.dt_us.min(playback.duration_us);
                        rebuild = true;
                    }
                    playback.playing = !playback.playing;
                }
                #[cfg(feature = "camera")]
                Feed::Live(_) => self.paused = !self.paused,
            },
            KeyCode::Left | KeyCode::Right => {
                if let Feed::File(feed) = &mut self.feed {
                    let second = if code == KeyCode::Left {
                        -1_000_000
                    } else {
                        1_000_000
                    };
                    feed.playback.seek(second);
                    rebuild = true;
                }
            }
            KeyCode::Char('+') | KeyCode::Char('=') | KeyCode::Char('-') => {
                if let Feed::File(feed) = &mut self.feed {
                    let factor = if code == KeyCode::Char('-') { 0.5 } else { 2.0 };
                    feed.playback.speed = (feed.playback.speed * factor).clamp(1.0 / 64.0, 64.0);
                }
            }
            KeyCode::Char('[') | KeyCode::Char(']') => {
                let dt_us = if code == KeyCode::Char('[') {
                    self.dt_us / 2
                } else {
                    self.dt_us * 2
                };
                self.dt_us = dt_us.clamp(DT_RANGE_US.0, DT_RANGE_US.1);
                rebuild = true;
            }
            KeyCode::Char('r') => {
                self.view_index = (self.view_index + 1) % (ReprSpec::NAMES.len() + 1);
                rebuild = true;
            }
            _ => {}
        }
        // Every name in `NAMES` builds, so this cannot fail for a key press.
        if rebuild {
            let _ = self.rebuild_view();
        }
        true
    }

    fn draw(&self, frame: &mut Frame) {
        let [body, footer] =
            Layout::vertical([Constraint::Min(1), Constraint::Length(1)]).areas(frame.area());
        let [picture, panel] =
            Layout::horizontal([Constraint::Min(10), Constraint::Length(36)]).areas(body);

        match (&self.image, &self.problem) {
            (Some(image), _) => frame.render_widget(
                HalfBlock {
                    image,
                    truecolor: self.truecolor,
                },
                picture,
            ),
            (None, Some(problem)) => {
                frame.render_widget(Paragraph::new(problem.as_str()).red(), picture)
            }
            (None, None) => {}
        }
        self.draw_panel(frame, panel);
        frame.render_widget(
            Line::from(" q quit · space pause · ←/→ seek · +/- speed · [/] window · r view").dim(),
            footer,
        );
    }

    fn draw_panel(&self, frame: &mut Frame, area: Rect) {
        let row = |key: &str, value: String| Line::from(format!("{key:>10}  {value}"));
        let (width, height) = self.sensor;
        let mut lines = vec![row("sensor", format!("{width} x {height}"))];
        match &self.feed {
            Feed::File(feed) => {
                let playback = &feed.playback;
                lines.push(row("events", format_count(feed.reader.n_events() as u64)));
                lines.push(row(
                    "position",
                    format!(
                        "{} / {}",
                        format_time(playback.cursor_us),
                        format_time(playback.duration_us)
                    ),
                ));
                lines.push(row(
                    "state",
                    if playback.playing {
                        format!("playing ×{}", format_speed(playback.speed))
                    } else {
                        "paused".to_owned()
                    },
                ));
            }
            #[cfg(feature = "camera")]
            Feed::Live(feed) => {
                let capture = &feed.capture;
                let state = if self.paused { "paused" } else { "live" };
                lines.push(row("state", state.to_owned()));
                lines.push(row("backlog", format!("{} buffers", capture.backlog())));
                lines.push(row("overflows", format_count(feed.overflows as u64)));
                lines.push(row(
                    "dropped",
                    format!(
                        "{} buffers",
                        format_count(capture.n_budget_dropped() as u64)
                    ),
                ));
                if let Some(bias) = capture.bias_state() {
                    lines.push(row("bias rate", format_rate(bias.event_rate)));
                    lines.push(row(
                        "target",
                        format!(
                            "{} – {}",
                            format_rate(bias.target_rate.0),
                            format_rate(bias.target_rate.1)
                        ),
                    ));
                    lines.push(row(
                        "control",
                        if bias.calibrating {
                            "calibrating".to_owned()
                        } else {
                            format!("{:?}, {} slow steps", bias.authority, bias.slow_steps)
                        },
                    ));
                }
            }
        }
        lines.push(row("window", format!("{} ms", self.dt_ms())));
        lines.push(row("view", self.view_name().to_owned()));
        lines.push(Line::default());

        let Stats {
            total,
            positive,
            rate,
        } = self.stats;
        let share = |count: usize| {
            if total == 0 {
                String::new()
            } else {
                format!(" ({:.0}%)", 100.0 * count as f64 / total as f64)
            }
        };
        lines.push(row("events", format_count(total as u64)));
        lines.push(row(
            "ON",
            format!("{}{}", format_count(positive as u64), share(positive)),
        ));
        let negative = total - positive;
        lines.push(row(
            "OFF",
            format!("{}{}", format_count(negative as u64), share(negative)),
        ));
        // Live rates count what was decoded; past the budget some buffers never were.
        let label = if matches!(self.feed, Feed::File(_)) {
            "rate"
        } else {
            "decoded"
        };
        lines.push(row(label, format_rate(rate)));

        let block = Block::bordered().title(format!(" {} ", self.name));
        let inner = block.inner(area);
        frame.render_widget(block, area);
        let text_height = (lines.len() as u16).min(inner.height);
        let [text, chart] =
            Layout::vertical([Constraint::Length(text_height), Constraint::Min(0)]).areas(inner);
        frame.render_widget(Paragraph::new(lines), text);
        let shown = usize::from(chart.width).min(self.history.len());
        let recent: Vec<u64> = self
            .history
            .iter()
            .skip(self.history.len() - shown)
            .copied()
            .collect();
        frame.render_widget(
            Sparkline::default()
                .data(&recent)
                .style(Style::new().cyan()),
            chart,
        );
    }
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

fn format_rate(rate: f64) -> String {
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

    fn parse(args: &[&str]) -> Result<TuiArgs, clap::Error> {
        Standalone::try_parse_from(std::iter::once("eventcv tui").chain(args.iter().copied()))
            .map(|standalone| standalone.args)
    }

    #[test]
    fn a_path_or_a_camera_is_required() {
        assert!(parse(&[]).is_err());
        assert!(parse(&["rec.h5", "--camera"]).is_err());
        assert_eq!(parse(&["rec.h5"]).unwrap().path.as_deref(), Some("rec.h5"));
        assert_eq!(parse(&["--camera"]).unwrap().camera, Some(None));
        assert_eq!(
            parse(&["--camera", "00051234"]).unwrap().camera,
            Some(Some("00051234".to_owned()))
        );
    }

    #[test]
    fn defaults_match_the_player() {
        let args = parse(&["rec.h5"]).unwrap();
        assert_eq!((args.dt_ms, args.speed, args.refresh_hz), (30.0, 1.0, 30.0));
        assert_eq!(
            (args.repr.as_str(), args.colormap.as_str()),
            ("raw", "viridis")
        );
    }

    fn example() -> App {
        let path = concat!(env!("CARGO_MANIFEST_DIR"), "/../../data/test/example.npz");
        App::new(&parse(&[path]).unwrap()).unwrap()
    }

    #[test]
    fn draws_a_recording() {
        let mut app = example();
        app.step(Duration::ZERO).unwrap();
        let backend = ratatui::backend::TestBackend::new(100, 30);
        let mut terminal = ratatui::Terminal::new(backend).unwrap();
        terminal.draw(|frame| app.draw(frame)).unwrap();
        let screen: String = terminal
            .backend()
            .buffer()
            .content()
            .iter()
            .map(|cell| cell.symbol())
            .collect();
        for text in ["example.npz", "640 x 480", "106,295", "playing", "▀"] {
            assert!(screen.contains(text), "{text:?} not on screen");
        }
    }

    #[test]
    fn every_view_draws() {
        let mut app = example();
        for _ in 0..=ReprSpec::NAMES.len() {
            app.step(Duration::from_millis(5)).unwrap();
            assert!(
                app.image.is_some(),
                "{}: {:?}",
                app.view_name(),
                app.problem
            );
            assert!(app.key(KeyCode::Char('r'), KeyModifiers::NONE));
        }
        assert_eq!(app.view_name(), "raw");
    }

    #[test]
    fn keys_steer_playback() {
        let mut app = example();
        assert!(app.key(KeyCode::Char(']'), KeyModifiers::NONE));
        assert_eq!(app.dt_us, 60_000);
        assert!(app.key(KeyCode::Char('-'), KeyModifiers::NONE));
        assert!(matches!(&app.feed, Feed::File(feed) if feed.playback.speed == 0.5));
        assert!(!app.key(KeyCode::Char('q'), KeyModifiers::NONE));
        assert!(!app.key(KeyCode::Char('c'), KeyModifiers::CONTROL));
    }

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
}
