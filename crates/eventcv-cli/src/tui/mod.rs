//! `eventcv tui`: a recording or a live camera, watched in the terminal.
//!
//! The quick look that does not need a GPU window — over SSH, on a headless capture box, or just
//! to check a camera is producing what it should. The events are drawn in half blocks next to the
//! diagnostics that matter when something is off: rates and polarity balance, and for a camera the
//! driver backlog, overflows, decode drops and the adaptive-bias state.

#[cfg(feature = "camera")]
mod biases;
mod draw;
mod home;
mod source;
mod theme;
mod ui;

use std::collections::VecDeque;
use std::ffi::OsString;
use std::io::IsTerminal;
use std::time::{Duration, Instant};

use clap::{ArgGroup, Args, Parser};
use eventcv_core::representation::ReprSpec;
use eventcv_core::viz::{Colormap, RawSurface, RenderView, Rgb8Image, Scale};
use eventcv_core::EventStream;
use ratatui::crossterm::event::{self, Event, KeyCode, KeyEventKind, KeyModifiers};

use crate::camera::CameraOptions;
use crate::commands::{duration_us, parse_colormap, Source};
use crate::Failure;
use home::{Action, Home, Target};
use source::{FileFeed, Step};
use theme::Theme;

/// Window lengths `[` and `]` stay within, as in the GUI player.
const DT_RANGE_US: (i64, i64) = (100, 1_000_000);
/// Rate samples kept for the sparkline — more than any terminal is wide.
const HISTORY: usize = 512;

// Neither a path nor `--camera` opens the home screen to pick one from.
#[derive(Args, Clone, Debug)]
#[command(group(ArgGroup::new("input").args(["path", "camera"])))]
pub(crate) struct TuiArgs {
    /// recording to play (default: the home screen, to pick a recording or camera)
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
    let mut home =
        (args.path.is_none() && args.camera.is_none()).then(|| Home::new(Theme::detect()));
    let mut viewer = match home {
        Some(_) => None,
        None => Some(App::new(&args)?),
    };
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
        match (&mut viewer, &mut home) {
            (Some(app), _) => app.step(now - last)?,
            (None, Some(home)) => home.tick(),
            (None, None) => {}
        }
        last = now;
        terminal.draw(|frame| match (&viewer, &home) {
            (Some(app), _) => app.draw(frame),
            (None, Some(home)) => home.draw(frame),
            (None, None) => unreachable!("there is always a viewer or the home screen"),
        })?;
        let deadline = now + tick;
        while let Some(remaining) = deadline.checked_duration_since(Instant::now()) {
            if !event::poll(remaining)? {
                break;
            }
            let Event::Key(key) = event::read()? else {
                continue;
            };
            // Windows reports releases too; acting on those would fire every key twice.
            if key.kind != KeyEventKind::Press {
                continue;
            }
            // Ctrl+C leaves from anywhere; `q` only closes the screen it is pressed on.
            if key.code == KeyCode::Char('c') && key.modifiers.contains(KeyModifiers::CONTROL) {
                return Ok(());
            }
            if let Some(app) = &mut viewer {
                if !app.key(key.code, key.modifiers) {
                    // Closing the viewer releases its camera, so the home screen lists it again.
                    viewer = None;
                    match &mut home {
                        Some(home) => home.refresh(),
                        None => return Ok(()),
                    }
                }
            } else if let Some(home) = &mut home {
                match home.key(key.code) {
                    Action::Stay => {}
                    Action::Quit => return Ok(()),
                    Action::Open(target) => {
                        let mut chosen = args.clone();
                        match target {
                            Target::File(path) => chosen.path = Some(path.display().to_string()),
                            Target::Camera(serial) => chosen.camera = Some(serial),
                        }
                        match App::new(&chosen) {
                            Ok(mut app) => {
                                app.back = true;
                                home.status = None;
                                viewer = Some(app);
                            }
                            Err(failure) => home.status = Some(failure.message),
                        }
                    }
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
    /// The bias panel, while `b` has it open over the camera box.
    #[cfg(feature = "camera")]
    biases: Option<biases::BiasPanel>,
    image: Option<Rgb8Image>,
    /// Why the current view could not be drawn, shown in place of the image.
    problem: Option<String>,
    stats: Stats,
    history: VecDeque<u64>,
    theme: Theme,
    /// Opened from the home screen, which `q` returns to.
    back: bool,
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
            #[cfg(feature = "camera")]
            biases: None,
            image: None,
            problem: None,
            stats: Stats::default(),
            history: VecDeque::with_capacity(HISTORY),
            theme: Theme::detect(),
            back: false,
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
            Feed::Live(feed) => {
                if let Some(panel) = &mut self.biases {
                    panel.refresh(&feed.capture);
                }
                feed.step(self.dt_us, self.paused)?
            }
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
        #[cfg(feature = "camera")]
        if self.bias_key(code) {
            return true;
        }
        let mut rebuild = false;
        match code {
            KeyCode::Char('q') => return false,
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
}

#[cfg(feature = "camera")]
impl App {
    /// Handles a key aimed at the bias panel: `b` opens and closes it, and while it is open the
    /// panel takes the keys it knows. `false` leaves the key to the viewer.
    fn bias_key(&mut self, code: KeyCode) -> bool {
        let Feed::Live(feed) = &mut self.feed else {
            return false;
        };
        if let Some(panel) = &mut self.biases {
            if panel.key(&mut feed.capture, code) {
                return true;
            }
        }
        match (&self.biases, code) {
            (Some(_), KeyCode::Char('b')) => self.biases = None,
            (None, KeyCode::Char('b')) => {
                self.biases = Some(biases::BiasPanel::open(&feed.capture));
            }
            _ => return false,
        }
        true
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
    fn a_path_and_a_camera_conflict() {
        let home = parse(&[]).unwrap();
        assert!(home.path.is_none() && home.camera.is_none());
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
        for text in [
            "EventCV",
            "example.npz",
            "640×480",
            "106,295",
            "stream",
            "▀",
        ] {
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
}
