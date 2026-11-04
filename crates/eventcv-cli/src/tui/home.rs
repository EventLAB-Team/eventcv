//! The home screen that a bare `eventcv` opens: the cameras plugged in, and the recordings in the
//! current folder, either of which opens in the viewer — which returns here when it is closed.
//!
//! ```text
//! ╭─ ◉ EventCV ─────────────────────────────────────────── v1.1.0 ─╮
//! │   ·   ███████ ██    ██ ███████ ███    ██ ████████  ██████ ██    ██ │
//! │       The open-source event-based computer vision library.   •    │
//! │       https://eventcv.net · https://github.com/EventLAB-Team/eventcv │
//! │  ╭─ cameras ───────────────────╮╭─ recordings · ~/data ───────╮  │
//! │  │ ▶ iniVation DAVIS 346       ││   ..                        │  │
//! │  │   00000286 · super · 2:5    ││   test/                     │  │
//! ╰─ ↑↓ select · tab switch · enter open · ⌫ up · r refresh · q quit ─╯
//! ```
//!
//! The title moves a little, the way an event camera would see it: every few seconds an edge sweeps
//! across the wordmark, lighting the letters it reaches in ON and the ones it leaves in OFF, and
//! bursts of events keep glinting off the letters' edges, as if light were catching them.

use std::path::PathBuf;
use std::time::Instant;

use ratatui::crossterm::event::KeyCode;
use ratatui::layout::{Constraint, Flex, Layout, Rect};
use ratatui::style::{Modifier, Style};
use ratatui::text::{Line, Span, Text};
use ratatui::widgets::{List, ListItem, ListState, Paragraph};
use ratatui::Frame;

use super::theme::Theme;
use super::ui::{badge, boxed, key_hints};

/// "EVENTCV" in block letters; the first five are drawn in ON red, the rest in OFF blue.
const WORDMARK: [[&str; 5]; 7] = [
    ["███████", "██     ", "█████  ", "██     ", "███████"],
    ["██    ██", "██    ██", "██    ██", " ██  ██ ", "  ████  "],
    ["███████", "██     ", "█████  ", "██     ", "███████"],
    [
        "███    ██",
        "████   ██",
        "██ ██  ██",
        "██  ██ ██",
        "██   ████",
    ],
    ["████████", "   ██   ", "   ██   ", "   ██   ", "   ██   "],
    [" ██████", "██     ", "██     ", "██     ", " ██████"],
    ["██    ██", "██    ██", "██    ██", " ██  ██ ", "  ████  "],
];

const TAGLINE: &str = "The open-source event-based computer vision library.";
const WEBSITE: &str = "https://eventcv.net";
const REPOSITORY: &str = "https://github.com/EventLAB-Team/eventcv";

/// Seconds between ripples, and how fast one crosses the wordmark (columns per second): a sweep
/// of about a second and a half, then a long rest.
const RIPPLE_PERIOD: f64 = 6.0;
const RIPPLE_SPEED: f64 = 45.0;
/// How long a sparkle lives, the gap between bursts (`min` plus up to `spread`), and how many
/// events a burst holds (`min` plus up to `spread`).
const SPARKLE_LIFE: f64 = 0.9;
const BURST_GAP: (f64, f64) = (0.4, 0.8);
const BURST_SIZE: (usize, usize) = (6, 8);

/// Where the ripple is over `column` at `clock` seconds: `Some(true)` just as the edge arrives (an
/// ON event), `Some(false)` just after it has passed (OFF), `None` otherwise.
fn ripple(column: usize, clock: f64) -> Option<bool> {
    // Starts a few columns off the left so the first letter is reached, not appeared on.
    let front = (clock % RIPPLE_PERIOD) * RIPPLE_SPEED - 6.0;
    match front - column as f64 {
        passed if (0.0..3.0).contains(&passed) => Some(true),
        passed if (3.0..7.0).contains(&passed) => Some(false),
        _ => None,
    }
}

/// One flicker, at a cell relative to the wordmark's top-left corner (so it may sit just outside).
struct Sparkle {
    x: i32,
    y: i32,
    born: f64,
    on: bool,
}

/// The wordmark's lit cells, row by row, letters one blank column apart.
fn wordmark_cells() -> Vec<Vec<bool>> {
    (0..5)
        .map(|row| {
            let glyphs: Vec<&str> = WORDMARK.iter().map(|letter| letter[row]).collect();
            glyphs.join(" ").chars().map(|cell| cell != ' ').collect()
        })
        .collect()
}

/// Lit cells with a dark side: where a glint off the letters would come from.
fn edges(cells: &[Vec<bool>]) -> Vec<(i32, i32)> {
    let lit = |x: i32, y: i32| {
        usize::try_from(y)
            .ok()
            .and_then(|y| cells.get(y))
            .zip(usize::try_from(x).ok())
            .is_some_and(|(row, x)| row.get(x).copied().unwrap_or(false))
    };
    let mut edges = Vec::new();
    for (y, row) in cells.iter().enumerate() {
        for x in 0..row.len() {
            let (x, y) = (x as i32, y as i32);
            let dark_side = [(1, 0), (-1, 0), (0, 1), (0, -1)]
                .iter()
                .any(|(dx, dy)| !lit(x + dx, y + dy));
            if lit(x, y) && dark_side {
                edges.push((x, y));
            }
        }
    }
    edges
}

/// xorshift64: plenty for where a sparkle lands, and no dependency for it.
struct Rng(u64);

impl Rng {
    fn unit(&mut self) -> f64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        (self.0 >> 11) as f64 / (1u64 << 53) as f64
    }
}

/// What the viewer should open.
pub(super) enum Target {
    File(PathBuf),
    /// A camera serial; `None` opens the first one found.
    Camera(Option<String>),
}

pub(super) enum Action {
    Stay,
    Open(Target),
    Quit,
}

#[derive(Clone, Copy, PartialEq)]
enum Pane {
    Cameras,
    Files,
}

struct Camera {
    name: String,
    detail: String,
    serial: Option<String>,
}

enum Entry {
    Parent,
    Dir(String),
    File(String, u64),
}

pub(super) struct Home {
    theme: Theme,
    dir: PathBuf,
    entries: Vec<Entry>,
    cameras: Vec<Camera>,
    /// Why the camera list may be empty when it should not be — or that this build has no cameras.
    camera_note: Option<String>,
    focus: Pane,
    selected_camera: usize,
    selected_file: usize,
    /// The last thing that went wrong, such as a camera that would not open.
    pub status: Option<String>,
    started: Instant,
    /// Seconds since the home screen opened, as of the last [`tick`](Self::tick).
    clock: f64,
    sparkles: Vec<Sparkle>,
    next_burst: f64,
    rng: Rng,
}

impl Home {
    pub fn new(theme: Theme) -> Self {
        let mut home = Self {
            theme,
            dir: std::env::current_dir().unwrap_or_else(|_| PathBuf::from(".")),
            entries: Vec::new(),
            cameras: Vec::new(),
            camera_note: None,
            focus: Pane::Files,
            selected_camera: 0,
            selected_file: 0,
            status: None,
            started: Instant::now(),
            clock: 0.0,
            sparkles: Vec::new(),
            next_burst: 1.0,
            // Seeded from the clock so each run sparkles differently; never zero, or xorshift sticks.
            rng: Rng(std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map_or(1, |since| since.as_nanos() as u64 | 1)),
        };
        home.refresh();
        if !home.cameras.is_empty() {
            home.focus = Pane::Cameras;
        }
        home
    }

    /// Moves the animation on to now.
    pub fn tick(&mut self) {
        self.advance(self.started.elapsed().as_secs_f64());
    }

    /// Moves the animation to `clock` seconds: retires faded sparkles and, when one is due, starts
    /// a burst — events scattered around a point on a letter's edge, born over a fraction of a
    /// second.
    fn advance(&mut self, clock: f64) {
        self.clock = clock;
        self.sparkles
            .retain(|sparkle| clock - sparkle.born < SPARKLE_LIFE);
        if clock < self.next_burst {
            return;
        }
        let edges = edges(&wordmark_cells());
        let (x, y) = edges[(self.rng.unit() * edges.len() as f64) as usize % edges.len()];
        let count = BURST_SIZE.0 + (self.rng.unit() * BURST_SIZE.1 as f64) as usize;
        for _ in 0..count {
            // Within three columns and one row of the edge: a cell is about twice as tall as wide.
            let sparkle = Sparkle {
                x: x + ((self.rng.unit() - 0.5) * 7.0).round() as i32,
                y: y + ((self.rng.unit() - 0.5) * 3.0).round() as i32,
                born: clock + self.rng.unit() * 0.35,
                on: self.rng.unit() < 0.5,
            };
            self.sparkles.push(sparkle);
        }
        self.next_burst = clock + BURST_GAP.0 + self.rng.unit() * BURST_GAP.1;
    }

    /// Lists the cameras and the current folder again — after a camera is plugged in, or when the
    /// viewer hands one back.
    pub fn refresh(&mut self) {
        self.scan_cameras();
        self.read_dir();
    }

    #[cfg(feature = "camera")]
    fn scan_cameras(&mut self) {
        let listing = eventcv_core::device::list_cameras();
        self.cameras = listing
            .cameras
            .into_iter()
            .map(|camera| Camera {
                detail: format!(
                    "{} · {} USB",
                    camera.serial.as_deref().unwrap_or("serial unknown"),
                    camera.speed
                ),
                name: camera.name,
                serial: camera.serial,
            })
            .collect();
        self.camera_note = listing.warning;
        self.selected_camera = self
            .selected_camera
            .min(self.cameras.len().saturating_sub(1));
    }

    #[cfg(not(feature = "camera"))]
    fn scan_cameras(&mut self) {
        self.camera_note = Some("this build of eventcv has no camera support".to_owned());
    }

    fn read_dir(&mut self) {
        let mut dirs = Vec::new();
        let mut files = Vec::new();
        match std::fs::read_dir(&self.dir) {
            Ok(listing) => {
                for entry in listing.flatten() {
                    let name = entry.file_name().to_string_lossy().into_owned();
                    let Ok(metadata) = entry.metadata() else {
                        continue;
                    };
                    if name.starts_with('.') {
                        continue;
                    }
                    let path = entry.path();
                    if metadata.is_dir() {
                        dirs.push(name);
                    // Every event format can be read back except E2VID's `.zip`, which is an export.
                    } else if eventcv_core::io::supports_event_append(&path)
                        && !eventcv_core::io::is_e2vid_target(&path, None)
                    {
                        files.push((name, metadata.len()));
                    }
                }
            }
            Err(error) => self.status = Some(format!("{}: {error}", self.dir.display())),
        }
        dirs.sort();
        files.sort();
        self.entries = self
            .dir
            .parent()
            .map(|_| Entry::Parent)
            .into_iter()
            .chain(dirs.into_iter().map(Entry::Dir))
            .chain(
                files
                    .into_iter()
                    .map(|(name, size)| Entry::File(name, size)),
            )
            .collect();
        self.selected_file = self.selected_file.min(self.entries.len().saturating_sub(1));
    }

    fn enter(&mut self, dir: PathBuf) {
        self.dir = dir;
        self.selected_file = 0;
        self.read_dir();
    }

    pub fn key(&mut self, code: KeyCode) -> Action {
        let (selected, len) = match self.focus {
            Pane::Cameras => (&mut self.selected_camera, self.cameras.len()),
            Pane::Files => (&mut self.selected_file, self.entries.len()),
        };
        match code {
            KeyCode::Char('q') | KeyCode::Esc => return Action::Quit,
            KeyCode::Tab | KeyCode::BackTab | KeyCode::Left | KeyCode::Right => {
                self.focus = match self.focus {
                    Pane::Cameras => Pane::Files,
                    Pane::Files => Pane::Cameras,
                };
            }
            KeyCode::Up | KeyCode::Char('k') => *selected = selected.saturating_sub(1),
            KeyCode::Down | KeyCode::Char('j') => {
                *selected = (*selected + 1).min(len.saturating_sub(1));
            }
            KeyCode::Backspace => {
                if let Some(parent) = self.dir.parent() {
                    self.enter(parent.to_path_buf());
                }
            }
            KeyCode::Char('r') => {
                self.status = None;
                self.refresh();
            }
            KeyCode::Enter => match self.focus {
                Pane::Cameras => {
                    if let Some(camera) = self.cameras.get(self.selected_camera) {
                        return Action::Open(Target::Camera(camera.serial.clone()));
                    }
                }
                Pane::Files => match self.entries.get(self.selected_file) {
                    Some(Entry::Parent) => {
                        if let Some(parent) = self.dir.parent() {
                            self.enter(parent.to_path_buf());
                        }
                    }
                    Some(Entry::Dir(name)) => self.enter(self.dir.join(name)),
                    Some(Entry::File(name, _)) => {
                        return Action::Open(Target::File(self.dir.join(name)))
                    }
                    None => {}
                },
            },
            _ => {}
        }
        Action::Stay
    }

    pub fn draw(&self, frame: &mut Frame) {
        let theme = &self.theme;
        let version = Line::from(vec![Span::styled(
            format!(" v{} ", env!("CARGO_PKG_VERSION")),
            theme.label(),
        )]);
        let keys = [
            ("↑↓", "select"),
            ("tab", "switch"),
            ("enter", "open"),
            ("⌫", "up"),
            ("r", "refresh"),
            ("q", "quit"),
        ];
        let block = boxed(theme, badge(theme))
            .title(version.right_aligned())
            .title_bottom(key_hints(theme, &keys));
        let body = block.inner(frame.area());
        frame.render_widget(block, frame.area());

        // The block letters need 60 columns and 5 rows; a smaller terminal gets one line instead.
        let wordmark = if body.width >= 64 && body.height >= 18 {
            5
        } else {
            0
        };
        // Centred as one block, with the lists no taller than a screenful of recordings needs.
        let [_, title, _, tagline, links, _, panes, status] = Layout::vertical([
            Constraint::Length(1),
            Constraint::Length(wordmark),
            Constraint::Length(1),
            Constraint::Length(1),
            Constraint::Length(1),
            Constraint::Length(1),
            Constraint::Max(18),
            Constraint::Length(1),
        ])
        .flex(Flex::Center)
        .areas(body);
        if wordmark > 0 {
            frame.render_widget(Paragraph::new(self.wordmark()).centered(), title);
        }
        frame.render_widget(
            Paragraph::new(Line::styled(
                TAGLINE,
                Style::new().fg(theme.muted).add_modifier(Modifier::ITALIC),
            ))
            .centered(),
            tagline,
        );
        // Plain text, so any terminal that recognises a URL makes it clickable.
        let link = Style::new()
            .fg(theme.off)
            .add_modifier(Modifier::UNDERLINED);
        frame.render_widget(
            Paragraph::new(Line::from(vec![
                Span::styled(WEBSITE, link),
                Span::styled("  ·  ", Style::new().fg(theme.border)),
                Span::styled(REPOSITORY, link),
            ]))
            .centered(),
            links,
        );
        if wordmark > 0 {
            self.draw_sparkles(frame, title);
        }

        let [panes] = Layout::horizontal([Constraint::Max(110)])
            .flex(Flex::Center)
            .areas(panes);
        let [cameras, files] =
            Layout::horizontal([Constraint::Percentage(45), Constraint::Percentage(55)])
                .spacing(1)
                .areas(panes);
        self.draw_cameras(frame, cameras);
        self.draw_files(frame, files);

        if let Some(message) = &self.status {
            frame.render_widget(
                Paragraph::new(Line::styled(message.as_str(), Style::new().fg(theme.on)))
                    .centered(),
                status,
            );
        }
    }

    /// The block letters, coloured cell by cell so the ripple can pass through them.
    fn wordmark(&self) -> Vec<Line<'static>> {
        (0..5)
            .map(|row| {
                let mut spans = Vec::new();
                let mut column = 0;
                for (index, letter) in WORDMARK.iter().enumerate() {
                    if index > 0 {
                        spans.push(Span::raw(" "));
                        column += 1;
                    }
                    let base = if index < 5 {
                        self.theme.on
                    } else {
                        self.theme.off
                    };
                    for cell in letter[row].chars() {
                        let color = match ripple(column, self.clock) {
                            Some(on) if cell != ' ' => self.theme.flash(on),
                            _ => base,
                        };
                        spans.push(Span::styled(cell.to_string(), Style::new().fg(color)));
                        column += 1;
                    }
                }
                Line::from(spans)
            })
            .collect()
    }

    /// Sparkles over the wordmark drawn centred in `title`: a dot where one lands on the dark,
    /// and a brief glint where one lands on a letter.
    fn draw_sparkles(&self, frame: &mut Frame, title: Rect) {
        let width = wordmark_cells()[0].len() as i32;
        let left = i32::from(title.x) + (i32::from(title.width) - width) / 2;
        let buf = frame.buffer_mut();
        for sparkle in &self.sparkles {
            let age = self.clock - sparkle.born;
            if age < 0.0 {
                continue; // not born yet: a burst spreads its events over a moment
            }
            let (Ok(x), Ok(y)) = (
                u16::try_from(left + sparkle.x),
                u16::try_from(i32::from(title.y) + sparkle.y),
            ) else {
                continue;
            };
            if !buf.area.contains((x, y).into()) {
                continue;
            }
            // Bright for most of its life, then gone quickly, as an event's afterglow is.
            let fade = (1.0 - age / SPARKLE_LIFE).sqrt();
            let cell = &mut buf[(x, y)];
            match cell.symbol() {
                " " => {
                    cell.set_char(if fade > 0.7 { '•' } else { '·' })
                        .set_fg(self.theme.polarity(sparkle.on, fade));
                }
                "█" if fade > 0.6 => {
                    cell.set_fg(self.theme.flash(sparkle.on));
                }
                _ => {}
            }
        }
    }

    /// A pane, its border lit when it has the focus, with its list and selection drawn inside.
    fn draw_list(
        &self,
        frame: &mut Frame,
        area: Rect,
        title: Line<'static>,
        pane: Pane,
        items: Vec<ListItem<'static>>,
        selected: usize,
    ) {
        let theme = &self.theme;
        let focused = self.focus == pane;
        let mut block = boxed(theme, title);
        if focused {
            block = block.border_style(Style::new().fg(theme.off));
        }
        let highlight = if focused {
            Style::new().bg(theme.border).add_modifier(Modifier::BOLD)
        } else {
            Style::new()
        };
        let list = List::new(items)
            .block(block)
            .highlight_style(highlight)
            .highlight_symbol(Line::styled("▶ ", Style::new().fg(theme.on)));
        let mut state = ListState::default().with_selected(focused.then_some(selected));
        frame.render_stateful_widget(list, area, &mut state);
    }

    fn draw_cameras(&self, frame: &mut Frame, area: Rect) {
        let theme = &self.theme;
        let items = if self.cameras.is_empty() {
            let mut lines = vec![Line::styled("  no cameras found", theme.label())];
            if let Some(note) = &self.camera_note {
                lines.push(Line::styled(
                    format!("  {note}"),
                    Style::new().fg(theme.warn),
                ));
            }
            vec![ListItem::new(Text::from(lines))]
        } else {
            self.cameras
                .iter()
                .map(|camera| {
                    ListItem::new(Text::from(vec![
                        Line::styled(camera.name.clone(), theme.value()),
                        Line::styled(camera.detail.clone(), theme.label()),
                    ]))
                })
                .collect()
        };
        let title = Line::from(vec![Span::styled(" cameras ", theme.value())]);
        self.draw_list(
            frame,
            area,
            title,
            Pane::Cameras,
            items,
            self.selected_camera,
        );
    }

    fn draw_files(&self, frame: &mut Frame, area: Rect) {
        let theme = &self.theme;
        // Inside the border and past the highlight symbol.
        let width = usize::from(area.width).saturating_sub(4);
        let items = self
            .entries
            .iter()
            .map(|entry| match entry {
                Entry::Parent => ListItem::new(Line::styled("..", Style::new().fg(theme.off))),
                Entry::Dir(name) => {
                    ListItem::new(Line::styled(format!("{name}/"), Style::new().fg(theme.off)))
                }
                Entry::File(name, size) => {
                    let size = format_bytes(*size);
                    let gap = width
                        .saturating_sub(name.chars().count() + size.len())
                        .max(1);
                    ListItem::new(Line::from(vec![
                        Span::styled(name.clone(), Style::new().fg(theme.text)),
                        Span::raw(" ".repeat(gap)),
                        Span::styled(size, theme.label()),
                    ]))
                }
            })
            .collect();
        let title = Line::from(vec![
            Span::styled(" recordings ", theme.value()),
            Span::styled(format!("{} ", shorten(&self.dir, width / 2)), theme.label()),
        ]);
        self.draw_list(frame, area, title, Pane::Files, items, self.selected_file);
    }
}

/// `~/data/test`, cut from the left to at most `width` characters so the deepest part survives.
fn shorten(dir: &std::path::Path, width: usize) -> String {
    let mut text = dir.display().to_string();
    if let Some(home) = std::env::var_os("HOME").map(PathBuf::from) {
        if let Ok(rest) = dir.strip_prefix(&home) {
            text = format!("~/{}", rest.display())
                .trim_end_matches('/')
                .to_owned();
        }
    }
    let count = text.chars().count();
    if count > width && width > 1 {
        let tail: String = text.chars().skip(count - (width - 1)).collect();
        text = format!("…{tail}");
    }
    text
}

fn format_bytes(bytes: u64) -> String {
    let mut value = bytes as f64;
    for unit in ["B", "KB", "MB", "GB"] {
        if value < 1000.0 {
            return if unit == "B" {
                format!("{bytes} B")
            } else {
                format!("{value:.1} {unit}")
            };
        }
        value /= 1000.0;
    }
    format!("{value:.1} TB")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wordmark_rows_line_up() {
        for letter in WORDMARK {
            let width = letter[0].chars().count();
            assert!(letter.iter().all(|row| row.chars().count() == width));
        }
    }

    #[test]
    fn the_ripple_is_on_where_the_edge_arrives_and_off_behind_it() {
        // At 1 s the front is at 45 - 6 = 39 columns.
        assert_eq!(ripple(38, 1.0), Some(true));
        assert_eq!(ripple(36, 1.0), Some(false));
        assert_eq!(ripple(30, 1.0), None);
        assert_eq!(ripple(45, 1.0), None);
        // And it rests between sweeps.
        assert!((0..60).all(|column| ripple(column, 5.0).is_none()));
    }

    #[test]
    fn sparkles_come_in_bursts_off_the_letters_and_fade() {
        let mut home = Home::new(Theme::new(true));
        home.advance(0.5);
        assert!(home.sparkles.is_empty());
        home.advance(1.0);
        let burst = home.sparkles.len();
        assert!((6..14).contains(&burst), "{burst}");
        assert!(home.next_burst >= 1.0 + BURST_GAP.0);
        // Every event within reach of a letter's edge.
        let edges = edges(&wordmark_cells());
        assert!(home.sparkles.iter().all(|sparkle| edges
            .iter()
            .any(|&(x, y)| (sparkle.x - x).abs() <= 4 && (sparkle.y - y).abs() <= 2)));
        // The first burst has faded by now (a new one may have begun).
        home.advance(1.0 + 0.35 + SPARKLE_LIFE);
        assert!(home
            .sparkles
            .iter()
            .all(|sparkle| sparkle.born > 1.0 + 0.35));
    }

    #[test]
    fn edges_are_lit_cells_beside_dark_ones() {
        let cells = wordmark_cells();
        assert_eq!(cells[0].len(), 60);
        let edges = edges(&cells);
        // The E's top-left corner is an edge; the middle of the T's stem is too (dark both sides).
        assert!(edges.contains(&(0, 0)));
        assert!(edges.iter().all(|&(x, y)| cells[y as usize][x as usize]));
    }

    #[test]
    fn sizes_read_naturally() {
        assert_eq!(format_bytes(512), "512 B");
        assert_eq!(format_bytes(2_500), "2.5 KB");
        assert_eq!(format_bytes(5_400_000_000), "5.4 GB");
    }

    #[test]
    fn long_folders_keep_their_tail() {
        let dir = std::path::Path::new("/media/adam/vprdatasets/data/test");
        assert_eq!(shorten(dir, 12), "…s/data/test");
    }

    #[test]
    fn folders_and_recordings_are_listed_and_entered() {
        let data = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../data");
        let mut home = Home::new(Theme::new(true));
        home.focus = Pane::Files;
        home.enter(data.canonicalize().unwrap());
        let test = home
            .entries
            .iter()
            .position(|entry| matches!(entry, Entry::Dir(name) if name == "test"))
            .unwrap();
        home.selected_file = test;
        assert!(matches!(home.key(KeyCode::Enter), Action::Stay));
        let names: Vec<_> = home
            .entries
            .iter()
            .filter_map(|entry| match entry {
                Entry::File(name, _) => Some(name.as_str()),
                _ => None,
            })
            .collect();
        assert!(names.contains(&"example.npz"));
        // E2VID's zip is write-only, so it is not offered.
        assert!(!names.iter().any(|name| name.ends_with(".zip")));
        home.selected_file = home
            .entries
            .iter()
            .position(|entry| matches!(entry, Entry::File(name, _) if name == "example.npz"))
            .unwrap();
        assert!(matches!(
            home.key(KeyCode::Enter),
            Action::Open(Target::File(_))
        ));
        assert!(matches!(home.key(KeyCode::Char('q')), Action::Quit));
    }
}
