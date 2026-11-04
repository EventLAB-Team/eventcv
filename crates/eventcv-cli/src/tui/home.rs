//! The home screen that a bare `eventcv` opens: the cameras plugged in, and the recordings in the
//! current folder, either of which opens in the viewer — which returns here when it is closed.
//!
//! ```text
//! ╭─ ◉ EventCV ─────────────────────────────────────────── v1.1.0 ─╮
//! │       ███████ ██    ██ ███████ ███    ██ ████████  ██████ ██    ██ │
//! │                 OpenCV for event-based vision                    │
//! │  ╭─ cameras ───────────────────╮╭─ recordings · ~/data ───────╮  │
//! │  │ ▶ iniVation DAVIS 346       ││   ..                        │  │
//! │  │   00000286 · super · 2:5    ││   test/                     │  │
//! ╰─ ↑↓ select · tab switch · enter open · ⌫ up · r refresh · q quit ─╯
//! ```

use std::path::PathBuf;

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
        };
        home.refresh();
        if !home.cameras.is_empty() {
            home.focus = Pane::Cameras;
        }
        home
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
        let [title, tagline, _, panes, status] = Layout::vertical([
            Constraint::Length(wordmark),
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
                "OpenCV for event-based vision",
                Style::new().fg(theme.muted).add_modifier(Modifier::ITALIC),
            ))
            .centered(),
            tagline,
        );

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

    fn wordmark(&self) -> Vec<Line<'static>> {
        (0..5)
            .map(|row| {
                let mut spans = Vec::new();
                for (index, letter) in WORDMARK.iter().enumerate() {
                    if index > 0 {
                        spans.push(Span::raw(" "));
                    }
                    let color = if index < 5 {
                        self.theme.on
                    } else {
                        self.theme.off
                    };
                    spans.push(Span::styled(letter[row], Style::new().fg(color)));
                }
                Line::from(spans)
            })
            .collect()
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
