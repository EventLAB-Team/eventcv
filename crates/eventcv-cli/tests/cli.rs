//! The `eventcv` binary end to end, on the repository's own test recording.

use std::path::{Path, PathBuf};
use std::process::{Command, Output, Stdio};

fn example() -> String {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../data/test/example.npz")
        .to_string_lossy()
        .into_owned()
}

fn eventcv(args: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_eventcv"))
        .args(args)
        .stdin(Stdio::null())
        .output()
        .unwrap()
}

fn stdout(output: &Output) -> String {
    String::from_utf8_lossy(&output.stdout).into_owned()
}

fn stderr(output: &Output) -> String {
    String::from_utf8_lossy(&output.stderr).into_owned()
}

/// A scratch directory unique to one test, so tests can run in parallel; removed when dropped.
struct Scratch(PathBuf);

impl Scratch {
    fn new(name: &str) -> Self {
        let dir = std::env::temp_dir().join(format!("eventcv-cli-{}-{name}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        Self(dir)
    }

    fn join(&self, file: &str) -> PathBuf {
        self.0.join(file)
    }
}

impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

#[test]
fn version_names_the_build() {
    let output = eventcv(&["--version"]);
    assert!(output.status.success());
    let text = stdout(&output);
    assert!(text.starts_with(&format!("eventcv {} (", env!("CARGO_PKG_VERSION"))));
    assert!(text.contains("standalone binary on"));
}

#[test]
fn no_command_prints_help() {
    let output = eventcv(&[]);
    assert!(output.status.success());
    assert!(stdout(&output).contains("Usage: eventcv"));
}

#[test]
fn info_summarises_a_recording() {
    let output = eventcv(&["info", &example(), "--rate-bin-ms", "5"]);
    assert!(output.status.success(), "{}", stderr(&output));
    let text = stdout(&output);
    for row in [
        "   sensor  640 x 480",
        "   events  106,295",
        " duration  0.050 s",
        "mean rate  2,137,958 ev/s",
        "     bins  10 x 5.0 ms",
    ] {
        assert!(text.contains(row), "missing {row:?} in\n{text}");
    }
}

#[test]
fn convert_round_trips() {
    let dir = Scratch::new("convert");
    let out = dir.join("copy.npz");
    let output = eventcv(&["convert", &example(), out.to_str().unwrap()]);
    assert!(output.status.success(), "{}", stderr(&output));
    assert_eq!(
        stdout(&output),
        format!("wrote {} (106,295 events, loaded)\n", out.display())
    );
    let info = stdout(&eventcv(&["info", out.to_str().unwrap()]));
    assert!(info.contains("events  106,295"));
}

#[test]
fn render_writes_an_animation() {
    let dir = Scratch::new("render");
    let out = dir.join("raw.gif");
    let output = eventcv(&["render", &example(), out.to_str().unwrap(), "--dt-ms", "10"]);
    assert!(output.status.success(), "{}", stderr(&output));
    assert_eq!(
        stdout(&output),
        format!("wrote {} (5 frames at 30 fps)\n", out.display())
    );
    assert!(std::fs::metadata(&out).unwrap().len() > 0);
}

#[test]
fn a_missing_file_names_itself() {
    let output = eventcv(&["info", "no-such-recording.npz"]);
    assert_eq!(output.status.code(), Some(1));
    let text = stderr(&output);
    assert!(text.starts_with("error: "), "{text}");
    assert!(text.contains("no-such-recording.npz"), "{text}");
}

#[test]
fn tui_refuses_a_pipe() {
    let output = eventcv(&["tui", &example()]);
    assert_eq!(output.status.code(), Some(1));
    assert_eq!(
        stderr(&output),
        "error: eventcv tui needs an interactive terminal\n"
    );
}

#[test]
fn tui_checks_its_arguments_first() {
    let output = eventcv(&["tui", &example(), "--repr", "nope"]);
    assert_eq!(output.status.code(), Some(1));
    assert_eq!(stderr(&output), "error: unknown representation: nope\n");
}

#[test]
fn simulate_streams_events_to_disk() {
    let ffmpeg = |args: &[&str]| Command::new("ffmpeg").args(args).output();
    if !ffmpeg(&["-version"]).is_ok_and(|output| output.status.success()) {
        eprintln!("skipped: no ffmpeg on PATH");
        return;
    }
    let dir = Scratch::new("simulate");
    let clip = dir.join("clip.mp4");
    let made = ffmpeg(&[
        "-loglevel",
        "error",
        "-y",
        "-f",
        "lavfi",
        "-i",
        "testsrc=size=64x48:rate=30",
        "-frames:v",
        "6",
        clip.to_str().unwrap(),
    ])
    .unwrap();
    assert!(made.status.success());
    let out = dir.join("events.npz");
    let output = eventcv(&["simulate", clip.to_str().unwrap(), out.to_str().unwrap()]);
    assert!(output.status.success(), "{}", stderr(&output));
    assert!(stdout(&output).ends_with("events from 6 frames)\n"));
}
