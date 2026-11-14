//! The standalone `eventcv` command: the same commands the Python package installs, for machines
//! that have no Python — installed by `scripts/install.sh` from the release binaries — plus the
//! terminal viewer, which the Python package runs from here too (`eventcv tui`).
//!
//! Every command that also exists in `python/eventcv/__main__.py` takes the same flags and prints
//! the same lines, so scripts and docs work against either. `play`, `reconstruct` and `ros2-*` stay
//! Python-only: they need the GPU viewer, an ONNX Runtime, or ROS.

mod camera;
mod commands;
pub mod tui;

use std::ffi::OsString;
use std::fmt::Display;
use std::io::{IsTerminal, Write};
use std::sync::atomic::{AtomicBool, Ordering};

use clap::{CommandFactory, Parser, Subcommand};
use eventcv_core::io::IoError;
use eventcv_core::representation::RepresentationError;

/// The optional features compiled into this binary, in a fixed order so `--version` is stable.
const FEATURES: &[&str] = &[
    #[cfg(feature = "hdf5")]
    "hdf5",
    #[cfg(feature = "camera")]
    "camera",
];

/// Set by the commands that stop cleanly on `Ctrl+C` (a recording keeps what it captured).
static GRACEFUL: AtomicBool = AtomicBool::new(false);
static INTERRUPTED: AtomicBool = AtomicBool::new(false);

/// The binary's `Ctrl+C` handler. Anything mid-way through a conversion or render exits at once
/// with the shell's 130, like the Python CLI; a [`graceful`] command is told instead, and a second
/// `Ctrl+C` still exits. Never installed inside Python, which has its own.
pub fn on_interrupt() {
    if GRACEFUL.load(Ordering::SeqCst) && !INTERRUPTED.swap(true, Ordering::SeqCst) {
        return;
    }
    eprintln!("interrupted");
    std::process::exit(130);
}

pub(crate) fn graceful() {
    GRACEFUL.store(true, Ordering::SeqCst);
}

pub(crate) fn interrupted() -> bool {
    INTERRUPTED.load(Ordering::SeqCst)
}

#[derive(Parser)]
#[command(
    name = "eventcv",
    about = "EventCV — OpenCV for event-based vision.",
    disable_version_flag = true,
    infer_long_args = true
)]
struct Cli {
    /// print the installed version and the features it was built with
    #[arg(short = 'V', long)]
    version: bool,
    #[command(subcommand)]
    command: Option<Command>,
}

#[derive(Subcommand)]
enum Command {
    /// summarise a recording (sensor size, event count, duration, rate)
    Info(commands::InfoArgs),
    /// convert a recording to another format (by output extension)
    Convert(commands::ConvertArgs),
    /// render a recording to .gif / .apng / .mp4 (.mp4 needs ffmpeg on PATH)
    Render(commands::RenderArgs),
    /// turn a video into synthetic events (needs ffmpeg on PATH)
    Simulate(commands::SimulateArgs),
    /// list connected USB event cameras
    Cameras,
    /// record a USB event camera to a file
    Record(camera::RecordArgs),
    /// watch a recording or a live camera in the terminal
    Tui(tui::TuiArgs),
}

/// Why a command failed, printed as `error: …` with exit status 1.
#[derive(Debug)]
pub(crate) struct Failure {
    message: String,
    /// A missing file: the OS message does not say *which*, so the input path is appended.
    not_found: bool,
}

impl From<String> for Failure {
    fn from(message: String) -> Self {
        Self {
            message,
            not_found: false,
        }
    }
}

impl From<&str> for Failure {
    fn from(message: &str) -> Self {
        message.to_owned().into()
    }
}

impl From<std::io::Error> for Failure {
    fn from(error: std::io::Error) -> Self {
        Self {
            not_found: error.kind() == std::io::ErrorKind::NotFound,
            message: error.to_string(),
        }
    }
}

impl From<IoError> for Failure {
    fn from(error: IoError) -> Self {
        match error {
            IoError::Io(error) => error.into(),
            other => other.to_string().into(),
        }
    }
}

impl From<RepresentationError> for Failure {
    fn from(error: RepresentationError) -> Self {
        error.to_string().into()
    }
}

/// Prints a line to stdout, ignoring a closed pipe: `eventcv info x | head -1` is not an error.
pub(crate) fn emit(line: impl Display) {
    let _ = writeln!(std::io::stdout().lock(), "{line}");
}

fn version_text() -> String {
    let features = if FEATURES.is_empty() {
        "no optional features".to_owned()
    } else {
        FEATURES.join(", ")
    };
    // The second line is what tells this binary apart from the pip-installed script when both are
    // on PATH — that one names the Python it runs under instead.
    format!(
        "eventcv {} ({features})\nstandalone binary on {}-{}",
        env!("CARGO_PKG_VERSION"),
        std::env::consts::OS,
        std::env::consts::ARCH
    )
}

/// Runs the command line in `argv` (program name first) and returns the exit status.
pub fn run<I, T>(argv: I) -> i32
where
    I: IntoIterator<Item = T>,
    T: Into<OsString> + Clone,
{
    let cli = match Cli::try_parse_from(argv) {
        Ok(cli) => cli,
        Err(error) => {
            let _ = error.print();
            return error.exit_code();
        }
    };
    if cli.version {
        emit(version_text());
        return 0;
    }
    let Some(command) = cli.command else {
        // At a terminal, a bare `eventcv` opens the home screen; anywhere else (a pipe, a script)
        // it prints the help it always has.
        if std::io::stdin().is_terminal() && std::io::stdout().is_terminal() {
            return tui::main(["eventcv"]);
        }
        let _ = Cli::command().print_help();
        return 0;
    };
    let (path, result) = match command {
        Command::Info(args) => (Some(args.path.clone()), commands::info(&args)),
        Command::Convert(args) => (Some(args.input.clone()), commands::convert(&args)),
        Command::Render(args) => (Some(args.input.clone()), commands::render(&args)),
        Command::Simulate(args) => (Some(args.input.clone()), commands::simulate(&args)),
        Command::Cameras => (None, camera::cameras()),
        Command::Record(args) => (None, camera::record(&args)),
        Command::Tui(args) => (args.path.clone(), tui::run(args)),
    };
    match result {
        Ok(()) => 0,
        Err(failure) => {
            report(failure, path.as_deref());
            1
        }
    }
}

pub(crate) fn report(failure: Failure, path: Option<&str>) {
    let mut message = failure.message;
    if let Some(path) = path.filter(|path| failure.not_found && !message.contains(path)) {
        message = format!("{message}: {path}");
    }
    eprintln!("error: {message}");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn command_line_is_well_formed() {
        Cli::command().debug_assert();
    }

    #[test]
    fn long_flags_take_unambiguous_prefixes() {
        // argparse accepts any unambiguous prefix, and the Python CLI's users rely on it.
        let cli = Cli::try_parse_from(["eventcv", "info", "rec.h5", "--rate", "5"]).unwrap();
        match cli.command {
            Some(Command::Info(args)) => assert_eq!(args.rate_bin_ms, Some(5.0)),
            _ => panic!("expected info"),
        }
    }

    #[test]
    fn only_a_missing_file_gets_its_path_appended() {
        let missing = IoError::Io(std::io::Error::from(std::io::ErrorKind::NotFound));
        assert!(Failure::from(missing).not_found);
        assert!(!Failure::from(IoError::Format("x.npz is broken".to_owned())).not_found);
    }
}
